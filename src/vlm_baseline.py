"""Zero-shot VLM baselines for well prediction (Gemini 3.1 Pro), FPV-only.

Two baselines, both with the same .predict(fpv_path, topview_path) interface as
the 3-stage EndToEndPipeline (topview is ignored — FPV-only by design):

  VLMMonolithicPipeline  — ONE call: N FPV frames -> find commit + read well.
  VLMDecomposedPipeline  — TWO calls: (1) pick dispensing frame from N FPV frames,
                           (2) read the well from that single full-res FPV frame.

Each .predict() returns a dict matching the 3-stage contract, plus per-call
timing and token cost:
  {clip_id_FPV, clip_id_Topview, wells_prediction, pipette_type, status,
   stages, timing{...,total}, cost{input_tokens,output_tokens,thinking_tokens,usd}}

Multi-channel handling is identical to the 3-stage pipeline: the model returns a
single anchor (row, col) + a pipette type, which is expanded to the full
row/column via expand_prediction — so scoring is apples-to-apples.
"""

import re
import time
from pathlib import Path

import cv2

from google import genai
from google.genai import types

PROJECT_DIR = Path(__file__).resolve().parent.parent

# ── Model + pricing (Gemini 3.1 Pro, per 1M tokens; output includes thinking) ──
GEMINI_MODEL = "gemini-3.1-pro-preview"
PRICE_IN_PER_M = 2.0
PRICE_OUT_PER_M = 12.0

# ── Frame sampling config ──
N_FRAMES = 8
MONO_MAX_DIM = 1024        # monolithic: 8 frames, downscaled to control tokens
SELECT_MAX_DIM = 768       # decomposed call 1: frame selection needs little detail
READ_MAX_DIM = None        # decomposed call 2: full-res single frame (the advantage)

MAX_RETRIES = 3


# ─────────────────────────── prompts ───────────────────────────

SYSTEM_PROMPT = """\
You are an expert at reading 96-well microplates from laboratory video.

Plate facts:
- 8 rows labeled A-H, ordered top to bottom.
- 12 columns labeled 1-12, ordered left to right.
- Well A1 is one corner; H12 is diagonally opposite.
- To orient the plate: read any printed labels on the plate edge (letters A-H,
  numbers 1-12) if legible; otherwise use the plate's notched/beveled corner,
  which marks A1.

Pipette types:
- single      -> dispenses into ONE well.
- 8-channel   -> dispenses into an entire COLUMN at once (8 wells, e.g. A5..H5).
- 12-channel  -> dispenses into an entire ROW at once (12 wells, e.g. C1..C12).

A "dispensing"/commit moment is when the pipette tip is lowered into a well and
releasing liquid - not merely approaching, hovering above, or retracting.
"""

MONO_PROMPT = """\
You are shown {n} time-ordered frames of a single pipetting action, captured from
a first-person (FPV) camera. The frames span the clip from approach through
dispensing to retraction.

Determine which well(s) the pipette dispenses into. Reason step by step:

1. ORIENTATION - Establish the plate's coordinate frame. State which on-screen
   direction is increasing column (1->12) and which is increasing row (A->H), and
   how you determined it (printed labels or beveled A1 corner).
2. COMMIT MOMENT - Identify which frame shows the tip lowered into a well and
   dispensing.
3. TIP LOCATION - At that moment, locate the pipette tip(s). Count columns from
   the left reference edge and rows from the top reference edge to the tip.
4. PIPETTE TYPE - single / 8-channel (full column) / 12-channel (full row).
5. ANSWER - Report the pipette_type and the (well_row, well_column) of the well
   the tip is centered over. For a multi-channel pipette, report any one well in
   the dispensed row/column.
"""

SELECT_PROMPT = """\
You are shown {n} time-ordered frames (indices 0-{last}) of a single pipetting
action, captured from a first-person (FPV) camera.

Identify the index of the frame that best shows the pipette ACTIVELY DISPENSING -
tip lowered into a well, releasing liquid - as opposed to approaching, hovering
above the plate, or retracting. If several qualify, choose the most clearly
committed one. Reason briefly, then give the index.
"""

READ_PROMPT = """\
This is the single dispensing moment of a pipetting action, shown at full
resolution from a first-person (FPV) camera.

Determine which well(s) the pipette is dispensing into. Reason step by step:

1. ORIENTATION - State which on-screen direction is increasing column (1->12) and
   increasing row (A->H), and how you determined it (printed labels / beveled A1
   corner).
2. TIP LOCATION - Locate the pipette tip(s). Count columns from the left
   reference edge and rows from the top reference edge to the tip.
3. PIPETTE TYPE - single / 8-channel (full column) / 12-channel (full row).
4. ANSWER - Report the pipette_type and the (well_row, well_column) of the well
   the tip is centered over. For a multi-channel pipette, report any one well in
   the dispensed row/column.
"""

ANSWER_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "reasoning": {"type": "STRING"},
        "pipette_type": {"type": "STRING", "enum": ["single", "8-channel", "12-channel"]},
        "well_row": {"type": "STRING"},
        "well_column": {"type": "STRING"},
    },
    "required": ["pipette_type", "well_row", "well_column"],
}

SELECT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "reasoning": {"type": "STRING"},
        "dispensing_frame_index": {"type": "INTEGER"},
    },
    "required": ["dispensing_frame_index"],
}

DEFAULT_WELL = [{"well_row": "D", "well_column": "6"}]  # match 3-stage fallback


# ─────────────────────────── helpers ───────────────────────────

def load_api_key():
    env_path = PROJECT_DIR / ".env"
    if env_path.exists():
        for line in open(env_path):
            m = re.match(r'\s*GEMINI_API_KEY\s*=\s*"?([^"\n]+)"?', line)
            if m:
                return m.group(1)
    import os
    return os.environ.get("GEMINI_API_KEY")


def _sample_indices(n_total, k):
    if n_total <= 0:
        return []
    if n_total <= k:
        return list(range(n_total))
    return [round(i * (n_total - 1) / (k - 1)) for i in range(k)]


def _downscale(frame, max_dim):
    if max_dim is None:
        return frame
    h, w = frame.shape[:2]
    s = max_dim / max(h, w)
    if s >= 1.0:
        return frame
    return cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def extract_fpv_frames(fpv_path, k=N_FRAMES, max_dim=None):
    """Return (list[(idx, jpeg_bytes)], total_frames)."""
    cap = cv2.VideoCapture(fpv_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in _sample_indices(n, k):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = _downscale(frame, max_dim)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            out.append((i, buf.tobytes()))
    cap.release()
    return out, n


def _map_type(vlm_type):
    """Map VLM type string to expand_prediction's vocabulary."""
    return {"single": "single",
            "12-channel": "multi-col",   # full row -> all 12 columns
            "8-channel": "multi-row"     # full column -> all 8 rows
            }.get(vlm_type, "single")


def expand_prediction(pred_row, pred_col, pipette_type):
    """Identical to the 3-stage's expansion (kept inline for a fair comparison)."""
    if pipette_type == "single":
        return [{"well_row": chr(pred_row + ord("A")), "well_column": str(pred_col + 1)}]
    elif pipette_type == "multi-col":
        row = chr(pred_row + ord("A"))
        return [{"well_row": row, "well_column": str(c)} for c in range(1, 13)]
    elif pipette_type == "multi-row":
        col = str(pred_col + 1)
        return [{"well_row": chr(r + ord("A")), "well_column": col} for r in range(8)]
    return [{"well_row": chr(pred_row + ord("A")), "well_column": str(pred_col + 1)}]


def _parse_anchor(data):
    """Parse {pipette_type, well_row, well_column} into (row_idx, col_idx, type_internal)."""
    vt = data.get("pipette_type", "single")
    row_s = str(data.get("well_row", "D")).strip().upper()[:1]
    col_s = re.sub(r"[^0-9]", "", str(data.get("well_column", "6"))) or "6"
    row = max(0, min(7, ord(row_s) - ord("A"))) if row_s.isalpha() else 3
    col = max(0, min(11, int(col_s) - 1))
    return row, col, _map_type(vt)


# ─────────────────────────── base ───────────────────────────

class _VLMBase:
    def __init__(self, model=GEMINI_MODEL, api_key=None, n_frames=N_FRAMES):
        self.model = model
        self.n_frames = n_frames
        self.client = genai.Client(api_key=api_key or load_api_key())

    def _call(self, parts, schema):
        """One Gemini call. Returns (data_dict, usd, usage_dict, latency_s)."""
        import json as _json
        cfg = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=schema,
        )
        last_err = None
        for attempt in range(MAX_RETRIES):
            t0 = time.perf_counter()
            try:
                r = self.client.models.generate_content(
                    model=self.model, contents=parts, config=cfg)
                latency = time.perf_counter() - t0
                um = r.usage_metadata
                in_tok = um.prompt_token_count or 0
                out_tok = um.candidates_token_count or 0
                think = getattr(um, "thoughts_token_count", None) or 0
                billed_out = out_tok + think
                usd = in_tok * PRICE_IN_PER_M / 1e6 + billed_out * PRICE_OUT_PER_M / 1e6
                usage = {"input_tokens": in_tok, "output_tokens": out_tok,
                         "thinking_tokens": think}
                data = _json.loads(r.text)
                return data, usd, usage, latency
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Gemini call failed after {MAX_RETRIES} tries: {last_err}")

    @staticmethod
    def _base_result(fpv_path, topview_path):
        return {
            "clip_id_FPV": Path(fpv_path).stem,
            "clip_id_Topview": Path(topview_path).stem,
            "wells_prediction": list(DEFAULT_WELL),
            "pipette_type": "single",
            "status": "ok",
            "stages": {},
            "timing": {},
            "cost": {"input_tokens": 0, "output_tokens": 0,
                     "thinking_tokens": 0, "usd": 0.0},
        }

    @staticmethod
    def _add_cost(result, usd, usage):
        c = result["cost"]
        c["input_tokens"] += usage["input_tokens"]
        c["output_tokens"] += usage["output_tokens"]
        c["thinking_tokens"] += usage["thinking_tokens"]
        c["usd"] = round(c["usd"] + usd, 6)


# ─────────────────────────── monolithic ───────────────────────────

class VLMMonolithicPipeline(_VLMBase):
    """Single call: N FPV frames -> find commit + read well."""

    def predict(self, fpv_path, topview_path, ground_truth=None, overlay_dir=None,
                pipette_type=None):
        result = self._base_result(fpv_path, topview_path)
        t_total = time.perf_counter()
        try:
            frames, _ = extract_fpv_frames(fpv_path, self.n_frames, MONO_MAX_DIM)
            parts = [MONO_PROMPT.format(n=len(frames))]
            for j, (_, jpg) in enumerate(frames):
                parts.append(f"Frame {j + 1}/{len(frames)}:")
                parts.append(types.Part.from_bytes(data=jpg, mime_type="image/jpeg"))
            data, usd, usage, latency = self._call(parts, ANSWER_SCHEMA)
            self._add_cost(result, usd, usage)
            result["timing"]["call_read"] = round(latency, 3)
            row, col, itype = _parse_anchor(data)
            used = pipette_type if pipette_type is not None else itype
            result["pipette_type"] = used
            result["wells_prediction"] = expand_prediction(row, col, used)
            result["stages"]["vlm"] = {
                "pred_row": row, "pred_col": col, "pred_type": itype,
                "reasoning": data.get("reasoning", "")[:1000],
            }
        except Exception as e:  # noqa: BLE001
            result["status"] = "vlm_fail"
            result["stages"]["vlm"] = {"error": str(e)[:300]}
        result["timing"]["total"] = round(time.perf_counter() - t_total, 3)
        return result


# ─────────────────────────── decomposed ───────────────────────────

class VLMDecomposedPipeline(_VLMBase):
    """Two calls: pick dispensing frame, then read well from that full-res frame."""

    def predict(self, fpv_path, topview_path, ground_truth=None, overlay_dir=None,
                pipette_type=None):
        result = self._base_result(fpv_path, topview_path)
        t_total = time.perf_counter()
        try:
            # Call 1: frame selection (downscaled frames)
            sel_frames, n_total = extract_fpv_frames(fpv_path, self.n_frames, SELECT_MAX_DIM)
            parts = [SELECT_PROMPT.format(n=len(sel_frames), last=len(sel_frames) - 1)]
            for j, (_, jpg) in enumerate(sel_frames):
                parts.append(f"Frame {j}:")
                parts.append(types.Part.from_bytes(data=jpg, mime_type="image/jpeg"))
            d1, usd1, usage1, lat1 = self._call(parts, SELECT_SCHEMA)
            self._add_cost(result, usd1, usage1)
            result["timing"]["call_select"] = round(lat1, 3)

            sel_j = int(d1.get("dispensing_frame_index", len(sel_frames) // 2))
            sel_j = max(0, min(len(sel_frames) - 1, sel_j))
            frame_idx = sel_frames[sel_j][0]
            result["stages"]["frame_select"] = {
                "selected_local": sel_j, "selected_frame_idx": frame_idx,
                "reasoning": d1.get("reasoning", "")[:500],
            }

            # Call 2: read well from the chosen frame at FULL resolution
            read_jpg = _read_single_frame(fpv_path, frame_idx, READ_MAX_DIM)
            parts2 = [READ_PROMPT,
                      types.Part.from_bytes(data=read_jpg, mime_type="image/jpeg")]
            d2, usd2, usage2, lat2 = self._call(parts2, ANSWER_SCHEMA)
            self._add_cost(result, usd2, usage2)
            result["timing"]["call_read"] = round(lat2, 3)

            row, col, itype = _parse_anchor(d2)
            used = pipette_type if pipette_type is not None else itype
            result["pipette_type"] = used
            result["wells_prediction"] = expand_prediction(row, col, used)
            result["stages"]["vlm"] = {
                "pred_row": row, "pred_col": col, "pred_type": itype,
                "reasoning": d2.get("reasoning", "")[:1000],
            }
        except Exception as e:  # noqa: BLE001
            result["status"] = "vlm_fail"
            result["stages"].setdefault("vlm", {})["error"] = str(e)[:300]
        result["timing"]["total"] = round(time.perf_counter() - t_total, 3)
        return result


def _read_single_frame(fpv_path, frame_idx, max_dim):
    cap = cv2.VideoCapture(fpv_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_idx} from {fpv_path}")
    frame = _downscale(frame, max_dim)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()
