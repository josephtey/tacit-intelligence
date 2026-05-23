"""Perception tools the agent can call against extracted frames.

These are deliberately backed by off-the-shelf models for the first working
pipeline (tesseract OCR + region cropping/upscaling). The interface is designed
so a FineBio-trained detector can be added later as another tool without
changing the agent loop.

A frame "region" is one of a small fixed vocabulary so the LLM can address
parts of a frame reliably without juggling pixel coordinates.
"""

from __future__ import annotations

import base64
import io
import threading
from pathlib import Path

import pytesseract
from PIL import Image

REGIONS = ["full", "center", "top-left", "top-right", "bottom-left", "bottom-right"]

# Default open-vocabulary lab equipment queries for GroundingDINO. Period-separated
# phrases, lowercase — the format the model expects.
LAB_QUERIES = (
    "a gloved hand. a pipette. a multichannel pipette. a microcentrifuge tube. "
    "a tube rack. a petri dish. a cell culture plate. a reagent bottle. a flask. "
    "a centrifuge. a vortex mixer. a marker pen. a pipette tip box. a beaker. "
    "a gel cassette. a serological pipette."
)

# Lazy GroundingDINO singleton (loaded once, guarded for thread-safe inference).
_GD_MODEL = None
_GD_PROCESSOR = None
_GD_LOCK = threading.Lock()
_GD_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


def _load_grounding_dino():
    """Thread-safe lazy load (double-checked locking).

    Loading must be serialized: concurrent first-use from multiple threads
    otherwise races the import and the meta-tensor → device copy, which fails.
    """
    global _GD_MODEL, _GD_PROCESSOR
    if _GD_MODEL is not None:
        return _GD_MODEL, _GD_PROCESSOR
    with _GD_LOCK:
        if _GD_MODEL is None:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            processor = AutoProcessor.from_pretrained(_GD_MODEL_ID)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(_GD_MODEL_ID).eval()
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
            _GD_PROCESSOR, _GD_MODEL = processor, model
    return _GD_MODEL, _GD_PROCESSOR


def warmup() -> None:
    """Force the detector to load now (single-threaded), before any pool starts."""
    _load_grounding_dino()


def _region_box(w: int, h: int, region: str) -> tuple[int, int, int, int]:
    """Pixel box (left, top, right, bottom) for a named region.

    Quadrants overlap the center slightly so a label straddling the midline
    isn't sliced in half.
    """
    cx, cy = w // 2, h // 2
    mx, my = int(w * 0.05), int(h * 0.05)  # small margin
    return {
        "full":         (0, 0, w, h),
        "center":       (w // 4, h // 4, 3 * w // 4, 3 * h // 4),
        "top-left":     (0, 0, cx + mx, cy + my),
        "top-right":    (cx - mx, 0, w, cy + my),
        "bottom-left":  (0, cy - my, cx + mx, h),
        "bottom-right": (cx - mx, cy - my, w, h),
    }[region]


def _crop(frame_path: Path, region: str, upscale: int = 2) -> Image.Image:
    img = Image.open(frame_path).convert("RGB")
    box = _region_box(img.width, img.height, region)
    crop = img.crop(box)
    if upscale != 1:
        crop = crop.resize((crop.width * upscale, crop.height * upscale), Image.LANCZOS)
    return crop


def zoom_in(frame_path: Path, region: str = "center", upscale: int = 2) -> str:
    """Return an upscaled crop of `region` as a base64 data URL (for re-feeding to the VLM)."""
    if region not in REGIONS:
        region = "center"
    crop = _crop(frame_path, region, upscale)
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def read_text(frame_path: Path, region: str = "full") -> str:
    """OCR the given region. Returns extracted text, or a sentinel if none found."""
    if region not in REGIONS:
        region = "full"
    crop = _crop(frame_path, region, upscale=2)
    text = pytesseract.image_to_string(crop).strip()
    if not text:
        return "(no text detected)"
    # Collapse whitespace noise that tesseract loves to emit.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines) if lines else "(no text detected)"


def detect_objects(frame_path: Path, box_threshold: float = 0.30, text_threshold: float = 0.25) -> str:
    """Open-vocabulary object detection over a fixed lab-equipment vocabulary.

    Returns a text list of detections (label, confidence, box) the reasoner can
    consume — e.g. "gloved hand (0.83) at [1,369,236,550]". Boxes are in the
    coordinate space of the (1024px) frame the agent is viewing.
    """
    import torch

    model, processor = _load_grounding_dino()
    img = Image.open(frame_path).convert("RGB")
    with _GD_LOCK:
        inputs = processor(images=img, text=LAB_QUERIES, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        res = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=[img.size[::-1]],
        )[0]

    dets = sorted(
        zip(res["labels"], res["scores"].tolist(), res["boxes"].tolist()),
        key=lambda x: -x[1],
    )
    if not dets:
        return "(no objects detected)"
    lines = []
    for lbl, score, box in dets:
        box_s = ",".join(f"{c:.0f}" for c in box)
        lines.append(f"- {lbl} ({score:.2f}) at [{box_s}]")
    return "\n".join(lines)
