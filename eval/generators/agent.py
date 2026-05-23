"""Tool-calling agent generator.

A frontier reasoner (GPT-5.5) is given the 32 frames plus two perception tools
— zoom_in (upscaled crop) and read_text (OCR) — and runs an agentic loop:
look, zoom/OCR where it needs to read labels or parameters, then write the
protocol. This is the first instantiation of the compound-system bet; the
perception tools are off-the-shelf for now (tesseract OCR + crop) and the
interface is built so FineBio-trained detectors can be added as more tools
without touching the loop.

The final protocol is returned to the base class for caching/scoring exactly
like any other generator. The full tool-call trace is written to a sidecar at
runs/agent_traces/{slice_id}.json for inspection.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from eval.agent import perception
from eval.agent.perception import REGIONS, detect_objects, zoom_in
from eval.generators.base import ProtocolGenerator

ROOT = Path(__file__).resolve().parent.parent.parent
TRACE_DIR = ROOT / "runs" / "agent_traces"

AGENT_INSTRUCTIONS = """You are reconstructing a stepwise experimental protocol from frames of a wet-lab biology session, sampled in chronological order.

You have two perception tools to help you ground what you see:
- detect_objects(frame_index): runs an object detector over a frame and returns the lab equipment present (hands, pipettes, tubes, racks, petri dishes, bottles, centrifuge, etc.) with confidence and bounding boxes. Use this to know which instruments and vessels are actually in play at each moment.
- zoom_in(frame_index, region): get an upscaled close-up of a region to inspect fine detail. region is one of: full, center, top-left, top-right, bottom-left, bottom-right.

Use the detector to ground each step in the actual equipment and vessels present, and zoom_in to inspect detail. Identify equipment, vessels, and any visible parameters; name reagents when you have visual evidence for them, but do not invent specific reagent names you cannot justify.

When you have gathered enough detail, output the final protocol. Output format:
- Number every step starting from 1.
- Each step: one short paragraph, a single discrete action, with reagents/materials and any observable parameters (volumes, durations, temperatures, equipment, concentrations).
- Write at protocol level, not visual-narration level. Do not split one logical step ("add reagent to tube") into many micro-actions ("pick up pipette", "attach tip", ...).
- No hedging ("appears to"), no preamble, no closing remarks. Output only the numbered steps.
"""

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "zoom_in",
            "description": "Upscaled close-up crop of a region of a frame, to see fine detail.",
            "parameters": {
                "type": "object",
                "properties": {
                    "frame_index": {"type": "integer", "description": "0-based frame index"},
                    "region": {"type": "string", "enum": REGIONS},
                },
                "required": ["frame_index", "region"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_objects",
            "description": "Run an object detector over a frame; returns lab equipment present (label, confidence, bounding box).",
            "parameters": {
                "type": "object",
                "properties": {
                    "frame_index": {"type": "integer", "description": "0-based frame index"},
                },
                "required": ["frame_index"],
            },
        },
    },
]


class AgentGenerator(ProtocolGenerator):
    name = "gpt-5.5-agent"
    model_id = "gpt-5.5"

    def __init__(self, max_iters: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.max_iters = max_iters
        self._client = OpenAI(api_key=self._require_env("OPENAI_API_KEY"))
        perception.warmup()  # load the detector single-threaded before any pool starts

    def _cache_key(self, prompt: str) -> str:
        # Fold the agent's actual behavior (instructions + tool set + loop depth)
        # into the cache key so changing any of them invalidates old predictions.
        import hashlib
        tool_names = ",".join(t["function"]["name"] for t in _TOOLS)
        cfg = hashlib.sha1((AGENT_INSTRUCTIONS + tool_names).encode()).hexdigest()[:8]
        return super()._cache_key(prompt + f"|agent|iters={self.max_iters}|cfg={cfg}")

    @staticmethod
    def _data_url(path: Path) -> str:
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def _chat(self, messages, use_tools: bool):
        return self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            tools=_TOOLS if use_tools else None,
            tool_choice="auto" if use_tools else None,
            max_completion_tokens=self.max_tokens,
        )

    def _call_api(self, prompt: str, frame_paths: list[Path]) -> str:
        slice_id = frame_paths[0].parent.name
        n = len(frame_paths)

        content: list[dict] = [{"type": "text", "text": AGENT_INSTRUCTIONS}]
        for i, fp in enumerate(frame_paths):
            content.append({"type": "text", "text": f"Frame {i}:"})
            content.append({"type": "image_url", "image_url": {"url": self._data_url(fp)}})
        messages: list = [{"role": "user", "content": content}]

        trace: list[dict] = []
        final_text = ""

        for _ in range(self.max_iters):
            resp = self._chat(messages, use_tools=True)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                final_text = msg.content or ""
                break
            messages.append(msg.model_dump(exclude_none=True))
            # OpenAI requires every tool_call_id to get a tool response immediately,
            # before any other message type. So append all tool responses first,
            # then any zoom crops as a single trailing user message.
            pending_images: list[tuple[int, str, str]] = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                fidx = max(0, min(int(args.get("frame_index", 0)), n - 1))
                region = args.get("region", "center")
                if tc.function.name == "detect_objects":
                    result = detect_objects(frame_paths[fidx])
                    trace.append({"tool": "detect_objects", "frame": fidx, "result": result})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                elif tc.function.name == "zoom_in":
                    crop_url = zoom_in(frame_paths[fidx], region)
                    trace.append({"tool": "zoom_in", "frame": fidx, "region": region})
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": f"Zoomed crop of frame {fidx} ({region}) attached below."})
                    pending_images.append((fidx, region, crop_url))
                else:
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": "unknown tool"})

            if pending_images:
                img_content: list[dict] = []
                for fidx, region, url in pending_images:
                    img_content.append({"type": "text", "text": f"Zoomed crop of frame {fidx} ({region}):"})
                    img_content.append({"type": "image_url", "image_url": {"url": url}})
                messages.append({"role": "user", "content": img_content})

        if not final_text:
            # Ran out of iterations — force a final answer without tools.
            messages.append({"role": "user", "content": "Stop using tools and output the final numbered protocol now."})
            resp = self._chat(messages, use_tools=False)
            final_text = resp.choices[0].message.content or ""

        self._save_trace(slice_id, trace, final_text)
        return final_text

    def _save_trace(self, slice_id: str, trace: list[dict], final_text: str) -> None:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        (TRACE_DIR / f"{slice_id}.json").write_text(json.dumps({
            "slice_id": slice_id,
            "model": self.name,
            "n_tool_calls": len(trace),
            "trace": trace,
            "final_protocol": final_text,
        }, indent=2))
