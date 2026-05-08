"""Gemini 2.5 Pro protocol generator.

Note: gemini-3.1-pro-preview is the latest as of May 2026 but was 503ing on
all endpoints during our setup. We use 2.5 Pro because (a) it's stable GA,
(b) it's the exact baseline LabOS reports 2.86/5 on, so our results are
directly comparable to their published number. To swap to 3.x later,
change model_id below and update the registry name.
"""

from __future__ import annotations

from pathlib import Path

from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from eval.generators.base import ProtocolGenerator


class GeminiGenerator(ProtocolGenerator):
    name = "gemini-2.5-pro"
    model_id = "gemini-2.5-pro"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = genai.Client(api_key=self._require_env("GOOGLE_API_KEY"))

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def _call_api(self, prompt: str, frame_paths: list[Path]) -> str:
        parts: list[types.Part] = []
        for fp in frame_paths:
            parts.append(types.Part.from_bytes(
                data=fp.read_bytes(),
                mime_type="image/jpeg",
            ))
        parts.append(types.Part.from_text(text=prompt))

        resp = self._client.models.generate_content(
            model=self.model_id,
            contents=parts,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=self.max_tokens,
            ),
        )
        return resp.text or ""
