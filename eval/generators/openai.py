"""OpenAI GPT-5.5 protocol generator."""

from __future__ import annotations

import base64
from pathlib import Path

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from eval.generators.base import ProtocolGenerator


class OpenAIGenerator(ProtocolGenerator):
    name = "gpt-5.5"
    model_id = "gpt-5.5"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = OpenAI(api_key=self._require_env("OPENAI_API_KEY"))

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def _call_api(self, prompt: str, frame_paths: list[Path]) -> str:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for fp in frame_paths:
            b64 = base64.b64encode(fp.read_bytes()).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""
