"""Claude Opus 4.7 judge."""

from __future__ import annotations

from anthropic import Anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from eval.judge.base import ProtocolJudge


class ClaudeJudge(ProtocolJudge):
    name = "claude-opus-4-7"
    model_id = "claude-opus-4-7"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = Anthropic(api_key=self._require_env("ANTHROPIC_API_KEY"))

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def _call_api(self, prompt: str) -> str:
        resp = self._client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if hasattr(b, "text"))
