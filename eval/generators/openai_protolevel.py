"""GPT-5.5 baseline with the agent's protocol-level prompt (no tools).

Exists purely for the prompt-isolation ablation: it differs from the plain
gpt-5.5 generator ONLY in the prompt (protocol-level guidance, the non-tool
part of the agent's instructions). Comparing
    gpt-5.5  <  gpt-5.5-protolevel  <  gpt-5.5-agent
separates the prompt effect from the perception-tool effect.
"""

from __future__ import annotations

from pathlib import Path

from eval.generators.openai import OpenAIGenerator

ROOT = Path(__file__).resolve().parent.parent.parent
_PROMPT = (ROOT / "eval" / "prompts" / "generation_protocol_level.txt").read_text()


class OpenAIProtoLevelGenerator(OpenAIGenerator):
    name = "gpt-5.5-protolevel"
    model_id = "gpt-5.5"

    def _call_api(self, prompt: str, frame_paths):
        # Ignore the harness-supplied prompt; use the protocol-level prompt.
        return super()._call_api(_PROMPT, frame_paths)

    def _cache_key(self, prompt: str) -> str:
        return super()._cache_key(_PROMPT)
