"""Abstract base class for protocol judges.

A ProtocolJudge takes a predicted protocol and a gold protocol (both as text)
and returns a structured score against a published rubric. Concrete subclasses
implement the actual API call.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
SCORE_DIR = ROOT / "runs" / "scores"

load_dotenv(ROOT / ".env")

DIMENSIONS = [
    "step_coverage",
    "step_hallucination",
    "ordering",
    "parameter_accuracy",
    "granularity_match",
]


class ProtocolJudge(ABC):
    name: str       # e.g., "claude-opus-4-7"
    model_id: str

    def __init__(self, max_tokens: int = 2048):
        self.max_tokens = max_tokens

    def score(
        self,
        slice_id: str,
        generator_name: str,
        predicted_protocol: str,
        gold_protocol: str,
        rubric_template: str,
    ) -> dict:
        cache_key = self._cache_key(predicted_protocol, gold_protocol, rubric_template)
        out_path = SCORE_DIR / self.name / generator_name / f"{slice_id}__{cache_key}.json"
        if out_path.is_file():
            return json.loads(out_path.read_text())

        prompt = rubric_template.format(
            gold_protocol=gold_protocol.strip(),
            predicted_protocol=predicted_protocol.strip(),
        )
        raw = self._call_api(prompt=prompt)
        parsed = self._parse_json(raw)
        composite = sum(parsed[d]["score"] for d in DIMENSIONS) / len(DIMENSIONS)

        record = {
            "slice_id": slice_id,
            "generator": generator_name,
            "judge": self.name,
            "judge_model_id": self.model_id,
            "rubric_hash": cache_key,
            "scores": {d: parsed[d] for d in DIMENSIONS},
            "composite": round(composite, 3),
            "summary": parsed.get("summary", ""),
            "raw": raw,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2))
        return record

    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """Send prompt to the judge model. Return raw text."""

    def _cache_key(self, predicted: str, gold: str, rubric: str) -> str:
        h = hashlib.sha1()
        h.update(predicted.encode())
        h.update(b"|||")
        h.update(gold.encode())
        h.update(b"|||")
        h.update(rubric.encode())
        h.update(f"|m={self.model_id}|t={self.max_tokens}".encode())
        return h.hexdigest()[:12]

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Extract a JSON object from the model's response.

        Models often wrap JSON in ```json ... ``` fences or add prose.
        This finds the first {...} block and parses it.
        """
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError(f"No JSON object found in judge output:\n{raw[:500]}")
        return json.loads(raw[start : end + 1])

    @staticmethod
    def _require_env(var: str) -> str:
        val = os.environ.get(var)
        if not val:
            raise RuntimeError(f"Environment variable {var} is not set. Check .env.")
        return val
