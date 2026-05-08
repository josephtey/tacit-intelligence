"""Abstract base class for protocol generators.

A ProtocolGenerator takes the path to a video and a prompt template, sends
those to a VLM, and returns the generated stepwise protocol as plain text.
Concrete subclasses handle SDK-specific details for OpenAI, Gemini, etc.

The base class also handles frame extraction (delegated to eval.frames) and
on-disk caching of outputs, so a subclass only needs to implement the actual
API call: _call_api(self, prompt, frame_paths) -> str.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

from dotenv import load_dotenv

from eval.frames import extract_frames

ROOT = Path(__file__).resolve().parent.parent.parent
PRED_DIR = ROOT / "runs" / "predictions"

load_dotenv(ROOT / ".env")


class ProtocolGenerator(ABC):
    """Subclasses set these as class attributes."""
    name: str           # e.g., "gpt-5"  — used as the directory name
    model_id: str       # e.g., "gpt-5"  — exact model string passed to the API

    def __init__(self, n_frames: int = 32, max_dim: int = 1024, max_tokens: int = 4096):
        self.n_frames = n_frames
        self.max_dim = max_dim
        self.max_tokens = max_tokens

    def generate(self, slice_id: str, video_path: str, prompt: str) -> dict:
        """Generate a protocol for one video. Cached on disk."""
        cache_key = self._cache_key(prompt)
        out_path = PRED_DIR / self.name / f"{slice_id}__{cache_key}.json"
        if out_path.is_file():
            return json.loads(out_path.read_text())

        frames = extract_frames(
            video_path=video_path,
            slice_id=slice_id,
            n_frames=self.n_frames,
            max_dim=self.max_dim,
        )
        protocol_text = self._call_api(prompt=prompt, frame_paths=frames)

        record = {
            "slice_id": slice_id,
            "model": self.name,
            "model_id": self.model_id,
            "n_frames": self.n_frames,
            "max_dim": self.max_dim,
            "max_tokens": self.max_tokens,
            "prompt_hash": cache_key,
            "protocol": protocol_text,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2))
        return record

    @abstractmethod
    def _call_api(self, prompt: str, frame_paths: list[Path]) -> str:
        """Send prompt + frames to the underlying API. Return raw text."""

    def _cache_key(self, prompt: str) -> str:
        h = hashlib.sha1()
        h.update(prompt.encode())
        h.update(f"|n={self.n_frames}|d={self.max_dim}|t={self.max_tokens}|m={self.model_id}".encode())
        return h.hexdigest()[:12]

    @staticmethod
    def _require_env(var: str) -> str:
        val = os.environ.get(var)
        if not val:
            raise RuntimeError(f"Environment variable {var} is not set. Check .env.")
        return val
