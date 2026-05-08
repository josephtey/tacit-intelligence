"""Lightweight tests for the eval pipeline. No API calls.

Run with:  pytest -v
"""

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "runs" / "manifests" / "bench_50_seed42.json"
GEN_PROMPT = ROOT / "eval" / "prompts" / "generation.txt"
RUBRIC_PROMPT = ROOT / "eval" / "prompts" / "rubric.txt"


# --------- Manifest ---------

def test_manifest_exists():
    assert MANIFEST.is_file()


def test_manifest_has_50_records():
    records = json.loads(MANIFEST.read_text())
    assert len(records) == 50


def test_manifest_records_well_formed():
    records = json.loads(MANIFEST.read_text())
    required = {"slice_id", "subset", "video_path", "protocol_path", "scene"}
    for r in records:
        assert required.issubset(r.keys())
        assert r["scene"] == "bench"
        assert (ROOT / r["video_path"]).is_file()
        assert (ROOT / r["protocol_path"]).is_file()


def test_manifest_subset_breakdown():
    records = json.loads(MANIFEST.read_text())
    by_subset = {}
    for r in records:
        by_subset[r["subset"]] = by_subset.get(r["subset"], 0) + 1
    assert by_subset == {"XMglass": 20, "DJI": 20, "Multiview": 10}


# --------- Prompts ---------

def test_generation_prompt_exists():
    assert GEN_PROMPT.is_file()
    text = GEN_PROMPT.read_text()
    # The prompt must instruct numbered output and forbid hedging
    assert "Number every step" in text or "Number each step" in text
    assert "step 1" in text.lower()


def test_rubric_prompt_has_all_dimensions():
    text = RUBRIC_PROMPT.read_text()
    for dim in ["step_coverage", "step_hallucination", "ordering",
                "parameter_accuracy", "granularity_match"]:
        assert dim in text, f"Rubric missing dimension: {dim}"
    # Must use template placeholders for substitution
    assert "{gold_protocol}" in text
    assert "{predicted_protocol}" in text


# --------- Frame extraction ---------

def test_frame_extraction_one_video():
    from eval.frames import extract_frames

    records = json.loads(MANIFEST.read_text())
    rec = records[0]
    paths = extract_frames(
        video_path=rec["video_path"],
        slice_id=rec["slice_id"],
        n_frames=4,         # small budget for fast test
        max_dim=512,
    )
    assert len(paths) == 4
    for p in paths:
        assert p.is_file()
        assert p.stat().st_size > 1000   # >1KB; not empty


# --------- Judge JSON parser ---------

def test_judge_parser_extracts_clean_json():
    from eval.judge.base import ProtocolJudge

    raw = '{"step_coverage": {"score": 4, "reasoning": "x"}}'
    parsed = ProtocolJudge._parse_json(raw)
    assert parsed["step_coverage"]["score"] == 4


def test_judge_parser_handles_fenced_json():
    from eval.judge.base import ProtocolJudge

    raw = """Here's my evaluation:
```json
{"step_coverage": {"score": 3, "reasoning": "y"}}
```
"""
    parsed = ProtocolJudge._parse_json(raw)
    assert parsed["step_coverage"]["score"] == 3


def test_judge_parser_raises_on_no_json():
    from eval.judge.base import ProtocolJudge

    with pytest.raises(ValueError):
        ProtocolJudge._parse_json("no json here")


# --------- Registry ---------

def test_generator_registry_has_both_providers():
    from eval.generators import registry
    names = registry.names()
    assert "gpt-5.5" in names
    assert "gemini-2.5-pro" in names


def test_generator_instantiation_requires_api_key():
    """Instantiation should fail loudly if env var missing — not silently."""
    from eval.generators import registry

    saved = os.environ.pop("OPENAI_API_KEY", None)
    saved_dotenv_path = ROOT / ".env"
    # If .env is present and contains the key, we can't easily simulate "missing".
    # Skip cleanly in that case.
    if saved or saved_dotenv_path.is_file():
        if saved:
            os.environ["OPENAI_API_KEY"] = saved
        pytest.skip("OPENAI_API_KEY is set in env or .env — can't test missing-key path")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        registry.get("gpt-5.5")
