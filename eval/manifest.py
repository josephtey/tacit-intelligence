"""Build a deterministic 50-video evaluation manifest from LSV.

Filters: Scene == "bench" AND Protocol field non-empty AND protocol file
exists on disk. Stratifies across XMglass / DJI / Multiview to maximize
procedural diversity (Multiview clips are time-aligned multi-phone recordings
of just 2 procedures, so we cap and dedupe them).

Output: runs/manifests/bench_50_seed42.json — one JSON list of records, each:
  {
    "slice_id": "XM_005",
    "subset": "XMglass",
    "video_path": "data/lsv/XMglass/XMvideo/VID_....mp4",
    "protocol_path": "data/lsv/XMglass/XMprotocol/PCR_Reaction_Setup_XW.txt",
    "operation": "PCR Reaction Setup",
    "length": "10'",
    "scene": "bench"
  }
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LSV = ROOT / "data" / "lsv"
OUT = ROOT / "runs" / "manifests" / "bench_50_seed42.json"

SUBSETS = {
    "XMglass":   {"csv": "XMglass/xm.csv",       "proto_dir": "XMglass/XMprotocol",   "video_dir": "XMglass/XMvideo",   "n": 20},
    "DJI":       {"csv": "DJI/dji.csv",          "proto_dir": "DJI/DJI-Protocol",     "video_dir": "DJI/DJI-Video",     "n": 20},
    "Multiview": {"csv": "Multiview/multi.csv",  "proto_dir": "Multiview/Protocols",  "video_dir": "Multiview/Videos",  "n": 10},
}

SEED = 42
MIN_PROTOCOL_CHARS = 50  # filter out stub protocol files


def load_subset(name: str, spec: dict) -> pd.DataFrame:
    df = pd.read_csv(LSV / spec["csv"])
    df.columns = [c.strip() for c in df.columns]
    df = df.assign(subset=name)

    df = df[df["Scene"].astype(str).str.lower().str.strip() == "bench"]
    df = df[df["Protocol"].fillna("").str.strip().ne("")]

    proto_dir = LSV / spec["proto_dir"]
    video_dir = LSV / spec["video_dir"]

    def video_path(name):
        p = video_dir / str(name)
        return str(p.relative_to(ROOT)) if p.is_file() else None

    def protocol_path(name):
        p = proto_dir / str(name)
        if not p.is_file() or p.stat().st_size < MIN_PROTOCOL_CHARS:
            return None
        return str(p.relative_to(ROOT))

    df["video_path"] = df["Video Name"].map(video_path)
    df["protocol_path"] = df["Protocol"].map(protocol_path)
    df = df[df["video_path"].notna() & df["protocol_path"].notna()]
    return df


def sample_subset(df: pd.DataFrame, n: int, name: str) -> pd.DataFrame:
    """Sample n rows, deduplicating multi-phone clips within Multiview."""
    if name == "Multiview":
        # Each (Operation, Length) tuple is one time-segment recorded by multiple phones.
        # Keep one phone per tuple, then sample.
        df = df.drop_duplicates(subset=["Operation", "Length"], keep="first")
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=SEED)


def to_record(row: pd.Series) -> dict:
    return {
        "slice_id": str(row["Slice_ID"]).strip(),
        "subset": row["subset"],
        "video_path": row["video_path"],
        "protocol_path": row["protocol_path"],
        "operation": str(row.get("Operation", "")).strip() or None,
        "length": str(row.get("Length", "")).strip() or None,
        "scene": str(row["Scene"]).strip(),
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    breakdown: dict[str, int] = {}
    for name, spec in SUBSETS.items():
        df = load_subset(name, spec)
        sampled = sample_subset(df, spec["n"], name)
        breakdown[name] = (len(df), len(sampled))
        records.extend(sampled.apply(to_record, axis=1).tolist())

    records.sort(key=lambda r: (r["subset"], r["slice_id"]))

    OUT.write_text(json.dumps(records, indent=2))

    print(f"Wrote {len(records)} records to {OUT.relative_to(ROOT)}")
    print(f"Seed: {SEED}")
    print(f"\nBreakdown (eligible → sampled):")
    for name, (eligible, sampled) in breakdown.items():
        print(f"  {name:10s}: {eligible:3d} eligible, {sampled:3d} sampled")
    print(f"  {'TOTAL':10s}: {sum(e for e, _ in breakdown.values()):3d} eligible, {len(records):3d} sampled")
    if len(records) < 50:
        print(f"\nWARNING: sampled {len(records)} < 50", file=sys.stderr)


if __name__ == "__main__":
    main()
