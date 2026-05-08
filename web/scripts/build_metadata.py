"""Build web/public/metadata.json from the three LSV CSVs.

Resolves Protocol filename -> protocol text (read from the matching *Protocol*/ folder).
Marks each entry's media type (mp4 / jpg / other) so the UI can render appropriately.
"""

import csv
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LSV = REPO_ROOT / "data" / "lsv"
OUT = REPO_ROOT / "web" / "public" / "metadata.json"

CONFIGS = [
    {
        "key": "XMglass",
        "csv": LSV / "XMglass" / "xm.csv",
        "video_dir": LSV / "XMglass" / "XMvideo",
        "protocol_dir": LSV / "XMglass" / "XMprotocol",
    },
    {
        "key": "DJI",
        "csv": LSV / "DJI" / "dji.csv",
        "video_dir": LSV / "DJI" / "DJI-Video",
        "protocol_dir": LSV / "DJI" / "DJI-Protocol",
    },
    {
        "key": "Multiview",
        "csv": LSV / "Multiview" / "multi.csv",
        "video_dir": LSV / "Multiview" / "Videos",
        "protocol_dir": LSV / "Multiview" / "Protocols",
    },
]


def media_kind(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in {".mp4", ".mov", ".m4v"}:
        return "video"
    if ext in {".jpg", ".jpeg", ".png"}:
        return "image"
    return "other"


def load_protocol(protocol_dir: Path, name: str) -> str | None:
    if not name:
        return None
    p = protocol_dir / name
    if p.is_file():
        return p.read_text(encoding="utf-8", errors="replace")
    # Some rows reference a .docx; try the .txt sibling.
    txt = p.with_suffix(".txt")
    if txt.is_file():
        return txt.read_text(encoding="utf-8", errors="replace")
    return None


def normalize(row: dict) -> dict:
    return {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k}


entries: list[dict] = []
protocol_texts: dict[str, dict[str, str]] = {}

for cfg in CONFIGS:
    if not cfg["csv"].is_file():
        print(f"warn: missing csv {cfg['csv']}")
        continue
    protocol_texts[cfg["key"]] = {}
    with cfg["csv"].open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            row = normalize(raw)
            slice_id = row.get("Slice_ID")
            video_name = row.get("Video Name") or ""
            if not slice_id or not video_name:
                continue
            video_path = cfg["video_dir"] / video_name
            kind = media_kind(video_name)
            protocol_name = row.get("Protocol") or ""
            protocol_text = load_protocol(cfg["protocol_dir"], protocol_name)
            if protocol_text and protocol_name:
                protocol_texts[cfg["key"]][protocol_name] = protocol_text
            entries.append(
                {
                    "config": cfg["key"],
                    "slice_id": slice_id,
                    "video_name": video_name,
                    "video_path": str(video_path),
                    "media_kind": kind,
                    "video_exists": video_path.is_file(),
                    "scene": row.get("Scene") or "",
                    "operation": row.get("Operation") or "",
                    "protocol_name": protocol_name,
                    "has_protocol": bool(protocol_text),
                    "issue": row.get("Issue (if any)") or "",
                    "length": row.get("Length") or "",
                    "time_stamp": row.get("Time_stamp") or "",
                    "tools": row.get("Tools") or "",
                    "gpt4o_output": row.get("GPT4o_output") or "",
                    "date": row.get("Date") or "",
                }
            )

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    json.dumps({"entries": entries, "protocols": protocol_texts}, indent=2),
    encoding="utf-8",
)

by_cfg = {}
for e in entries:
    by_cfg.setdefault(e["config"], {"total": 0, "playable": 0, "with_protocol": 0, "with_gpt4o": 0})
    s = by_cfg[e["config"]]
    s["total"] += 1
    if e["media_kind"] == "video" and e["video_exists"]:
        s["playable"] += 1
    if e["has_protocol"]:
        s["with_protocol"] += 1
    if e["gpt4o_output"]:
        s["with_gpt4o"] += 1

print(f"wrote {OUT} ({len(entries)} entries)")
for k, v in by_cfg.items():
    print(f"  {k}: {v}")
