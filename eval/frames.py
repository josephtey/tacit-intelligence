"""Uniform frame extraction from video via ffmpeg.

Identical frames are passed to every generator, so the per-model comparison
isolates model capability rather than sampling pipeline."""

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRAME_CACHE = ROOT / "runs" / "frames"


def _probe_duration_seconds(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def extract_frames(
    video_path: str | Path,
    slice_id: str,
    n_frames: int = 32,
    max_dim: int = 1024,
) -> list[Path]:
    """Extract n_frames JPEGs uniformly spaced across the video.

    Cached: if the expected number of frames already exist on disk for this
    slice_id, returns the cached paths without re-extracting.
    """
    video = ROOT / video_path
    out_dir = FRAME_CACHE / slice_id
    out_dir.mkdir(parents=True, exist_ok=True)

    expected = [out_dir / f"{i:03d}.jpg" for i in range(n_frames)]
    if all(p.is_file() for p in expected):
        return expected

    # Wipe partial state and re-extract.
    for p in out_dir.glob("*.jpg"):
        p.unlink()

    duration = _probe_duration_seconds(video)
    # Sample at the midpoint of n_frames equal-width windows; avoids the very
    # first/last frames which often carry artifacts (caps off, hands not in shot).
    timestamps = [duration * (i + 0.5) / n_frames for i in range(n_frames)]

    for i, ts in enumerate(timestamps):
        out_path = out_dir / f"{i:03d}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", f"{ts:.3f}",
             "-i", str(video),
             "-frames:v", "1",
             "-vf", f"scale='min({max_dim},iw)':-2",
             "-q:v", "3",   # JPEG quality ~85
             str(out_path)],
            check=True,
        )

    return expected


def clear_cache(slice_id: str | None = None) -> None:
    if slice_id:
        target = FRAME_CACHE / slice_id
        if target.exists():
            shutil.rmtree(target)
    else:
        if FRAME_CACHE.exists():
            shutil.rmtree(FRAME_CACHE)
