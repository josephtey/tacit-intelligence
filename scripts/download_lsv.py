"""Resilient downloader for the labos1/LSV dataset.

snapshot_download is idempotent and resumes from .incomplete files, so this
just wraps it in an outer retry loop for transient network errors on a
~319 GB pull.
"""

import os
import sys
import time

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

from huggingface_hub import snapshot_download

REPO_ID = "labos1/LSV"
LOCAL_DIR = "/large_storage/hsulab/joetey/tacit-intelligence/lsv"
MAX_WORKERS = 4
MAX_ATTEMPTS = 50

for attempt in range(1, MAX_ATTEMPTS + 1):
    try:
        path = snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            max_workers=MAX_WORKERS,
        )
        print(f"DONE attempt={attempt} path={path}", flush=True)
        sys.exit(0)
    except Exception as e:
        backoff = min(60, 5 * attempt)
        print(
            f"attempt={attempt} failed: {type(e).__name__}: {e}; "
            f"retrying in {backoff}s",
            flush=True,
        )
        time.sleep(backoff)

print("FAILED after max attempts", flush=True)
sys.exit(1)
