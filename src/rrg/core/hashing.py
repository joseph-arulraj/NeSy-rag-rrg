"""Content hashes for the version manifest (docs/pipeline.md 7.1)."""
from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: str | Path, chunk_bytes: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_bytes)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
