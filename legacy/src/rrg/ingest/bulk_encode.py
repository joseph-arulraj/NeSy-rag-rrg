"""Shared parallel image loading + CLEAR encoding for any script that needs embeddings for many
studies at once (offline/build_faiss_index.py, cbm/train.py, fusion/calibrate.py).

Image decoding (2500x3000 JPEGs) is the bottleneck, not the GPU forward pass -- decode/resize
must happen in DataLoader worker PROCESSES (runtime.num_workers), with only the model forward
running in the main process. A serial `[load_record_image(r, s) for r in batch]` loop, run in
the main process, leaves the GPU idle waiting on single-threaded JPEG decoding -- confirmed
directly on HPC (H100 at 0% util, ~4.8 img/s instead of the ~300 img/s build_faiss_index.py's
parallel loader achieves). This module exists so that mistake can only be made once.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..core.config import Settings
from ..core.errors import PipelineError
from ..datasets.mimic_cxr import StudyRecord
from ..ingest.image_loader import load_record_image
from ..perception.clear_encoder import ClearEncoder


class ImageDataset:
    """Map-style dataset (torch's Dataset protocol: __len__ / __getitem__). Returns (row, tensor,
    ok) -- decode failures are reported, not raised, so one bad file doesn't kill a worker."""

    def __init__(self, records: list[StudyRecord], settings: Settings, preprocess, shape: tuple[int, int, int]):
        self.records, self.settings, self.preprocess, self.shape = records, settings, preprocess, shape

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        import torch

        try:
            decoded, _ = load_record_image(self.records[i], self.settings)
            return i, self.preprocess(decoded.image), True
        except (PipelineError, OSError, ValueError):
            return i, torch.zeros(self.shape), False


def encode_records(
    records: list[StudyRecord],
    settings: Settings,
    encoder: ClearEncoder,
    log: Callable[[str], None] = print,
) -> tuple[np.ndarray, list[str]]:
    """[N] StudyRecord -> (embeddings [N, D] float32, failed_dicom_ids). Rows for failed decodes
    are left as all-zero -- caller decides whether that's acceptable (build_faiss_index.py fails
    the build past `retrieval.max_failed_fraction`; callers here should check similarly rather
    than silently training/calibrating on zero vectors)."""
    from PIL import Image
    from torch.utils.data import DataLoader

    rc = settings.retrieval
    shape = tuple(encoder.preprocess(Image.new("L", (64, 64))).shape)
    ds = ImageDataset(records, settings, encoder.preprocess, shape)
    nw = settings.runtime.num_workers
    loader = DataLoader(
        ds, batch_size=settings.clear.batch_size, shuffle=False, num_workers=nw,
        pin_memory=encoder.device.type == "cuda", persistent_workers=False,
    )

    n, d = len(records), settings.clear.embed_dim
    emb = np.zeros((n, d), dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
    import time
    t0 = time.time()
    done = 0
    for b, (rows, xs, good) in enumerate(loader, start=1):
        good = good.bool()
        if good.any():
            feats = encoder.encode_tensors(xs[good]).numpy().astype(np.float32, copy=False)
            sel = rows[good].numpy()
            emb[sel] = feats
            ok[sel] = True
        done += len(rows)
        if b % rc.log_every_batches == 0 or done == n:
            rate = done / max(time.time() - t0, 1e-9)
            log(f"  encoded {done:,}/{n:,}  ({rate:,.1f} img/s, eta {(n - done) / max(rate, 1e-9) / 60:.1f} min)")

    failed = [records[i].dicom_id for i in np.flatnonzero(~ok)]
    return emb, failed
