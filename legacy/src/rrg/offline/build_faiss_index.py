"""Offline (build once): encode every TRAIN study's one frontal image with the frozen CLEAR encoder in fp32,
L2-normalise, and write the exact FAISS inner-product index + id table + manifest.

Image decoding is the bottleneck (2500x3000 JPEGs), so images are decoded / preprocessed in DataLoader
workers (runtime.num_workers) and only the model forward runs in the main process. Images that cannot be
decoded are recorded in the manifest; the build aborts if more than retrieval.max_failed_fraction fail.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from ..core.config import Settings
from ..core.errors import IndexBuildError, PipelineError
from ..datasets.mimic_cxr import MimicCxrIndex, StudyRecord
from ..ingest.image_loader import load_record_image
from ..perception.clear_encoder import ClearEncoder
from ..retrieval.faiss_index import FaissRetriever, IdTable, IndexManifest, write_index


class _ImageDataset:
    """Map-style dataset (torch's Dataset protocol: __len__ / __getitem__). Returns (row, tensor, ok)."""

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


def build_faiss_index(
    settings: Settings,
    device=None,
    limit: Optional[int] = None,
    log: Callable[[str], None] = print,
) -> IndexManifest:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader

    rc = settings.retrieval
    index = MimicCxrIndex.from_settings(settings)
    records: list[StudyRecord] = []
    stats: dict = {}
    for split in rc.corpus_splits:                       # validated to be ['train'] in the config
        recs, st = index.study_records(split)
        records += recs
        stats[split] = st
    if not records:
        raise IndexBuildError(f"no images to index for splits {rc.corpus_splits}")
    if limit is not None:
        records = records[:limit]
    log(f"corpus: {len(records):,} studies from {rc.corpus_splits} | policy stats: {stats}")

    encoder = ClearEncoder(settings, device)
    sha = encoder.checkpoint_sha256                       # always hashed for the manifest
    shape = tuple(encoder.preprocess(Image.new("L", (64, 64))).shape)
    ds = _ImageDataset(records, settings, encoder.preprocess, shape)
    nw = settings.runtime.num_workers
    loader = DataLoader(ds, batch_size=settings.clear.batch_size, shuffle=False, num_workers=nw,
                        pin_memory=encoder.device.type == "cuda", persistent_workers=False)

    n, d = len(records), settings.clear.embed_dim
    emb = np.zeros((n, d), dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
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
    if len(failed) > rc.max_failed_fraction * n:
        raise IndexBuildError(f"{len(failed)} of {n} images failed to decode (> retrieval.max_failed_fraction="
                              f"{rc.max_failed_fraction}); first: {failed[:5]}")
    if failed:
        log(f"WARNING: {len(failed)} image(s) could not be decoded and were left out: {failed[:5]}...")
    kept = [r for r, g in zip(records, ok) if g]
    manifest = write_index(
        settings.paths.index_dir, rc.index_name, emb[ok], IdTable.from_records(kept),
        corpus_splits=rc.corpus_splits, clear_checkpoint_name=encoder.checkpoint_name, clear_checkpoint_sha256=sha,
        view_policy={"view_preference": list(settings.dataset.accepted_view_positions),
                     "one_image_per_study": settings.dataset.one_image_per_study, "tie_break": "smallest dicom_id"},
        selection_stats=stats, failed_dicom_ids=failed, save_embeddings=rc.save_embeddings, built_limit=limit,
        backend=rc.backend,
    )

    # sanity: the saved index must load with these settings and return every probe vector as its own top hit
    retriever = FaissRetriever.load(settings, expected_checkpoint_sha256=sha, exclude_same_patient=False)
    probe = np.linspace(0, len(kept) - 1, num=min(8, len(kept)), dtype=int)
    for pi, hits in zip(probe, retriever.search(emb[ok][probe], k=1)):
        if not hits or hits[0].dicom_id != kept[pi].dicom_id or hits[0].similarity < 0.999:
            raise IndexBuildError(f"sanity check failed: row {pi} ({kept[pi].dicom_id}) did not retrieve itself: {hits}")
    log(f"[{rc.backend}] wrote {manifest.n_vectors:,} x {manifest.embed_dim} fp32 vectors to {settings.paths.index_dir / rc.index_name}.* "
        f"in {(time.time() - t0) / 60:.1f} min")
    return manifest
