"""N17 CBM head training (pipeline.md §4.7). Trains a single linear layer on the 67 gathered
concept scores from the OFFICIAL TRAIN split, against MIMIC's own official CheXpert labels.

Label handling: CheXpert's convention distinguishes "not mentioned" (blank) from "mentioned and
negative" (0.0) from "uncertain" (-1.0). Blank and uncertain are both EXCLUDED from that label's
loss term for that study (masked, not guessed as positive or negative) -- a deliberate,
documented default, not the only reasonable one (the CheXpert literature also has "U-Ones"/
"U-Zeros" conventions for uncertain labels; masking is the conservative choice, consistent with
this project's "never silently guess" pattern elsewhere).

  python scripts/train_cbm.py --config configs/hpc.yaml
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .concepts import load_or_resolve_concept_ids
from .model import CBMHeadInfo, build_head, save_head
from ..concepts.bank import ConceptBank
from ..concepts.similarity import score_concepts_batch
from ..core.config import Settings
from ..core.runtime import configure_runtime
from ..core.types import Polarity
from ..datasets.chexpert_labels import ChexpertLabels, load_chexpert_labels
from ..datasets.mimic_cxr import MimicCxrIndex
from ..ingest.image_loader import load_record_image
from ..perception.clear_encoder import ClearEncoder


@dataclass(frozen=True)
class CBMTrainingReport:
    n_studies: int
    n_labels: int
    per_label_n_positive: dict[str, int]
    per_label_n_negative: dict[str, int]
    final_loss: float


def _build_label_matrix(records, labels: ChexpertLabels, label_space: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """-> (y [N, L] float32 in {0,1}, mask [N, L] float32 in {0,1}; 0 = excluded from loss)."""
    n = len(records)
    y = np.zeros((n, len(label_space)), dtype=np.float32)
    mask = np.zeros((n, len(label_space)), dtype=np.float32)
    for i, rec in enumerate(records):
        study_labels = labels.get(rec.subject_id, rec.study_id)
        for j, finding in enumerate(label_space):
            pol = study_labels.get(finding)
            if pol == Polarity.PRESENT:
                y[i, j], mask[i, j] = 1.0, 1.0
            elif pol == Polarity.ABSENT:
                y[i, j], mask[i, j] = 0.0, 1.0
            # UNCERTAIN and "not mentioned" (no entry at all) both leave mask[i, j] == 0.
    return y, mask


def train_cbm(settings: Settings, log: Callable[[str], None] = print, limit: "int | None" = None) -> CBMTrainingReport:
    import torch
    import torch.nn as nn

    s = settings
    device = configure_runtime(s)
    bank = ConceptBank.load(s)
    concept_ids = load_or_resolve_concept_ids(s.cbm.concepts_path, s.cbm.concept_ids_cache, bank.concepts)
    log(f"resolved {len(concept_ids)} CBM concepts to bank ids")

    index = MimicCxrIndex.from_settings(s)
    records, _ = index.study_records("train")
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError("no train-split studies found -- check paths.mimic_root / dataset.splits")
    labels = load_chexpert_labels(s)
    y, mask = _build_label_matrix(records, labels, s.cbm.label_space)
    log(f"{len(records)} train studies; per-label known counts: "
        + ", ".join(f"{s.cbm.label_space[j]}={int(mask[:, j].sum())}" for j in range(len(s.cbm.label_space))))

    encoder = ClearEncoder(s, device)
    bs = s.clear.batch_size
    feats_n = np.empty((len(records), len(concept_ids)), dtype=np.float32)
    for start in range(0, len(records), bs):
        batch = records[start:start + bs]
        decoded = [load_record_image(r, s)[0] for r in batch]
        feats = encoder.encode_batch(decoded)
        scores = score_concepts_batch(feats, bank, device, s.concept_bank.score_batch_size)
        feats_n[start:start + len(batch)] = scores[:, concept_ids]
        if (start // bs) % 10 == 0:
            log(f"  encoded {start + len(batch):,}/{len(records):,}")

    X = torch.from_numpy(feats_n)
    Y = torch.from_numpy(y)
    M = torch.from_numpy(mask)

    head = build_head(len(concept_ids), len(s.cbm.label_space))
    opt = torch.optim.Adam(head.parameters(), lr=s.cbm.train_lr, weight_decay=s.cbm.train_weight_decay)

    if s.cbm.positive_class_weight == "balanced":
        n_pos = (Y * M).sum(dim=0).clamp_min(1.0)
        n_neg = ((1 - Y) * M).sum(dim=0).clamp_min(1.0)
        pos_weight = n_neg / n_pos
    else:
        pos_weight = torch.ones(len(s.cbm.label_space))
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    n = X.shape[0]
    final_loss = float("nan")
    for epoch in range(s.cbm.train_epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, s.cbm.train_batch_size):
            idx = perm[start:start + s.cbm.train_batch_size]
            logits = head(X[idx])
            raw = criterion(logits, Y[idx])
            weighted = raw * (1 + (pos_weight - 1) * Y[idx]) * M[idx]
            denom = M[idx].sum().clamp_min(1.0)
            loss = weighted.sum() / denom
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss) * idx.shape[0]
        final_loss = epoch_loss / n
        if epoch % max(1, s.cbm.train_epochs // 10) == 0 or epoch == s.cbm.train_epochs - 1:
            log(f"  epoch {epoch + 1}/{s.cbm.train_epochs}  loss={final_loss:.4f}")

    info = CBMHeadInfo(
        n_concepts=len(concept_ids), n_labels=len(s.cbm.label_space), label_space=list(s.cbm.label_space),
        bank_version_concepts_sha256=bank.info.concepts_sha256,
    )
    save_head(s.cbm.head_checkpoint, head, info)
    log(f"wrote {s.cbm.head_checkpoint}")

    return CBMTrainingReport(
        n_studies=n, n_labels=len(s.cbm.label_space),
        per_label_n_positive={s.cbm.label_space[j]: int((Y[:, j] * M[:, j]).sum()) for j in range(len(s.cbm.label_space))},
        per_label_n_negative={s.cbm.label_space[j]: int(((1 - Y[:, j]) * M[:, j]).sum()) for j in range(len(s.cbm.label_space))},
        final_loss=final_loss,
    )
