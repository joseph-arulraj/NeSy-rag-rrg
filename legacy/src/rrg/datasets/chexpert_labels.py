"""MIMIC-CXR's own official CheXpert-labeler output (`dataset.chexpert_csv`): per-study
presence/absence/uncertain labels for the 14 CheXpert findings. Used as Evidence C (CBM)
training ground truth (pipeline.md §4.7, GAP-1 resolved per pipeline.md §10).
"""
from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from typing import Optional

from ..core.config import Settings, require_file
from ..core.types import Polarity

# CheXpert CSV column header -> our canonical snake_case finding label (configs/finding_synonyms.yaml).
# Verified against the real column headers in data/mimic_cxr_sample/mimic-cxr-2.0.0-chexpert.csv.gz.
CHEXPERT_COLUMN_TO_CANONICAL: dict[str, str] = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Consolidation": "consolidation",
    "Edema": "edema",
    "Enlarged Cardiomediastinum": "enlarged_cardiomediastinum",
    "Fracture": "fracture",
    "Lung Lesion": "lung_lesion",
    "Lung Opacity": "lung_opacity",
    "No Finding": "no_finding",
    "Pleural Effusion": "pleural_effusion",
    "Pleural Other": "pleural_other",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
    "Support Devices": "support_devices",
}


def _parse_value(raw: str) -> Optional[Polarity]:
    """CheXpert labeler convention: 1.0 = positive, 0.0 = negative, -1.0 = uncertain,
    blank = not mentioned at all (returns None -- distinct from Polarity.ABSENT)."""
    raw = (raw or "").strip()
    if raw == "":
        return None
    val = float(raw)
    if val == 1.0:
        return Polarity.PRESENT
    if val == 0.0:
        return Polarity.ABSENT
    if val == -1.0:
        return Polarity.UNCERTAIN
    raise ValueError(f"unexpected CheXpert label value {raw!r} (expected 1.0/0.0/-1.0/blank)")


@dataclass(frozen=True)
class ChexpertLabels:
    """(subject_id, study_id) -> {canonical_finding: Polarity}. A finding absent from the inner
    dict means the labeler didn't mention it at all -- treat as unknown, not absent."""

    by_study: dict[tuple[int, int], dict[str, Polarity]]

    def get(self, subject_id: int, study_id: int) -> dict[str, Polarity]:
        return self.by_study.get((int(subject_id), int(study_id)), {})


def load_chexpert_labels(settings: Settings) -> ChexpertLabels:
    path = require_file(settings.paths.mimic_root / settings.dataset.chexpert_csv, "dataset.chexpert_csv")
    by_study: dict[tuple[int, int], dict[str, Polarity]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing_cols = set(CHEXPERT_COLUMN_TO_CANONICAL) - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(f"{path}: missing expected CheXpert columns {sorted(missing_cols)}")
        for row in reader:
            key = (int(row["subject_id"]), int(row["study_id"]))
            labels: dict[str, Polarity] = {}
            for col, canonical in CHEXPERT_COLUMN_TO_CANONICAL.items():
                pol = _parse_value(row[col])
                if pol is not None:
                    labels[canonical] = pol
            by_study[key] = labels
    return ChexpertLabels(by_study=by_study)
