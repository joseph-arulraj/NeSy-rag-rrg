"""Study-level design matrices: one frontal CLEAR embedding per study, labels, splits."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import kg, manifests as M, paths as P

import os

# Embedding variant (2026-10-06): "stretch" (released hub preprocess, the original store) or "letterbox"
# (the CLEAR authors' aspect-preserving resize + zero padding). Chosen by the NESY_EMB environment
# variable so every stage script runs unchanged on either; default stretch.
VARIANT = os.environ.get("NESY_EMB", "stretch")
if VARIANT not in ("stretch", "letterbox"):
    raise ValueError(f"NESY_EMB must be stretch or letterbox, got {VARIANT}")
_SUFFIX = "v1" if VARIANT == "stretch" else "lb_v1"
EMB = P.V2 / f"data/embeddings/clear_frontal_{_SUFFIX}.npy"
EMB_INDEX = P.V2 / f"data/embeddings/clear_frontal_{_SUFFIX}_index.parquet"
RETRIEVAL = P.FEATURES / ("retrieval_v1.parquet" if VARIANT == "stretch" else "retrieval_lb_v1.parquet")
LABELS = P.V2 / "data/labels/labels_v2.parquet"
FINDINGS = [f for f in kg.finding_ids()]          # 13 findings + the root any_abnormality (no_finding = its absent band)


def study_table() -> pd.DataFrame:
    """One row per study with a frontal embedding: row (into EMB), ids, split, view, labels."""
    idx = pd.read_parquet(EMB_INDEX)
    lab = pd.read_parquet(LABELS)
    view = M.load("mimic", columns=["dicom_id", "view"])[["dicom_id", "view"]]
    t = idx.merge(lab.drop(columns=["subject_id"]), on="study_id", how="left", validate="one_to_one")
    t = t.merge(view, on="dicom_id", how="left")
    if t.split.isna().any():
        raise ValueError("studies without a split")
    return t


def embeddings(rows: np.ndarray | None = None) -> np.ndarray:
    e = np.load(EMB, mmap_mode="r")
    return np.asarray(e if rows is None else e[rows], dtype=np.float32)
