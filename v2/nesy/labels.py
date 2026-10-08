"""Label rules R1-R4 (scripts/build_labels.py) as a function of per-finding source codes, so the same rules can
be applied to another label source (the MIMIC-CXR 2.1.0 radiologist test labels) for evaluation.

Codes: 1 positive, 0 negative, -1 uncertain, 9 blank (not mentioned), -9 no label row.
R1 blank -> 0; R2 uncertain -> masked (NaN); R0 no row -> masked; R3 positive child -> every ancestor positive;
R4 uncertain child with a blank parent -> parent masked, cascading upward. any_abnormality (root) starts blank.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import kg

RADIOLOGIST_COLUMNS = {  # mimic-cxr-2.1.0-test-set-labeled.csv column -> finding id
    "Enlarged Cardiomediastinum": "enlarged_cardiomediastinum", "Cardiomegaly": "cardiomegaly", "Lung Lesion": "lung_lesion",
    "Airspace Opacity": "lung_opacity", "Edema": "edema", "Consolidation": "consolidation", "Pneumonia": "pneumonia",
    "Atelectasis": "atelectasis", "Pneumothorax": "pneumothorax", "Pleural Effusion": "pleural_effusion",
    "Pleural Other": "pleural_other", "Fracture": "fracture", "Support Devices": "support_devices"}


def from_codes(src: pd.DataFrame) -> pd.DataFrame:
    """src: index study_id, one integer code column per finding id (missing finding columns = blank)."""
    F = kg.finding_ids()
    src = src.copy()
    for f in F:
        if f not in src:
            src[f] = 9
    val = pd.DataFrame(index=src.index, dtype=float)
    for f in F:
        c = src[f].values
        val[f] = np.select([c == 1, c == 0, c == 9, c == -1, c == -9], [1.0, 0.0, 0.0, np.nan, np.nan])
    order = kg.topo_order()
    for f in order:                                        # R3, children first
        pos = val[f] == 1
        for p in kg.parents_of(f):
            val.loc[pos & (val[p] != 1), p] = 1.0
    unc = pd.DataFrame({f: src[f] == -1 for f in F})
    for f in order:                                        # R4
        for p in kg.parents_of(f):
            change = unc[f] & (src[p] == 9) & (val[p] == 0)
            val.loc[change, p] = np.nan
            unc.loc[change, p] = True
    return val


def radiologist_codes(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw radiologist rows (study_id + label columns, values 1 / 0 / -1 / blank) -> code table."""
    src = pd.DataFrame(index=raw.study_id.values)
    for c, f in RADIOLOGIST_COLUMNS.items():
        v = raw[c].values.astype(float)
        src[f] = np.where(np.isnan(v), 9, v).astype(int)
    return src
