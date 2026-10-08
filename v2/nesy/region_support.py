"""Region evidence for the belief graph, from a Stage 4 run's CheXmask-derived region scores (set B, the
primary region set). Per study (its frontal image) and finding:

  support   max score over the finding's relevant regions (heart for cardiomegaly; lungs and lung
            thirds otherwise)
  left / right   max score over the patient-left / patient-right lung regions
  zone      upper / middle / lower: the highest-scoring third on a given side

Scores are Stage 4 logits (out-of-fold for fit-split images). "High" means >= a per-finding threshold
fitted on the thresh split (scripts/rules_eval.py), never on validate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .regions import CHEXMASK_REGIONS

LEFT = [j for j, r in enumerate(CHEXMASK_REGIONS) if r.startswith("lung_left")]
RIGHT = [j for j, r in enumerate(CHEXMASK_REGIONS) if r.startswith("lung_right")]
THIRDS = {side: {r.rsplit("_", 1)[1]: j for j, r in enumerate(CHEXMASK_REGIONS)
                 if r.startswith(f"lung_{side}_")} for side in ("left", "right")}
ZONE_NAME = {"upper": "upper", "middle": "middle", "lower": "lower"}


def relevant(f: str) -> list[int]:
    if f in ("cardiomegaly", "enlarged_cardiomediastinum"):
        return [CHEXMASK_REGIONS.index("heart")]
    return LEFT + RIGHT


def load(s4_run: str | Path):
    """-> (findings list, {study_id: row}, scores [n_img, 9, n_find] float32 memmap-backed)."""
    import sys
    s4 = Path(s4_run)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from stage4_regions import FIND
    idx = pd.read_parquet(s4 / "scores_index.parquet")
    sc = np.load(s4 / "scores_B_chexmask.npy", mmap_mode="r")
    return list(FIND), idx, sc


def evidence(sc_img: np.ndarray, fi: int, f: str) -> dict:
    """sc_img: [9, n_find] scores of one image. NaN regions are missing."""
    v = np.asarray(sc_img[:, fi], np.float32)
    rel = [j for j in relevant(f) if not np.isnan(v[j])]
    if not rel:
        return {"support": None, "left": None, "right": None, "zone": {}}
    L = [v[j] for j in LEFT if not np.isnan(v[j])]
    R = [v[j] for j in RIGHT if not np.isnan(v[j])]
    zone = {}
    for side in ("left", "right"):
        th = {z: v[j] for z, j in THIRDS[side].items() if not np.isnan(v[j])}
        if th:
            z = max(th, key=th.get)
            zone[side] = (ZONE_NAME[z], float(th[z]))
    return {"support": float(max(v[j] for j in rel)), "left": float(max(L)) if L else None,
            "right": float(max(R)) if R else None, "zone": zone}


def region_side(ev: dict, thr: float | None) -> str | None:
    """Side the region scores point to: bilateral if both lungs score >= thr, else the one side that does."""
    if thr is None or ev["left"] is None or ev["right"] is None:
        return None
    lh, rh = ev["left"] >= thr, ev["right"] >= thr
    if lh and rh:
        return "bilateral"
    return "left" if lh else "right" if rh else None
