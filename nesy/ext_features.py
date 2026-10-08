"""Feature assembly for images outside the MIMIC study table (external test sets), for the C+A+R head.

Inputs per image set (all cached; no labels):
  - region cache data/features/regions/<cache>_shards: CLEAR global embedding (= the head's image embedding,
    cosine 1.000 checked on 500 MIMIC images), CheXmask-region pooled vectors [9, 768] and their presence
  - CheXmask anatomy features data/features/chexmask_<set>.parquet (scripts/chexmask_features_ext.py)
  - Stage 4 region scores, computed here from SAVED Stage 4 fold models (mean over folds, as for every
    non-fit MIMIC image), or read from a Stage 4 run's cached scores (MIMIC development check only)
Every feature is standardised with the MIMIC fit-split statistics the head was trained with, recomputed
here with the same transforms as scripts/stage3_factorised.concept_features and scripts/probe.build_features.
The transforms are duplicated, so `check_against_runner` (run_pipeline.py --feature-path external on MIMIC
val) must show the same design matrix as the standard path before an external run is trusted.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths as P
from .regions import CHEXMASK_REGIONS

CHEXMASK_COLS = ["ctr", "ctr_maxrow", "heart_width", "thorax_width", "lung_area_frac_left", "lung_height_frac_left",
                 "lung_base_diff_left_minus_right", "lung_apex_diff_left_minus_right", "heart_shift_to_left",
                 "lung_l_area", "lung_r_area", "heart_area", "rca_mean"]
L_IDX = [j for j, r in enumerate(CHEXMASK_REGIONS) if r.startswith("lung_left")]
R_IDX = [j for j, r in enumerate(CHEXMASK_REGIONS) if r.startswith("lung_right")]


def load_region_cache(name: str, ids: list[str] | None = None) -> dict:
    """-> {image_id [n], global [n,768] f32, chexmask [n,9,768] f16, present [n,9] bool} in cache order,
    restricted to `ids` (in that order) when given."""
    d = P.FEATURES / "regions" / f"{name}_shards"
    idf = P.FEATURES / "regions" / f"{name}_ids.parquet"
    all_ids = pd.read_parquet(idf if idf.exists() else d / "ids.parquet").image_id.values   # external caches keep ids in the shard dir
    ks = sorted(int(Path(f).name.split("_")[1]) for f in glob.glob(str(d / "shard_*_global.npy")))
    rows = np.concatenate([np.load(d / f"shard_{k:04d}_rows.npy") for k in ks])
    want = None if ids is None else set(ids)
    G, CM, PR, keep_rows = [], [], [], []
    for k in ks:
        r = np.load(d / f"shard_{k:04d}_rows.npy")
        m = np.ones(len(r), bool) if want is None else np.isin(all_ids[r], list(want))
        if not m.any():
            continue
        G.append(np.load(d / f"shard_{k:04d}_global.npy", mmap_mode="r")[m].astype(np.float32))
        CM.append(np.load(d / f"shard_{k:04d}_chexmask.npy", mmap_mode="r")[m])
        PR.append(np.load(d / f"shard_{k:04d}_chexmask_present.npy", mmap_mode="r")[m])
        keep_rows.append(r[m])
    out = {"image_id": all_ids[np.concatenate(keep_rows)], "global": np.concatenate(G), "chexmask": np.concatenate(CM),
           "present": np.concatenate(PR).astype(bool), "n_cache": len(rows)}
    if ids is not None:
        pos = {d_: i for i, d_ in enumerate(out["image_id"])}
        order = np.array([pos[i] for i in ids if i in pos], dtype=int)
        out = {k: (v[order] if isinstance(v, np.ndarray) and len(v) == len(out["image_id"]) else v) for k, v in out.items()}
    return out


def score_regions(models: dict, V: np.ndarray, present: np.ndarray, findings: list[str]) -> np.ndarray:
    """Stage 4 set-B scores [n, 9, len(findings)] (NaN where a region is absent or a finding has no model):
    mean over the fold models of ((x - mu) / sd) @ w + b on [region vector, region one-hot]."""
    n, R, d = V.shape
    out = np.full((n, R, len(findings)), np.nan, np.float32)
    onehot = np.eye(R, dtype=np.float32)
    for fi, f in enumerate(findings):
        ms = models.get(f)
        if not ms:
            continue
        for j in range(R):
            ok = present[:, j]
            if not ok.any():
                continue
            x = np.asarray(V[ok, j], np.float32)
            s = np.zeros(int(ok.sum()), np.float64)
            for m in ms:
                wv = m["w"][:d] / m["sd"][:d]
                const = m["b"] - (m["mu"][:d] / m["sd"][:d]) @ m["w"][:d] + ((onehot[j] - m["mu"][d:]) / m["sd"][d:]) @ m["w"][d:]
                s += x @ wv + const
            out[ok, j, fi] = (s / len(ms)).astype(np.float32)
    return out


def chexmask_raw(cm: pd.DataFrame, ids: np.ndarray, view: np.ndarray):
    """(raw [n, 15], missing [n], ap [n]) exactly as probe.build_features group 'chexmask'."""
    df = pd.DataFrame({"dicom_id": ids, "view": view}).merge(cm[["dicom_id", "qc_ok", *CHEXMASK_COLS]], on="dicom_id", how="left")
    miss_s = ~(df.qc_ok.fillna(False).astype(bool))
    vals = df[CHEXMASK_COLS].where(~miss_s, np.nan, axis=0)            # as probe.py: row-wise mask
    miss = miss_s.values
    ap = (df.view == "AP").astype(float).values
    raw = np.hstack([vals.values, vals[["ctr", "ctr_maxrow"]].values * ap[:, None]])
    return raw.astype(float), miss, ap


def s4_raw(sc: np.ndarray, findings: list[str], names: list[str]):
    """Region-score features in the head's column order (names 's4B_<f>_<max|left|right>')."""
    cols = [n for n in names if n.startswith("s4B_")]
    fi = {f: i for i, f in enumerate(findings)}
    mats = []
    with np.errstate(all="ignore"):
        for c in cols:
            f, nm = c[4:].rsplit("_", 1)
            a = sc[:, :, fi[f]]
            sel = slice(None) if nm == "max" else (L_IDX if nm == "left" else R_IDX)
            mats.append(np.nanmax(a[:, sel], axis=1))
    return np.stack(mats, 1), cols


def fit_stats(head_cfg: dict, fit_rows: np.ndarray, fit_dicom: np.ndarray, fit_view: np.ndarray, s4_run: Path,
              names: list[str], concept_C: np.ndarray, emb_loader) -> dict:
    """MIMIC fit-split means / SDs of every raw feature group, as used in training."""
    X = emb_loader(fit_rows)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    S = X @ concept_C.T
    st = {"concepts": (S.mean(0), S.std(0) + 1e-6)}
    cm = pd.read_parquet(P.FEATURES / "chexmask_mimic.parquet")
    raw, _, _ = chexmask_raw(cm, fit_dicom, fit_view)
    st["chexmask"] = (np.nanmean(raw, 0), np.nanstd(raw, 0) + 1e-6)
    sidx = pd.read_parquet(s4_run / "scores_index.parquet")[["dicom_id"]]
    pos = {d: i for i, d in enumerate(sidx.dicom_id)}
    r = np.array([pos.get(d, -1) for d in fit_dicom])
    sc = np.load(s4_run / "scores_B_chexmask.npy", mmap_mode="r")
    a = np.asarray(sc[np.where(r >= 0, r, 0)], np.float32)
    a[r < 0] = np.nan
    import sys
    sys.path.insert(0, str(P.V2 / "scripts"))
    from stage4_regions import FIND
    raw4, _ = s4_raw(a, list(FIND), names)
    st["s4"] = (np.nanmean(raw4, 0), np.nanstd(raw4, 0) + 1e-6)
    return st


def design(st: dict, names: list[str], glob_emb: np.ndarray, concept_C: np.ndarray, cm_raw, sc: np.ndarray, s4_findings) -> np.ndarray:
    """Standardised design matrix in the head's feature order: concepts | chexmask (+missing, view_ap) | s4 (+missing)."""
    X = glob_emb / np.linalg.norm(glob_emb, axis=1, keepdims=True)
    mu, sd = st["concepts"]
    Zc = ((X @ concept_C.T - mu) / sd).astype(np.float32)
    raw, miss, ap = cm_raw
    mu, sd = st["chexmask"]
    Zm = np.hstack([(np.where(np.isnan(raw), mu, raw) - mu) / sd, miss[:, None].astype(float), ap[:, None]])
    raw4, _ = s4_raw(sc, s4_findings, names)
    mu, sd = st["s4"]
    m4 = np.isnan(raw4)
    Z4 = np.hstack([(np.where(m4, mu, raw4) - mu) / sd, m4.any(1, keepdims=True).astype(float)])
    return np.hstack([Zc, Zm, Z4]).astype(np.float32)
