"""The only way later code reads datasets: manifests, with the project split joined in."""
from __future__ import annotations

import pandas as pd

from . import paths as P

SPLIT_VERSION = "v1"   # frozen 2026-10-06, confirmed by the user
FITTING_SPLITS = ("train", "calib", "thresh")


def load_splits(version: str = SPLIT_VERSION) -> pd.DataFrame:
    return pd.read_parquet(P.SPLITS / f"mimic_splits_{version}.parquet")


def load(name: str, version: str = SPLIT_VERSION, columns: list[str] | None = None) -> pd.DataFrame:
    """Manifest `name` (mimic, mscxr, imagenome_silver, imagenome_gold, vindr, padchest_gr).
    MIMIC-derived manifests get the project `split` column joined by subject_id; external sets get
    their fixed role."""
    path = P.MANIFESTS / f"{name}.parquet"
    if columns is not None:
        import pyarrow.parquet as pq
        have = set(pq.read_schema(path).names)
        columns = list(dict.fromkeys(columns + [c for c in ("subject_id", "role") if c in have]))
    df = pd.read_parquet(path, columns=columns)
    if "subject_id" in df.columns:
        sp = load_splits(version)[["subject_id", "split", "split_version", "excluded_from_fitting"]]
        n = len(df)
        df = df.merge(sp, on="subject_id", how="left", validate="many_to_one")
        if len(df) != n or df.split.isna().any():
            raise ValueError(f"{name}: {df.split.isna().sum()} rows have no split")
    elif "role" in df.columns:
        df["split"] = df["role"]
    return df
