"""CheXmask anatomy features for the external test sets (VinDr-CXR test, PadChest-GR): the same per-image
features as scripts/chexmask_features.py (same `features()` function), for the A input of the head.
Images and masks only: the manifests are read with the id and view columns, never labels.

Source: CheXmask/OriginalResolution/{VinDr-CXR,Padchest}.csv (the Preprocessed files for these sets were
deleted on 2026-10-07). All features are ratios or normalised by mask size; Preprocessed vs OriginalResolution
agreed to <= 0.004 on ctr / ctr_maxrow / heart shift (runs/20261006-0922_chexmask-smoke). Images with no
CheXmask row stay missing (qc_ok False -> the head's cm_missing indicator), never zero.

Output: data/features/chexmask_{vindr_test,padchest_gr}.parquet, keyed by `dicom_id` (= the dataset's image id,
same column name as the MIMIC file so the feature builder can be reused).
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from chexmask_features import RCA_OK, features  # noqa: E402
from nesy import paths as P  # noqa: E402
from nesy.runlog import Progress, Run  # noqa: E402

SETS = {  # name: (CheXmask file, id column there, id list of the cached image features, manifest)
    "vindr_test": ("VinDr-CXR.csv", "image_id", "ext_vindr_test_ids.parquet", "vindr"),
    "padchest_gr": ("Padchest.csv", "ImageID", "ext_padchest_gr_ids.parquet", "padchest_gr"),
}
COLS = ["Dice RCA (Mean)", "Dice RCA (Max)", "Left Lung", "Right Lung", "Heart", "Height", "Width"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="vindr_test,padchest_gr")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--run-dir", default=None)
    a = ap.parse_args()
    with Run("chexmask-features-ext", vars(a), a.run_dir) as run:
        with Pool(a.workers) as pool:
            for name in a.sets.split(","):
                fn, key, ids_fn, man = SETS[name]
                ids = pd.read_parquet(P.FEATURES / "regions" / ids_fn).image_id
                keep = set(ids)
                view = pd.read_parquet(P.MANIFESTS / f"{man}.parquet", columns=["image_id", "view"]).drop_duplicates("image_id")
                run.log(f"{name}: {len(keep):,} images (ids of the cached image features); reading {fn}")
                rows, n_in = [], 0
                prog = Progress(run, len(keep), f"{name} images found", every_s=60)
                for ch in pd.read_csv(P.CHEXMASK / "OriginalResolution" / fn, usecols=[key, *COLS], chunksize=2000):
                    n_in += len(ch)
                    ch = ch[ch[key].isin(keep)]
                    if len(ch):
                        rows += pool.map(features, ch[[key, *COLS]].itertuples(index=False, name=None), chunksize=16)
                        prog.update(len(rows))
                df = pd.DataFrame(rows)
                dup = int(df.dicom_id.duplicated().sum())
                if dup:
                    run.log(f"  WARNING {dup} duplicate ids; keeping the highest rca_mean")
                    df = df.sort_values("rca_mean", ascending=False).drop_duplicates("dicom_id")
                df = df.merge(view.rename(columns={"image_id": "dicom_id"}), on="dicom_id", how="left")
                df["qc_ok"] = (df.rca_mean >= RCA_OK) & df.masks_complete
                df["ctr_reliable_view"] = df.view == "PA"
                out = P.FEATURES / f"chexmask_{name}.parquet"
                df.to_parquet(out, index=False)
                conv = df.loc[df.masks_complete, "left_lung_on_image_right"].mean()
                run.log(f"  CSV rows read {n_in:,}; features for {len(df):,} / {len(keep):,} images; missing (no CheXmask row) "
                        f"{len(keep) - len(df):,}; qc_ok {int(df.qc_ok.sum()):,}; masks incomplete {int((~df.masks_complete).sum()):,}; "
                        f"side convention {conv:.2%}; views {df.view.value_counts(dropna=False).to_dict()} -> {out}")
                run.metric(set=name, n_images=len(keep), n_features=len(df), qc_ok=int(df.qc_ok.sum()))


if __name__ == "__main__":
    main()
