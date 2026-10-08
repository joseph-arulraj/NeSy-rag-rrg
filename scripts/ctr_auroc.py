"""AUROC of the cardiothoracic ratio ALONE for cardiomegaly (and enlarged cardiomediastinum), by view.

No model is fitted: the CheXmask CTR is used directly as the score, so the fit split can be reported
too (large n) next to val. Only CheXmask rows passing QC (Dice RCA mean >= 0.7, all masks) are used;
the number excluded is reported. Two CTR definitions: bounding-box widths (ctr) and widest-row
widths (ctr_maxrow).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from nesy import data as D, paths as P  # noqa: E402
from nesy.runlog import Run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("ctr-auroc", vars(args), args.run_dir) as run:
        t = D.study_table()
        cm = pd.read_parquet(P.FEATURES / "chexmask_mimic.parquet", columns=["dicom_id", "ctr", "ctr_maxrow", "qc_ok"])
        df = t.merge(cm, on="dicom_id", how="left")
        run.log(f"studies {len(df):,}; CheXmask QC fail or missing: {int((~df.qc_ok.fillna(False).astype(bool)).sum()):,} (excluded)")
        df = df[df.qc_ok.fillna(False).astype(bool)]
        rows = []
        for split in ("train", "val"):
            for view in ("PA", "AP", "all"):
                s = df[df.split == split] if view == "all" else df[(df.split == split) & (df.view == view)]
                for f in ("cardiomegaly", "enlarged_cardiomediastinum"):
                    m = s[f].notna()
                    y = s.loc[m, f].astype(int)
                    if y.nunique() < 2:
                        continue
                    for feat in ("ctr", "ctr_maxrow"):
                        rows.append({"split": split, "view": view, "finding": f, "feature": feat, "n": int(m.sum()),
                                     "positives": int(y.sum()), "auroc": round(roc_auc_score(y, s.loc[m, feat]), 4),
                                     "median_ctr_pos": round(float(s.loc[m & (s[f] == 1), feat].median()), 3),
                                     "median_ctr_neg": round(float(s.loc[m & (s[f] == 0), feat].median()), 3)})
        out = pd.DataFrame(rows)
        pd.set_option("display.width", 200)
        run.log("CTR ALONE:\n" + out.to_string(index=False))
        out.to_csv(run.dir / "ctr_auroc.csv", index=False)


if __name__ == "__main__":
    main()
