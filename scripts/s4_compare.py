"""Compare a retrained Stage 4 run's set-B region scores (and its saved models) with the cached run the head was
trained on. Reports, per split group, max |diff| and correlation of the scores; and checks that scoring images from
the saved models (nesy/ext_features.score_regions) reproduces the retrained run's own non-fit scores."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
sys.path.insert(0, str(V2 / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="runs/20261006-224243_lb-stage4")
    ap.add_argument("--new", required=True)
    ap.add_argument("--run-dir", default=None)
    a = ap.parse_args()
    import joblib
    from nesy import ext_features as XF
    from nesy.runlog import Run
    from stage4_regions import FIND
    with Run("s4-compare", vars(a), a.run_dir) as run:
        oi = pd.read_parquet(Path(a.old) / "scores_index.parquet")
        ni = pd.read_parquet(Path(a.new) / "scores_index.parquet")
        assert (oi.dicom_id.values == ni.dicom_id.values).all(), "image order differs"
        so = np.load(Path(a.old) / "scores_B_chexmask.npy").astype(np.float32)
        sn = np.load(Path(a.new) / "scores_B_chexmask.npy").astype(np.float32)
        out = {}
        for grp, m in (("fit", oi.split.values == "train"), ("val", oi.split.values == "val"), ("all_non_fit", oi.split.values != "train")):
            x, y = so[m].ravel(), sn[m].ravel()
            k = ~np.isnan(x) & ~np.isnan(y)
            out[grp] = {"n_scores": int(k.sum()), "nan_mismatch": int((np.isnan(x) != np.isnan(y)).sum()),
                        "max_abs_diff": float(np.abs(x[k] - y[k]).max()), "mean_abs_diff": float(np.abs(x[k] - y[k]).mean()),
                        "corr": float(np.corrcoef(x[k], y[k])[0, 1])}
            run.log(f"{grp}: {out[grp]}")
        mdl = joblib.load(Path(a.new) / "models.joblib")
        vm = np.nonzero(oi.split.values == "val")[0]
        rc = XF.load_region_cache("mimic_full_lb", list(oi.dicom_id.values[vm]))
        sc = XF.score_regions(mdl["models"]["B_chexmask"], rc["chexmask"], rc["present"], list(FIND))
        ref = sn[vm]
        k = ~np.isnan(sc) & ~np.isnan(ref)
        out["models_vs_new_run_val"] = {"max_abs_diff": float(np.abs(sc[k] - ref[k]).max()),
                                        "nan_mismatch": int((np.isnan(sc) != np.isnan(ref)).sum())}
        run.log(f"scores from saved models vs the new run's own val scores (float16-stored): {out['models_vs_new_run_val']}")
        (run.dir / "s4_compare.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
