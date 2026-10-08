"""Patient-level bootstrap on val: AUROC and ECE differences between models (e.g. Stages 1, 2, 3).

Resamples val PATIENTS with replacement (all of a patient's studies move together), recomputes each
model's AUROC and ECE per finding and the macro means, and reports the 95 % percentile interval of
each pairwise difference. A difference "excludes zero" if both interval ends have the same sign.

  python -u scripts/bootstrap_compare.py --runs S1=<dir> S2=<dir> S3=<dir> --n-boot 2000
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from nesy import data as D  # noqa: E402
from nesy.evaluate import ece  # noqa: E402
from nesy.runlog import Progress, Run  # noqa: E402


def metrics(y, p):
    m = ~np.isnan(y)
    y, p = y[m].astype(int), p[m]
    if y.min() == y.max():
        return np.nan, np.nan
    return roc_auc_score(y, p), ece(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="NAME=run_dir")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("bootstrap-compare", vars(args), args.run_dir) as run:
        runs = dict(x.split("=", 1) for x in args.runs)
        lab = D.study_table()[["study_id", "subject_id", *D.FINDINGS]]
        val = lab[lab.study_id.isin(pd.read_parquet(Path(next(iter(runs.values()))) / "predictions_non_test.parquet").query("split == 'val'").study_id)]
        P = {}
        for name, rd in runs.items():
            pr = pd.read_parquet(Path(rd) / "predictions_non_test.parquet").set_index("study_id")
            P[name] = pr.loc[val.study_id, [f"p_{f}" for f in D.FINDINGS]].values
        Y = val[D.FINDINGS].values.astype(float)
        pats = val.subject_id.values
        upat, inv = np.unique(pats, return_inverse=True)
        idx_by_pat = [np.nonzero(inv == i)[0] for i in range(len(upat))]
        run.log(f"val: {len(val):,} studies, {len(upat):,} patients; models {list(runs)}; {args.n_boot} resamples")
        rng = np.random.default_rng(0)
        F = len(D.FINDINGS)
        point = {n: np.array([metrics(Y[:, j], P[n][:, j]) for j in range(F)]) for n in runs}      # [F, 2]
        boot = {n: np.full((args.n_boot, F, 2), np.nan) for n in runs}
        prog = Progress(run, args.n_boot, "bootstrap resamples", every_s=60)
        for b in range(args.n_boot):
            sel = np.concatenate([idx_by_pat[i] for i in rng.integers(0, len(upat), len(upat))])
            for n in runs:
                for j in range(F):
                    boot[n][b, j] = metrics(Y[sel, j], P[n][sel, j])
            prog.update(b + 1)
        rows = []
        for a, c in itertools.combinations(runs, 2):
            for k, metric in enumerate(("AUROC", "ECE")):
                d_point = point[c][:, k] - point[a][:, k]
                d_boot = boot[c][:, :, k] - boot[a][:, :, k]
                for j, f in enumerate(D.FINDINGS + ["MACRO"]):
                    if f == "MACRO":
                        dp, db = np.nanmean(d_point), np.nanmean(d_boot, axis=1)
                    else:
                        dp, db = d_point[j], d_boot[:, j]
                    lo, hi = np.nanpercentile(db, [2.5, 97.5])
                    rows.append({"comparison": f"{c} - {a}", "metric": metric, "finding": f, "diff": round(float(dp), 4),
                                 "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
                                 "excludes_zero": bool(lo > 0 or hi < 0)})
        out = pd.DataFrame(rows)
        out.to_csv(run.dir / "bootstrap_differences.csv", index=False)
        pd.set_option("display.width", 220)
        pd.set_option("display.max_rows", 500)
        run.log("PAIRED PATIENT-BOOTSTRAP DIFFERENCES ON VAL (95 % percentile CI):\n" + out.to_string(index=False))
        sig = out[out.excludes_zero]
        run.log("Differences whose CI excludes zero:\n" + (sig.to_string(index=False) if len(sig) else "none"))


if __name__ == "__main__":
    main()
