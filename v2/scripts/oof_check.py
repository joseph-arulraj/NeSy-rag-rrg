"""Out-of-fold check for the region-classifier (s4) features: AUROC of each feature alone against the
study label on the fit (train) split vs a comparison set of splits (default calib + thresh pooled; validate
is too small to read). In-sample train scores would separate train studies clearly better. No test rows.
Patient-bootstrap 95% CI of the comparison-split AUROC so the gap can be read against its noise.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from nesy import data as D  # noqa: E402
from nesy.runlog import Run  # noqa: E402
from probe import build_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="s4")
    ap.add_argument("--compare", default="calib,thresh")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("oof-check", {**vars(args), "NESY_S4_RUN": os.environ.get("NESY_S4_RUN"),
                           "NESY_S4_SETS": os.environ.get("NESY_S4_SETS")}, args.run_dir) as run:
        cmp_splits = args.compare.split(",")
        t = D.study_table()
        t = t[t.split.isin(["train", *cmp_splits])].reset_index(drop=True)
        fit = (t.split == "train").values
        cm = t.split.isin(cmp_splits).values
        run.log(f"train {int(fit.sum()):,} studies; comparison ({'+'.join(cmp_splits)}) {int(cm.sum()):,}")
        rng = np.random.default_rng(0)
        rows = []
        for g in args.groups.split(","):
            X, names = build_features(t, [g], fit, run)
            for j, n in enumerate(names):
                if not (n.endswith("_max") or n.startswith("nb10_frac_")):
                    continue
                f = n.split("_", 1)[1].rsplit("_max", 1)[0] if n.startswith("s4") else n.replace("nb10_frac_", "")
                if f not in D.FINDINGS:
                    continue
                y = t[f].values
                k = ~np.isnan(y)
                a_tr = roc_auc_score(y[fit & k], X[fit & k, j])
                mm = cm & k
                a_c = roc_auc_score(y[mm], X[mm, j])
                sub = t[mm].reset_index(drop=True)
                ys, xs = y[mm], X[mm, j]
                pats = sub.subject_id.unique()
                grp = {p: np.nonzero(sub.subject_id.values == p)[0] for p in pats} if len(pats) < 20000 else None
                bs = []
                for _ in range(args.n_boot):
                    ii = np.concatenate([grp[p] for p in rng.choice(pats, len(pats))])
                    if len(np.unique(ys[ii])) == 2:
                        bs.append(roc_auc_score(ys[ii], xs[ii]))
                lo, hi = np.percentile(bs, [2.5, 97.5])
                rows.append({"feature": n, "finding": f, "n_pos_compare": int(ys.sum()), "auroc_train": a_tr,
                             "auroc_compare": a_c, "ci_low": lo, "ci_high": hi, "gap": a_tr - a_c,
                             "train_inside_ci": bool(lo <= a_tr <= hi)})
                run.log(f"  {n:40s} train {a_tr:.3f} vs {'+'.join(cmp_splits)} {a_c:.3f} [{lo:.3f}, {hi:.3f}] "
                        f"(pos {int(ys.sum())}) gap {a_tr - a_c:+.3f}")
        d = pd.DataFrame(rows)
        d.to_csv(run.dir / "oof_check.csv", index=False)
        run.log(f"features {len(d)}; mean gap {d.gap.mean():+.4f}; median {d.gap.median():+.4f}; "
                f"train AUROC inside the comparison CI for {int(d.train_inside_ci.sum())}/{len(d)}; "
                f"above the CI for {int((d.auroc_train > d.ci_high).sum())}")
        run.metric(summary="oof", mean_gap=float(d.gap.mean()), n_above_ci=int((d.auroc_train > d.ci_high).sum()), n=len(d))


if __name__ == "__main__":
    main()
