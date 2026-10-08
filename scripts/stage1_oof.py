"""Out-of-fold Stage 1 logits, the inputs of the association-links stage (brief: "stage two adds the
stage-one logits of findings linked by associated-with edges"). Same features and per-finding C as the
given Stage 1 run; 2 folds by patient (hash of subject_id, as Stage 4) within the fit split: fit-split
studies get the logit of the fold model that did not see them, other splits the mean of the two fold
models. Output: data/features/stage1_oof_<variant>.parquet (study_id, s1_<finding>).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from nesy import data as D, paths as P  # noqa: E402
from nesy.runlog import Run  # noqa: E402
from probe import build_features  # noqa: E402


def fold(subject_id, k=2):
    return int(hashlib.sha256(f"s1fold:{subject_id}".encode()).hexdigest()[:8], 16) % k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-run", required=True)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("stage1-oof", vars(args), args.run_dir) as run:
        bundle = joblib.load(Path(args.stage1_run) / "model.joblib")
        t = D.study_table()
        fit = (t.split == "train").values
        X, names = build_features(t, bundle["groups"], fit, run)
        assert names == bundle["feature_names"]
        fo = np.array([fold(s) for s in t.subject_id.values])
        out = {"study_id": t.study_id.values}
        for f in D.FINDINGS:
            y = t[f].values
            known = ~np.isnan(y)
            C = bundle["models"][f]["C"]
            logit = np.zeros(len(t))
            per = []
            for k in (0, 1):
                tr = fit & known & (fo != k)
                m = LogisticRegression(C=C, max_iter=3000, tol=1e-4).fit(X[tr], y[tr].astype(int))
                per.append(m.decision_function(X))
            per = np.stack(per)
            logit = per.mean(0)
            logit[fit] = per[fo[fit], np.nonzero(fit)[0]]
            out[f"s1_{f}"] = logit.astype(np.float32)
            a_tr = roc_auc_score(y[fit & known].astype(int), logit[fit & known])
            vm = (t.split == "val").values & known
            a_va = roc_auc_score(y[vm].astype(int), logit[vm])
            run.log(f"  {f:28s} C={C}: out-of-fold train AUROC {a_tr:.4f}; val AUROC {a_va:.4f}")
            run.metric(finding=f, oof_train_auroc=a_tr, val_auroc=a_va)
        o = P.FEATURES / f"stage1_oof_{D.VARIANT}.parquet"
        pd.DataFrame(out).to_parquet(o, index=False)
        run.log(f"wrote {o} ({len(t):,} studies)")


if __name__ == "__main__":
    main()
