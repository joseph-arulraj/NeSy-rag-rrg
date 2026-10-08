"""Item B evaluation on validate: do the k similar fit-split cases agree with the model's band, and are
present calls that few similar cases share more often wrong? Bands come from the final head with the kept
knowledge rules (as in Stage 8). Case evidence changes nothing; this only measures it.

Per finding:
  agreement     band present/possible and >= ceil(k/2) of k neighbours positive, or band absent and
                < k/2 positive (silent band excluded)
  present calls split by case support: low (<= 1 of k positive) vs the rest; error rate (label negative)
                in each group, difference with a patient-bootstrap 95% CI (pooled over findings too)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nesy import data as D, paths as P, pipeline_graph as PG, region_support as RS  # noqa: E402
from nesy.runlog import Progress, Run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-run", required=True)
    ap.add_argument("--rules-run", required=True)
    ap.add_argument("--s4-run", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--low", type=int, default=1, help="'few similar cases': at most this many of k positive")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("case-eval", vars(args), args.run_dir) as run:
        rules = json.loads((Path(args.rules_run) / "rules_eval.json").read_text())
        pr = Path(args.pred_run)
        thr = json.loads((pr / "thresholds_4band.json").read_text())
        preds = pd.read_parquet(pr / "predictions_non_test.parquet")
        t = D.study_table()[["study_id", "subject_id", "dicom_id", *D.FINDINGS]]
        va = preds[preds.split == "val"].merge(t, on="study_id")
        cases = pd.read_parquet(P.FEATURES / f"cases_{D.VARIANT}_k{args.k}.parquet").set_index("study_id")
        FIND, idx, sc = RS.load(args.s4_run)
        pos = {d: i for i, d in enumerate(idx.dicom_id)}
        rows = []
        prog = Progress(run, len(va), "val studies", every_s=60)
        for n_, row in enumerate(va.to_dict("records")):
            i = pos.get(row["dicom_id"])
            ev = {} if i is None else {f: RS.evidence(np.asarray(sc[i], np.float32), fi, f) for fi, f in enumerate(FIND) if f in D.FINDINGS}
            g = PG.build(row["study_id"], row, thr, ev, rules["region_thresholds"], use_r1=rules["R1"]["keep"], use_r2=rules["R2"]["keep"])
            c = cases.loc[row["study_id"]]
            for f in D.FINDINGS:
                if f == "any_abnormality":
                    continue
                rows.append({"study_id": row["study_id"], "subject_id": row["subject_id"], "finding": f, "band": g.nodes[f].band,
                             "y": row[f], "case_pos": int(c[f"case_pos_{f}"]), "case_known": int(c[f"case_known_{f}"])})
            prog.update(n_ + 1)
        d = pd.DataFrame(rows)
        half = math.ceil(args.k / 2)
        st = d[d.band != "silent"].copy()
        st["agree"] = np.where(st.band.isin(["present", "possible"]), st.case_pos >= half, st.case_pos < args.k / 2)
        agree = st.groupby("finding").agg(n=("agree", "size"), agreement=("agree", "mean"))
        agree.loc["ALL"] = [len(st), st.agree.mean()]
        p = d[(d.band == "present") & d.y.notna()].copy()
        p["low"] = p.case_pos <= args.low
        p["wrong"] = p.y == 0

        def err(x):
            lo, hi = x[x.low], x[~x.low]
            return (lo.wrong.mean() if len(lo) else np.nan), (hi.wrong.mean() if len(hi) else np.nan), len(lo), len(hi)

        tab = []
        rng = np.random.default_rng(0)
        for f, x in list(p.groupby("finding")) + [("ALL", p)]:
            e_lo, e_hi, n_lo, n_hi = err(x)
            pats = x.subject_id.unique()
            gx = {q: y for q, y in x.groupby("subject_id")}
            bs = []
            for _ in range(args.n_boot):
                b = pd.concat([gx[q] for q in rng.choice(pats, len(pats))])
                a, c_, _, _ = err(b)
                bs.append(a - c_)
            lo_, hi_ = np.nanpercentile(bs, [2.5, 97.5]) if np.isfinite(bs).any() else (np.nan, np.nan)
            tab.append({"finding": f, "present_calls": len(x), "n_low_support": n_lo, "n_other": n_hi,
                        "error_low_support": e_lo, "error_other": e_hi, "diff": e_lo - e_hi, "ci_low": lo_, "ci_high": hi_,
                        "excludes_zero": bool(lo_ > 0 or hi_ < 0)})
        tab = pd.DataFrame(tab).set_index("finding")
        agree.to_csv(run.dir / "case_agreement.csv")
        tab.to_csv(run.dir / "present_error_by_case_support.csv")
        d.to_parquet(run.dir / "case_rows_val.parquet", index=False)
        pd.set_option("display.width", 220)
        run.log("agreement of similar cases with the model's band (val, silent excluded):\n" + agree.round(3).to_string())
        run.log(f"present calls: error rate when <= {args.low} of {args.k} similar cases are positive vs the rest:\n" + tab.round(3).to_string())
        run.metric(summary="cases", agreement=float(st.agree.mean()), err_low=float(tab.loc["ALL", "error_low_support"]),
                   err_other=float(tab.loc["ALL", "error_other"]), diff_ci_low=float(tab.loc["ALL", "ci_low"]))


if __name__ == "__main__":
    main()
