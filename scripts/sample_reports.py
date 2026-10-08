"""Item 12 sample run: belief graphs + template reports for a small val sample from a probe run.

Thresholds: the user's four bands (nesy/thresholds.py), refitted on the thresh split from the probe
run's own predictions: present PPV >= 0.7, possible PPV >= 0.4, absent <= 5 % positives missed
(pneumothorax <= 2 %), silent otherwise; >= 10 true positives for every PPV tier. D4 (absent band
reachable on val) is checked over all val studies.
Grounding: no A2 localisations exist yet, so findings carry no anatomy; CTR comes from CheXmask (G5).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nesy import belief as B, data as D, grounding as G, paths as P, report as R  # noqa: E402
from nesy.runlog import Run  # noqa: E402


def provisional(cands: pd.DataFrame) -> dict:
    thr = {}
    for f, g in cands.groupby("finding"):
        def pick(rules):
            for r in rules:
                x = g[(g.rule == r) & g.thr.notna()]
                if len(x):
                    return float(x.thr.iloc[0]), r
            return None, None
        hi, rhi = pick(["PPV>=0.8", "PPV>=0.7"])
        lo, rlo = pick(["NPV>=0.98", "NPV>=0.95"])
        if hi is None or lo is None or lo >= hi:
            thr[f] = {"status": "abstain", "reason": f"present {rhi}, absent {rlo}"}
        else:
            thr[f] = {"present": hi, "absent": lo, "rule_present": rhi, "rule_absent": rlo, "status": "provisional"}
    return thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-run", required=True)
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("sample-reports", vars(args), args.run_dir) as run:
        pr = Path(args.pred_run)
        preds = pd.read_parquet(pr / "predictions_non_test.parquet")
        from nesy import thresholds as TH
        lab = D.study_table()[["study_id", *D.FINDINGS]]
        th = preds[preds.split == "thresh"].merge(lab, on="study_id")
        thr = {f: TH.fit(th[f].values, th[f"p_{f}"].values, f) for f in D.FINDINGS}
        run.log("4-band thresholds refitted on thresh (n=%d):" % len(th))
        for f, v in thr.items():
            run.log(f"  {f:28s} present {v['present']}, possible {v['possible']}, absent {v['absent']}")
        (run.dir / "thresholds_used.json").write_text(json.dumps(thr, indent=2))
        val = preds[preds.split == "val"]
        bands = B.check_bands({f: val[f"p_{f}"].tolist() for f in D.FINDINGS}, thr)
        run.log("D4 band check on val PASSED. Band counts per finding:\n" + pd.DataFrame(bands).T.to_string())

        t = D.study_table()
        cm = pd.read_parquet(P.FEATURES / "chexmask_mimic.parquet", columns=["dicom_id", "ctr", "qc_ok"])
        sample = val.sample(min(args.n, len(val)), random_state=0).merge(t[["study_id", "dicom_id", "view", *D.FINDINGS]], on="study_id")
        sample = sample.merge(cm, on="dicom_id", how="left")
        out = []
        for r in sample.itertuples():
            probs = {f: getattr(r, f"p_{f}") for f in D.FINDINGS}
            ctr = G.ctr(r.ctr if bool(r.qc_ok) else None, r.view)
            grounding = {}
            for f in D.FINDINGS:                                   # side head (Stage 3 models only)
                cols = [c for c in sample.columns if c.startswith(f"pside_{f}_")]
                if cols:
                    vals = {c.rsplit("_", 1)[1]: getattr(r, c) for c in cols}
                    side = max(vals, key=vals.get)
                    grounding[f] = {"localisations": [], "side": side, "side_conf": float(vals[side])}
            g = B.build(r.study_id, probs, thr, ctr=ctr, grounding=grounding)
            rep = R.render(g)
            truth = {f: getattr(r, f) for f in D.FINDINGS}
            out.append({"study_id": int(r.study_id), "view": r.view, "report": rep, "graph": g.to_dict(),
                        "labels": {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in truth.items()}})
            run.log(f"--- study {r.study_id} ({r.view}); no_acute_abnormality={g.no_acute_abnormality}; audit entries {len(g.audit)}\n"
                    f"    FINDINGS: {rep['findings']}\n    IMPRESSION: {rep['impression']}\n"
                    f"    labels positive: {[f for f, v in truth.items() if v == 1]}")
        (run.dir / "sample_reports.json").write_text(json.dumps(out, indent=1, default=str))
        # three examples: normal, single finding, several findings (by number of stated present/possible findings)
        def n_stated(o):
            return sum(1 for s in o["report"]["sentences"] if s["band"] in ("present", "possible"))
        picks = {"normal": next((o for o in out if o["graph"]["no_acute_abnormality"]), None),
                 "single": next((o for o in out if n_stated(o) == 1), None),
                 "several": max(out, key=n_stated)}
        for k, o in picks.items():
            if o is None:
                run.log(f"EXAMPLE {k}: none in this sample")
                continue
            run.log(f"EXAMPLE {k}: study {o['study_id']} ({o['view']})\n  FINDINGS: {o['report']['findings']}\n"
                    f"  IMPRESSION: {o['report']['impression']}\n  labels positive: {[f for f, v in o['labels'].items() if v == 1]}")
            gr = o["graph"]
            lines = [f"    {f:28s} p={n['prob']:.3f} band={n['band']:8s} side={n['side']} side_conf={n['side_conf']}"
                     for f, n in gr["nodes"].items()]
            run.log("  BELIEF GRAPH nodes:\n" + "\n".join(lines) + f"\n    no_acute_abnormality={gr['no_acute_abnormality']}")
            run.log("  AUDIT (changes after D1):\n" + "\n".join(
                f"    {a['rule']} {a['target']}.{a['field']}: {a['before']} -> {a['after']} ({a['reason'][:90]}){' edge ' + a['edge'] if a['edge'] else ''}"
                for a in gr["audit"] if a["rule"] != "D1"))
        (run.dir / "examples.json").write_text(json.dumps(picks, indent=1, default=str))
        bc = pd.DataFrame(bands).T
        run.log("val band fractions: " + "; ".join(f"{f}: present {r.present / 1733:.0%}, possible {r.possible / 1733:.0%}, "
                                                   f"silent {r.silent / 1733:.0%}, absent {r.absent / 1733:.0%}" for f, r in bc.iterrows()))
        run.log(f"wrote {len(out)} belief graphs + reports to {run.dir / 'sample_reports.json'}")


if __name__ == "__main__":
    main()
