"""Runner vs the existing C+A+R results on validate (no labels used).

Reference: head run predictions_non_test.parquet (val rows) and the Stage 8 run's results.jsonl (graph per study,
expected statements and prompt derived from the template, final text = template for fallback studies).
Checks:
  1. probabilities (p_* and side-head pside_*): same study set, max |diff| <= --tol
  2. belief graphs: band, side, zone (anatomy) per finding, and the normal call, identical
  3. template reports: (a) identical to the stored template text for every fallback study; (b) identical expected
     statements and LLM prompt for every study; (c) identical to the template re-rendered from the REFERENCE
     probabilities with the reference code path (no case evidence attached) for every study.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
sys.path.insert(0, str(V2 / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True, help="run_pipeline output dir")
    ap.add_argument("--head-run", default="runs/20261007-065311_a1-s3-s4b")
    ap.add_argument("--stage8", default="runs/20261007-0925_lane-final/stage8_results/results.jsonl")
    ap.add_argument("--rules-run", default="runs/20261007-093124_rules-set")
    ap.add_argument("--s4-run", default="runs/20261006-224243_lb-stage4")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--run-dir", default=None)
    a = ap.parse_args()
    os.environ.setdefault("NESY_EMB", "letterbox")
    from nesy import data as D, pipeline_graph as PG, region_support as RS, report as R, rg_match as M
    from nesy.runlog import Run
    from stage8_phrase import prompt_for
    with Run("equivalence-check", vars(a), a.run_dir) as run:
        res = {}
        rn = pd.read_parquet(Path(a.runner) / "predictions.parquet").set_index("study_id")
        ref = pd.read_parquet(Path(a.head_run) / "predictions_non_test.parquet")
        ref = ref[ref.split == "val"].set_index("study_id")
        same = set(rn.index) == set(ref.index)
        cols = [c for c in ref.columns if c.startswith("p")]
        d = np.abs(rn.loc[ref.index, cols].values - ref[cols].values)
        res["probabilities"] = {"n_runner": len(rn), "n_reference": len(ref), "same_study_set": same,
                                "columns": len(cols), "max_abs_diff": float(np.nanmax(d)),
                                "nan_mismatch": int((np.isnan(rn.loc[ref.index, cols].values) != np.isnan(ref[cols].values)).sum())}
        res["probabilities"]["pass"] = same and res["probabilities"]["max_abs_diff"] <= a.tol and res["probabilities"]["nan_mismatch"] == 0
        run.log(f"1. probabilities: {res['probabilities']}")

        G = {json.loads(l)["study_id"]: json.loads(l) for l in open(Path(a.runner) / "graphs.jsonl")}
        S8 = {json.loads(l)["study_id"]: json.loads(l) for l in open(a.stage8)}
        S8 = {k: v for k, v in S8.items() if k in ref.index}
        diffs = {"band": 0, "side": 0, "zone": 0, "normal_call": 0}
        ex = []
        for sid, r in S8.items():
            gn, rn_ = G[sid]["graph"]["nodes"], r["graph"]["nodes"]
            for f in rn_:
                for fld in ("band", "side"):
                    if gn[f].get(fld) != rn_[f].get(fld):
                        diffs[fld] += 1
                        ex.append((sid, f, fld, rn_[f].get(fld), gn[f].get(fld)))
                z = lambda n: [x.get("zone") for x in (n.get("anatomy") or [])]  # noqa: E731
                if z(gn[f]) != z(rn_[f]):
                    diffs["zone"] += 1
                    ex.append((sid, f, "zone", z(rn_[f]), z(gn[f])))
            if G[sid]["graph"]["no_acute_abnormality"] != r["graph"]["no_acute_abnormality"]:
                diffs["normal_call"] += 1
        res["graphs"] = {"n": len(S8), **diffs, "pass": not any(diffs.values()), "examples": ex[:10]}
        run.log(f"2. graphs: {res['graphs']}")

        # 3. templates
        fb_bad = 0
        n_fb = 0
        exp_bad = prompt_bad = 0
        for sid, r in S8.items():
            tpl = G[sid]["template"]
            if r["fallback"]:
                n_fb += 1
                fb_bad += (tpl["findings"], tpl["impression"]) != (r["final_text"]["findings"], r["final_text"]["impression"])
            e = M.expected(tpl, G[sid]["graph"]["no_acute_abnormality"])
            exp_bad += e != r["expected"]
            prompt_bad += prompt_for(e, G[sid]["graph"]["no_acute_abnormality"]) != r["prompt"]
        rules = json.loads((Path(a.rules_run) / "rules_eval.json").read_text())
        thr = json.loads((Path(a.head_run) / "thresholds_4band.json").read_text())
        FIND, idx, sc = RS.load(a.s4_run)
        pos = {dd: i for i, dd in enumerate(idx.dicom_id)}
        t = pd.read_parquet(D.EMB_INDEX)[["study_id", "dicom_id"]]
        rr = ref.reset_index().merge(t, on="study_id")
        rerender_bad, rex = 0, []
        for row in rr.to_dict("records"):
            i = pos.get(row["dicom_id"])
            ev = {} if i is None else {f: RS.evidence(np.asarray(sc[i], np.float32), fi, f) for fi, f in enumerate(FIND) if f in D.FINDINGS}
            g = PG.build(row["study_id"], row, thr, ev, rules["region_thresholds"], use_r1=rules["R1"]["keep"], use_r2=rules["R2"]["keep"])
            t0 = R.render(g)
            t1 = G[row["study_id"]]["template"]
            if (t0["findings"], t0["impression"]) != (t1["findings"], t1["impression"]):
                rerender_bad += 1
                rex.append((row["study_id"], t0["findings"], t1["findings"]))
        res["templates"] = {"fallback_studies": n_fb, "fallback_text_differs": fb_bad, "expected_statements_differ": exp_bad,
                            "prompt_differs": prompt_bad, "rerendered_from_reference_differs": rerender_bad,
                            "examples": rex[:5]}
        res["templates"]["pass"] = not (fb_bad or exp_bad or prompt_bad or rerender_bad)
        run.log(f"3. templates: {res['templates']}")
        res["all_pass"] = all(res[k]["pass"] for k in ("probabilities", "graphs", "templates"))
        (run.dir / "equivalence.json").write_text(json.dumps(res, indent=1, default=str))
        run.log(f"ALL PASS: {res['all_pass']}")


if __name__ == "__main__":
    main()
