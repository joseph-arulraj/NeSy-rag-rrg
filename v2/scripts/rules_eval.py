"""Lane A item 5: two knowledge-based band rules on validate (bands / sides only, never probabilities).

Region threshold per finding (fitted on the THRESH split, never validate): the 5th percentile of the
region support (max relevant CheXmask-region score from Stage 4) among thresh studies that are in the
present band and label-positive, i.e. on thresh the rule would demote at most ~5% of present-band true
positives. Findings with fewer than 10 such studies get no threshold (the rule does not apply to them).

  R1 localisation support: present finding with support < threshold -> possible.
     Metric: precision of the present tier (pooled over findings with a Stage 4 classifier; micro).
     Also reported: present-tier sensitivity, number demoted, per finding.
  R2 side agreement: state a side only when the side head and the region-derived side agree.
     Metric: side accuracy among stated sides (explicit ImaGenome side labels only).
     Also reported: share of stated lateralisable findings that keep a side.
A rule is kept only if its metric improves on validate with a patient-level bootstrap (2,000) 95% CI of
the change that excludes zero (the project's standard for every keep/drop decision).
Writes rules_eval.json (decisions) and region_thresholds.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nesy import data as D, kg, paths as P, pipeline_graph as PG, region_support as RS  # noqa: E402
from nesy import report as R  # noqa: E402
from nesy.runlog import Progress, Run  # noqa: E402


def graphs(df, thr, evs, rthr, r1, r2):
    out = {}
    for row in df.to_dict("records"):
        sid = row["study_id"]
        g = PG.build(sid, row, thr, evs.get(sid, {}), rthr, use_r1=r1, use_r2=r2)
        out[sid] = g
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-run", required=True, help="head run (predictions_non_test.parquet, thresholds_4band.json)")
    ap.add_argument("--s4-run", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("rules-eval", vars(args), args.run_dir) as run:
        pr = Path(args.pred_run)
        preds = pd.read_parquet(pr / "predictions_non_test.parquet")
        preds = preds[preds.split.isin(["thresh", "val"])]
        thr = json.loads((pr / "thresholds_4band.json").read_text())
        t = D.study_table()[["study_id", "subject_id", "dicom_id", *D.FINDINGS]]
        df = preds.merge(t, on="study_id", suffixes=("", "_lab"))
        FIND, idx, sc = RS.load(args.s4_run)
        pos = {d: i for i, d in enumerate(idx.dicom_id)}
        evs = {}
        for r in df[["study_id", "dicom_id"]].itertuples():
            i = pos.get(r.dicom_id)
            if i is None:
                continue
            a = np.asarray(sc[i], np.float32)
            evs[r.study_id] = {f: RS.evidence(a, fi, f) for fi, f in enumerate(FIND) if f in D.FINDINGS}
        run.log(f"thresh {int((df.split == 'thresh').sum()):,} and val {int((df.split == 'val').sum()):,} studies; "
                f"region evidence for {len(evs):,}; findings with a region classifier {FIND}")

        # ---- region thresholds on THRESH
        th = df[df.split == "thresh"]
        g0 = graphs(th, thr, evs, {}, False, False)
        thl = th.set_index("study_id")
        rthr = {}
        for f in FIND:
            if f not in D.FINDINGS:
                continue
            s = [evs[sid][f]["support"] for sid, g in g0.items() if g.nodes[f].band == "present"
                 and thl.at[sid, f] == 1 and sid in evs and evs[sid][f]["support"] is not None]
            if len(s) >= 10:
                rthr[f] = float(np.percentile(s, 5))
            run.log(f"  region threshold {f:28s} n present TP on thresh {len(s):4d} -> {rthr.get(f)}")
        (run.dir / "region_thresholds.json").write_text(json.dumps(rthr, indent=2))

        # ---- validate: before / after
        va = df[df.split == "val"].reset_index(drop=True)
        lab = va.set_index("study_id")
        side_lab = pd.read_parquet(P.FEATURES / "imagenome_side_explicit_v1.parquet")
        side_lab = side_lab[side_lab.side_text != "default"].drop_duplicates(["dicom_id", "finding"]).set_index(["dicom_id", "finding"]).side_text
        base = graphs(va, thr, evs, rthr, False, False)
        G1 = graphs(va, thr, evs, rthr, True, False)
        G2 = graphs(va, thr, evs, rthr, False, True)
        R1F = [f for f in rthr]
        rows = []
        for sid in va.study_id:
            pat = lab.at[sid, "subject_id"]
            for f in R1F:
                y = lab.at[sid, f]
                if np.isnan(y):
                    continue
                rows.append({"study_id": sid, "subject_id": pat, "finding": f, "y": int(y),
                             "pres0": base[sid].nodes[f].band == "present", "pres1": G1[sid].nodes[f].band == "present"})
        r1 = pd.DataFrame(rows)
        srows = []
        for sid in va.study_id:
            did = lab.at[sid, "dicom_id"]
            for f in D.FINDINGS:
                if not kg.load_findings()["_by_id"][f]["lateralisable"]:
                    continue
                n0, n2 = base[sid].nodes[f], G2[sid].nodes[f]
                if n0.band not in ("present", "possible"):
                    continue
                s0, s2 = R.shown_side(f, n0), R.shown_side(f, n2)
                truth = side_lab.get((did, f))
                srows.append({"study_id": sid, "subject_id": lab.at[sid, "subject_id"], "finding": f, "truth": truth,
                              "side0": s0, "side2": s2})
        r2 = pd.DataFrame(srows)

        def m1(d):
            p0, p1 = d[d.pres0], d[d.pres1]
            return {"prec0": p0.y.mean(), "prec1": p1.y.mean(), "sens0": d[d.y == 1].pres0.mean(), "sens1": d[d.y == 1].pres1.mean(),
                    "n_present0": int(d.pres0.sum()), "n_present1": int(d.pres1.sum()), "demoted": int((d.pres0 & ~d.pres1).sum()),
                    "demoted_true_pos": int((d.pres0 & ~d.pres1 & (d.y == 1)).sum())}

        def m2(d):
            k = d[d.truth.notna()]
            a0 = k[k.side0.notna()]
            a2 = k[k.side2.notna()]
            return {"acc0": (a0.side0 == a0.truth).mean() if len(a0) else np.nan, "acc2": (a2.side2 == a2.truth).mean() if len(a2) else np.nan,
                    "n_stated_with_truth0": len(a0), "n_stated_with_truth2": len(a2),
                    "keep_side0": d.side0.notna().mean(), "keep_side2": d.side2.notna().mean(), "n_lateral_stated": len(d)}

        o1, o2 = m1(r1), m2(r2)
        rng = np.random.default_rng(0)
        pats = va.subject_id.unique()
        g1 = {p: d for p, d in r1.groupby("subject_id")}
        g2 = {p: d for p, d in r2.groupby("subject_id")}
        d1, d2 = [], []
        prog = Progress(run, args.n_boot, "bootstrap resamples", every_s=60)
        for b in range(args.n_boot):
            pb = rng.choice(pats, len(pats), replace=True)
            b1 = pd.concat([g1[p] for p in pb if p in g1])
            b2 = pd.concat([g2[p] for p in pb if p in g2])
            x1, x2 = m1(b1), m2(b2)
            d1.append(x1["prec1"] - x1["prec0"])
            d2.append(x2["acc2"] - x2["acc0"])
            prog.update(b + 1)
        ci1 = np.nanpercentile(d1, [2.5, 97.5]).tolist()
        ci2 = np.nanpercentile(d2, [2.5, 97.5]).tolist()
        keep1 = bool(ci1[0] > 0)
        keep2 = bool(ci2[0] > 0)
        per1 = pd.DataFrame({f: m1(d) for f, d in r1.groupby("finding")}).T
        per2 = pd.DataFrame({f: m2(d) for f, d in r2.groupby("finding")}).T
        run.log("R1 localisation support (val, pooled over " + ", ".join(R1F) + "):\n"
                + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o1.items()})
                + f"\n  precision change {o1['prec1'] - o1['prec0']:+.4f}, 95% CI [{ci1[0]:+.4f}, {ci1[1]:+.4f}] -> "
                + ("KEEP (precision improves, CI excludes 0)" if keep1 else "DROP (no improvement with CI excluding 0)")
                + "\nper finding:\n" + per1.round(3).to_string())
        run.log("R2 side agreement (val, stated lateralisable findings):\n"
                + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o2.items()})
                + f"\n  side-accuracy change {o2['acc2'] - o2['acc0']:+.4f}, 95% CI [{ci2[0]:+.4f}, {ci2[1]:+.4f}] -> "
                + ("KEEP (side accuracy improves, CI excludes 0)" if keep2 else "DROP (no improvement with CI excluding 0)")
                + "\nper finding:\n" + per2.round(3).to_string())
        res = {"R1": {**o1, "ci": ci1, "keep": bool(keep1)}, "R2": {**o2, "ci": ci2, "keep": bool(keep2)},
               "region_thresholds": rthr, "pred_run": str(pr), "s4_run": args.s4_run}
        (run.dir / "rules_eval.json").write_text(json.dumps(res, indent=2, default=float))
        per1.to_csv(run.dir / "R1_per_finding.csv")
        per2.to_csv(run.dir / "R2_per_finding.csv")
        run.metric(summary="rules", r1_prec0=o1["prec0"], r1_prec1=o1["prec1"], r1_keep=bool(keep1),
                   r2_acc0=o2["acc0"], r2_acc2=o2["acc2"], r2_keep=bool(keep2))


if __name__ == "__main__":
    main()
