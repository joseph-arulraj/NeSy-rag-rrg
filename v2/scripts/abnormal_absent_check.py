"""Normal-call ("No acute cardiopulmonary abnormality") check for a model's predictions. PROVISIONAL.

Rules compared (thresholds refitted on the thresh split from this model's predictions):
  root 5 %        root any_abnormality absent (<= 5 % of abnormal thresh studies missed); no critical list
  root 2 %        same with <= 2 %
  root 5 % + A    root absent (5 %) AND pneumothorax, pleural_effusion each in their own absent band
  root 5 % + B    root absent (5 %) AND pneumothorax, pleural_effusion, edema, consolidation absent
D2 applies throughout (a present/possible child lifts the root). Reported separately on val and on the
calibration split, and pooled, because the event counts are small. NOTE: Platt scaling was fitted on the
calibration split, so calib-split probabilities are in-sample for calibration (the thresholds are not:
they come from thresh).
Columns: normal calls, share of the split, share of normal calls with >= 1 positive pathology label
(support devices excluded), abnormal studies missed / all abnormal studies, and missed findings by type.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from nesy import belief as B, data as D, thresholds as TH  # noqa: E402
from nesy.evaluate import threshold_candidates  # noqa: E402
from nesy.runlog import Run  # noqa: E402

PATH = [f for f in D.FINDINGS if f not in ("any_abnormality", "support_devices")]
LISTS = {"none": [], "A": ["pneumothorax", "pleural_effusion"], "B": ["pneumothorax", "pleural_effusion", "edema", "consolidation"]}
RULES = [("root 5%", 0.05, "none"), ("root 2%", 0.02, "none"), ("root 5% + A", 0.05, "A"), ("root 5% + B", 0.05, "B")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-run", default=str(Path(__file__).resolve().parents[1] / "runs/20261006-1218_stage1-v2"))
    ap.add_argument("--label", default="stage1-v2")
    ap.add_argument("--rules", default="root 5%;root 2%;root 5% + A;root 5% + B")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("normal-call-check", vars(args), args.run_dir) as run:
        pr = pd.read_parquet(Path(args.pred_run) / "predictions_non_test.parquet")
        df = pr.merge(D.study_table()[["study_id", *D.FINDINGS]], on="study_id")
        th = df[df.split == "thresh"]
        thr_base = {f: TH.fit(th[f].values, th[f"p_{f}"].values, f) for f in D.FINDINGS}
        root_thr = {}
        for miss in (0.05, 0.02):
            c = {r["rule"]: r for r in threshold_candidates(th.any_abnormality.values, th.p_any_abnormality.values, miss_targets=(miss,))}
            root_thr[miss] = float(c[f"missed_pos<={miss}"]["thr"])
        run.log(f"root thresholds (thresh split): 5 % -> {root_thr[0.05]:.4f}, 2 % -> {root_thr[0.02]:.4f}")
        rules = [r for r in RULES if r[0] in args.rules.split(";")]
        calls = {}
        for name, miss, lst in rules:
            thr = dict(thr_base)
            thr["any_abnormality"] = {"present": None, "possible": None, "absent": root_thr[miss]}
            for sp in ("val", "calib"):
                sub = df[df.split == sp]
                calls[(name, sp)] = pd.Series([B.build(s, {f: getattr(r, f"p_{f}") for f in D.FINDINGS}, thr, critical=LISTS[lst]).no_acute_abnormality
                                               for s, r in zip(sub.study_id, sub.itertuples())], index=sub.index)
        rows, detail = [], []
        for name, miss, lst in rules:
            for sp in ("val", "calib", "val+calib"):
                sps = ["val", "calib"] if sp == "val+calib" else [sp]
                mask = pd.concat([calls[(name, s)] for s in sps])
                sub = df.loc[mask.index]
                called = sub[mask.values]
                abn = (sub[PATH] == 1).any(axis=1)
                missed = (called[PATH] == 1).any(axis=1)
                rows.append({"model": args.label, "rule": name, "split": sp, "n_split": len(sub), "normal_calls": int(mask.sum()),
                             "share_of_split": round(float(mask.mean()), 3),
                             "share_calls_with_positive": round(float(missed.mean()), 3) if len(called) else None,
                             "n_calls_with_positive": int(missed.sum()),
                             "abnormal_missed_over_all_abnormal": f"{int(missed.sum())}/{int(abn.sum())} = {missed.sum() / max(abn.sum(), 1):.1%}"})
                cnt = (called[PATH] == 1).sum()
                detail.append({"rule": name, "split": sp, **{f: int(v) for f, v in cnt.items() if v > 0}})
        out = pd.DataFrame(rows)
        det = pd.DataFrame(detail).fillna(0)
        cols = ["rule", "split"] + [c for c in det.columns if c not in ("rule", "split")]
        det = det[cols]
        for c in det.columns[2:]:
            det[c] = det[c].astype(int)
        pd.set_option("display.width", 250)
        run.log("NORMAL-CALL TABLE (provisional):\n" + out.to_string(index=False))
        run.log("missed findings by type among normal calls (studies per finding; parents include children):\n" + det.to_string(index=False))
        out.to_csv(run.dir / "normal_call_table.csv", index=False)
        det.to_csv(run.dir / "normal_call_missed_by_type.csv", index=False)


if __name__ == "__main__":
    main()
