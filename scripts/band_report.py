"""Per-finding band shares and tier precision/sensitivity for a model's predictions (four bands v2:
present p >= 0.70, possible 0.40-0.70, absent by the miss rule refitted on thresh, silent otherwise).

Bands here are each finding's OWN band (before the D2 parent raise), so tier precision/sensitivity
describe that finding's probability alone. Reported on val, on calib (Platt was fitted there, so
calib probabilities are in-sample for calibration), and pooled.
  precision(tier)   = positives among studies in the tier / studies in the tier
  sensitivity(tier) = positives in the tier / all positives
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nesy import belief as B, data as D, thresholds as TH  # noqa: E402
from nesy.runlog import Run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-run", required=True)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("band-report", vars(args), args.run_dir) as run:
        pr = pd.read_parquet(Path(args.pred_run) / "predictions_non_test.parquet")
        df = pr.merge(D.study_table()[["study_id", *D.FINDINGS]], on="study_id")
        th = df[df.split == "thresh"]
        thr = {f: TH.fit(th[f].values, th[f"p_{f}"].values, f) for f in D.FINDINGS}
        run.log("thresholds: " + "; ".join(f"{f}: present {v['present']}, possible {v['possible']}, absent "
                                           f"{None if v['absent'] is None else round(v['absent'], 4)}" for f, v in thr.items()))
        rows = []
        for sp in ("val", "calib", "val+calib"):
            sub = df[df.split.isin(["val", "calib"])] if sp == "val+calib" else df[df.split == sp]
            for f in D.FINDINGS:
                p, y = sub[f"p_{f}"].values, sub[f].values
                bands = np.array([B.band(x, thr[f]) for x in p])
                known = ~np.isnan(y)
                r = {"split": sp, "finding": f, "n": len(sub), "pos": int((y == 1).sum()), "max_p": round(float(p.max()), 3)}
                for b in ("present", "possible", "silent", "absent"):
                    r[f"share_{b}"] = round(float((bands == b).mean()), 3)
                for tier in ("present", "possible"):
                    m = (bands == tier) & known
                    r[f"{tier}_n"] = int(m.sum())
                    r[f"{tier}_precision"] = round(float((y[m] == 1).mean()), 3) if m.any() else None
                    r[f"{tier}_sensitivity"] = round(float((y[m] == 1).sum() / max((y[known] == 1).sum(), 1)), 3)
                ma = (bands == "absent") & known
                r["absent_missed_pos"] = int((y[ma] == 1).sum())
                rows.append(r)
        out = pd.DataFrame(rows)
        pd.set_option("display.width", 260)
        pd.set_option("display.max_rows", 200)
        for sp in ("val", "calib", "val+calib"):
            run.log(f"BANDS ({sp}):\n" + out[out.split == sp].drop(columns="split").to_string(index=False))
        out.to_csv(run.dir / "band_report.csv", index=False)


if __name__ == "__main__":
    main()
