"""Val metrics split by view (PA / AP) for several models: AUROC, AUPRC, ECE, n, positives.

  python -u scripts/view_metrics.py --runs S1=<dir> S2=<dir> S3=<dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from nesy import data as D  # noqa: E402
from nesy.evaluate import binary_metrics  # noqa: E402
from nesy.runlog import Run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("view-metrics", vars(args), args.run_dir) as run:
        t = D.study_table()[["study_id", "view", *D.FINDINGS]]
        rows = []
        for spec in args.runs:
            name, rd = spec.split("=", 1)
            pr = pd.read_parquet(Path(rd) / "predictions_non_test.parquet")
            df = pr[pr.split == "val"].merge(t, on="study_id")
            for view in ("PA", "AP", "all"):
                s = df if view == "all" else df[df.view == view]
                for f in D.FINDINGS:
                    m = binary_metrics(s[f].values, s[f"p_{f}"].values)
                    rows.append({"model": name, "view": view, "finding": f, **m})
        out = pd.DataFrame(rows)
        out.to_csv(run.dir / "view_metrics.csv", index=False)
        pd.set_option("display.width", 250)
        pd.set_option("display.max_rows", 300)
        for metric in ("auroc", "ece"):
            piv = out.pivot_table(index="finding", columns=["view", "model"], values=metric)
            run.log(f"{metric.upper()} by view (val):\n" + piv.round(3).to_string())
            run.log(f"macro {metric}: " + out.groupby(["view", "model"])[metric].mean().round(4).to_string().replace("\n", "; "))
        cnt = out[out.model == out.model.iloc[0]].pivot_table(index="finding", columns="view", values=["n", "pos"])
        run.log("n / positives by view (val):\n" + cnt.to_string())


if __name__ == "__main__":
    main()
