"""Appends report-draft points 7 and 8 to RESULTS.md once the final head and lane C's bootstrap exist:
(7) final head on letterbox: AUROC/AUPRC/ECE macro and per finding, AUROC by PA and AP;
(8) per-finding DenseNet vs final head differences with the patient bootstrap;
plus the final head's hierarchy violations (same three numbers as the table for the other models).
Each section is written independently; a missing input is reported as missing, not guessed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from nesy import data as D, kg  # noqa: E402
from nesy.evaluate import binary_metrics  # noqa: E402
from nesy.runlog import Run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--boot", default="", help="lane C c-bootstrap run dir (S1, HEAD, DenseNet)")
    ap.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "RESULTS.md"))
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("results-7-8", vars(args), args.run_dir) as run:
        head = Path(args.head)
        out = [f"\n### Report-draft points 7 and 8 (written automatically by `scripts/results_7_8.py`, run `{run.dir.name}`; val, seen by backbone)",
               f"Final head run: `{head}` (concepts + CheXmask anatomy + region set B + retrieval k=25)."]
        try:
            p = pd.read_parquet(head / "predictions_non_test.parquet")
            t = D.study_table()[["study_id", "view", *D.FINDINGS]]
            v = p[p.split == "val"].merge(t, on="study_id")
            rows = []
            for view in ("all", "PA", "AP"):
                s = v if view == "all" else v[v.view == view]
                for f in D.FINDINGS:
                    rows.append({"view": view, "finding": f, **binary_metrics(s[f].values, s[f"p_{f}"].values)})
            m = pd.DataFrame(rows)
            m.to_csv(run.dir / "final_head_view_metrics.csv", index=False)
            mac = m.groupby("view")[["auroc", "auprc", "ece"]].mean()
            out.append(f"- **7. Final head:** macro AUROC {mac.loc['all', 'auroc']:.4f}, **AUPRC {mac.loc['all', 'auprc']:.4f}**, ECE {mac.loc['all', 'ece']:.4f}; "
                       f"**AUROC PA {mac.loc['PA', 'auroc']:.4f} / AP {mac.loc['AP', 'auroc']:.4f}** (n PA {int((v.view == 'PA').sum())}, AP {int((v.view == 'AP').sum())}).")
            pf = m.pivot(index="finding", columns="view", values="auroc")[["all", "PA", "AP"]]
            pf["auprc_all"] = m[m.view == "all"].set_index("finding").auprc
            out.append("  Per finding (AUROC all / PA / AP, AUPRC all):\n\n```\n" + pf.round(3).to_string() + "\n```\n")
            edges = kg.load_findings()["is_a"]
            anyv = pd.Series(False, index=v.index)
            per = {}
            for e in edges:
                b = v[f"p_{e['child']}"] > v[f"p_{e['parent']}"] + 1e-12
                per[e["id"]] = int(b.sum())
                anyv |= b
            out.append(f"- **Final head hierarchy violations (val):** per edge {per}; total {sum(per.values())}; studies with ≥ 1 violation {int(anyv.sum())}.")
        except Exception as exc:                              # noqa: BLE001
            out.append(f"- 7 / violations: NOT WRITTEN ({type(exc).__name__}: {exc})")
            run.log(f"point 7 failed: {exc}")
        try:
            b = pd.read_csv(Path(args.boot) / "bootstrap_differences.csv")
            want = b[b.comparison.isin(["HEAD - DenseNet", "DenseNet - HEAD"])].copy()
            if want.empty:
                raise ValueError("no HEAD vs DenseNet rows")
            flip = want.comparison == "DenseNet - HEAD"
            for c in ("diff",):
                want.loc[flip, c] = -want.loc[flip, c]
            lo, hi = want.ci_low.copy(), want.ci_high.copy()
            want.loc[flip, "ci_low"], want.loc[flip, "ci_high"] = -hi[flip], -lo[flip]
            want["comparison"] = "final head - DenseNet"
            tab = want[want.metric == "AUROC"][["finding", "diff", "ci_low", "ci_high", "excludes_zero"]].set_index("finding")
            tab_e = want[want.metric == "ECE"][["finding", "diff", "ci_low", "ci_high", "excludes_zero"]].set_index("finding")
            out.append(f"- **8. Final head − DenseNet (main, inner-holdout epoch), patient bootstrap 2,000 (`{Path(args.boot).name}`):** AUROC per finding:\n\n"
                       + tab.round(4).to_string() + "\n\n  ECE (negative = final head better calibrated):\n\n" + tab_e.round(4).to_string() + "\n")
        except Exception as exc:                              # noqa: BLE001
            out.append(f"- 8: NOT WRITTEN ({type(exc).__name__}: {exc})")
            run.log(f"point 8 failed: {exc}")
        text = "\n".join(out) + "\n"
        (run.dir / "appended.md").write_text(text)
        with open(args.results, "a") as fh:
            fh.write(text)
        run.log(f"appended to {args.results}:\n{text}")


if __name__ == "__main__":
    main()
