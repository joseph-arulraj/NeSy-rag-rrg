"""Lane A items 1-2 decision: which feature groups the final head keeps. Rules fixed before seeing the
results; revised by the user on 2026-10-07 07:50, before this step first ran. From the patient-bootstrap
differences on validate:

CheXmask anatomy features (A) and region features (R, CheXmask-derived set B): NO-HARM rule. They have
other jobs (measurement, side, location), so redundancy for detection is not a reason to drop them.
  A: drop only if the head WITHOUT A is significantly better (95% CI excludes 0 in its favour) on macro
     AUROC or macro ECE, compared with region features present (S3s4B vs S3s4BnoCM); otherwise keep.
     The same comparison without region features (S3 vs S3noCM) and the cardiomegaly / enlarged
     cardiomediastinum AUROC differences are reported alongside.
  R: drop only if the head without R (S3) is significantly better than with R (S3s4B) on macro AUROC or
     macro ECE; otherwise keep.
Retrieval (accuracy-only component; S3nbK - S3 macro AUROC, K = 5, 10, 25):
  keep the K whose CI excludes 0 in favour of retrieval (largest difference if several); if none, drop
  retrieval and record a negative result.
Writes best_head.json: {"chexmask": bool, "nb": K or null, "extra": "...", "reuse_run": dir or null}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from nesy.runlog import Run  # noqa: E402


def row(b, comp, metric, finding="MACRO"):
    r = b[(b.comparison == comp) & (b.metric == metric) & (b.finding == finding)]
    if r.empty:
        raise KeyError(f"{comp} {metric} {finding} not in bootstrap table")
    return r.iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1-boot", required=True)
    ap.add_argument("--a2-boot", required=True)
    ap.add_argument("--s3s4b", required=True)
    ap.add_argument("--s3s4b-nocm", required=True)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("decide-head", vars(args), args.run_dir) as run:
        b1 = pd.read_csv(Path(args.a1_boot) / "bootstrap_differences.csv")
        b2 = pd.read_csv(Path(args.a2_boot) / "bootstrap_differences.csv")

        def diff(b, a, c, metric, finding="MACRO"):
            """a - c, whichever order the table holds."""
            try:
                r = row(b, f"{a} - {c}", metric, finding)
                return r["diff"], r.ci_low, r.ci_high
            except KeyError:
                r = row(b, f"{c} - {a}", metric, finding)
                return -r["diff"], -r.ci_high, -r.ci_low

        lines, better, worse = [], [], []
        for ctx, w, wo in (("with region features (decides)", "S3s4B", "S3s4BnoCM"), ("without region features", "S3", "S3noCM")):
            for metric, fnd in (("AUROC", "MACRO"), ("ECE", "MACRO"), ("AUROC", "cardiomegaly"), ("AUROC", "enlarged_cardiomediastinum")):
                d, lo, hi = diff(b1, w, wo, metric, fnd)
                good = (lo > 0) if metric == "AUROC" else (hi < 0)            # ECE: lower is better
                bad = (hi < 0) if metric == "AUROC" else (lo > 0)
                lines.append(f"  CheXmask {ctx}: {metric} {fnd}: with - without = {d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                             + (" (better with)" if good else " (worse with)" if bad else ""))
                if w == "S3s4B":
                    if good:
                        better.append(f"{metric} {fnd}")
                    if bad and fnd == "MACRO":
                        worse.append(f"{metric} {fnd}")
        keep_cm = not worse                                   # no-harm rule
        run.log("CheXmask anatomy ablation:\n" + "\n".join(lines))
        run.log(f"-> CheXmask anatomy {'KEEP' if keep_cm else 'DROP'} (no-harm rule: removing it significantly better on "
                f"{worse or 'nothing'}; significantly better WITH it on {better or 'nothing'})")
        r_lines, r_worse = [], []
        for metric in ("AUROC", "ECE"):
            d, lo, hi = diff(b1, "S3s4B", "S3", metric)        # with R - without R
            bad = (hi < 0) if metric == "AUROC" else (lo > 0)
            r_lines.append(f"  region features: {metric} MACRO: with - without = {d:+.4f} [{lo:+.4f}, {hi:+.4f}]" + (" (worse with)" if bad else ""))
            if bad:
                r_worse.append(f"{metric} MACRO")
        keep_r = not r_worse
        run.log("Region features (set B) ablation:\n" + "\n".join(r_lines))
        run.log(f"-> region features {'KEEP' if keep_r else 'DROP'} (no-harm rule; removing them significantly better on {r_worse or 'nothing'})")

        nb_lines, cands = [], []
        for k in (5, 10, 25):
            d, lo, hi = diff(b2, f"S3nb{k}", "S3", "AUROC")
            de, loe, hie = diff(b2, f"S3nb{k}", "S3", "ECE")
            nb_lines.append(f"  k={k}: AUROC {d:+.4f} [{lo:+.4f}, {hi:+.4f}]; ECE {de:+.4f} [{loe:+.4f}, {hie:+.4f}]")
            if lo > 0:
                cands.append((d, k))
        nb = max(cands)[1] if cands else None
        run.log("Retrieval (Stage 3 + nbK vs Stage 3, macro):\n" + "\n".join(nb_lines))
        run.log(f"-> retrieval {'KEEP k=' + str(nb) if nb else 'DROP (no interval excludes zero): negative result'}")

        extra = ",".join([g for g in (("chexmask" if keep_cm else None), ("s4" if keep_r else None), (f"nb{nb}" if nb else None)) if g])
        reuse = None
        if not nb and keep_r:
            reuse = args.s3s4b if keep_cm else args.s3s4b_nocm
        if not nb and not keep_r and keep_cm:
            reuse = None                                          # concepts + chexmask = the existing letterbox Stage 3; retrained for a uniform record
        res = {"chexmask": keep_cm, "regions": keep_r, "nb": nb, "extra": extra, "s4_sets": "B", "reuse_run": reuse,
               "rules": "A, R: no-harm; retrieval: CI must exclude 0", "chexmask_better_with": better,
               "chexmask_worse_without": worse, "regions_worse_without": r_worse}
        (run.dir / "best_head.json").write_text(json.dumps(res, indent=2))
        run.log(f"best head: concepts + {extra} (region set B)" + (f"; reuses {reuse}" if reuse else "; needs a new run"))


if __name__ == "__main__":
    main()
