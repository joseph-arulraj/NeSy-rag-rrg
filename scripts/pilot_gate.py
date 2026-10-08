"""Automatic pass/fail gate for the letterbox pilot (user rule 2026-10-06: if the checks pass, continue;
if they fail, stop). Exit code 0 = pass, 1 = fail. Checks, against the stretch pilot on the same images:
  * left/right flip check >= 99.5 % of images
  * median same-side grid IoU (ImaGenome lung box vs CheXmask lung) within 0.08 of the stretch pilot
  * median share of the CheXmask left lung inside its ImaGenome box >= 0.95
  * no image-size mismatches
  * letterbox global vector vs an independent encode_image of the letterboxed image: cosine >= 0.999
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from nesy.runlog import Run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--stretch-pilot", required=True)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    ok = True
    with Run("pilot-gate", vars(args), args.run_dir) as run:
        lb = pd.read_csv(Path(args.pilot) / "lung_alignment.csv")
        st = pd.read_csv(Path(args.stretch_pilot) / "lung_alignment.csv")
        flip = ((lb.iou_left_same > lb.iou_left_cross) & (lb.iou_right_same > lb.iou_right_cross)).mean()
        checks = [("flip check >= 0.995", flip, flip >= 0.995)]
        for c in ("iou_left_same", "iou_right_same"):
            d = abs(lb[c].median() - st[c].median())
            checks.append((f"median {c} within 0.08 of stretch ({st[c].median():.3f})", lb[c].median(), d <= 0.08))
        cov = lb.mask_coverage_in_box_left.median()
        checks.append(("median left-lung mask share inside its box >= 0.95", cov, cov >= 0.95))
        log = (Path(args.pilot) / "log.txt").read_text()
        mm = [ln for ln in log.splitlines() if "mismatches:" in ln]
        n_mm = int(mm[-1].rsplit("mismatches:", 1)[1].strip()) if mm else -1
        checks.append(("image size mismatches == 0", n_mm, n_mm == 0))
        met = [json.loads(x) for x in (Path(args.pilot) / "metrics.jsonl").read_text().splitlines()]
        cosv = [m["global_cos_min_independent"] for m in met if "global_cos_min_independent" in m]
        checks.append(("global vs independent encode cosine >= 0.999", cosv[-1] if cosv else None, bool(cosv) and cosv[-1] >= 0.999))
        for name, val, passed in checks:
            run.log(f"{'PASS' if passed else 'FAIL'}  {name}: {val}")
            ok &= bool(passed)
        run.log("GATE " + ("PASSED" if ok else "FAILED"))
        (Path(run.dir) / "GATE").write_text("PASSED\n" if ok else "FAILED\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
