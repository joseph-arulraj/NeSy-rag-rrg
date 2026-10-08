#!/usr/bin/env python
"""Diagnostic: show what each (source, finding) calibrator actually does -- how much data it
was fit on, and what it maps a spread of raw scores onto. Built to answer a concrete question:
why do most fused findings come out as "probable"/"possible" rather than "assertive" -- is it
genuinely under-confident evidence, or a calibrator fit on too little data collapsing everything
toward the middle?

Works on a calibrators.pkl saved before `n_samples`/`n_positive` were added to Calibrator --
falls back to "?" for those fields rather than crashing, so this doesn't require a recalibration
rerun just to inspect what you already have.

  python scripts/inspect_calibration.py --config configs/hpc.yaml
  python scripts/inspect_calibration.py --config configs/hpc.yaml --finding pneumonia
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rrg.core.config import load_settings  # noqa: E402

# Representative raw-score probe points. B's raw score is a cosine similarity (roughly -0.3..0.6
# in practice); C's raw score is already a sigmoid probability (0..1).
_B_PROBES = [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
_C_PROBES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--finding", default=None, help="only show this finding (default: all)")
    args = ap.parse_args()

    s = load_settings(args.config, args.overrides)
    with s.calibration.isotonic_cache.open("rb") as fh:
        calibrators = pickle.load(fh)

    findings = sorted({finding for (_source, finding) in calibrators})
    if args.finding:
        findings = [f for f in findings if f == args.finding]
        if not findings:
            sys.exit(f"no calibrator for finding {args.finding!r}")

    for finding in findings:
        print(f"\n=== {finding} ===")
        for source, probes in (("B", _B_PROBES), ("C", _C_PROBES)):
            cal = calibrators.get((source, finding))
            if cal is None:
                print(f"  {source}: no calibrator (finding not covered by this source)")
                continue
            n = getattr(cal, "n_samples", "?")
            n_pos = getattr(cal, "n_positive", "?")
            print(f"  {source}: kind={cal.kind} n_samples={n} n_positive={n_pos} base_rate={cal.base_rate:.3f}")
            if cal.kind == "isotonic":
                curve = ", ".join(f"{p:+.1f}->{cal.predict_proba(p):.3f}" for p in probes)
                print(f"       curve: {curve}")


if __name__ == "__main__":
    main()
