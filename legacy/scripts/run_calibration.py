#!/usr/bin/env python
"""Fit B/C calibrators + per-finding admission thresholds on the official VALIDATE split
(pipeline.md §10.2). Requires the CBM head to already be trained (scripts/train_cbm.py) and
concept tags to be fresh (scripts/build_concept_tags.py).

  python scripts/run_calibration.py --config configs/hpc.yaml
"""
from __future__ import annotations

import os

# Must be set before `radgraph`/`transformers` are imported anywhere below (it's read once at
# import time) -- not relying on the caller having exported it in their shell, since that's easy
# to forget across separate terminal sessions and doesn't reliably suppress the warning anyway
# (RadGraph's own internal AllenNLP-based multiprocessing can retrigger it regardless).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rrg.core.config import load_settings  # noqa: E402
from rrg.fusion.calibrate import run_calibration  # noqa: E402


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--limit", type=int, default=None, help="calibrate on only the first N validate-split studies (smoke test)")
    args = ap.parse_args()

    s = load_settings(args.config, args.overrides)
    artifacts = run_calibration(s, limit=args.limit)
    print(f"wrote {s.calibration.isotonic_cache} and {s.calibration.threshold_cache}")
    print(f"{len(artifacts.calibrators)} calibrators, {len(artifacts.thresholds)} thresholds")


if __name__ == "__main__":
    main()
