#!/usr/bin/env python
"""RadGraph-parse all 368,294 concepts and tag anatomy/laterality/polarity/temporal_class/
canonical_finding (pipeline.md 4.6 rev 3). Resumable: interrupting and rerunning picks up from
the last completed checkpoint chunk rather than re-parsing from scratch.

  python scripts/build_concept_tags.py                       # full run (~2h on CPU, measured)
  python scripts/build_concept_tags.py --limit 500            # smoke test
  python scripts/build_concept_tags.py --config configs/hpc.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rrg.core.config import load_settings  # noqa: E402
from rrg.offline.build_concept_bank import main as build  # noqa: E402


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    build(load_settings(args.config, args.overrides), args.limit)
