#!/usr/bin/env python
"""Parse a downloaded RadLex.owl once into the compact local snapshot used at runtime.

  python scripts/build_radlex_snapshot.py
  python scripts/build_radlex_snapshot.py --config configs/hpc.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rrg.core.config import load_settings  # noqa: E402
from rrg.offline.build_radlex_snapshot import main as build  # noqa: E402


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    build(load_settings(args.config, args.overrides))
