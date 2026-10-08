#!/usr/bin/env python
"""Train N17's CBM head (pipeline.md §4.7) on the official TRAIN split's CheXpert labels.

  python scripts/train_cbm.py --config configs/hpc.yaml
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rrg.cbm.train import train_cbm  # noqa: E402
from rrg.core.config import load_settings  # noqa: E402


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--limit", type=int, default=None, help="train on only the first N train-split studies (smoke test)")
    args = ap.parse_args()

    s = load_settings(args.config, args.overrides)
    report = train_cbm(s, limit=args.limit)
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
