#!/usr/bin/env python
"""Build the FAISS retrieval index from the TRAIN split (offline, run once per encoder checkpoint).

  python scripts/build_faiss_index.py                            # Mac sample (50 train studies)
  python scripts/build_faiss_index.py --config configs/hpc.yaml  # full MIMIC-CXR train split on the cluster
  python scripts/build_faiss_index.py --limit 200                # smoke test on the first 200 studies (marked in the manifest)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rrg.core.config import load_settings  # noqa: E402
from rrg.core.runtime import configure_runtime  # noqa: E402
from rrg.offline.build_faiss_index import build_faiss_index  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--limit", type=int, default=None, help="index only the first N studies (smoke test)")
    args = ap.parse_args()
    s = load_settings(args.config, args.overrides)
    device = configure_runtime(s)
    print(f"device={device}  workers={s.runtime.num_workers}  precision={s.clear.precision}")
    m = build_faiss_index(s, device, args.limit)
    print(json.dumps({k: v for k, v in asdict(m).items() if k != "failed_dicom_ids"}, indent=2))


if __name__ == "__main__":
    main()
