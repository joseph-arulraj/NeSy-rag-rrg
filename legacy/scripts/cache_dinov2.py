#!/usr/bin/env python
"""Populate the torch-hub cache with the DINOv2 *source code* that CLEAR's loader needs.

CLEAR builds its ViT-B/14 image tower from facebookresearch/dinov2 (code only; the weights come from
best_model.pt). By default torch.hub fetches that code from GitHub at load time, which fails on
compute nodes without internet. Run this ONCE on a machine with internet (login node), then set
runtime.torch_hub_dir to the same folder and clear.local_files_only=true (configs/hpc.yaml does).

  python scripts/cache_dinov2.py --config configs/hpc.yaml        # uses runtime.torch_hub_dir
  python scripts/cache_dinov2.py --dest /some/dir
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rrg.core.config import load_settings  # noqa: E402

REPO = "https://github.com/facebookresearch/dinov2"
DIRNAME = "facebookresearch_dinov2_main"   # the exact folder name clear.hub looks for


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--dest", default=None, help="hub directory (default: runtime.torch_hub_dir)")
    args = ap.parse_args()

    s = load_settings(args.config, args.overrides)
    hub_dir = Path(args.dest).expanduser() if args.dest else s.runtime.torch_hub_dir
    if hub_dir is None:
        sys.exit("set runtime.torch_hub_dir in your config (or pass --dest)")
    if "CHANGE_ME" in str(hub_dir):
        sys.exit(f"runtime.torch_hub_dir still has the CHANGE_ME placeholder: {hub_dir}")
    target = hub_dir / DIRNAME
    if (target / "hubconf.py").is_file():
        print(f"already cached: {target}")
        return
    hub_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO, str(target)], check=True)
    if not (target / "hubconf.py").is_file():
        sys.exit(f"clone finished but {target}/hubconf.py is missing")
    print(f"cached DINOv2 source at {target}\nset runtime.torch_hub_dir: {hub_dir}  and  clear.local_files_only: true")


if __name__ == "__main__":
    main()
