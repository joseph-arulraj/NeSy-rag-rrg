"""Build the letterbox embedding store (data/embeddings/clear_frontal_lb_v1.npy + index) from the global
vectors of the letterbox region-feature shards, in exactly the row order of the stretch store index, so
NESY_EMB=letterbox runs every stage on the same studies. Written with np.save (no memmap; 335 MB).
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nesy import paths as P  # noqa: E402
from nesy.runlog import Run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="mimic_full_lb")
    ap.add_argument("--ids", default=str(P.FEATURES / "regions/mimic_full_ids.parquet"))
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("build-lb-store", vars(args), args.run_dir) as run:
        ids = pd.read_parquet(args.ids).image_id.values
        d = P.FEATURES / "regions" / f"{args.shards}_shards"
        G = np.full((len(ids), 768), np.nan, np.float16)
        n = 0
        for f in sorted(glob.glob(str(d / "shard_*_global.npy"))):
            rr = np.load(f.replace("_global.npy", "_rows.npy"))
            G[rr] = np.load(f)
            n += len(rr)
        run.log(f"global vectors from shards: {n:,} of {len(ids):,}")
        pos = {k: i for i, k in enumerate(ids)}
        src = pd.read_parquet(P.V2 / "data/embeddings/clear_frontal_v1_index.parquet")
        missing = [x for x in src.dicom_id if x not in pos]
        if missing:
            raise RuntimeError(f"{len(missing)} store images not in the pass ids")
        out = G[[pos[x] for x in src.dicom_id]]
        bad = np.isnan(out.astype(np.float32)).any(1).sum()
        if bad:
            raise RuntimeError(f"{bad} store rows have no letterbox vector (pass incomplete)")
        nrm = np.linalg.norm(out.astype(np.float32), axis=1)
        run.log(f"rows {len(out):,}; norm min {nrm.min():.4f} max {nrm.max():.4f}")
        dst = P.V2 / "data/embeddings/clear_frontal_lb_v1.npy"
        np.save(dst, out)
        idx = src.copy()
        idx["source"] = f"letterbox region pass ({args.shards})"
        idx.to_parquet(P.V2 / "data/embeddings/clear_frontal_lb_v1_index.parquet", index=False)
        run.log(f"wrote {dst} ({dst.stat().st_size / 1e6:.0f} MB) and index")


if __name__ == "__main__":
    main()
