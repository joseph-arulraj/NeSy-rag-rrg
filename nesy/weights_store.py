"""CheXmask region-weight stores: either one uint8 .npy (+ _index.parquet) or compressed chunk files
(<name>_chunks/chunk_NNNNN.npz with ids + weights, + _index.parquet with image_id, chunk, pos). Compressed
chunks are used for the letterbox weights (2026-10-06) to save space: the maps are mostly zeros."""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from . import paths as P

D = P.FEATURES / "regions"


def load(names: str):
    """-> (DataFrame indexed by image_id with columns found, rca_mean, H, W, slot), weights uint8 [n, 9, 32, 32])."""
    idxs, arrs, off = [], [], 0
    for name in names.split(","):
        chunks = sorted(glob.glob(str(D / f"{name}_chunks" / "chunk_*.npz")))
        if chunks:
            ids, ws = [], []
            for c in chunks:
                z = np.load(c, allow_pickle=True)
                ids += list(z["ids"])
                ws.append(z["w"])
            w = np.concatenate(ws) if ws else np.zeros((0, 9, 32, 32), np.uint8)
            found = pd.read_parquet(D / f"{name}_found.parquet").drop_duplicates("image_id").set_index("image_id")
            df = pd.DataFrame({"image_id": ids, "slot": np.arange(len(ids)) + off}).drop_duplicates("image_id", keep="last")
            df = df.join(found, on="image_id")
        else:
            wi = pd.read_parquet(D / f"{name}_index.parquet")
            wi = wi[wi.found]
            w = np.load(D / f"{name}_weights.npy", mmap_mode="r")
            df = wi.assign(slot=wi.row.values + off)
            w = w  # memmap; concatenation below forces RAM only if several stores are mixed
        df["found"] = True
        idxs.append(df)
        arrs.append(w)
        off += len(w)
    W = arrs[0] if len(arrs) == 1 else np.concatenate([np.asarray(a) for a in arrs])
    return pd.concat(idxs).drop_duplicates("image_id").set_index("image_id"), W
