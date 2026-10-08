"""Lane C, item 10: cache of the one-frontal-per-study MIMIC images used by the pipeline (the study table's
dicom_id, all splits), letterboxed to 256 px (LANCZOS, aspect kept, zero pad; same function as the CLEAR
letterbox at a different size), stored as compressed uint8 shards: data/image_cache_256/shard_NNNN.npz with
arrays img [n, 256, 256] and row [n] (row index into ids.parquet). Resumable per shard; --part i/n splits
the shards between allocations. The first shard written logs a size projection; --max-gb stops the run
before writing more if the projection exceeds it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nesy import data as D, manifests as M, paths as P  # noqa: E402
from nesy.runlog import Progress, Run  # noqa: E402

OUT = P.V2 / "data/image_cache_256"
SIZE = 256


class DS:
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        from letterbox_test import letterbox
        try:
            return i, np.asarray(letterbox(Image.open(self.paths[i]), SIZE), np.uint8), True
        except Exception:                                           # noqa: BLE001
            return i, np.zeros((SIZE, SIZE), np.uint8), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-size", type=int, default=4096)
    ap.add_argument("--part", default="0/1", help="i/n: this process writes shards k with k %% n == i")
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--max-gb", type=float, default=15.0)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("image-cache-256", vars(args), args.run_dir) as run:
        from torch.utils.data import DataLoader
        pi, pn = map(int, args.part.split("/"))
        OUT.mkdir(parents=True, exist_ok=True)
        t = D.study_table()[["dicom_id", "study_id", "subject_id", "split", "view"]]
        man = M.load("mimic", columns=["dicom_id", "image_path"])[["dicom_id", "image_path"]]
        t = t.merge(man, on="dicom_id", how="left", validate="one_to_one")
        assert t.image_path.notna().all()
        if pi == 0:
            t.drop(columns=["image_path"]).to_parquet(OUT / "ids.parquet", index=False)
        N = len(t)
        K = (N + args.shard_size - 1) // args.shard_size
        mine = [k for k in range(K) if k % pn == pi]
        todo = [k for k in mine if not (OUT / f"shard_{k:04d}.npz").exists()]
        run.log(f"studies {N:,} (one frontal image each), splits {t.split.value_counts().to_dict()}; shards {K}, "
                f"this part {len(mine)}, already written {len(mine) - len(todo)}, to do {len(todo)}")
        prog = Progress(run, len(todo) * args.shard_size, "images", every_s=120)
        done, failed, t0 = 0, 0, time.time()
        for n_sh, k in enumerate(todo):
            rr = np.arange(k * args.shard_size, min((k + 1) * args.shard_size, N))
            img = np.zeros((len(rr), SIZE, SIZE), np.uint8)
            dl = DataLoader(DS(t.image_path.values[rr]), batch_size=64, num_workers=args.workers)
            for li, x, ok in dl:
                img[li.numpy()] = x.numpy()
                failed += int((~ok.numpy().astype(bool)).sum())
            tmp = OUT / f"shard_{k:04d}.tmp.npz"
            np.savez_compressed(tmp, img=img, row=rr)
            tmp.replace(OUT / f"shard_{k:04d}.npz")
            done += len(rr)
            prog.update(done)
            if n_sh == 0:
                per = (OUT / f"shard_{k:04d}.npz").stat().st_size / len(rr)
                proj = per * N / 1e9
                run.log(f"size: {per / 1e3:.1f} kB per image -> projected {proj:.1f} GB for all {N:,} images")
                if proj > args.max_gb:
                    run.log(f"projection exceeds {args.max_gb} GB: stopping to ask (shard {k} kept)")
                    raise SystemExit(3)
        run.log(f"wrote {done:,} images in {len(todo)} shards ({failed} unreadable -> zeros); {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
