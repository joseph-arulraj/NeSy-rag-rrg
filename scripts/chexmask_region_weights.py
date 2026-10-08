"""Stream a CheXmask OriginalResolution CSV once and store, for each wanted image, the 9 CheXmask
region weight maps on CLEAR's 32x32 patch grid (uint8, 255 = patch fully inside the region).
Masks themselves are decoded in memory and discarded.

  python -u scripts/chexmask_region_weights.py --dataset mimic --ids <parquet with image ids> --out-name <name>

Output: data/features/regions/<out-name>_weights.npy (uint8 [N, 9, 32, 32]) + <out-name>_index.parquet
(image_id, row, rca_mean, found). Resumable per chunk: chunks already stored are skipped.
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from chexmask_features import rle_decode  # noqa: E402
from nesy import paths as P  # noqa: E402
from nesy.regions import CHEXMASK_REGIONS, chexmask_region_masks, mask_to_grid  # noqa: E402
from nesy.runlog import Progress, Run  # noqa: E402

SRC = {"mimic": ("MIMIC-CXR-JPG.csv", "dicom_id"), "vindr": ("VinDr-CXR.csv", "image_id"), "padchest": ("Padchest.csv", "ImageID")}
CHUNK = 1000


GEOMETRY = "stretch"


def weights(row):
    iid, rca, l, r, h, H, W = row
    H, W = int(H), int(W)
    ml, mr, mh = rle_decode(l, H, W), rle_decode(r, H, W), rle_decode(h, H, W)
    reg = chexmask_region_masks(ml, mr, mh)
    g = np.stack([mask_to_grid(reg[k], GEOMETRY) for k in CHEXMASK_REGIONS])
    return iid, float(rca), (g * 255 + 0.5).astype(np.uint8), (H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(SRC), required=True)
    ap.add_argument("--ids", required=True, help="parquet with an image_id column (the images wanted, in output order)")
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--geometry", choices=["stretch", "letterbox"], default="stretch")
    ap.add_argument("--compressed", action="store_true",
                    help="write compressed chunk files (<out-name>_chunks/*.npz) instead of one .npy (saves space)")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    global GEOMETRY
    GEOMETRY = args.geometry
    fname, idcol = SRC[args.dataset]
    src = P.CHEXMASK / "OriginalResolution" / fname
    with Run(f"chexmask-weights-{args.out_name}", {**vars(args), "src": str(src), "regions": CHEXMASK_REGIONS}, args.run_dir) as run:
        want = pd.read_parquet(args.ids).image_id.astype(str).tolist()
        pos = {k: i for i, k in enumerate(want)}
        out_dir = P.FEATURES / "regions"
        out_dir.mkdir(parents=True, exist_ok=True)
        wpath = out_dir / f"{args.out_name}_weights.npy"
        from nesy import safe_memmap
        wshape = (len(want), len(CHEXMASK_REGIONS), 32, 32)
        cdir = out_dir / f"{args.out_name}_chunks"
        if args.compressed:
            cdir.mkdir(exist_ok=True)
            W = None
        else:
            W = np.lib.format.open_memmap(wpath, mode="r+") if wpath.exists() else safe_memmap.create(wpath, np.uint8, wshape)
        done_path = out_dir / f"{args.out_name}_found.parquet"
        found = pd.read_parquet(done_path) if done_path.exists() else pd.DataFrame(columns=["image_id", "rca_mean", "H", "W"])
        have = set(found.image_id)
        run.log(f"wanted {len(want):,} images from {src.name}; already stored {len(have):,}")
        prog = Progress(run, len(want), "images with region weights", every_s=60)
        rows_in = 0
        cols = [idcol, "Dice RCA (Mean)", "Left Lung", "Right Lung", "Heart", "Height", "Width"]
        with Pool(args.workers) as pool:
            for chunk in pd.read_csv(src, usecols=cols, chunksize=CHUNK, dtype={idcol: str}):
                rows_in += len(chunk)
                chunk = chunk[chunk[idcol].isin(pos) & ~chunk[idcol].isin(have)]
                if len(chunk) == 0:
                    continue
                new, cids, cws = [], [], []
                for iid, rca, g, (h, w) in pool.imap(weights, chunk[cols].itertuples(index=False, name=None), chunksize=8):
                    if W is not None:
                        W[pos[iid]] = g
                    else:
                        cids.append(iid)
                        cws.append(g)
                    new.append({"image_id": iid, "rca_mean": rca, "H": h, "W": w})
                    have.add(iid)
                if W is not None:
                    W.flush()
                else:
                    k = len(list(cdir.glob("chunk_*.npz")))
                    tmp = cdir / f"chunk_{k:05d}.tmp.npz"
                    np.savez_compressed(tmp, ids=np.array(cids, dtype=object), w=np.stack(cws))
                    tmp.replace(cdir / f"chunk_{k:05d}.npz")
                found = pd.concat([found, pd.DataFrame(new)], ignore_index=True)
                found.to_parquet(done_path, index=False)
                prog.update(len(have))
                if len(have) == len(want):
                    break
        prog.update(len(have), force=True)
        if W is not None:
            safe_memmap.close(W, wpath, np.uint8, wshape)
        idx = pd.DataFrame({"image_id": want, "row": np.arange(len(want))}).merge(found, on="image_id", how="left")
        idx["found"] = idx.rca_mean.notna()
        idx.to_parquet(out_dir / f"{args.out_name}_index.parquet", index=False)
        run.log(f"CSV rows read {rows_in:,}; images found {int(idx.found.sum()):,} / {len(want):,}; missing {int((~idx.found).sum()):,}")
        if W is not None:
            run.log(f"wrote {wpath} ({wpath.stat().st_size / 1e9:.2f} GB) and index")
        else:
            gb = sum(f.stat().st_size for f in cdir.glob("chunk_*.npz")) / 1e9
            run.log(f"wrote {len(list(cdir.glob('chunk_*.npz')))} compressed chunks in {cdir} ({gb:.2f} GB) and index")


if __name__ == "__main__":
    main()
