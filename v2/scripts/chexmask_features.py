"""Stage 0 step 4: decode CheXmask masks for MIMIC frontal images into named side and
cardiothoracic-ratio features. Masks are decoded in memory and discarded; only features are kept.

Source: CheXmask/Preprocessed/MIMIC-CXR-JPG.csv (masks resampled to 1024x1024). All features are
ratios or are normalised by the mask size, so they do not depend on the resampling, provided it is
uniform in x and y. `--check-orig N` compares against OriginalResolution on N images to verify this.

Side convention: CheXmask "Left Lung" / "Right Lung" are the PATIENT's left/right. In a standard
frontal display the patient's left is on the image right. That is checked for every image
(`left_lung_on_image_right`), not assumed, and the feature names below use the patient's sides.

Resumable: the CSV is streamed in fixed chunks, and each chunk's features are written to their own
shard. Shards that already exist are skipped on restart.

  python -u v2/scripts/chexmask_features.py --workers 16 [--limit-chunks 1] [--check-orig 200]
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nesy import paths as P  # noqa: E402
from nesy.runlog import Progress, Run  # noqa: E402

SRC = P.CHEXMASK / "Preprocessed/MIMIC-CXR-JPG.csv"
ORIG = P.CHEXMASK / "OriginalResolution/MIMIC-CXR-JPG.csv"
OUT = P.FEATURES / "chexmask_mimic.parquet"
SHARDS = P.FEATURES / "chexmask_mimic_shards"
CHUNK = 2000
RCA_OK = 0.7   # CheXmask README: use masks with Dice RCA (Mean) >= 0.7


def rle_decode(rle: str, h: int, w: int) -> np.ndarray:
    """CheXmask RLE: 'start length start length ...', 1-based starts, row-major."""
    if not isinstance(rle, str) or not rle.strip():
        return np.zeros((h, w), dtype=bool)
    r = np.fromstring(rle, dtype=np.int64, sep=" ")
    starts, lengths = r[0::2] - 1, r[1::2]
    flat = np.zeros(h * w + 1, dtype=np.int8)
    np.add.at(flat, starts, 1)
    np.add.at(flat, starts + lengths, -1)
    return (np.cumsum(flat[:-1]) > 0).reshape(h, w)


def _shape(m: np.ndarray, h: int, w: int, p: str) -> dict:
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return {f"{p}_area": 0.0, **{f"{p}_{k}": np.nan for k in ("x_min", "x_max", "y_min", "y_max", "cx", "cy")}}
    return {f"{p}_area": xs.size / (h * w), f"{p}_x_min": xs.min() / w, f"{p}_x_max": (xs.max() + 1) / w,
            f"{p}_y_min": ys.min() / h, f"{p}_y_max": (ys.max() + 1) / h, f"{p}_cx": xs.mean() / w, f"{p}_cy": ys.mean() / h}


def max_row_width(m: np.ndarray) -> float:
    """Widest horizontal extent of the mask over all rows (pixels)."""
    rows = np.nonzero(m.any(axis=1))[0]
    if rows.size == 0:
        return np.nan
    sub = m[rows]
    left = sub.argmax(axis=1)
    right = sub.shape[1] - sub[:, ::-1].argmax(axis=1)
    return float((right - left).max())


def features(row: tuple) -> dict:
    dicom_id, rca_mean, rca_max, rle_l, rle_r, rle_h, h, w = row
    h, w = int(h), int(w)
    ml, mr, mh = rle_decode(rle_l, h, w), rle_decode(rle_r, h, w), rle_decode(rle_h, h, w)
    f = {"dicom_id": dicom_id, "rca_mean": float(rca_mean), "rca_max": float(rca_max), "mask_h": h, "mask_w": w}
    f.update(_shape(ml, h, w, "lung_l"))
    f.update(_shape(mr, h, w, "lung_r"))
    f.update(_shape(mh, h, w, "heart"))
    lungs = ml | mr
    ok = f["lung_l_area"] > 0 and f["lung_r_area"] > 0
    f["masks_complete"] = bool(ok and f["heart_area"] > 0)
    f["left_lung_on_image_right"] = bool(ok and f["lung_l_cx"] > f["lung_r_cx"])
    if ok:
        thorax_x0 = min(f["lung_l_x_min"], f["lung_r_x_min"])
        thorax_x1 = max(f["lung_l_x_max"], f["lung_r_x_max"])
        thorax_w = thorax_x1 - thorax_x0
        f["thorax_width"] = thorax_w
        f["thorax_width_maxrow"] = max_row_width(lungs) / w
        f["heart_width"] = (f["heart_x_max"] - f["heart_x_min"]) if f["heart_area"] > 0 else np.nan
        f["heart_width_maxrow"] = max_row_width(mh) / w if f["heart_area"] > 0 else np.nan
        f["ctr"] = f["heart_width"] / thorax_w
        f["ctr_maxrow"] = f["heart_width_maxrow"] / f["thorax_width_maxrow"]
        tot = f["lung_l_area"] + f["lung_r_area"]
        f["lung_area_frac_left"] = f["lung_l_area"] / tot
        hl = f["lung_l_y_max"] - f["lung_l_y_min"]
        hr = f["lung_r_y_max"] - f["lung_r_y_min"]
        f["lung_height_frac_left"] = hl / (hl + hr)
        f["lung_base_diff_left_minus_right"] = f["lung_l_y_max"] - f["lung_r_y_max"]   # >0: left base lower in image
        f["lung_apex_diff_left_minus_right"] = f["lung_l_y_min"] - f["lung_r_y_min"]
        mid = (thorax_x0 + thorax_x1) / 2
        sign = 1.0 if f["left_lung_on_image_right"] else -1.0
        # heart centroid offset from mid-thorax, towards the patient's left (+) or right (-), in thorax widths
        f["heart_shift_to_left"] = sign * (f["heart_cx"] - mid) / thorax_w if f["heart_area"] > 0 else np.nan
    return f


def iter_chunks(path: Path, keep: set, limit_chunks: int | None):
    cols = ["dicom_id", "Dice RCA (Mean)", "Dice RCA (Max)", "Left Lung", "Right Lung", "Heart", "Height", "Width"]
    for i, chunk in enumerate(pd.read_csv(path, usecols=cols, chunksize=CHUNK)):
        if limit_chunks is not None and i >= limit_chunks:
            return
        n_in = len(chunk)
        chunk = chunk[chunk.dicom_id.isin(keep)]
        yield i, n_in, chunk[cols]


def check_orig(run: Run, n: int, pool: Pool) -> None:
    """Features from the preprocessed 1024x1024 masks vs the original-resolution masks."""
    run.log(f"check-orig: comparing {n} images between Preprocessed and OriginalResolution")
    pre = pd.read_csv(SRC, nrows=n)
    ids = set(pre.dicom_id)
    orig = next(c for c in [pd.read_csv(ORIG, nrows=n)])
    both = sorted(ids & set(orig.dicom_id))
    run.log(f"  {len(both)} of the first {n} rows share a dicom_id in both files")
    cols = ["dicom_id", "Dice RCA (Mean)", "Dice RCA (Max)", "Left Lung", "Right Lung", "Heart", "Height", "Width"]
    fp = pd.DataFrame(pool.map(features, pre[pre.dicom_id.isin(both)][cols].itertuples(index=False, name=None))).set_index("dicom_id")
    fo = pd.DataFrame(pool.map(features, orig[orig.dicom_id.isin(both)][cols].itertuples(index=False, name=None))).set_index("dicom_id")
    fo = fo.loc[fp.index]
    for c in ["ctr", "ctr_maxrow", "lung_area_frac_left", "heart_shift_to_left", "lung_base_diff_left_minus_right"]:
        d = (fp[c] - fo[c]).abs()
        run.log(f"  {c:32s} median |diff| {d.median():.4f}, 95th pct {d.quantile(0.95):.4f}, max {d.max():.4f}")
        run.metric(check_orig=c, median_abs_diff=float(d.median()), p95_abs_diff=float(d.quantile(0.95)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit-chunks", type=int, default=None, help="smoke test: only the first N chunks of the CSV")
    ap.add_argument("--check-orig", type=int, default=0, help="compare N images against OriginalResolution")
    ap.add_argument("--shard-dir", default=str(SHARDS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    cfg = {**vars(args), "src": str(SRC), "chunk": CHUNK, "rca_ok": RCA_OK}
    with Run("chexmask-features", cfg, args.run_dir) as run:
        man = pd.read_parquet(P.MANIFESTS / "mimic.parquet", columns=["dicom_id", "is_frontal", "view"])
        frontal = man[man.is_frontal]
        keep = set(frontal.dicom_id)
        run.log(f"MIMIC frontal (PA/AP) images in manifest: {len(keep):,}")
        shard_dir = Path(args.shard_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)
        # The MIMIC CheXmask file has ~243k rows; total chunks are known only after one pass, so
        # progress is reported against the frontal-image target.
        prog = Progress(run, len(keep), "frontal images processed")
        n_rows_in = n_kept = n_skipped_resume = 0
        with Pool(args.workers) as pool:
            if args.check_orig:
                check_orig(run, args.check_orig, pool)
            for i, n_in, chunk in iter_chunks(SRC, keep, args.limit_chunks):
                n_rows_in += n_in
                shard = shard_dir / f"chunk_{i:05d}.parquet"
                if shard.exists():
                    n_skipped_resume += len(chunk)
                    n_kept += len(chunk)
                    prog.update(n_kept)
                    continue
                feats = pool.map(features, chunk.itertuples(index=False, name=None), chunksize=32)
                tmp = shard.with_suffix(".tmp")
                pd.DataFrame(feats).to_parquet(tmp, index=False)
                tmp.replace(shard)
                n_kept += len(chunk)
                prog.update(n_kept)
        prog.update(n_kept, force=True)
        run.log(f"CSV rows read {n_rows_in:,}; kept (frontal in manifest) {n_kept:,}; dropped non-frontal or not in manifest "
                f"{n_rows_in - n_kept:,}; reused from existing shards {n_skipped_resume:,}")

        df = pd.concat([pd.read_parquet(p) for p in sorted(shard_dir.glob("chunk_*.parquet"))], ignore_index=True)
        dup = df.dicom_id.duplicated().sum()
        if dup:
            run.log(f"WARNING: {dup} duplicate dicom_ids in CheXmask; keeping the row with the highest rca_mean")
            df = df.sort_values("rca_mean", ascending=False).drop_duplicates("dicom_id")
        df = df.merge(frontal[["dicom_id", "view"]], on="dicom_id", how="left")
        df["qc_ok"] = (df.rca_mean >= RCA_OK) & df.masks_complete
        df["ctr_reliable_view"] = df.view == "PA"    # AP magnifies the heart; CTR on AP is flagged, not dropped
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out.with_suffix(".tmp"), index=False)
        out.with_suffix(".tmp").replace(out)

        cov = len(df) / len(keep)
        conv = df.loc[df.masks_complete, "left_lung_on_image_right"].mean()
        run.log(f"features for {len(df):,} / {len(keep):,} frontal images ({cov:.1%}); missing {len(keep) - len(df):,}")
        run.log(f"qc_ok (rca_mean >= {RCA_OK} and all 3 masks): {df.qc_ok.sum():,} ({df.qc_ok.mean():.1%}); "
                f"masks incomplete: {(~df.masks_complete).sum():,}")
        run.log(f"side convention: 'Left Lung' centroid on image right in {conv:.2%} of complete masks")
        q = df[df.qc_ok]
        for v in ["PA", "AP"]:
            c = q.loc[q.view == v, "ctr"]
            run.log(f"CTR {v}: n {len(c):,}, median {c.median():.3f}, IQR {c.quantile(.25):.3f}-{c.quantile(.75):.3f}")
        run.metric(n_features=len(df), n_frontal=len(keep), coverage=cov, qc_ok=int(df.qc_ok.sum()), side_convention_rate=conv)
        run.log(f"wrote {out}")


if __name__ == "__main__":
    main()
