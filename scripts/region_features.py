"""GPU: CLEAR global embedding + patch tokens pooled in two region sets, for MIMIC images.

  a. Chest ImaGenome region boxes (36 per image, original pixel coordinates)
  b. CheXmask-derived regions (9: each lung, its upper/middle/lower thirds, heart), from
     data/features/regions/<weights-name>_weights.npy (OriginalResolution masks on the 32x32 grid)

Tokens: DINOv2 backbone.forward_features()["x_norm_patchtokens"] (32x32 grid at 448 px), passed
through CLEAR's visual projection (linear, so pooling-then-projecting == projecting-then-pooling).
Global: projection(CLS), L2-normalised, which is exactly encode_image (checked against the cache).
Saved per image as float16: global [768], imagenome [36, 768], chexmask [9, 768] (+ presence masks).
Raw patch tokens are never saved. Resumable: rows already marked done are skipped.

  python -u scripts/region_features.py --ids data/features/regions/mimic_pilot_ids.parquet \
      --weights-name mimic_pilot --out-name mimic_pilot --overlays 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nesy import clear_model, manifests as M, paths as P  # noqa: E402
from nesy.regions import CHEXMASK_REGIONS, GRID, box_weights  # noqa: E402
from nesy.runlog import Progress, Run  # noqa: E402

OUT = P.FEATURES / "regions"
IG_REGIONS = sorted(pd.read_csv(P.IMAGENOME / "semantics/objects_detectable_by_bbox_pipeline_v1.txt", header=None)[0]
                    .str.strip().str.rstrip(",").tolist())


class DS:
    draft = False     # PIL JPEG draft decode at >= 2x the 448 target, then the unchanged CLEAR preprocess
    geometry = "stretch"   # "letterbox": the CLEAR authors' aspect-preserving resize + zero padding before preprocess

    def __init__(self, df, pre):
        self.df, self.pre = df.reset_index(drop=True), pre

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        from PIL import Image
        r = self.df.iloc[i]
        im = Image.open(r.image_path)
        W, H = im.size                       # original size, used for region coordinates
        if self.draft:
            im.draft("L", (896, 896))
        im.load()
        if self.geometry == "letterbox":
            from letterbox_test import letterbox
            im = letterbox(im)
        return i, self.pre(im), W, H


def imagenome_weights(boxes: pd.DataFrame, W: int, H: int, geometry: str = "stretch") -> tuple[np.ndarray, np.ndarray]:
    w = np.zeros((len(IG_REGIONS), GRID, GRID), np.float32)
    present = np.zeros(len(IG_REGIONS), bool)
    for b in boxes.itertuples():
        if b.region in IG_INDEX and b.original_x2 > b.original_x1 and b.original_y2 > b.original_y1:
            k = IG_INDEX[b.region]
            w[k] = box_weights(b.original_x1, b.original_y1, b.original_x2, b.original_y2, W, H, geometry)
            present[k] = w[k].sum() > 0
    return w, present


IG_INDEX = {r: i for i, r in enumerate(IG_REGIONS)}


def overlay(img448: np.ndarray, wig: np.ndarray, pig: np.ndarray, wcm: np.ndarray, pcm: np.ndarray, title: str):
    from PIL import Image, ImageDraw
    base = Image.fromarray(img448).convert("RGB")
    S = 448 / GRID
    col = {"lung_left": (255, 60, 60), "lung_right": (60, 120, 255), "heart": (255, 220, 0)}
    tint = np.asarray(base, np.float32)
    for k, name in enumerate(CHEXMASK_REGIONS[:3]):
        if pcm[k]:
            m = np.kron(wcm[k], np.ones((14, 14)))[..., None]
            tint = tint * (1 - 0.35 * m) + np.array(col[name], np.float32) * 0.35 * m
    im = Image.fromarray(tint.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for k, name in enumerate(CHEXMASK_REGIONS[3:], start=3):       # thirds: outline the band boundaries
        if pcm[k]:
            ys, xs = np.nonzero(wcm[k] > 0.5)
            if len(ys):
                c = col["lung_left"] if "left" in name else col["lung_right"]
                d.rectangle([xs.min() * S, ys.min() * S, (xs.max() + 1) * S, (ys.max() + 1) * S], outline=c, width=1)
    for name, c in (("left lung", (255, 160, 160)), ("right lung", (160, 200, 255)), ("cardiac silhouette", (255, 255, 120))):
        k = IG_INDEX[name]
        if pig[k]:
            ys, xs = np.nonzero(wig[k] > 0)
            d.rectangle([xs.min() * S, ys.min() * S, (xs.max() + 1) * S - 1, (ys.max() + 1) * S - 1], outline=c, width=3)
    for name in ("left lower lung zone", "right lower lung zone", "left apical zone", "right apical zone"):
        k = IG_INDEX[name]
        if pig[k]:
            ys, xs = np.nonzero(wig[k] > 0)
            d.rectangle([xs.min() * S, ys.min() * S, (xs.max() + 1) * S - 1, (ys.max() + 1) * S - 1], outline=(0, 255, 0), width=1)
    d.text((4, 4), title, fill=(255, 255, 255))
    d.text((4, 432), "image left = patient RIGHT", fill=(160, 200, 255))
    d.text((300, 432), "image right = patient LEFT", fill=(255, 160, 160))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--weights-name", required=True)
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--overlays", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=4096)
    ap.add_argument("--draft", action="store_true", help="fast JPEG draft decode (checked against the cache)")
    ap.add_argument("--geometry", choices=["stretch", "letterbox"], default="stretch")
    ap.add_argument("--exclude-ids", default=None, help="csv/parquet with dicom_id: region features set missing (both sets)")
    ap.add_argument("--min-rca", type=float, default=0.7, help="CheXmask Dice RCA below this -> CheXmask regions missing")
    ap.add_argument("--only-with-weights", action="store_true",
                    help="process only images whose CheXmask weights exist (rerun later for the rest)")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run(f"region-features-{args.out_name}", {**vars(args), "imagenome_regions": IG_REGIONS,
                                                   "chexmask_regions": CHEXMASK_REGIONS}, args.run_dir) as run:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        ids = pd.read_parquet(args.ids).image_id.tolist()
        man = M.load("mimic", columns=["dicom_id", "image_path", "view", "rows", "cols"]).set_index("dicom_id")
        df = man.loc[ids].reset_index()
        from nesy import weights_store
        widx, wall = weights_store.load(args.weights_name)
        run.log(f"CheXmask weights: {len(widx):,} images from {args.weights_name}")
        excl = set()
        if args.exclude_ids:
            ex = pd.read_parquet(args.exclude_ids) if args.exclude_ids.endswith(".parquet") else pd.read_csv(args.exclude_ids)
            excl = set(ex.dicom_id)
        low_rca = set(widx.index[widx.rca_mean < args.min_rca])
        run.log(f"geometry {args.geometry}; excluded from region features (left/right check failures): {len(excl & set(ids))}; "
                f"CheXmask Dice RCA < {args.min_rca} (CheXmask regions missing): {len(low_rca & set(ids))}")
        regions = pd.read_parquet(P.FEATURES / "imagenome_regions.parquet",
                                  filters=[("dicom_id", "in", ids)]) if len(ids) < 5000 else None
        if regions is None:
            regions = pd.read_parquet(P.FEATURES / "imagenome_regions.parquet")
            regions = regions[regions.dicom_id.isin(set(ids))]
        reg_by = dict(tuple(regions.groupby("dicom_id")))
        run.log(f"images {len(df):,}; with ImaGenome boxes {len(reg_by):,}; with CheXmask weights {int(widx.found.sum()):,}")

        N = len(df)
        OUT.mkdir(parents=True, exist_ok=True)
        shard_dir = OUT / f"{args.out_name}_shards"
        shard_dir.mkdir(exist_ok=True)
        # Output = shard files of SHARD consecutive rows of the id list (written sequentially with np.save,
        # then renamed). Large np.memmap files on cephfs lost their headers when a step was killed
        # (2026-10-06), so no memmaps are used for outputs. A shard is processed only if all its images
        # have CheXmask weights (or with --allow-missing-weights); finished shards are skipped on resume.
        SHARD = args.shard_size
        n_shards = (N + SHARD - 1) // SHARD
        have_w = df.dicom_id.isin(widx.index).values
        todo_shards = []
        for k in range(n_shards):
            if (shard_dir / f"shard_{k:04d}_global.npy").exists():
                continue
            rr = np.arange(k * SHARD, min((k + 1) * SHARD, N))
            if args.only_with_weights and not have_w[rr].all():
                continue
            todo_shards.append(k)
        todo = np.concatenate([np.arange(k * SHARD, min((k + 1) * SHARD, N)) for k in todo_shards]) if todo_shards else np.array([], int)
        run.log(f"shards: {n_shards} of {SHARD}; already written {len(list(shard_dir.glob('shard_*_global.npy')))}; "
                f"to do now {len(todo_shards)} ({len(todo):,} images)")

        model, pre = clear_model.load("cuda")
        vis = model.visual
        DS.draft = args.draft
        DS.geometry = args.geometry
        prog = Progress(run, len(todo), "images", every_s=60)
        t0, n_done, size_mismatch = time.time(), 0, 0
        ov_dir = run.dir / "overlays"
        if args.overlays:
            ov_dir.mkdir(exist_ok=True)
        iou_rows = []
        mean = torch.tensor(pre.transforms[3].mean).view(3, 1, 1)
        std = torch.tensor(pre.transforms[3].std).view(3, 1, 1)
        tm = {"wait_data": 0.0, "gpu": 0.0, "weights": 0.0, "pool": 0.0, "write": 0.0}
        for k in todo_shards:
            rr = np.arange(k * SHARD, min((k + 1) * SHARD, N))
            buf = {"global": np.zeros((len(rr), 768), np.float16),
                   "imagenome": np.zeros((len(rr), len(IG_REGIONS), 768), np.float16),
                   "chexmask": np.zeros((len(rr), len(CHEXMASK_REGIONS), 768), np.float16),
                   "imagenome_present": np.zeros((len(rr), len(IG_REGIONS)), bool),
                   "chexmask_present": np.zeros((len(rr), len(CHEXMASK_REGIONS)), bool)}
            dl = DataLoader(DS(df.iloc[rr], pre), batch_size=args.batch, num_workers=args.workers, pin_memory=True)
            tl = time.time()
            with torch.inference_mode():
                for local, x, Wd, Hd in dl:
                    tm["wait_data"] += time.time() - tl
                    t1 = time.time()
                    li = local.numpy()
                    rows = rr[li]
                    ff = vis.backbone.forward_features(x.cuda(non_blocking=True))
                    glob = F.normalize(vis.projection(ff["x_norm_clstoken"]).float(), dim=-1)
                    tok = vis.projection(ff["x_norm_patchtokens"]).float()              # [B, 1024, 768]
                    torch.cuda.synchronize()
                    tm["gpu"] += time.time() - t1
                    t1 = time.time()
                    wig = np.zeros((len(rows), len(IG_REGIONS), GRID * GRID), np.float32)
                    pig = np.zeros((len(rows), len(IG_REGIONS)), bool)
                    wcm = np.zeros((len(rows), len(CHEXMASK_REGIONS), GRID * GRID), np.float32)
                    pcm = np.zeros((len(rows), len(CHEXMASK_REGIONS)), bool)
                    for b, r in enumerate(rows):
                        did = df.dicom_id.iat[r]
                        W, H = int(Wd[b]), int(Hd[b])
                        if (H, W) != (int(df.rows.iat[r]), int(df.cols.iat[r])):
                            size_mismatch += 1
                        if did in reg_by:
                            w, p = imagenome_weights(reg_by[did], W, H, args.geometry)
                            wig[b], pig[b] = w.reshape(len(IG_REGIONS), -1), p
                        if have_w[r]:
                            w = np.asarray(wall[int(widx.at[did, "slot"])], np.float32) / 255.0
                            wcm[b] = w.reshape(len(CHEXMASK_REGIONS), -1)
                            pcm[b] = wcm[b].sum(1) > 0
                    tm["weights"] += time.time() - t1
                    t1 = time.time()

                    def pool(wt):
                        wt_t = torch.from_numpy(wt).cuda()
                        sm = wt_t.sum(-1, keepdim=True).clamp_min(1e-6)
                        return (torch.einsum("brp,bpd->brd", wt_t, tok) / sm).cpu().numpy()
                    buf["global"][li] = glob.cpu().numpy().astype(np.float16)
                    dids = df.dicom_id.values[rows]
                    ex_img = np.array([d in excl for d in dids])
                    ex_cm = ex_img | np.array([d in low_rca for d in dids])
                    pig_s, pcm_s = pig & ~ex_img[:, None], pcm & ~ex_cm[:, None]
                    buf["imagenome"][li] = np.where(pig_s[..., None], pool(wig), np.nan).astype(np.float16)
                    buf["chexmask"][li] = np.where(pcm_s[..., None], pool(wcm), np.nan).astype(np.float16)
                    buf["imagenome_present"][li], buf["chexmask_present"][li] = pig_s, pcm_s
                    tm["pool"] += time.time() - t1
                    for b in range(len(rows)):
                        lI, rI = IG_INDEX["left lung"], IG_INDEX["right lung"]
                        if pig[b, lI] and pig[b, rI] and pcm[b, 0] and pcm[b, 1]:
                            def iou(a, c):
                                return float(np.minimum(a, c).sum() / max(np.maximum(a, c).sum(), 1e-6))
                            bl, br = (wig[b, lI] > 0).astype(float), (wig[b, rI] > 0).astype(float)
                            ml, mr = (wcm[b, 0] > 0.5).astype(float), (wcm[b, 1] > 0.5).astype(float)
                            iou_rows.append({"dicom_id": df.dicom_id.iat[rows[b]], "iou_left_same": iou(bl, ml), "iou_right_same": iou(br, mr),
                                             "iou_left_cross": iou(bl, mr), "iou_right_cross": iou(br, ml),
                                             "mask_coverage_in_box_left": float((ml * bl).sum() / max(ml.sum(), 1e-6))})
                    if args.overlays and len(list(ov_dir.glob("*.png"))) < args.overlays:
                        for b in range(min(len(rows), args.overlays - len(list(ov_dir.glob('*.png'))))):
                            img = ((x[b] * std + mean)[0].numpy() * 255).clip(0, 255).astype(np.uint8)
                            r = rows[b]
                            overlay(img, wig[b].reshape(-1, GRID, GRID), pig[b], wcm[b].reshape(-1, GRID, GRID), pcm[b],
                                    f"{df.dicom_id.iat[r][:17]} {df.view.iat[r]}").save(ov_dir / f"{df.dicom_id.iat[r]}.png")
                    n_done += len(rows)
                    prog.update(n_done)
                    tl = time.time()
            t1 = time.time()
            np.save(shard_dir / f"shard_{k:04d}_rows.tmp.npy", rr)
            for name, arr in buf.items():
                np.save(shard_dir / f"shard_{k:04d}_{name}.tmp.npy", arr)
            for name in ["rows", *[n for n in buf if n != "global"], "global"]:      # global last = completion marker
                (shard_dir / f"shard_{k:04d}_{name}.tmp.npy").replace(shard_dir / f"shard_{k:04d}_{name}.npy")
            tm["write"] += time.time() - t1
            run.log(f"  shard {k} written ({len(rr):,} images); time split (s): " + ", ".join(f"{a} {v:.0f}" for a, v in tm.items()))
            run.metric(shard=k, n=len(rr), **{f"t_{a}": round(v, 1) for a, v in tm.items()})
        el = time.time() - t0
        rate = n_done / max(el, 1e-9)
        run.log(f"done {n_done:,} images in {el / 60:.1f} min -> {rate:.1f} img/s; image size vs metadata/CheXmask mismatches: {size_mismatch}")
        per_img = (1 + len(IG_REGIONS) + len(CHEXMASK_REGIONS)) * 768 * 2
        run.log(f"storage per image {per_img / 1e3:.1f} kB (float16: global + {len(IG_REGIONS)} ImaGenome + {len(CHEXMASK_REGIONS)} CheXmask vectors)")
        for n_full, label in ((218187, "MIMIC full pass"), (18000 + 4555, "VinDr+PadChest (global + CheXmask only)")):
            pi = per_img if label.startswith("MIMIC") else (1 + len(CHEXMASK_REGIONS)) * 768 * 2
            run.log(f"projection {label}: {n_full:,} images -> {n_full / rate / 3600:.1f} h at {rate:.0f} img/s, {n_full * pi / 1e9:.1f} GB")
        run.metric(img_per_s=rate, per_image_bytes=per_img, n=n_done)
        if iou_rows:
            ious = pd.DataFrame(iou_rows)
            ious.to_csv(run.dir / "lung_alignment.csv", index=False)
            s = ious.describe().T[["mean", "25%", "50%", "min"]]
            run.log("ImaGenome lung box vs CheXmask lung (grid IoU), n=%d:\n%s" % (len(ious), s.round(3).to_string()))
            flip_ok = ((ious.iou_left_same > ious.iou_left_cross) & (ious.iou_right_same > ious.iou_right_cross)).mean()
            run.log(f"same-side IoU > cross-side IoU for both lungs in {flip_ok:.1%} of images (left/right flip check)")
            run.metric(flip_ok=float(flip_ok), iou_left_median=float(ious.iou_left_same.median()))
        if args.geometry == "letterbox":
            # independent check: re-encode 64 images through model.encode_image on the letterboxed image
            from letterbox_test import letterbox
            from PIL import Image
            first = sorted(shard_dir.glob("shard_*_global.npy"))[:1]
            if first:
                rr0 = np.load(str(first[0]).replace("_global.npy", "_rows.npy"))[:64]
                g0 = np.load(first[0])[:64].astype(np.float32)
                x = torch.stack([pre(letterbox(Image.open(df.image_path.iat[r]))) for r in rr0]).cuda()
                with torch.inference_mode():
                    ind = F.normalize(model.encode_image(x).float(), dim=-1).cpu().numpy()
                cos = (ind * g0).sum(1) / np.linalg.norm(g0, axis=1)
                run.log(f"letterbox global vs independent encode_image on {len(rr0)} images: cosine min {cos.min():.5f}")
                run.metric(global_cos_min_independent=float(cos.min()))
            return
        # consistency of the global vector with the cached embedding store
        idx = pd.read_parquet(P.V2 / "data/embeddings/clear_frontal_v1_index.parquet").set_index("dicom_id")
        first = sorted(shard_dir.glob("shard_*_global.npy"))[:1]
        common = []
        if first:
            rr0 = np.load(str(first[0]).replace("_global.npy", "_rows.npy"))
            g0 = np.load(first[0])
            common = [j for j, r in enumerate(rr0) if df.dicom_id.iat[r] in idx.index][:500]
        if common:
            cached = np.load(P.V2 / "data/embeddings/clear_frontal_v1.npy", mmap_mode="r")
            a = np.asarray(g0[common], np.float32)
            c = np.asarray(cached[idx.loc[df.dicom_id.iloc[rr0[common]], "row"].values], np.float32)
            cos = (a * c).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(c, axis=1)
            run.log(f"global vs cached store on {len(common)} images: cosine min {cos.min():.5f}, median {np.median(cos):.5f}")
            run.metric(global_cos_min=float(cos.min()))


if __name__ == "__main__":
    main()
