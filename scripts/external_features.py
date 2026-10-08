"""External feature cache (CACHING ONLY; no labels are read and no metric is computed): VinDr-CXR test and
PadChest-GR, letterbox geometry, percentile windowing (approved 2026-10-06).

Per image, float16: global CLEAR embedding + 9 CheXmask-region vectors (lungs, thirds, heart) from the
CheXmask OriginalResolution masks (weights stores ext_vindr_test_lb, ext_padchest_gr_lb). Images without
CheXmask masks (245 PadChest-GR) or with Dice RCA < 0.7 get missing region features (present = False,
vectors NaN); they are not dropped. Output: data/features/regions/ext_<set>_lb_shards/shard_NNNN_*.npy,
resumable per shard.
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

from nesy import clear_model, external as X, manifests as M, paths as P, weights_store  # noqa: E402
from nesy.regions import CHEXMASK_REGIONS, GRID  # noqa: E402
from nesy.runlog import Progress, Run  # noqa: E402

OUT = P.FEATURES / "regions"


class DS:
    def __init__(self, df, pre, kind):
        self.df, self.pre, self.kind = df.reset_index(drop=True), pre, kind

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        from letterbox_test import letterbox
        r = self.df.iloc[i]
        if self.kind == "vindr":
            im, _ = X.load_vindr(r.image_path)
        else:
            im, _ = X.load_padchest(r.archive_prefix, r.archive_member)
        W, H = im.size
        return i, self.pre(letterbox(im)), W, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-size", type=int, default=2048)
    ap.add_argument("--min-rca", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0, help="test mode: first N images per set")
    ap.add_argument("--tag", default="", help="suffix for weight-store and output names (test mode)")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("external-features", vars(args), args.run_dir) as run:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        model, pre = clear_model.load("cuda")
        vis = model.visual
        sets = []
        v = M.load("vindr")
        sets.append(("vindr_test", "vindr", v[v.source_split == "test"].reset_index(drop=True), "ext_vindr_test_lb"))
        g = M.load("padchest_gr")
        sets.append(("padchest_gr", "padchest", g.reset_index(drop=True), "ext_padchest_gr_lb"))
        for name, kind, df, wname in sets:
            if args.limit:
                df = df.head(args.limit).reset_index(drop=True)
            if args.tag:
                wname = {"vindr": "exttest_vindr_lb", "padchest": "exttest_padchest_lb"}[kind]
            widx, wall = weights_store.load(wname)
            good = set(widx.index[widx.rca_mean >= args.min_rca])
            run.log(f"{name}: {len(df):,} images; CheXmask masks {int(df.image_id.isin(widx.index).sum()):,}; "
                    f"usable (RCA >= {args.min_rca}) {int(df.image_id.isin(good).sum()):,}; the rest get missing region features")
            sd = OUT / f"ext_{name}_lb{args.tag}_shards"
            sd.mkdir(parents=True, exist_ok=True)
            df[["image_id"]].to_parquet(sd / "ids.parquet", index=False)
            N = len(df)
            prog = Progress(run, N, f"{name} images", every_s=120)
            done = 0
            for k in range((N + args.shard_size - 1) // args.shard_size):
                rr = np.arange(k * args.shard_size, min((k + 1) * args.shard_size, N))
                if (sd / f"shard_{k:04d}_global.npy").exists():
                    done += len(rr)
                    continue
                G = np.zeros((len(rr), 768), np.float16)
                CM = np.full((len(rr), len(CHEXMASK_REGIONS), 768), np.nan, np.float16)
                CP = np.zeros((len(rr), len(CHEXMASK_REGIONS)), bool)
                dl = DataLoader(DS(df.iloc[rr], pre, kind), batch_size=64, num_workers=args.workers)
                with torch.inference_mode():
                    for li, x, Wd, Hd in dl:
                        li = li.numpy()
                        ff = vis.backbone.forward_features(x.cuda())
                        G[li] = F.normalize(vis.projection(ff["x_norm_clstoken"]).float(), dim=-1).cpu().numpy()
                        tok = vis.projection(ff["x_norm_patchtokens"]).float()
                        wcm = np.zeros((len(li), len(CHEXMASK_REGIONS), GRID * GRID), np.float32)
                        for b, j in enumerate(li):
                            iid = df.image_id.iat[rr[j]]
                            if iid in good:
                                if (int(widx.at[iid, "H"]), int(widx.at[iid, "W"])) != (int(Hd[b]), int(Wd[b])):
                                    run.log(f"  size mismatch {iid}: mask {widx.at[iid, 'H']}x{widx.at[iid, 'W']} image {int(Hd[b])}x{int(Wd[b])}; regions missing")
                                    continue
                                wcm[b] = np.asarray(wall[int(widx.at[iid, "slot"])], np.float32).reshape(len(CHEXMASK_REGIONS), -1) / 255.0
                        pres = wcm.sum(-1) > 0
                        wt = torch.from_numpy(wcm).cuda()
                        pooled = (torch.einsum("brp,bpd->brd", wt, tok) / wt.sum(-1, keepdim=True).clamp_min(1e-6)).cpu().numpy()
                        CM[li] = np.where(pres[..., None], pooled, np.nan).astype(np.float16)
                        CP[li] = pres
                np.save(sd / f"shard_{k:04d}_rows.npy", rr)
                np.save(sd / f"shard_{k:04d}_chexmask.npy", CM)
                np.save(sd / f"shard_{k:04d}_chexmask_present.npy", CP)
                np.save(sd / f"shard_{k:04d}_global.tmp.npy", G)
                (sd / f"shard_{k:04d}_global.tmp.npy").replace(sd / f"shard_{k:04d}_global.npy")
                done += len(rr)
                prog.update(done)
            n_miss = sum(int((~np.load(f)).all(1).sum()) for f in sorted(sd.glob("shard_*_chexmask_present.npy")))
            run.log(f"{name}: cached {done:,} images in {sd}; images with all CheXmask regions missing: {n_miss:,}")


if __name__ == "__main__":
    main()
