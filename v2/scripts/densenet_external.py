"""DenseNet-121 baseline on the external sets, PREDICTIONS ONLY (user, 2026-10-07): VinDr-CXR test and
PadChest-GR. No labels are read (only image id / path columns are taken from the manifests) and no metric
is computed; this is stored for the final external comparison.

Model: the inner-holdout-selected epoch (best_model.pt) with its Platt calibration (platt.json) from the
given DenseNet run. Preprocessing as for MIMIC: image -> 8-bit grayscale (external DICOMs: percentile
windowing 0.5-99.5, MONOCHROME1 inverted, nesy/external.py, as approved for the external caches) ->
letterbox to 256 px (same function) -> centre 224 crop -> ImageNet normalisation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nesy import external as X, manifests as M  # noqa: E402
from nesy.runlog import Progress, Run  # noqa: E402

KEEP = {"vindr": ["image_id", "image_path", "source_split"], "padchest_gr": ["image_id", "archive_prefix", "archive_member"]}


class DS:
    def __init__(self, df, kind):
        self.df, self.kind = df.reset_index(drop=True), kind

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        from letterbox_test import letterbox
        r = self.df.iloc[i]
        try:
            im = X.load_vindr(r.image_path)[0] if self.kind == "vindr" else X.load_padchest(r.archive_prefix, r.archive_member)[0]
            a = np.asarray(letterbox(im, 256), np.uint8)[16:240, 16:240]
            return i, a, True
        except Exception:                                       # noqa: BLE001
            return i, np.zeros((224, 224), np.uint8), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--densenet-run", required=True)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("densenet-external", vars(args), args.run_dir) as run:
        import torch
        from torch.utils.data import DataLoader
        from densenet_train import build_model, MEAN, STD
        dr = Path(args.densenet_run)
        ck = torch.load(dr / "best_model.pt", map_location="cpu", weights_only=False)
        F = ck["findings"]
        platt = json.loads((dr / "platt.json").read_text())
        model = build_model(len(F))
        model.load_state_dict(ck["model"])
        model = model.cuda().eval().to(memory_format=torch.channels_last)
        run.log(f"model epoch {ck['epoch']} from {dr}; outputs {F}; Platt from {dr / 'platt.json'}")
        mean = torch.tensor(MEAN, device="cuda")[None, :, None, None]
        std = torch.tensor(STD, device="cuda")[None, :, None, None]
        for name in ("vindr", "padchest_gr"):
            df = M.load(name, columns=KEEP[name])[KEEP[name]]             # image columns only: no label column is read
            if name == "vindr":
                df = df[df.source_split == "test"]
            df = df.reset_index(drop=True)
            run.log(f"{name}: {len(df):,} images (columns read: {list(df.columns)})")
            logits = np.zeros((len(df), len(F)), np.float32)
            ok = np.zeros(len(df), bool)
            prog = Progress(run, len(df), f"{name} images", every_s=60)
            done = 0
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for li, x, good in DataLoader(DS(df, "vindr" if name == "vindr" else "padchest"), batch_size=128, num_workers=args.workers):
                    t = x.cuda().float().div_(255.0)[:, None].expand(-1, 3, -1, -1)
                    t = ((t - mean) / std).contiguous(memory_format=torch.channels_last)
                    logits[li.numpy()] = model(t).float().cpu().numpy()
                    ok[li.numpy()] = good.numpy().astype(bool)
                    done += len(li)
                    prog.update(done)
            out = pd.DataFrame({"image_id": df.image_id, "read_ok": ok})
            for j, f in enumerate(F):
                z = platt[f]["coef"] * logits[:, j] + platt[f]["intercept"]
                out[f"p_{f}"] = np.where(ok, 1 / (1 + np.exp(-z)), np.nan)
                out[f"logit_{f}"] = np.where(ok, logits[:, j], np.nan)
            out.to_parquet(run.dir / f"predictions_{name}.parquet", index=False)
            run.log(f"{name}: predictions for {int(ok.sum()):,} images ({int((~ok).sum())} unreadable -> NaN); no labels read, no metrics")


if __name__ == "__main__":
    main()
