"""Can the H100 decode + letterbox the JPEGs (nvJPEG via torchvision) instead of 16 CPUs?
Compares, on 256 pilot images, the CLEAR global embedding from the GPU path against the reference PIL path
(LANCZOS letterbox); switch only if cosine >= 0.999 everywhere. Also times both paths."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nesy import clear_model, manifests as M, paths as P  # noqa: E402
from nesy.runlog import Run  # noqa: E402


def gpu_letterbox(raw: "torch.Tensor", mean, std, size=448):
    """raw: uint8 [1, H, W] on cuda -> normalised [3, 448, 448] letterboxed (antialiased bicubic)."""
    import torch
    import torch.nn.functional as F
    _, H, W = raw.shape
    r = size / max(W, H)
    nw, nh = int(W * r), int(H * r)
    x = raw.float()[None]                                             # [1,1,H,W] 0..255
    x = F.interpolate(x, size=(nh, nw), mode="bicubic", antialias=True, align_corners=False).clamp(0, 255).round()
    canvas = torch.zeros((1, 1, size, size), device=raw.device)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    canvas[..., oy:oy + nh, ox:ox + nw] = x
    c = canvas[0].repeat(3, 1, 1) / 255.0
    return (c - mean[:, None, None]) / std[:, None, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("gpu-decode-test", vars(args), args.run_dir) as run:
        import torch
        import torch.nn.functional as F
        import torchvision
        from torchvision.io import decode_jpeg, read_file
        from PIL import Image
        from letterbox_test import letterbox
        run.log(f"torchvision {torchvision.__version__}")
        ids = pd.read_parquet(P.FEATURES / "regions/mimic_pilot_ids.parquet").image_id.tolist()[: args.n]
        man = M.load("mimic", columns=["dicom_id", "image_path"]).set_index("dicom_id")
        paths = man.loc[ids, "image_path"].tolist()
        model, pre = clear_model.load("cuda")
        mean = torch.tensor(pre.transforms[3].mean, device="cuda")
        std = torch.tensor(pre.transforms[3].std, device="cuda")
        t0 = time.time()
        ref = torch.stack([pre(letterbox(Image.open(p))) for p in paths])
        t_cpu = time.time() - t0
        t0 = time.time()
        datas = [read_file(p) for p in paths]
        imgs = decode_jpeg(datas, device="cuda", mode=torchvision.io.ImageReadMode.GRAY)
        gx = torch.stack([gpu_letterbox(im, mean, std) for im in imgs])
        torch.cuda.synchronize()
        t_gpu = time.time() - t0
        with torch.inference_mode():
            a = F.normalize(model.encode_image(ref.cuda()).float(), dim=-1)
            b = F.normalize(model.encode_image(gx).float(), dim=-1)
        cos = (a * b).sum(1).cpu().numpy()
        pix = (ref.cuda() - gx).abs().mean().item()
        run.log(f"{len(paths)} images: CPU PIL path {len(paths) / t_cpu:.1f} img/s (1 process); GPU nvJPEG path {len(paths) / t_gpu:.1f} img/s")
        run.log(f"embedding cosine GPU vs PIL: min {cos.min():.5f}, median {np.median(cos):.5f}; mean |pixel diff| (normalised) {pix:.4f}")
        run.log("VERDICT: " + ("GPU path usable (cosine >= 0.999)" if cos.min() >= 0.999 else "GPU path NOT equivalent; keep PIL"))


if __name__ == "__main__":
    main()
