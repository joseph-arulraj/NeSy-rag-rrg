"""Region definitions on CLEAR's 32x32 patch grid, for both preprocessing geometries.

Geometries (canvas = the 448x448 image CLEAR sees; DINOv2 ViT-B/14 -> 32x32 patches of 14 px):
  stretch    clear.hub.build_cxr_preprocess as released: resize to exactly 448x448, x and y scaled
             independently (aspect ratio not preserved), no crop, no padding
  letterbox  the CLEAR authors' dataset preprocessing (switched to on 2026-10-06): aspect-preserving
             resize so the longest side is 448 (r = 448 / max(W, H)), zero-padded symmetrically:
             x' = x * r + (448 - int(W r)) // 2, y' = y * r + (448 - int(H r)) // 2
A region is a weight map w[32, 32] in [0, 1]: the fraction of each patch covered by the region.
Pooled vector = sum(w * token) / sum(w).
  * Chest ImaGenome / radiologist boxes (original pixels) -> exact rectangle coverage per patch
  * CheXmask masks (OriginalResolution, original pixels) -> area-averaged mask per patch
CheXmask region set (patient sides): lung_left, lung_right, heart, and each lung split into upper,
middle and lower thirds of its own vertical extent.
"""
from __future__ import annotations

import numpy as np

GRID = 32
CANVAS = 448
CELL = CANVAS / GRID          # 14 px
CHEXMASK_REGIONS = ["lung_left", "lung_right", "heart",
                    "lung_left_upper", "lung_left_middle", "lung_left_lower",
                    "lung_right_upper", "lung_right_middle", "lung_right_lower"]
GEOMETRIES = ("stretch", "letterbox")


def canvas_map(W: int, H: int, geometry: str):
    """-> (sx, sy, ox, oy) with canvas = original * s + o."""
    if geometry == "stretch":
        return CANVAS / W, CANVAS / H, 0.0, 0.0
    if geometry == "letterbox":
        r = CANVAS / max(W, H)
        nw, nh = int(W * r), int(H * r)
        return r, r, float((CANVAS - nw) // 2), float((CANVAS - nh) // 2)
    raise ValueError(f"unknown geometry {geometry}")


def box_weights(x1, y1, x2, y2, W, H, geometry: str = "stretch") -> np.ndarray:
    """Fraction of each patch covered by the box [x1, x2) x [y1, y2) given in original pixels."""
    sx, sy, ox, oy = canvas_map(W, H, geometry)
    cx1, cx2 = x1 * sx + ox, x2 * sx + ox
    cy1, cy2 = y1 * sy + oy, y2 * sy + oy
    e = np.arange(GRID + 1) * CELL
    fx = np.clip(np.minimum(e[1:], cx2) - np.maximum(e[:-1], cx1), 0, None) / CELL
    fy = np.clip(np.minimum(e[1:], cy2) - np.maximum(e[:-1], cy1), 0, None) / CELL
    return np.outer(fy, fx).astype(np.float32)


def mask_to_grid(mask: np.ndarray, geometry: str = "stretch") -> np.ndarray:
    """Area fraction of a full-resolution boolean mask inside each patch cell."""
    from PIL import Image
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    if geometry == "stretch":
        return np.asarray(im.resize((GRID, GRID), Image.BOX), dtype=np.float32) / 255.0
    H, W = mask.shape
    _, _, ox, oy = canvas_map(W, H, "letterbox")
    r = CANVAS / max(W, H)
    nw, nh = int(W * r), int(H * r)
    small = np.asarray(im.resize((nw, nh), Image.BOX), dtype=np.float32) / 255.0
    canvas = np.zeros((CANVAS, CANVAS), np.float32)
    canvas[int(oy):int(oy) + nh, int(ox):int(ox) + nw] = small
    return canvas.reshape(GRID, 14, GRID, 14).mean(axis=(1, 3))


def chexmask_region_masks(ml: np.ndarray, mr: np.ndarray, mh: np.ndarray) -> dict[str, np.ndarray]:
    """Full-resolution masks for the 9 regions (ml/mr = patient-left/right lung)."""
    out = {"lung_left": ml, "lung_right": mr, "heart": mh}
    for name, m in (("lung_left", ml), ("lung_right", mr)):
        rows = np.nonzero(m.any(1))[0]
        if rows.size == 0:
            for part in ("upper", "middle", "lower"):
                out[f"{name}_{part}"] = np.zeros_like(m)
            continue
        y0, y1 = rows.min(), rows.max() + 1
        cuts = np.linspace(y0, y1, 4).round().astype(int)
        for k, part in enumerate(("upper", "middle", "lower")):
            band = np.zeros(m.shape[0], bool)
            band[cuts[k]:cuts[k + 1]] = True
            out[f"{name}_{part}"] = m & band[:, None]
    return out
