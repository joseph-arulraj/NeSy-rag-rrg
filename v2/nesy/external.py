"""Loaders for the external test sets (VinDr-CXR DICOM, PadChest-GR 16-bit PNG in a split zip).

Preprocessing (proposed in item 11, applied identically to both sets, to be confirmed by the user):
  1. read raw pixels (DICOM: pixel_array; PadChest: 16-bit PNG read from the archive in memory)
  2. invert if PhotometricInterpretation is MONOCHROME1 (VinDr only; PadChest PNGs are MONOCHROME2)
  3. window to the 0.5th-99.5th intensity percentiles of the image and scale to 8 bits. The DICOM
     WindowCenter/Width are not used: in the files checked they span the full bit range
     (e.g. 2047/4095), so they add no contrast information and are not present in PadChest.
  4. return a PIL 'L' image; CLEAR's own preprocess (RGB, bicubic resize to 448x448, normalise) then
     applies exactly as for MIMIC.
Nothing is written to disk; no metric is computed on these sets here.
"""
from __future__ import annotations

import io
from functools import lru_cache

import numpy as np

LOW_PCT, HIGH_PCT = 0.5, 99.5


def window_to_uint8(a: np.ndarray, invert: bool = False) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, [LOW_PCT, HIGH_PCT])
    if hi <= lo:
        hi = lo + 1.0
    a = np.clip((a - lo) / (hi - lo), 0, 1)
    if invert:
        a = 1.0 - a
    return (a * 255.0 + 0.5).astype(np.uint8)


def load_vindr(path: str):
    import pydicom
    from PIL import Image
    d = pydicom.dcmread(path)
    a = d.pixel_array
    if a.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D image, got {a.shape}")
    inv = str(d.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1"
    return Image.fromarray(window_to_uint8(a, invert=inv), mode="L"), {"photometric": str(d.PhotometricInterpretation),
                                                                       "bits_stored": int(d.get("BitsStored", 0)),
                                                                       "shape": a.shape}


@lru_cache(maxsize=2)
def _padchest_zip(prefix: str):
    from .multizip import open_split_zip
    return open_split_zip(prefix)


def load_padchest(archive_prefix: str, member: str):
    from PIL import Image
    z = _padchest_zip(archive_prefix)
    im = Image.open(io.BytesIO(z.read(member)))
    a = np.asarray(im)
    if a.ndim != 2:
        a = np.asarray(im.convert("I;16"))
    return Image.fromarray(window_to_uint8(a), mode="L"), {"mode": im.mode, "shape": a.shape}
