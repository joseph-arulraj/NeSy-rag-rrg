"""Create .npy memmaps safely on cephfs.

np.lib.format.open_memmap(mode="w+") writes the 128-byte header with a buffered file write and then maps
the file from offset 0 (the header shares the first page). On cephfs the mapping can read that page
before the header write lands, and the final flush then writes zeros over the header (seen twice on
2026-10-06: a 2 GB weights file and three region-feature files). Here the header file is created,
closed and fsync'ed first, and only then reopened for writing.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def create(path, dtype, shape):
    path = Path(path)
    m = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    del m
    _rewrite_header(path, dtype, shape)
    return np.lib.format.open_memmap(path, mode="r+")


def _rewrite_header(path, dtype, shape):
    import io
    buf = io.BytesIO()
    np.lib.format.write_array_header_1_0(buf, {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
                                              "fortran_order": False, "shape": tuple(shape)})
    with open(path, "r+b") as f:
        f.write(buf.getvalue())
        f.flush()
        os.fsync(f.fileno())


def close(m, path, dtype, shape):
    """Flush a memmap and re-assert its header (cheap; protects against the cephfs race)."""
    m.flush()
    del m
    _rewrite_header(path, dtype, shape)
