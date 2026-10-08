"""Read members of a split zip (name.zip.001, .002, ...) in place, without extracting.

PadChest-GR's images are a 37-part zip whose members are STORED (no compression), so a member can
be read straight from the parts by offset. Nothing is written to disk.
"""
from __future__ import annotations

import bisect
import glob
import io
import os
import zipfile


class ConcatFile(io.RawIOBase):
    """A read-only, seekable view of several files laid end to end."""

    def __init__(self, paths: list[str]):
        self.paths = paths
        self.starts = [0]
        for p in paths:
            self.starts.append(self.starts[-1] + os.path.getsize(p))
        self.total = self.starts[-1]
        self.pos = 0
        self._fh: dict[int, object] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.total + off
        return self.pos

    def readinto(self, b) -> int:
        if self.pos >= self.total:
            return 0
        i = bisect.bisect_right(self.starts, self.pos) - 1
        fh = self._fh.get(i)
        if fh is None:
            fh = self._fh[i] = open(self.paths[i], "rb")
        fh.seek(self.pos - self.starts[i])
        data = fh.read(min(len(b), self.starts[i + 1] - self.pos))
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)

    def close(self) -> None:
        for fh in self._fh.values():
            fh.close()
        super().close()


def open_split_zip(prefix: str) -> zipfile.ZipFile:
    """`prefix` is the path without the .NNN suffix, e.g. .../PadChest_GR.zip"""
    parts = sorted(glob.glob(prefix + ".[0-9][0-9][0-9]"))
    if not parts:
        raise FileNotFoundError(f"no parts found for {prefix}.NNN")
    return zipfile.ZipFile(io.BufferedReader(ConcatFile(parts), 1 << 20))
