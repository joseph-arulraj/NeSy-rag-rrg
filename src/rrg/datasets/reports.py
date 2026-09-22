"""MIMIC-CXR radiology reports (one .txt per study), read from a zip (local sample) or a directory (HPC)."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Optional

from ..core.config import Settings, require_dir, require_file

_WS = re.compile(r"[ \t]+")


def _relative(subject_id: int, study_id: int) -> str:
    sid = str(subject_id)
    return f"p{sid[:2]}/p{sid}/s{study_id}.txt"


class ReportStore:
    def __init__(self, zip_path: Optional[Path] = None, directory: Optional[Path] = None):
        if (zip_path is None) == (directory is None):
            raise ValueError("give exactly one of zip_path / directory")
        self._zip = zipfile.ZipFile(zip_path) if zip_path is not None else None
        self._dir = directory
        self._names = set(self._zip.namelist()) if self._zip else set()

    @classmethod
    def from_settings(cls, settings: Settings) -> "ReportStore":
        p = settings.paths
        if p.mimic_reports_zip is not None:
            return cls(zip_path=require_file(p.mimic_reports_zip, "paths.mimic_reports_zip"))
        if p.mimic_reports_dir is not None:
            return cls(directory=require_dir(p.mimic_reports_dir, "paths.mimic_reports_dir"))
        raise ValueError("set paths.mimic_reports_zip or paths.mimic_reports_dir")

    def get(self, subject_id: int, study_id: int) -> str:
        rel = _relative(subject_id, study_id)
        if self._zip is not None:
            for name in (f"files/{rel}", rel):
                if name in self._names:
                    return self._zip.read(name).decode("utf-8", errors="replace")
        else:
            for base in (self._dir, self._dir / "files"):  # type: ignore[operator]
                path = base / rel
                if path.is_file():
                    return path.read_text(encoding="utf-8", errors="replace")
        raise FileNotFoundError(f"no report for subject {subject_id}, study {study_id}")

    def get_clean(self, subject_id: int, study_id: int) -> str:
        """One-line, whitespace-collapsed text (for display)."""
        return _WS.sub(" ", " ".join(self.get(subject_id, study_id).split())).strip()
