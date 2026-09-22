"""MIMIC-CXR-JPG index: official split + metadata -> image records, plus the v1 study-level policy.

Only the official MIMIC-CXR splits are used; no custom splitting happens anywhere.
Policy (dataset.*): one frontal image per study; PA is preferred over AP (order of
dataset.accepted_view_positions); ties are broken by the smallest dicom_id; studies with no
accepted view (lateral-only etc.) are excluded. The index lists every image; the policy is applied
by study_records() / one_image_per_study().
"""
from __future__ import annotations

import csv
import gzip
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

from ..core.config import Settings, require_dir, require_file
from ..core.errors import ConfigError, SplitLeakageError


@dataclass(frozen=True)
class StudyRecord:
    subject_id: int
    study_id: int
    dicom_id: str
    split: str
    view_position: Optional[str]
    patient_orientation: Optional[str]   # MIMIC: posture (Erect/Recumbent), see StudyMeta note
    procedure: Optional[str]
    image_path: Path


def image_path_for(root: Path, subject_id: int, study_id: int, dicom_id: str) -> Path:
    """MIMIC-CXR-JPG layout: files/p<first 2 digits>/p<subject>/s<study>/<dicom_id>.jpg"""
    sid = str(subject_id)
    return root / "files" / f"p{sid[:2]}" / f"p{sid}" / f"s{study_id}" / f"{dicom_id}.jpg"


def _read_csv_gz(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:  # type: ignore[operator]
        yield from csv.DictReader(fh)


def _clean(v: Optional[str]) -> Optional[str]:
    v = (v or "").strip()
    return v or None


# --------------------------------------------------------------------------- policy
def one_image_per_study(
    records: Iterable[StudyRecord], view_preference: Sequence[str]
) -> tuple[list[StudyRecord], dict]:
    """Keep exactly one image per study: the most-preferred accepted view (PA before AP by default),
    ties broken by the smallest dicom_id. Studies with no accepted view are dropped. The result is
    ordered by (subject_id, study_id), so it does not depend on the input order."""
    rank = {v: i for i, v in enumerate(view_preference)}
    by_study: dict[tuple[int, int], list[StudyRecord]] = defaultdict(list)
    for r in records:
        by_study[(r.subject_id, r.study_id)].append(r)
    kept: list[StudyRecord] = []
    no_view = 0
    for key in sorted(by_study):
        cands = [r for r in by_study[key] if r.view_position in rank]
        if not cands:
            no_view += 1
            continue
        kept.append(min(cands, key=lambda r: (rank[r.view_position], r.dicom_id)))  # type: ignore[index]
    stats = {
        "n_studies_seen": len(by_study),
        "n_studies_kept": len(kept),
        "n_studies_dropped_no_accepted_view": no_view,
        "kept_by_view": dict(Counter(r.view_position for r in kept)),
    }
    return kept, stats


def assert_patient_disjoint(subject_splits: dict[int, set[str]]) -> None:
    """Patients must not appear in more than one split, otherwise train-only retrieval can leak."""
    bad = {sid: sorted(sp) for sid, sp in subject_splits.items() if len(sp) > 1}
    if bad:
        example = dict(list(bad.items())[:3])
        raise SplitLeakageError(
            f"{len(bad)} patient(s) appear in more than one official split, e.g. {example}. "
            "The retrieval / evaluation protocol assumes patient-disjoint splits."
        )


# ---------------------------------------------------------------------------- index
class MimicCxrIndex:
    def __init__(
        self,
        records: Sequence[StudyRecord],
        root: Path,
        dropped: dict[str, int],
        view_preference: Sequence[str],
        one_per_study: bool,
    ):
        self.records = list(records)
        self.root = root
        self.dropped = dropped
        self.view_preference = list(view_preference)
        self.one_per_study = one_per_study
        self._by_dicom = {r.dicom_id: r for r in self.records}

    # ---- construction
    @classmethod
    def from_settings(cls, settings: Settings, verify_files: bool = False) -> "MimicCxrIndex":
        root = require_dir(settings.paths.mimic_root, "paths.mimic_root")
        ds = settings.dataset
        meta_path = require_file(root / ds.metadata_csv, "dataset.metadata_csv")
        split_path = require_file(root / ds.split_csv, "dataset.split_csv")

        splits: dict[str, str] = {}
        subject_splits: dict[int, set[str]] = defaultdict(set)
        for row in _read_csv_gz(split_path):
            splits[row["dicom_id"]] = row["split"]
            subject_splits[int(row["subject_id"])].add(row["split"])
        assert_patient_disjoint(subject_splits)          # checked on the whole split file, not just the exposed splits

        wanted_splits = set(ds.splits)
        records: list[StudyRecord] = []
        dropped: Counter = Counter()
        for row in _read_csv_gz(meta_path):
            dicom = row["dicom_id"]
            split = splits.get(dicom)
            if split is None:
                dropped["no_split_entry"] += 1
                continue
            if split not in wanted_splits:
                dropped["split_not_selected"] += 1
                continue
            sid, stid = int(row["subject_id"]), int(row["study_id"])
            path = image_path_for(root, sid, stid, dicom)
            if verify_files and not path.is_file():
                dropped["image_file_missing"] += 1
                continue
            records.append(
                StudyRecord(
                    subject_id=sid,
                    study_id=stid,
                    dicom_id=dicom,
                    split=split,
                    view_position=_clean(row.get("ViewPosition")),
                    patient_orientation=_clean(row.get("PatientOrientationCodeSequence_CodeMeaning")),
                    procedure=_clean(row.get("PerformedProcedureStepDescription")),
                    image_path=path,
                )
            )
        if not records:
            raise ConfigError(f"no usable records found under {root} (dropped: {dict(dropped)})")
        return cls(records, root, dict(dropped), ds.accepted_view_positions, ds.one_image_per_study)

    # ---- access
    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[StudyRecord]:
        return iter(self.records)

    def get(self, dicom_id: str) -> StudyRecord:
        try:
            return self._by_dicom[dicom_id]
        except KeyError as exc:
            raise KeyError(f"unknown dicom_id {dicom_id}") from exc

    def select(
        self,
        split: Optional[str] = None,
        view_positions: Optional[Iterable[str]] = None,
    ) -> list[StudyRecord]:
        """Raw filter over every image (no study-level policy)."""
        views = set(view_positions) if view_positions is not None else None
        return [
            r for r in self.records
            if (split is None or r.split == split) and (views is None or r.view_position in views)
        ]

    def study_records(self, split: Optional[str] = None) -> tuple[list[StudyRecord], dict]:
        """The images the pipeline actually uses, per the dataset policy (see module docstring).
        Returns (records ordered by (subject, study), policy statistics)."""
        pool = self.select(split=split)
        if self.one_per_study:
            return one_image_per_study(pool, self.view_preference)
        allowed = set(self.view_preference)
        kept = sorted((r for r in pool if r.view_position in allowed), key=lambda r: (r.subject_id, r.study_id, r.dicom_id))
        return kept, {"n_studies_kept": len({(r.subject_id, r.study_id) for r in kept}), "n_images_kept": len(kept),
                      "one_image_per_study": False}

    def summary(self) -> dict:
        return {
            "n_images": len(self.records),
            "n_studies": len({(r.subject_id, r.study_id) for r in self.records}),
            "by_split": dict(Counter(r.split for r in self.records)),
            "by_view": dict(Counter(r.view_position for r in self.records)),
            "dropped": self.dropped,
        }
