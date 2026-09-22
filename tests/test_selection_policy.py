import csv
import gzip
import random
from pathlib import Path

import pytest

from rrg.core.config import load_settings
from rrg.core.errors import SplitLeakageError
from rrg.datasets.mimic_cxr import MimicCxrIndex, StudyRecord, assert_patient_disjoint, one_image_per_study


def rec(subject, study, dicom, view, split="train"):
    return StudyRecord(subject, study, dicom, split, view, None, None, Path(f"/x/{dicom}.jpg"))


PREF = ["PA", "AP"]


def test_pa_is_preferred_over_ap_and_lateral_is_ignored():
    kept, st = one_image_per_study([rec(1, 1, "b", "AP"), rec(1, 1, "z", "PA"), rec(1, 1, "a", "LATERAL")], PREF)
    assert [r.dicom_id for r in kept] == ["z"]
    assert st["kept_by_view"] == {"PA": 1}


def test_same_view_ties_go_to_the_smallest_dicom_id():
    kept, _ = one_image_per_study([rec(1, 1, "m", "PA"), rec(1, 1, "c", "PA"), rec(1, 1, "x", "PA")], PREF)
    assert [r.dicom_id for r in kept] == ["c"]


def test_lateral_only_and_unknown_views_are_excluded_and_counted():
    kept, st = one_image_per_study(
        [rec(1, 1, "a", "LATERAL"), rec(2, 2, "b", None), rec(3, 3, "c", "LL"), rec(4, 4, "d", "AP")], PREF)
    assert [r.dicom_id for r in kept] == ["d"]
    assert st["n_studies_seen"] == 4 and st["n_studies_dropped_no_accepted_view"] == 3


def test_result_is_independent_of_input_order_and_sorted_by_subject_study():
    base = [rec(2, 9, "d1", "AP"), rec(2, 9, "d0", "PA"), rec(1, 5, "e1", "AP"), rec(1, 3, "f1", "PA"), rec(1, 3, "f0", "PA")]
    want = [("f0", 1, 3), ("e1", 1, 5), ("d0", 2, 9)]
    for seed in range(20):
        shuffled = base[:]
        random.Random(seed).shuffle(shuffled)
        kept, _ = one_image_per_study(shuffled, PREF)
        assert [(r.dicom_id, r.subject_id, r.study_id) for r in kept] == want


def test_preference_order_is_configurable():
    kept, _ = one_image_per_study([rec(1, 1, "a", "PA"), rec(1, 1, "b", "AP")], ["AP", "PA"])
    assert kept[0].dicom_id == "b"


def test_patient_disjointness_check():
    assert_patient_disjoint({1: {"train"}, 2: {"test"}})
    with pytest.raises(SplitLeakageError, match="1 patient"):
        assert_patient_disjoint({1: {"train", "test"}, 2: {"train"}})


# ---- wiring through MimicCxrIndex with a synthetic MIMIC folder
def _write_gz(path, header, rows):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _mimic(tmp_path, rows):
    """rows: (subject, study, dicom, view, split)"""
    _write_gz(tmp_path / "mimic-cxr-2.0.0-metadata.csv.gz",
              ["dicom_id", "subject_id", "study_id", "PerformedProcedureStepDescription", "ViewPosition",
               "PatientOrientationCodeSequence_CodeMeaning"],
              [(d, s, st, "CHEST", v, "Erect") for s, st, d, v, _ in rows])
    _write_gz(tmp_path / "mimic-cxr-2.0.0-split.csv.gz", ["dicom_id", "study_id", "subject_id", "split"],
              [(d, st, s, sp) for s, st, d, _, sp in rows])
    return load_settings(overrides=[f"paths.mimic_root={tmp_path}"])


ROWS = [
    (10, 100, "a1", "LATERAL", "train"), (10, 100, "a2", "PA", "train"), (10, 100, "a3", "AP", "train"),
    (10, 101, "b1", "AP", "train"), (11, 102, "c1", "LATERAL", "train"),
    (12, 103, "d1", "PA", "validate"), (13, 104, "e1", "AP", "test"),
]


def test_index_applies_the_policy_per_split(tmp_path):
    idx = MimicCxrIndex.from_settings(_mimic(tmp_path, ROWS))
    train, st = idx.study_records("train")
    assert [r.dicom_id for r in train] == ["a2", "b1"]                     # PA over AP; lateral-only study 102 dropped
    assert st["n_studies_dropped_no_accepted_view"] == 1
    assert [r.dicom_id for r in idx.study_records("validate")[0]] == ["d1"]
    assert idx.summary()["by_split"] == {"train": 5, "validate": 1, "test": 1}


def test_all_accepted_images_are_kept_when_the_policy_is_off(tmp_path):
    s = load_settings(overrides=[f"paths.mimic_root={_mimic(tmp_path, ROWS).paths.mimic_root}", "dataset.one_image_per_study=false"])
    assert [r.dicom_id for r in MimicCxrIndex.from_settings(s).study_records("train")[0]] == ["a2", "a3", "b1"]


def test_patient_in_two_splits_is_rejected_even_if_only_train_is_exposed(tmp_path):
    rows = ROWS + [(10, 999, "z1", "PA", "test")]                          # patient 10 also in test
    s = _mimic(tmp_path, rows)
    s2 = load_settings(overrides=[f"paths.mimic_root={s.paths.mimic_root}", "dataset.splits=[train]"])
    with pytest.raises(SplitLeakageError):
        MimicCxrIndex.from_settings(s2)


@pytest.mark.sample
def test_sample_studies_are_patient_disjoint_and_one_image_each(settings, need_sample):
    idx = MimicCxrIndex.from_settings(settings)                             # raises SplitLeakageError otherwise
    by_subject = {}
    for r in idx:
        by_subject.setdefault(r.subject_id, set()).add(r.split)
    assert all(len(v) == 1 for v in by_subject.values())
    for split in ("train", "test"):
        recs, _ = idx.study_records(split)
        assert len({(r.subject_id, r.study_id) for r in recs}) == len(recs) == 50
