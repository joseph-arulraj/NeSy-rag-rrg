"""Split invariants (Stage 0 'must show'). Run: python -m pytest v2/tests -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from nesy import manifests as M  # noqa: E402
from nesy import paths as P  # noqa: E402

sys.path.insert(0, str(P.V2 / "scripts"))
import build_splits  # noqa: E402


@pytest.fixture(scope="module")
def splits():
    return M.load_splits()


@pytest.fixture(scope="module")
def mimic():
    return M.load("mimic", columns=["subject_id", "study_id", "dicom_id", "source_split"])


def test_one_row_per_patient(splits):
    assert splits.subject_id.is_unique


def test_every_mimic_patient_assigned(splits, mimic):
    assert set(mimic.subject_id) == set(splits.subject_id)
    assert mimic.split.notna().all()


def test_no_patient_in_two_splits(mimic):
    assert (mimic.groupby("subject_id").split.nunique() == 1).all()


def test_no_study_in_two_splits(mimic):
    assert (mimic.groupby("study_id").split.nunique() == 1).all()


def test_mscxr_patients_never_in_fitting_splits(splits):
    ms = set(pd.read_csv(P.MSCXR / "MS_CXR_Local_Alignment_v1.1.0.csv").path.str.extract(r"/p(\d{8})/")[0].astype(int))
    fit = set(splits[splits.split.isin(M.FITTING_SPLITS)].subject_id)
    assert len(ms) == 851
    assert not (ms & fit)


def test_imagenome_gold_patients_never_in_fitting_splits(splits):
    g = pd.read_csv(P.IMAGENOME / "gold_dataset/gold_bbox_coordinate_annotations_1000images.csv", usecols=["image_id"])
    meta = pd.read_csv(P.MIMIC / "mimic-cxr-2.0.0-metadata.csv.gz", usecols=["dicom_id", "subject_id"])
    gold = set(meta[meta.dicom_id.isin(set(g.image_id.str.replace(".dcm", "", regex=False)))].subject_id)
    fit = set(splits[splits.split.isin(M.FITTING_SPLITS)].subject_id)
    assert len(gold) == 500
    assert not (gold & fit)


def test_official_split_respected(splits):
    assert set(splits[splits.official_split == "test"].split) == {"test"}
    assert set(splits[splits.official_split == "validate"].split) == {"val"}
    assert set(splits[splits.official_split == "train"].split) <= {"train", "calib", "thresh", "heldout_loc"}


def test_calib_thresh_test_are_different_patients(splits):
    groups = {s: set(splits[splits.split == s].subject_id) for s in ("calib", "thresh", "test", "val", "train")}
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (groups[a] & groups[b]), (a, b)
    assert len(groups["calib"]) > 1000 and len(groups["thresh"]) > 1000


def test_assignment_is_deterministic(splits):
    redo = [build_splits.assign(o, e, build_splits.unit_hash(s))
            for s, o, e in zip(splits.subject_id, splits.official_split, splits.excluded_from_fitting)]
    assert redo == splits.split.tolist()


def test_frozen_v1(splits):
    assert (splits.split_version == "v1").all()


def test_v1_identical_to_reviewed_draft(splits):
    draft = pd.read_parquet(P.SPLITS / "mimic_splits_v0-DRAFT.parquet")
    assert draft[["subject_id", "split"]].equals(splits[["subject_id", "split"]])


def test_external_sets_have_no_fitting_role():
    v = M.load("vindr", columns=["image_id", "role"])
    p = M.load("padchest_gr", columns=["image_id", "role"])
    assert set(v.split) == {"vindr_train", "external_test"}
    assert (v[v.image_id.isin(v.image_id)].split.value_counts()["external_test"]) == 3000
    assert set(p.split) == {"external_test"}
