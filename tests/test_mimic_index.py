import pytest

from rrg.datasets.mimic_cxr import MimicCxrIndex, image_path_for

pytestmark = pytest.mark.sample


def test_path_layout():
    p = image_path_for(__import__("pathlib").Path("/r"), 10161682, 50933388, "abc")
    assert str(p) == "/r/files/p10/p10161682/s50933388/abc.jpg"


def test_sample_index_matches_the_documented_subset(settings, need_sample):
    idx = MimicCxrIndex.from_settings(settings, verify_files=True)
    s = idx.summary()
    assert s["n_images"] == 100 and s["n_studies"] == 100        # one image per study
    assert s["by_split"] == {"train": 50, "test": 50}            # official splits, 50 + 50
    assert set(s["by_view"]) <= {"PA", "AP"}                     # frontal only
    assert not s["dropped"]
    assert all(r.image_path.is_file() for r in idx)


def test_select_and_get(settings, need_sample):
    idx = MimicCxrIndex.from_settings(settings)
    train = idx.select(split="train")
    assert len(train) == 50 and all(r.split == "train" for r in train)
    assert idx.get(train[0].dicom_id) is train[0]
    assert idx.select(view_positions=["LATERAL"]) == []
    with pytest.raises(KeyError):
        idx.get("nope")
