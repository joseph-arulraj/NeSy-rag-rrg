import zipfile

import pytest

from rrg.datasets.mimic_cxr import MimicCxrIndex
from rrg.datasets.reports import ReportStore


def test_directory_mode_with_and_without_files_prefix(tmp_path):
    (tmp_path / "a" / "p10" / "p10161682").mkdir(parents=True)
    (tmp_path / "a" / "p10" / "p10161682" / "s5.txt").write_text("HELLO\n\n  WORLD")
    (tmp_path / "b" / "files" / "p10" / "p10161682").mkdir(parents=True)
    (tmp_path / "b" / "files" / "p10" / "p10161682" / "s5.txt").write_text("X")
    assert ReportStore(directory=tmp_path / "a").get_clean(10161682, 5) == "HELLO WORLD"
    assert ReportStore(directory=tmp_path / "b").get(10161682, 5) == "X"
    with pytest.raises(FileNotFoundError):
        ReportStore(directory=tmp_path / "a").get(10161682, 6)


def test_zip_mode(tmp_path):
    z = tmp_path / "r.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("files/p12/p12345678/s9.txt", "IMPRESSION: ok")
    assert ReportStore(zip_path=z).get(12345678, 9) == "IMPRESSION: ok"


def test_exactly_one_source_is_required(tmp_path):
    with pytest.raises(ValueError):
        ReportStore()
    with pytest.raises(ValueError):
        ReportStore(zip_path=tmp_path / "a.zip", directory=tmp_path)


@pytest.mark.sample
def test_every_sample_study_has_a_report(settings, need_sample):
    store = ReportStore.from_settings(settings)
    recs, _ = MimicCxrIndex.from_settings(settings).study_records()
    for r in recs:
        assert store.get_clean(r.subject_id, r.study_id)
