import pytest
from PIL import Image

from rrg.core.errors import InputDecodeError, InputModalityError
from rrg.datasets.mimic_cxr import MimicCxrIndex, StudyRecord
from rrg.ingest.image_loader import load_image, load_record_image


def _rec(path, view="PA"):
    return StudyRecord(1, 2, "d", "train", view, None, None, path)


def test_decodes_grayscale_jpeg(settings, tmp_path):
    p = tmp_path / "x.jpg"
    Image.new("L", (300, 200), 128).save(p)
    img, meta = load_image(p, settings)
    assert (img.width, img.height, img.mode) == (300, 200, "L")
    assert meta.spatial_unavailable is True                       # N02 paused -> Evidence A absent
    assert meta.image_id == "x"


def test_record_metadata_flows_into_studymeta(settings, tmp_path):
    p = tmp_path / "x.jpg"
    Image.new("L", (100, 100)).save(p)
    _, meta = load_record_image(_rec(p, "AP"), settings)
    assert (meta.study_uid, meta.subject_id, meta.split, meta.view_position) == ("2", "1", "train", "AP")


def test_lateral_view_is_rejected_before_decoding(settings, tmp_path):
    with pytest.raises(InputModalityError):
        load_record_image(_rec(tmp_path / "does_not_exist.jpg", "LATERAL"), settings)


@pytest.mark.parametrize("name,maker,msg", [
    ("a.jpg", lambda p: p.write_bytes(b"not an image"), "cannot decode"),
    ("b.txt", lambda p: p.write_text("x"), "extension"),
    ("c.jpg", lambda p: Image.new("L", (10, 10)).save(p), "smaller than"),
    ("d.png", lambda p: Image.new("I;16", (100, 100)).save(p), "unsupported pixel mode"),
    ("e.jpg", lambda p: None, "file not found"),
])
def test_bad_inputs_raise_decode_errors(settings, tmp_path, name, maker, msg):
    p = tmp_path / name
    maker(p)
    with pytest.raises(InputDecodeError, match=msg):
        load_image(p, settings)


def test_truncated_jpeg_fails_at_load_time(settings, tmp_path):
    good = tmp_path / "g.jpg"
    Image.new("L", (400, 400), 90).save(good)
    bad = tmp_path / "t.jpg"
    bad.write_bytes(good.read_bytes()[:200])
    with pytest.raises(InputDecodeError):
        load_image(bad, settings)


@pytest.mark.sample
def test_every_sample_image_decodes(settings, need_sample):
    idx = MimicCxrIndex.from_settings(settings)
    for r in idx.records[:10]:
        img, meta = load_record_image(r, settings)
        assert img.mode == "L" and min(img.width, img.height) >= settings.input.min_image_side
