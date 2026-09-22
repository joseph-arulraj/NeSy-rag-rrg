"""Real encoder + real sample: build the index from the train studies, then query with test / train images."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("clear")

from rrg.core.config import load_settings  # noqa: E402
from rrg.core.errors import IndexBuildError  # noqa: E402
from rrg.datasets.mimic_cxr import MimicCxrIndex  # noqa: E402
from rrg.ingest.image_loader import load_record_image  # noqa: E402
from rrg.offline import build_faiss_index as builder  # noqa: E402
from rrg.perception.clear_encoder import ClearEncoder  # noqa: E402
from rrg.retrieval.faiss_index import FaissRetriever  # noqa: E402

pytestmark = [pytest.mark.assets, pytest.mark.sample]


def _backends():
    import sys
    return ["numpy"] if sys.platform == "darwin" else ["numpy", "faiss"]


@pytest.fixture(scope="module")
def base(dinov2_hub_dir, assets_ok, sample_ok):
    if not (assets_ok and sample_ok):
        pytest.skip("needs real weights and the MIMIC sample")
    torch.hub.set_dir(str(dinov2_hub_dir))
    return [f"runtime.torch_hub_dir={dinov2_hub_dir}", "clear.local_files_only=true"]


@pytest.fixture(scope="module", params=_backends())
def built(request, base, tmp_path_factory):
    d = tmp_path_factory.mktemp("index")
    s = load_settings(overrides=base + [f"paths.index_dir={d}", f"retrieval.backend={request.param}"])
    manifest = builder.build_faiss_index(s, log=lambda *_: None)
    return s, manifest, d


def test_manifest_records_what_was_built(built):
    s, m, d = built
    assert m.corpus_splits == ["train"] and m.n_vectors == 50 and m.embed_dim == 768 and m.precision == "fp32"
    assert m.selection_stats["train"]["n_studies_kept"] == 50 and m.n_failed == 0 and m.built_limit is None
    assert len(m.clear_checkpoint_sha256) == 64 and m.view_policy["view_preference"] == ["PA", "AP"]
    emb = np.load(d / "train_embeddings.npy")
    assert emb.dtype == np.float32 and emb.shape == (50, 768)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)


def test_only_train_studies_are_indexed_and_queries_never_leak(built):
    s, m, d = built
    idx = MimicCxrIndex.from_settings(s)
    train_subjects = {r.subject_id for r in idx.study_records("train")[0]}
    enc = ClearEncoder(s)
    retriever = FaissRetriever.load(s, expected_checkpoint_sha256=enc.checkpoint_sha256)
    emb = np.load(d / "train_embeddings.npy")
    assert {int(x) for x in retriever.table.subject_id} == train_subjects            # nothing but train patients

    for rec in idx.study_records("test")[0][:5]:                                     # test queries: disjoint patients
        e = enc.encode(load_record_image(rec, s)[0])
        hits = retriever.search_embedding(e, subject_id=rec.subject_id)
        assert len(hits) == 5 and all(h.subject_id in train_subjects and h.subject_id != rec.subject_id for h in hits)
        sims = emb @ e.vector                                                        # brute-force oracle on saved embeddings
        want = sorted(range(len(sims)), key=lambda i: (-float(sims[i]), i))[:5]
        assert [h.dicom_id for h in hits] == [retriever.table.dicom_id[i] for i in want]

    rec = idx.study_records("train")[0][0]                                           # train query: own patient removed
    e = enc.encode(load_record_image(rec, s)[0])
    assert all(h.subject_id != rec.subject_id for h in retriever.search_embedding(e, subject_id=rec.subject_id))
    leaky = FaissRetriever.load(s, exclude_same_patient=False).search_embedding(e, subject_id=rec.subject_id)
    assert leaky[0].dicom_id == rec.dicom_id and leaky[0].similarity > 0.999          # without exclusion: itself


def test_undecodable_images_are_recorded_and_bounded(base, tmp_path, monkeypatch):
    real = builder.load_record_image
    bad = MimicCxrIndex.from_settings(load_settings()).study_records("train")[0][3].dicom_id

    def flaky(record, settings):
        if record.dicom_id == bad:
            raise builder.PipelineError("simulated decode failure")
        return real(record, settings)

    monkeypatch.setattr(builder, "load_record_image", flaky)
    s = load_settings(overrides=base + [f"paths.index_dir={tmp_path}", "retrieval.max_failed_fraction=0.05"])
    m = builder.build_faiss_index(s, log=lambda *_: None)
    assert m.n_failed == 1 and m.failed_dicom_ids == [bad] and m.n_vectors == 49
    assert bad not in FaissRetriever.load(s).table.dicom_id
    strict = load_settings(overrides=base + [f"paths.index_dir={tmp_path / 's'}", "retrieval.max_failed_fraction=0.0"])
    with pytest.raises(IndexBuildError, match="failed to decode"):
        builder.build_faiss_index(strict, log=lambda *_: None)


def test_limit_marks_a_smoke_index(base, tmp_path):
    s = load_settings(overrides=base + [f"paths.index_dir={tmp_path}"])
    m = builder.build_faiss_index(s, limit=7, log=lambda *_: None)
    assert m.n_vectors == 7 and m.built_limit == 7
