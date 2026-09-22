"""Retrieval: exactness, patient exclusion, manifest guards. numpy + (optionally) faiss; no torch needed.

macOS caveat: pip's torch and faiss-cpu clash on their OpenMP runtimes, so the faiss-backend cases skip when torch is
already loaded in this process. Run them on their own:  pytest tests/test_faiss_retrieval.py
"""
import json
import sys

import numpy as np
import pytest

from rrg.core.config import load_settings
from rrg.core.errors import (EmbeddingContractError, IndexBuildError, IndexVersionMismatchError, SplitLeakageError)
from rrg.core.types import ImageEmbedding
from rrg.retrieval.faiss_index import FaissRetriever, IdTable, write_index

D = 16
SHA = "a" * 64


def faiss_ok() -> bool:
    if sys.platform == "darwin" and "torch" in sys.modules:
        return False
    try:
        import faiss  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(params=["numpy", "faiss"])
def backend(request):
    if request.param == "faiss" and not faiss_ok():
        pytest.skip("faiss unavailable (or torch already loaded on macOS: run this file alone)")
    return request.param


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def corpus(n=200, seed=0):
    rng = np.random.default_rng(seed)
    emb = unit(rng.normal(size=(n, D)))
    subjects = rng.integers(1, 40, size=n)
    table = IdTable([f"d{i:04d}" for i in range(n)], np.arange(n) + 1000, subjects, ["PA"] * n)
    return emb, table


def settings_for(tmp_path, backend, **extra):
    ov = [f"paths.index_dir={tmp_path}", f"clear.embed_dim={D}", f"concept_bank.expected_dim={D}",
          f"retrieval.backend={backend}", "retrieval.k=5"] + [f"{k}={v}" for k, v in extra.items()]
    return load_settings(overrides=ov)


def build(tmp_path, backend, emb=None, table=None, **kw):
    if emb is None:
        emb, table = corpus()
    args = dict(corpus_splits=["train"], clear_checkpoint_name="best_model.pt", clear_checkpoint_sha256=SHA,
                view_policy={}, selection_stats={}, backend=backend)
    args.update(kw)
    write_index(tmp_path, "train", emb, table, **args)
    return emb, table


def brute(emb, table, q, k, exclude=None):
    sims = emb @ q
    order = sorted(range(len(sims)), key=lambda i: (-float(sims[i]), i))
    if exclude is not None:
        order = [i for i in order if table.subject_id[i] != exclude]
    return order[:k], sims


def test_matches_brute_force_exactly(tmp_path, backend):
    emb, table = build(tmp_path, backend)
    r = FaissRetriever.load(settings_for(tmp_path, backend))
    qs = unit(np.random.default_rng(1).normal(size=(12, D)))
    for q, hits in zip(qs, r.search(qs, k=7)):
        want, sims = brute(emb, table, q, 7)
        assert [table.dicom_id[i] for i in want] == [h.dicom_id for h in hits]
        np.testing.assert_allclose([sims[i] for i in want], [h.similarity for h in hits], atol=1e-5)
        assert [h.rank for h in hits] == list(range(1, 8)) and hits[0].report_id == f"s{hits[0].study_id}"


def test_patient_exclusion_is_exact_even_when_one_patient_dominates(tmp_path, backend):
    """One patient owns 30 near-duplicates of the query: fetching k+1 (self-exclusion only) would return them."""
    rng = np.random.default_rng(3)
    emb, table = corpus(150, seed=3)
    q = unit(rng.normal(size=D))
    emb = emb.copy()
    emb[:30] = unit(q + 0.01 * rng.normal(size=(30, D)))
    subjects = table.subject_id.copy()
    subjects[:30] = 777
    table = IdTable(table.dicom_id, table.study_id, subjects, table.view_position)
    build(tmp_path, backend, emb, table)
    r = FaissRetriever.load(settings_for(tmp_path, backend))

    leaked = r.search(q[None], k=5, exclude_subject_ids=None)[0]
    assert {h.subject_id for h in leaked} == {777}                                   # without exclusion: all one patient
    hits = r.search(q[None], k=5, exclude_subject_ids=[777])[0]
    want, _ = brute(emb, table, q, 5, exclude=777)
    assert [table.dicom_id[i] for i in want] == [h.dicom_id for h in hits] and len(hits) == 5
    assert all(h.subject_id != 777 for h in hits)


def test_exclusion_is_per_query_in_a_batch_and_can_be_switched_off(tmp_path, backend):
    emb, table = build(tmp_path, backend)
    r = FaissRetriever.load(settings_for(tmp_path, backend))
    qs = emb[[0, 1, 2]]                                                                # queries ARE corpus rows
    subs = [int(table.subject_id[i]) for i in (0, 1, 2)]
    for i, (q, hits) in enumerate(zip(qs, r.search(qs, k=5, exclude_subject_ids=subs))):
        want, _ = brute(emb, table, q, 5, exclude=subs[i])
        assert [table.dicom_id[j] for j in want] == [h.dicom_id for h in hits]
    off = FaissRetriever.load(settings_for(tmp_path, backend, **{"retrieval.exclude_same_patient": "false"}))
    assert off.search(qs, k=1, exclude_subject_ids=subs)[0][0].dicom_id == table.dicom_id[0]   # itself comes back
    assert r.search(qs, k=5, exclude_subject_ids=[None, None, None])[0][0].dicom_id == table.dicom_id[0]


def test_k_larger_than_corpus_floor_and_ties(tmp_path, backend):
    emb, table = corpus(20, seed=5)
    emb = emb.copy()
    emb[7] = emb[3]                                                                    # exact duplicate -> tie
    emb[11] = emb[3]
    build(tmp_path, backend, emb, table)
    r = FaissRetriever.load(settings_for(tmp_path, backend))
    hits = r.search(emb[3][None], k=999)[0]
    assert len(hits) == 20
    tied = [h.dicom_id for h in hits if h.similarity > 0.9999]
    assert tied == ["d0003", "d0007", "d0011"]                                         # ties ordered by row index
    floor = hits[5].similarity
    cut = r.search(emb[3][None], k=999, similarity_floor=floor)[0]
    assert all(h.similarity >= floor for h in cut) and len(cut) == 6
    assert r.search(emb[3][None], k=3, similarity_floor=1.5)[0] == []                     # nothing above 1.5 -> empty, not an error


def test_query_contract_is_enforced(tmp_path, backend):
    emb, _ = build(tmp_path, backend)
    r = FaissRetriever.load(settings_for(tmp_path, backend))
    q = emb[:1]
    for bad, msg in [(q.astype(np.float64), "float32"), (np.ones((1, D + 1), np.float32), r"\[B, 16\]"),
                     (q * 2, "L2-normalised"), (np.where(np.arange(D) == 0, np.nan, q).astype(np.float32), "NaN")]:
        with pytest.raises(EmbeddingContractError, match=msg):
            r.search(bad)
    with pytest.raises(EmbeddingContractError, match="k must"):
        r.search(q, k=0)
    with pytest.raises(EmbeddingContractError, match="one entry per query"):
        r.search(q, exclude_subject_ids=[1, 2])


def test_search_embedding_checks_precision_and_checkpoint(tmp_path, backend):
    emb, table = build(tmp_path, backend)
    r = FaissRetriever.load(settings_for(tmp_path, backend), expected_checkpoint_sha256=SHA)
    ok = ImageEmbedding(emb[0], D, "best_model.pt", SHA)
    assert r.search_embedding(ok, subject_id=int(table.subject_id[0]))
    with pytest.raises(IndexVersionMismatchError, match="precision"):
        r.search_embedding(ImageEmbedding(emb[0], D, "x", SHA, precision="bf16"))
    with pytest.raises(IndexVersionMismatchError, match="different CLEAR checkpoint"):
        r.search_embedding(ImageEmbedding(emb[0], D, "x", "b" * 64))


def test_load_guards(tmp_path, backend):
    build(tmp_path, backend)
    s = settings_for(tmp_path, backend)
    with pytest.raises(IndexVersionMismatchError, match="different CLEAR checkpoint"):
        FaissRetriever.load(s, expected_checkpoint_sha256="c" * 64)
    ids = tmp_path / "train_ids.json"
    good = ids.read_text()
    d = json.loads(good)
    d["subject_id"][0] += 1
    ids.write_text(json.dumps(d))
    with pytest.raises(IndexVersionMismatchError, match="out of sync"):
        FaissRetriever.load(s)
    ids.write_text(good)
    with pytest.raises(IndexVersionMismatchError, match="clear.embed_dim"):
        FaissRetriever.load(load_settings(overrides=[f"paths.index_dir={tmp_path}", "retrieval.backend=" + backend]))
    m = tmp_path / "train_manifest.json"
    man = json.loads(m.read_text())
    man["corpus_splits"] = ["train", "test"]
    m.write_text(json.dumps(man))
    with pytest.raises(SplitLeakageError):
        FaissRetriever.load(s)


def test_missing_files_and_backend_mismatch(tmp_path):
    with pytest.raises(IndexVersionMismatchError, match="build the index first"):
        FaissRetriever.load(settings_for(tmp_path, "numpy"))
    build(tmp_path, "numpy")
    if faiss_ok():
        with pytest.raises(IndexVersionMismatchError, match="build the index first|rebuild"):
            FaissRetriever.load(settings_for(tmp_path, "faiss"))


def test_write_index_validation(tmp_path):
    emb, table = corpus(10)
    for bad_emb, msg in [(emb.astype(np.float64), "float32"), (emb * 2, "L2-normalised"), (emb[:5], "ids")]:
        with pytest.raises(IndexBuildError, match=msg):
            build(tmp_path, "numpy", bad_emb, table)
    nan = emb.copy()
    nan[0, 0] = np.nan
    with pytest.raises(IndexBuildError, match="NaN"):
        build(tmp_path, "numpy", nan, table)
    with pytest.raises(SplitLeakageError):
        build(tmp_path, "numpy", emb, table, corpus_splits=["train", "validate"])
    with pytest.raises(IndexBuildError, match="save_embeddings"):
        build(tmp_path, "numpy", emb, table, save_embeddings=False)
    with pytest.raises(IndexBuildError, match="duplicate"):
        IdTable(["a", "a"], [1, 2], [1, 1], ["PA", "PA"])


def test_id_table_round_trip_and_subject_counts():
    _, table = corpus(50, seed=9)
    t2 = IdTable.from_json(table.to_json())
    assert t2.dicom_id == table.dicom_id and np.array_equal(t2.subject_id, table.subject_id)
    s = int(table.subject_id[0])
    assert table.subject_count(s) == int((table.subject_id == s).sum()) and table.subject_count(-5) == 0


def test_faiss_and_numpy_backends_agree(tmp_path):
    if not faiss_ok():
        pytest.skip("faiss unavailable (or torch already loaded on macOS: run this file alone)")
    emb, table = corpus(300, seed=11)
    a, b = tmp_path / "a", tmp_path / "b"
    build(a, "numpy", emb, table)
    build(b, "faiss", emb, table)
    ra, rb = FaissRetriever.load(settings_for(a, "numpy")), FaissRetriever.load(settings_for(b, "faiss"))
    qs = unit(np.random.default_rng(12).normal(size=(20, D)))
    subs = [int(table.subject_id[i]) for i in range(20)]
    for x, y in zip(ra.search(qs, k=8, exclude_subject_ids=subs), rb.search(qs, k=8, exclude_subject_ids=subs)):
        assert [h.dicom_id for h in x] == [h.dicom_id for h in y]
        np.testing.assert_allclose([h.similarity for h in x], [h.similarity for h in y], atol=1e-5)
