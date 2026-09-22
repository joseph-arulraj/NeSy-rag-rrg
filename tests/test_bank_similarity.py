import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import tiny_settings, write_bank  # noqa: E402
from rrg.concepts.bank import ConceptBank  # noqa: E402
from rrg.concepts.similarity import gather, score_concepts, score_concepts_batch, top_k  # noqa: E402
from rrg.core.errors import BankDimMismatchError, BankLoadError, EmbeddingContractError, NonFiniteScoreError  # noqa: E402
from rrg.core.types import ImageEmbedding  # noqa: E402

TEXTS = ["a", "b", "c", "d", "e"]


def make_bank(tmp_path, emb=None, texts=TEXTS, **kw):
    g = torch.Generator().manual_seed(0)
    emb = torch.randn(len(TEXTS), 8, generator=g) * 3.0 if emb is None else emb   # deliberately NOT unit-norm
    write_bank(tmp_path, texts, emb)
    return ConceptBank.load(tiny_settings(tmp_path, n=len(texts), **kw)), emb


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def test_bank_normalises_once_and_records_raw_norms(tmp_path):
    bank, emb = make_bank(tmp_path)
    assert torch.allclose(bank.embeddings.norm(dim=1), torch.ones(5), atol=1e-6)
    assert bank.info.prenormalised is False and bank.info.norm_max > 1.5
    assert len(bank) == 5 and bank.embed_dim == 8 and bank.text(2) == "c"
    assert torch.equal(torch.load(tmp_path / "e.pt", weights_only=True), emb)      # file untouched


def test_bank_alignment_and_shape_errors(tmp_path):
    with pytest.raises(BankLoadError, match="misaligned"):
        write_bank(tmp_path, TEXTS, torch.randn(4, 8))
        ConceptBank.load(tiny_settings(tmp_path))
    with pytest.raises(BankDimMismatchError):
        write_bank(tmp_path, TEXTS, torch.randn(5, 7))
        ConceptBank.load(tiny_settings(tmp_path))
    bad = torch.randn(5, 8)
    bad[2, 3] = float("nan")
    write_bank(tmp_path, TEXTS, bad)
    with pytest.raises(BankLoadError, match="non-finite"):
        ConceptBank.load(tiny_settings(tmp_path))
    write_bank(tmp_path, TEXTS, torch.zeros(5, 8))
    with pytest.raises(BankLoadError, match="all-zero"):
        ConceptBank.load(tiny_settings(tmp_path))


def test_score_is_cosine_similarity(tmp_path):
    bank, emb = make_bank(tmp_path)
    x = unit(np.random.default_rng(1).normal(size=8))
    e = ImageEmbedding(vector=x, embed_dim=8, checkpoint="t")
    got = score_concepts(e, bank)
    want = unit(emb.numpy()) @ x
    assert got.shape == (5,) and got.dtype == np.float32
    np.testing.assert_allclose(got, want, atol=1e-6)
    assert np.all(np.abs(got) <= 1 + 1e-6)


def test_batch_equals_single_and_respects_batch_size(tmp_path):
    bank, _ = make_bank(tmp_path)
    xs = unit(np.random.default_rng(2).normal(size=(7, 8)))
    b = score_concepts_batch(xs, bank, batch_size=3)
    assert b.shape == (7, 5)
    for i in range(7):
        np.testing.assert_allclose(b[i], score_concepts(ImageEmbedding(xs[i], 8, "t"), bank), atol=1e-6)


def test_top_k_order_ties_and_gather(tmp_path):
    bank, _ = make_bank(tmp_path)
    scores = np.array([0.1, 0.9, 0.9, -0.2, 0.5], dtype=np.float32)
    hits = top_k(scores, bank, 3)
    assert [h.concept_id for h in hits] == [1, 2, 4]              # tie 1 vs 2 -> lower id first
    assert hits[0].text == "b" and hits[0].score == pytest.approx(0.9)
    assert len(top_k(scores, bank, 99)) == 5
    np.testing.assert_allclose(gather(scores, [4, 0]), [0.5, 0.1], atol=1e-7)
    with pytest.raises(BankDimMismatchError):
        top_k(scores[:3], bank, 1)


def test_interface_contract_violations(tmp_path):
    bank, _ = make_bank(tmp_path)
    with pytest.raises(BankDimMismatchError):
        score_concepts_batch(unit(np.ones((1, 7))), bank)
    with pytest.raises(EmbeddingContractError, match="L2-normalised"):
        score_concepts_batch(np.ones((1, 8), dtype=np.float32), bank)
    with pytest.raises(EmbeddingContractError, match=r"\[B, D\]"):
        score_concepts_batch(np.ones((1, 2, 8), dtype=np.float32), bank)
    nan = unit(np.ones((1, 8)))
    nan[0, 0] = np.nan
    with pytest.raises(NonFiniteScoreError):
        score_concepts_batch(nan, bank)


@pytest.mark.assets
def test_real_bank_is_aligned_and_scores_are_bounded(settings, need_assets):
    bank = ConceptBank.load(settings)
    assert len(bank) == 368294 and bank.embed_dim == 768
    assert bank.info.prenormalised                      # the released file is already unit-norm
    x = unit(np.random.default_rng(0).normal(size=768))
    s = score_concepts_batch(x[None], bank)[0]
    assert s.shape == (368294,) and np.abs(s).max() <= 1 + 1e-4
