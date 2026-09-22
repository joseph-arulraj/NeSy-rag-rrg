"""Needs torch, the cloned CLEAR package, the real weights and the DINOv2 source cache."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("clear")

from rrg.concepts.bank import ConceptBank  # noqa: E402
from rrg.concepts.similarity import score_concepts, top_k  # noqa: E402
from rrg.core.config import load_settings  # noqa: E402
from rrg.core.errors import ClearLoadError, ConfigError  # noqa: E402
from rrg.core.runtime import resolve_device  # noqa: E402
from rrg.datasets.mimic_cxr import MimicCxrIndex  # noqa: E402
from rrg.ingest.image_loader import load_record_image  # noqa: E402
from rrg.perception.clear_encoder import ClearEncoder  # noqa: E402

pytestmark = pytest.mark.assets


@pytest.fixture(scope="module")
def s(dinov2_hub_dir):
    st = load_settings(overrides=[f"runtime.torch_hub_dir={dinov2_hub_dir}", "clear.local_files_only=true"])
    torch.hub.set_dir(str(dinov2_hub_dir))
    return st


@pytest.fixture(scope="module")
def encoder(s, assets_ok, sample_ok):
    if not (assets_ok and sample_ok):
        pytest.skip("needs real weights and the MIMIC sample")
    return ClearEncoder(s)


@pytest.fixture(scope="module")
def images(s, encoder):
    recs = MimicCxrIndex.from_settings(s).select(split="train", view_positions=s.dataset.accepted_view_positions)[:3]
    return [load_record_image(r, s)[0] for r in recs]


def test_embedding_contract(encoder, images):
    e = encoder.encode(images[0])
    assert e.vector.shape == (768,) and e.vector.dtype == np.float32 and e.normalised
    assert abs(np.linalg.norm(e.vector) - 1.0) < 1e-4 and np.isfinite(e.vector).all()
    assert e.checkpoint == "best_model.pt"


def test_deterministic_and_batch_invariant(encoder, images):
    a, b = encoder.encode(images[0]).vector, encoder.encode(images[0]).vector
    np.testing.assert_allclose(a, b, atol=1e-6)
    batch = encoder.encode_batch(images).numpy()
    assert batch.shape == (3, 768)
    for i in range(3):
        assert float(batch[i] @ encoder.encode(images[i]).vector) > 0.9999


def test_encode_tensors_is_the_dataloader_path(encoder, images):
    """Bulk index building preprocesses in DataLoader workers and calls encode_tensors: it must equal encode()."""
    x = torch.stack([encoder.preprocess(images[0].image)])
    np.testing.assert_allclose(encoder.encode_tensors(x)[0].numpy(), encoder.encode(images[0]).vector, atol=1e-6)
    assert encoder.encode(images[0]).precision == "fp32"
    assert len(encoder.checkpoint_sha256) == 64


def test_matches_clear_reference_computation(s, encoder, images):
    """Our encode + score == CLEAR's own recipe (README/quickstart): normalise(encode_image(preprocess(img)))
    against per-chunk-normalised raw concept embeddings -- computed here without any rrg code."""
    import torch.nn.functional as F

    img = images[0]
    with torch.inference_mode():
        x = torch.stack([encoder.preprocess(img.image)]).to(encoder.device)
        ref_feat = F.normalize(encoder.model.encode_image(x), dim=-1).cpu()
    ours = encoder.encode(img)
    assert float(ref_feat[0].numpy() @ ours.vector) > 0.99999

    raw = torch.load(s.paths.concept_embeddings, map_location="cpu", weights_only=True).float()
    ref_scores = (F.normalize(raw, dim=-1) @ ref_feat[0]).numpy()
    bank = ConceptBank.load(s)
    our_scores = score_concepts(ours, bank, encoder.device)
    np.testing.assert_allclose(our_scores, ref_scores, atol=2e-4)
    assert [h.concept_id for h in top_k(our_scores, bank, 10)] == np.argsort(-ref_scores)[:10].tolist()


def test_cpu_and_default_device_agree(s, encoder, images):
    if encoder.device.type == "cpu":
        pytest.skip("default device is already cpu")
    cpu = ClearEncoder(s, torch.device("cpu"))
    a, b = encoder.encode(images[1]).vector, cpu.encode(images[1]).vector
    assert float(a @ b) > 0.999                          # different backends -> tiny numeric drift only


def test_config_guards(s, tmp_path):
    with pytest.raises(ConfigError, match="must be 'fp32'"):
        load_settings(overrides=["clear.precision=bf16"])
    with pytest.raises(ConfigError, match="not found"):
        ClearEncoder(load_settings(overrides=[f"paths.clear_checkpoint={tmp_path / 'nope.pt'}"]), torch.device("cpu"))
    bad = tmp_path / "bad.pt"
    torch.save({"x": torch.zeros(1)}, bad)
    with pytest.raises(ClearLoadError):
        ClearEncoder(load_settings(overrides=[f"paths.clear_checkpoint={bad}", "runtime.device=cpu"]), torch.device("cpu"))


def test_forced_unavailable_device_raises():
    if not torch.cuda.is_available():
        with pytest.raises(ConfigError, match="CUDA is not available"):
            resolve_device("cuda")
    assert resolve_device("cpu").type == "cpu"
