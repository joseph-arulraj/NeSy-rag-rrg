from __future__ import annotations

import os
from pathlib import Path

import pytest

from rrg.core.config import Settings, load_settings


@pytest.fixture(scope="session")
def settings() -> Settings:
    return load_settings()


@pytest.fixture(scope="session")
def sample_ok(settings: Settings) -> bool:
    return (settings.paths.mimic_root / "mimic-cxr-2.0.0-split.csv.gz").is_file()


@pytest.fixture
def need_sample(sample_ok: bool) -> None:
    if not sample_ok:
        pytest.skip("MIMIC-CXR local sample not found under paths.mimic_root")


@pytest.fixture(scope="session")
def assets_ok(settings: Settings) -> bool:
    p = settings.paths
    return all(x.is_file() for x in (p.clear_checkpoint, p.concepts_csv, p.concept_embeddings))


@pytest.fixture
def need_assets(assets_ok: bool) -> None:
    if not assets_ok:
        pytest.skip("CLEAR weights / concept files not found under paths.*")


def tiny_settings(tmp_path: Path, n: int = 5, d: int = 8) -> Settings:
    """Settings pointing at a synthetic n x d bank in tmp_path."""
    return load_settings(
        overrides=[
            f"paths.concepts_csv={tmp_path / 'c.csv'}",
            f"paths.concept_embeddings={tmp_path / 'e.pt'}",
            f"concept_bank.expected_n_concepts={n}",
            f"concept_bank.expected_dim={d}",
            f"clear.embed_dim={d}",
        ]
    )


def write_bank(tmp_path: Path, texts: list[str], emb) -> None:
    import csv

    import torch

    with open(tmp_path / "c.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["concept", "concept_idx"])
        for i, t in enumerate(texts):
            w.writerow([t, i])
    torch.save(emb, tmp_path / "e.pt")


@pytest.fixture(scope="session")
def dinov2_hub_dir(settings: Settings) -> Path:
    """A torch-hub dir that already holds the DINOv2 source (no network in tests)."""
    candidates = [os.environ.get("RRG_TEST_TORCH_HUB_DIR"), settings.runtime.torch_hub_dir]
    try:
        import torch

        candidates.append(torch.hub.get_dir())
    except Exception:  # noqa: BLE001
        pass
    for c in candidates:
        if c and (Path(c) / "facebookresearch_dinov2_main" / "hubconf.py").is_file():
            return Path(c)
    pytest.skip("DINOv2 source not cached: run scripts/cache_dinov2.py or set RRG_TEST_TORCH_HUB_DIR")
