import pytest

from rrg.core.config import ConfigError, load_settings, require_file


def test_default_profile_loads_and_resolves_paths(settings):
    assert settings.paths.clear_checkpoint.is_absolute()
    assert settings.paths.clear_checkpoint.name == "best_model.pt"
    assert settings.dataset.accepted_view_positions == ["PA", "AP"]
    assert settings.spatial_enabled is False
    assert settings.concept_bank.expected_n_concepts == 368294


def test_hpc_profile_overlays_only_what_it_lists():
    s = load_settings(config="configs/hpc.yaml")
    assert "CHANGE_ME" in str(s.paths.mimic_root)
    assert s.clear.precision == "fp32" and s.runtime.device == "cuda"
    assert s.retrieval.backend == "faiss" and s.runtime.num_workers == 16
    assert s.concept_bank.expected_dim == 768          # untouched default
    assert s.paths.mimic_reports_zip is None and s.paths.mimic_reports_dir is not None


def test_placeholder_paths_fail_with_a_clear_message():
    s = load_settings(config="configs/hpc.yaml")
    with pytest.raises(ConfigError, match="CHANGE_ME"):
        require_file(s.paths.clear_checkpoint, "paths.clear_checkpoint")


def test_cli_override_and_yaml_typing():
    s = load_settings(overrides=["clear.batch_size=7", "demo.split=test", "runtime.torch_hub_dir=null"])
    assert s.clear.batch_size == 7 and s.demo.split == "test" and s.runtime.torch_hub_dir is None


@pytest.mark.parametrize(
    "override, fragment",
    [
        (["clear.bogus=1"], "unknown key"),
        (["clear.batch_size=abc"], "expected an integer"),
        (["clear.batch_size=0"], "must be positive"),
        (["clear.precision=bf16"], "must be 'fp32'"),
        (["clear.precision=fp8"], "must be 'fp32'"),
        (["retrieval.corpus_splits=[train, validate]"], "only training-split"),
        (["retrieval.corpus_splits=[test]"], "only training-split"),
        (["retrieval.corpus_splits=[]"], "only training-split"),
        (["retrieval.backend=annoy"], "faiss|numpy"),
        (["retrieval.k=0"], "must be positive"),
        (["retrieval.similarity_floor=1.5"], "cosine similarity"),
        (["retrieval.max_failed_fraction=1.0"], "max_failed_fraction"),
        (["runtime.num_workers=-1"], "num_workers"),
        (["dataset.accepted_view_positions=[PA, PA]"], "without duplicates"),
        (["runtime.device=tpu"], "auto|cuda|mps|cpu"),
        (["clear.embed_dim=512"], "must equal concept_bank.expected_dim"),
        (["paths.mimic_reports_dir=/x"], "only one of"),
        (["nosuchsection.x=1"], "unknown key"),
        (["broken"], "section.key=value"),
    ],
)
def test_invalid_config_is_rejected(override, fragment):
    with pytest.raises(ConfigError, match=fragment):
        load_settings(overrides=override)


def test_missing_key_is_reported(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("clear: {batch_size: 4}\nproject_root: /x\n")
    with pytest.raises(ConfigError, match="derived, not configurable"):
        load_settings(config=p)


def test_retrieval_and_view_policy_defaults(settings):
    assert settings.retrieval.corpus_splits == ["train"] and settings.retrieval.k == 5
    assert settings.retrieval.exclude_same_patient is True and settings.retrieval.similarity_floor is None
    assert settings.dataset.accepted_view_positions == ["PA", "AP"] and settings.dataset.one_image_per_study is True
    assert settings.paths.index_dir.is_absolute()
