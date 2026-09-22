"""Configuration loader.

Single source of truth for every path and tunable is configs/default.yaml. The dataclasses
below deliberately have NO defaults: a key that is missing from the YAML is an error, so the
YAML can never silently drift from what the code expects. Unknown keys are errors too
(typo protection). Profiles (configs/hpc.yaml) are deep-merged over the default file.
"""
from __future__ import annotations

import os
import types
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union, get_args, get_origin, get_type_hints

import yaml

from .errors import ConfigError

CONFIG_ENV = "RRG_CONFIG"
ROOT_ENV = "RRG_PROJECT_ROOT"


# --------------------------------------------------------------------------- schema
@dataclass
class PathsConfig:
    mimic_root: Path
    mimic_reports_zip: Optional[Path]
    mimic_reports_dir: Optional[Path]
    model_weights_dir: Path
    clear_checkpoint: Path
    concepts_csv: Path
    concept_embeddings: Path
    output_dir: Path
    index_dir: Path


@dataclass
class RuntimeConfig:
    device: str
    seed: int
    mps_fallback: bool
    torch_hub_dir: Optional[Path]
    torch_num_threads: Optional[int]
    num_workers: int


@dataclass
class DatasetConfig:
    metadata_csv: str
    split_csv: str
    chexpert_csv: str
    negbio_csv: str
    accepted_view_positions: list[str]   # ordered by preference (first = most preferred)
    one_image_per_study: bool
    splits: list[str]


@dataclass
class InputConfig:
    accepted_extensions: list[str]
    min_image_side: int


@dataclass
class ClearConfig:
    local_files_only: bool
    precision: str
    embed_dim: int
    batch_size: int
    hash_checkpoint: bool


@dataclass
class ConceptBankConfig:
    expected_n_concepts: int
    expected_dim: int
    normalise_embeddings: bool
    mmap_embeddings: bool
    hash_files: bool
    score_batch_size: int


@dataclass
class RetrievalConfig:
    backend: str
    corpus_splits: list[str]
    index_name: str
    k: int
    similarity_floor: Optional[float]
    exclude_same_patient: bool
    save_embeddings: bool
    max_failed_fraction: float
    log_every_batches: int


@dataclass
class RadLexConfig:
    raw_owl_path: Path
    snapshot_path: Path
    version: str
    chest_scope_root_label: str
    additional_scope_roots: list[str]


@dataclass
class RadGraphConfig:
    model_type: str
    device: str
    batch_size: int
    version: str


@dataclass
class TaggingConfig:
    checkpoint_dir: Path
    checkpoint_every_batches: int
    laterality_gazetteer: list[str]
    temporal_implies_present: list[str]
    temporal_implies_absent: list[str]
    temporal_indeterminate: list[str]
    negation_cues: list[str]
    finding_synonyms_path: Path
    observation_min_frequency: int


@dataclass
class GroupingConfig:
    top_k_groups: int
    contributing_concepts_per_group: int
    present_absent_count_threshold: float
    dedup_threshold: float


@dataclass
class ExploreConfig:
    sample_size: int
    top_tokens: int
    examples_per_category: int
    seed: int


@dataclass
class DemoConfig:
    split: str
    num_images: int
    top_k: int


@dataclass
class Settings:
    paths: PathsConfig
    runtime: RuntimeConfig
    dataset: DatasetConfig
    input: InputConfig
    clear: ClearConfig
    concept_bank: ConceptBankConfig
    retrieval: RetrievalConfig
    radlex: RadLexConfig
    radgraph: RadGraphConfig
    tagging: TaggingConfig
    grouping: GroupingConfig
    explore: ExploreConfig
    demo: DemoConfig
    spatial_enabled: bool
    project_root: Path  # not read from YAML; set by the loader

    def summary(self) -> dict[str, Any]:
        """Plain-dict view (Paths as str) for logging / saving next to results."""
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = _to_plain(v)
        return out


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_plain(x) for x in obj]
    return obj


# ------------------------------------------------------------------- strict building
_UNION_TYPES = tuple(t for t in (Union, getattr(types, "UnionType", None)) if t is not None)


def _coerce(value: Any, hint: Any, where: str) -> Any:
    origin = get_origin(hint)
    if origin in _UNION_TYPES:
        args = get_args(hint)
        if value is None and type(None) in args:
            return None
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _coerce(value, non_none[0], where)
        raise ConfigError(f"{where}: unsupported union type {hint}")
    if value is None:
        raise ConfigError(f"{where}: null is not allowed here (expected {hint})")
    if hint is Path:
        if not isinstance(value, (str, Path)):
            raise ConfigError(f"{where}: expected a path string, got {type(value).__name__}")
        return Path(str(value)).expanduser()
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list, got {type(value).__name__}")
        (inner,) = get_args(hint)
        return [_coerce(v, inner, f"{where}[{i}]") for i, v in enumerate(value)]
    raise ConfigError(f"{where}: unsupported config type {hint}")


def _build(cls: type, data: Any, where: str, skip: frozenset[str] = frozenset()) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"'{where}' must be a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    names = [f.name for f in fields(cls) if f.name not in skip]
    unknown = sorted(set(data) - set(names))
    if unknown:
        raise ConfigError(f"unknown key(s) {unknown} in '{where}'; valid keys: {names}")
    missing = sorted(set(names) - set(data))
    if missing:
        raise ConfigError(f"missing key(s) {missing} in '{where}' (add them to configs/default.yaml)")
    kwargs: dict[str, Any] = {}
    for name in names:
        hint = hints[name]
        sub = f"{where}.{name}"
        if is_dataclass(hint):
            kwargs[name] = _build(hint, data[name], sub)
        else:
            kwargs[name] = _coerce(data[name], hint, sub)
    return kwargs if skip else cls(**kwargs)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def _apply_overrides(data: dict, overrides: Sequence[str]) -> dict:
    data = _deep_merge({}, data)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override '{item}' must look like section.key=value")
        dotted, raw = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = data
        for k in keys[:-1]:
            node = node.setdefault(k, {})
            if not isinstance(node, dict):
                raise ConfigError(f"override '{item}': '{k}' is not a section")
        node[keys[-1]] = yaml.safe_load(raw)
    return data


# ------------------------------------------------------------------------- public API
def find_project_root(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get(ROOT_ENV)
    if env:
        return Path(env).expanduser().resolve()
    candidates = [Path(__file__).resolve().parents[3], Path.cwd()]
    for c in candidates:
        if (c / "configs" / "default.yaml").is_file():
            return c
    raise ConfigError(
        f"cannot locate the project root (a folder containing configs/default.yaml). "
        f"Set {ROOT_ENV} or run from the repository root."
    )


def load_settings(
    config: str | Path | None = None,
    overrides: Sequence[str] | None = None,
    project_root: str | Path | None = None,
) -> Settings:
    """Load configs/default.yaml, overlay a profile and CLI overrides, validate, resolve paths."""
    root = find_project_root(project_root)
    data = _load_yaml(root / "configs" / "default.yaml")

    profile = config if config is not None else os.environ.get(CONFIG_ENV)
    if profile:
        p = Path(profile).expanduser()
        if not p.is_file() and not p.is_absolute() and (root / p).is_file():
            p = root / p
        data = _deep_merge(data, _load_yaml(p))
    if overrides:
        data = _apply_overrides(data, overrides)

    if "project_root" in data:
        raise ConfigError("'project_root' is derived, not configurable; remove it from the YAML")
    parts = _build(Settings, data, "settings", skip=frozenset({"project_root"}))
    settings = Settings(project_root=root, **parts)
    _resolve_paths(settings.paths, root)
    if settings.runtime.torch_hub_dir is not None:
        settings.runtime.torch_hub_dir = _abs(settings.runtime.torch_hub_dir, root)
    _validate(settings)
    return settings


def _abs(p: Path, root: Path) -> Path:
    return p if p.is_absolute() else (root / p)


def _resolve_paths(paths: PathsConfig, root: Path) -> None:
    for f in fields(paths):
        v = getattr(paths, f.name)
        if v is not None:
            setattr(paths, f.name, _abs(v, root))


def _validate(s: Settings) -> None:
    if s.runtime.device not in {"auto", "cuda", "mps", "cpu"}:
        raise ConfigError(f"runtime.device must be auto|cuda|mps|cpu, got {s.runtime.device!r}")
    if s.clear.precision != "fp32":
        raise ConfigError(
            f"clear.precision must be 'fp32', got {s.clear.precision!r}: the FAISS index and every query embedding "
            "are fp32-only at this stage (no bf16 / mixed precision)"
        )
    if set(s.retrieval.corpus_splits) - {"train"} or not s.retrieval.corpus_splits:
        raise ConfigError(
            f"retrieval.corpus_splits must be ['train'] (got {s.retrieval.corpus_splits}): only training-split images "
            "may be retrieved; validation is for tuning and test is kept untouched for final evaluation"
        )
    if s.retrieval.backend not in {"faiss", "numpy"}:
        raise ConfigError(f"retrieval.backend must be faiss|numpy, got {s.retrieval.backend!r}")
    if s.retrieval.k <= 0 or s.retrieval.log_every_batches <= 0:
        raise ConfigError("retrieval.k and retrieval.log_every_batches must be positive")
    if not 0.0 <= s.retrieval.max_failed_fraction < 1.0:
        raise ConfigError("retrieval.max_failed_fraction must be in [0, 1)")
    if s.retrieval.similarity_floor is not None and not -1.0 <= s.retrieval.similarity_floor <= 1.0:
        raise ConfigError("retrieval.similarity_floor must be null or a cosine similarity in [-1, 1]")
    if s.runtime.num_workers < 0:
        raise ConfigError("runtime.num_workers must be >= 0")
    if s.radgraph.device not in {"auto", "cuda", "mps", "cpu"}:
        raise ConfigError(f"radgraph.device must be auto|cuda|mps|cpu, got {s.radgraph.device!r}")
    if s.radgraph.batch_size <= 0:
        raise ConfigError("radgraph.batch_size must be positive")
    if s.grouping.top_k_groups <= 0 or s.grouping.contributing_concepts_per_group <= 0:
        raise ConfigError("grouping.top_k_groups and grouping.contributing_concepts_per_group must be positive")
    if not 0.0 <= s.grouping.present_absent_count_threshold <= 1.0:
        raise ConfigError("grouping.present_absent_count_threshold must be in [0, 1] (it thresholds a cosine-derived score)")
    if s.tagging.checkpoint_every_batches <= 0:
        raise ConfigError("tagging.checkpoint_every_batches must be positive")
    if s.tagging.observation_min_frequency <= 0:
        raise ConfigError("tagging.observation_min_frequency must be positive")
    if not s.dataset.accepted_view_positions or len(set(s.dataset.accepted_view_positions)) != len(s.dataset.accepted_view_positions):
        raise ConfigError("dataset.accepted_view_positions must be a non-empty list without duplicates (ordered by preference)")
    for name, val in (
        ("clear.batch_size", s.clear.batch_size),
        ("concept_bank.score_batch_size", s.concept_bank.score_batch_size),
        ("input.min_image_side", s.input.min_image_side),
        ("clear.embed_dim", s.clear.embed_dim),
        ("concept_bank.expected_dim", s.concept_bank.expected_dim),
        ("concept_bank.expected_n_concepts", s.concept_bank.expected_n_concepts),
    ):
        if val <= 0:
            raise ConfigError(f"{name} must be positive, got {val}")
    if s.clear.embed_dim != s.concept_bank.expected_dim:
        raise ConfigError(
            f"clear.embed_dim ({s.clear.embed_dim}) must equal concept_bank.expected_dim "
            f"({s.concept_bank.expected_dim}): the image feature is dotted with the concept embeddings"
        )
    if s.paths.mimic_reports_zip is not None and s.paths.mimic_reports_dir is not None:
        raise ConfigError("set only one of paths.mimic_reports_zip / paths.mimic_reports_dir")


def require_file(path: Path, key: str) -> Path:
    """Fail with a message that names the config key, and flags un-edited HPC placeholders."""
    if "CHANGE_ME" in str(path):
        raise ConfigError(f"{key} still contains the CHANGE_ME placeholder ({path}); edit your profile YAML")
    if not path.is_file():
        raise ConfigError(f"{key}: file not found: {path}")
    return path


def require_dir(path: Path, key: str) -> Path:
    if "CHANGE_ME" in str(path):
        raise ConfigError(f"{key} still contains the CHANGE_ME placeholder ({path}); edit your profile YAML")
    if not path.is_dir():
        raise ConfigError(f"{key}: directory not found: {path}")
    return path
