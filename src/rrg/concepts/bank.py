"""N08: the concept bank = ordered vocabulary + aligned 368,294 x 768 text embeddings.

Read-only at runtime. Row i of the embedding matrix is concept i of the vocabulary
(ConceptID == row index). Embeddings are L2-normalised ONCE at load so that
image . embedding is a cosine similarity; the raw file is never modified.
RadLex anatomy/laterality tags are a separate, later layer and are not part of this loader.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ..core.config import Settings, require_file
from ..core.errors import BankDimMismatchError, BankLoadError
from ..core.hashing import file_sha256
from .vocabulary import load_concept_texts

if TYPE_CHECKING:  # torch is imported lazily so vocabulary-only tools need no torch
    import torch

_CHUNK_ROWS = 65536


@dataclass(frozen=True)
class BankInfo:
    concepts_path: str
    embeddings_path: str
    n_concepts: int
    embed_dim: int
    file_dtype: str
    norm_min: float               # of the raw file, before normalisation
    norm_max: float
    prenormalised: bool           # raw rows already unit-norm (within 1e-3)
    l2_normalised_in_memory: bool
    concepts_sha256: Optional[str] = None
    embeddings_sha256: Optional[str] = None


def _load_embedding_tensor(path: Path, mmap: bool) -> "torch.Tensor":
    import torch

    kwargs: dict[str, Any] = dict(map_location="cpu", weights_only=True)
    payload = None
    if mmap:
        try:
            payload = torch.load(path, mmap=True, **kwargs)
        except (TypeError, RuntimeError, ValueError):
            payload = None  # older torch / legacy file format: fall back to a normal read
    if payload is None:
        try:
            payload = torch.load(path, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- surface any load failure with the file name
            raise BankLoadError(f"{path}: cannot load embeddings ({exc})") from exc
    if isinstance(payload, dict):
        for key in ("embeddings", "concept_embeddings"):
            if key in payload:
                payload = payload[key]
                break
    if not torch.is_tensor(payload) or payload.ndim != 2:
        raise BankLoadError(f"{path}: expected one 2-D tensor, got {type(payload).__name__}")
    return payload


class ConceptBank:
    def __init__(
        self,
        concepts: tuple[str, ...],
        embeddings: "torch.Tensor",
        info: BankInfo,
        tags: Optional[list] = None,   # list[ConceptTag], positionally aligned; None until tagged
    ):
        self.concepts = concepts
        self.embeddings = embeddings          # [N, D] float32, CPU
        self.info = info
        self.tags = tags
        self._device_cache: dict[str, "torch.Tensor"] = {}

    def with_tags(self, tags: list) -> "ConceptBank":
        """Returns a new ConceptBank sharing this one's embeddings (no copy) with tags attached."""
        if len(tags) != len(self.concepts):
            raise BankLoadError(f"{len(tags)} tags for {len(self.concepts)} concepts -- misaligned")
        return ConceptBank(self.concepts, self.embeddings, self.info, tags)

    # ---- construction
    @classmethod
    def load(cls, settings: Settings) -> "ConceptBank":
        import torch

        cb = settings.concept_bank
        csv_path = require_file(settings.paths.concepts_csv, "paths.concepts_csv")
        emb_path = require_file(settings.paths.concept_embeddings, "paths.concept_embeddings")

        concepts = load_concept_texts(csv_path, cb.expected_n_concepts)
        raw = _load_embedding_tensor(emb_path, cb.mmap_embeddings)

        n, d = int(raw.shape[0]), int(raw.shape[1])
        if n != len(concepts):
            raise BankLoadError(
                f"misaligned bank: {len(concepts)} concepts in {csv_path.name} but {n} embedding rows in {emb_path.name}"
            )
        if d != cb.expected_dim:
            raise BankDimMismatchError(f"{emb_path.name}: embedding dim {d}, expected concept_bank.expected_dim={cb.expected_dim}")

        out = torch.empty((n, d), dtype=torch.float32)
        nmin, nmax = float("inf"), 0.0
        for start in range(0, n, _CHUNK_ROWS):
            chunk = raw[start : start + _CHUNK_ROWS].to(torch.float32)
            if not torch.isfinite(chunk).all():
                raise BankLoadError(f"{emb_path.name}: non-finite values in rows {start}..{start + chunk.shape[0]}")
            norms = chunk.norm(dim=1, keepdim=True)
            nmin, nmax = min(nmin, float(norms.min())), max(nmax, float(norms.max()))
            out[start : start + chunk.shape[0]] = chunk / norms.clamp_min(1e-6) if cb.normalise_embeddings else chunk
        if nmin <= 1e-6:
            raise BankLoadError(f"{emb_path.name}: contains an all-zero embedding row (min norm {nmin})")

        info = BankInfo(
            concepts_path=str(csv_path),
            embeddings_path=str(emb_path),
            n_concepts=n,
            embed_dim=d,
            file_dtype=str(raw.dtype).replace("torch.", ""),
            norm_min=nmin,
            norm_max=nmax,
            prenormalised=abs(nmin - 1.0) < 1e-3 and abs(nmax - 1.0) < 1e-3,
            l2_normalised_in_memory=cb.normalise_embeddings,
            concepts_sha256=file_sha256(csv_path) if cb.hash_files else None,
            embeddings_sha256=file_sha256(emb_path) if cb.hash_files else None,
        )
        return cls(concepts, out, info)

    # ---- access
    def __len__(self) -> int:
        return len(self.concepts)

    @property
    def embed_dim(self) -> int:
        return int(self.embeddings.shape[1])

    def text(self, concept_id: int) -> str:
        return self.concepts[concept_id]

    def device_embeddings(self, device: "torch.device") -> "torch.Tensor":
        """The normalised matrix on `device`, copied once and cached (1.1 GB in fp32)."""
        if device.type == "cpu":
            return self.embeddings
        key = str(device)
        if key not in self._device_cache:
            self._device_cache[key] = self.embeddings.to(device)
        return self._device_cache[key]


# ============================================================================================
# Tag sidecar (pipeline.md 4.6 rev 3): per-concept (anatomy, laterality, resolved_polarity,
# temporal_class, canonical_finding, group_key). Built offline by offline/build_concept_bank.py
# (concepts/tagging/*). Loaded here, read-only, alongside the embedding matrix.
# ============================================================================================
from dataclasses import dataclass as _dataclass  # noqa: E402 (kept near the class it augments)
import json as _json
import gzip as _gzip

from ..core.types import AnatomyID, Laterality, Polarity  # noqa: E402


@_dataclass(frozen=True)
class ConceptTag:
    anatomy: Optional[AnatomyID]
    laterality: Laterality
    resolved_polarity: Optional[Polarity]   # None means: excluded from grouping (indeterminate temporal, or ambiguous combination)
    temporal_class: str                      # "stationary" | "implies_present" | "implies_absent" | "indeterminate"
    canonical_finding: Optional[str]         # None if unmapped
    group_key: Optional[str]                 # f"{canonical_finding}|{anatomy}|{laterality}", None if canonical_finding is None


def group_key_for(canonical_finding: Optional[str], anatomy: Optional[str], laterality: Laterality) -> Optional[str]:
    if canonical_finding is None:
        return None
    return f"{canonical_finding}|{anatomy or '-'}|{laterality.value}"


def save_tags(path: Path, tags: list[ConceptTag]) -> None:
    """One JSON object per line (jsonl), gzip-compressed. Row i == concept i, positional
    alignment matches the embedding matrix -- never resort/filter this file independently."""
    tmp = path.with_name(path.name + ".tmp")
    with _gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for t in tags:
            fh.write(_json.dumps({
                "anatomy": t.anatomy, "laterality": t.laterality.value,
                "resolved_polarity": t.resolved_polarity.value if t.resolved_polarity else None,
                "temporal_class": t.temporal_class, "canonical_finding": t.canonical_finding,
                "group_key": t.group_key,
            }, separators=(",", ":")) + "\n")
    tmp.replace(path)


def load_tags(path: Path, expected_n: Optional[int] = None) -> list[ConceptTag]:
    if not path.is_file():
        raise BankLoadError(f"concept tag sidecar not found: {path}")
    tags: list[ConceptTag] = []
    with _gzip.open(path, "rt", encoding="utf-8") as fh:
        for row in fh:
            d = _json.loads(row)
            tags.append(ConceptTag(
                anatomy=d["anatomy"], laterality=Laterality(d["laterality"]),
                resolved_polarity=Polarity(d["resolved_polarity"]) if d["resolved_polarity"] else None,
                temporal_class=d["temporal_class"], canonical_finding=d["canonical_finding"],
                group_key=d["group_key"],
            ))
    if expected_n is not None and len(tags) != expected_n:
        raise BankLoadError(f"{path}: {len(tags)} tags, expected {expected_n} (misaligned with the concept bank)")
    return tags
