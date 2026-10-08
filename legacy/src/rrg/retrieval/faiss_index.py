"""N09: exact cosine retrieval over the TRAIN-split corpus with FAISS (IndexFlatIP).

* Corpus = one fp32, L2-normalised CLEAR vector X per training study (see dataset policy). On unit vectors the
  inner product IS the cosine similarity. Exact search (Flat), so results are deterministic; ~220k x 768 is small.
* Only the train split may be indexed, so validation / test images can never be retrieved.
* Query = the same X (fp32, unit-norm). Leakage control: every index entry of the query's OWN PATIENT is removed
  (not just the same image). This is exact: we over-fetch by the number of index rows that belong to that
  patient, filter them out, then keep the top k. Official MIMIC splits are patient-disjoint, so this only ever
  removes anything for train-split queries (e.g. when sanity-checking on train data).
* Files: <name>.faiss, <name>_ids.json (row -> dicom/study/subject/view), <name>_manifest.json (+ optional
  <name>_embeddings.npy). The manifest pins the encoder checkpoint, precision and the ids-file hash so a
  mismatched index fails loudly at load time.

This module deliberately has no torch dependency (numpy + faiss only).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..core.config import Settings
from ..core.errors import (
    EmbeddingContractError,
    IndexBuildError,
    IndexVersionMismatchError,
    SplitLeakageError,
)
from ..core.types import ImageEmbedding

INDEX_TYPE = "IndexFlatIP"                 # backend "faiss"
NUMPY_INDEX_TYPE = "ExactInnerProduct(numpy)"  # backend "numpy": identical exact search, plain matmul
METRIC = "inner_product"
_UNIT_TOL = 1e-3
_NUMPY_CHUNK = 64


def _faiss():
    if sys.platform == "darwin" and "torch" in sys.modules and not os.environ.get("KMP_DUPLICATE_LIB_OK"):
        raise RuntimeError(
            "importing faiss after torch aborts the process on macOS (both pip wheels bundle their own OpenMP runtime). "
            "Set retrieval.backend: numpy (exact, identical results) for local Mac runs; use backend: faiss on Linux/HPC."
        )
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover
        raise ImportError("FAISS is required for retrieval.backend=faiss: pip install '.[faiss]' (or faiss-cpu)") from exc
    return faiss


class _NumpyFlatIP:
    """Exact inner-product search as a matmul over the fp32 corpus embeddings; same contract as faiss.IndexFlatIP.search."""

    def __init__(self, embeddings: np.ndarray):
        self.emb = embeddings
        self.ntotal, self.d = int(embeddings.shape[0]), int(embeddings.shape[1])

    def search(self, q: np.ndarray, k: int):
        sims = np.empty((q.shape[0], k), dtype=np.float32)
        rows = np.empty((q.shape[0], k), dtype=np.int64)
        for s in range(0, q.shape[0], _NUMPY_CHUNK):
            scores = q[s : s + _NUMPY_CHUNK] @ self.emb.T                      # [b, N]
            top = np.argpartition(-scores, k - 1, axis=1)[:, :k] if k < self.ntotal else np.tile(np.arange(self.ntotal), (scores.shape[0], 1))
            top_scores = np.take_along_axis(scores, top, axis=1)
            order = np.argsort(-top_scores, axis=1, kind="stable")
            rows[s : s + _NUMPY_CHUNK] = np.take_along_axis(top, order, axis=1)
            sims[s : s + _NUMPY_CHUNK] = np.take_along_axis(top_scores, order, axis=1)
        return sims, rows


# ----------------------------------------------------------------------------- types
@dataclass(frozen=True)
class RetrievedReport:
    rank: int                 # 1 = most similar
    similarity: float         # cosine
    dicom_id: str
    study_id: int
    subject_id: int
    report_id: str            # MIMIC report file stem, "s<study_id>"


class IdTable:
    """Row i of the FAISS index <-> (dicom_id, study_id, subject_id, view_position)."""

    def __init__(self, dicom_id: Sequence[str], study_id: Sequence[int], subject_id: Sequence[int], view_position: Sequence[Optional[str]]):
        n = len(dicom_id)
        if not (len(study_id) == len(subject_id) == len(view_position) == n):
            raise IndexBuildError("id table columns have different lengths")
        if len(set(dicom_id)) != n:
            raise IndexBuildError("duplicate dicom_id in the id table")
        self.dicom_id = tuple(dicom_id)
        self.study_id = np.asarray(study_id, dtype=np.int64)
        self.subject_id = np.asarray(subject_id, dtype=np.int64)
        self.view_position = tuple(view_position)
        self._subject_counts: Optional[dict[int, int]] = None

    def __len__(self) -> int:
        return len(self.dicom_id)

    @classmethod
    def from_records(cls, records) -> "IdTable":
        return cls([r.dicom_id for r in records], [r.study_id for r in records],
                   [r.subject_id for r in records], [r.view_position for r in records])

    def to_json(self) -> str:
        return json.dumps({"dicom_id": list(self.dicom_id), "study_id": self.study_id.tolist(),
                           "subject_id": self.subject_id.tolist(), "view_position": list(self.view_position)},
                          separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "IdTable":
        d = json.loads(text)
        return cls(d["dicom_id"], d["study_id"], d["subject_id"], d["view_position"])

    def subject_count(self, subject_id: int) -> int:
        """Number of index rows that belong to `subject_id` (exact over-fetch size for patient exclusion)."""
        if self._subject_counts is None:
            u, c = np.unique(self.subject_id, return_counts=True)
            self._subject_counts = dict(zip(u.tolist(), c.tolist()))
        return self._subject_counts.get(int(subject_id), 0)


@dataclass(frozen=True)
class IndexManifest:
    name: str
    corpus_splits: list
    n_vectors: int
    embed_dim: int
    precision: str
    normalised: bool
    metric: str
    index_type: str
    faiss_version: Optional[str]
    clear_checkpoint_name: str
    clear_checkpoint_sha256: Optional[str]
    ids_sha256: str
    embeddings_file: Optional[str]
    view_policy: dict
    selection_stats: dict
    n_failed: int
    failed_dicom_ids: list
    patient_disjoint_verified: bool
    built_limit: Optional[int]          # not None => a smoke-test index built from the first N studies only
    built_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ------------------------------------------------------------------------------ files
def _paths(index_dir: Path, name: str) -> dict[str, Path]:
    return {
        "index": index_dir / f"{name}.faiss",
        "ids": index_dir / f"{name}_ids.json",
        "manifest": index_dir / f"{name}_manifest.json",
        "embeddings": index_dir / f"{name}_embeddings.npy",
    }


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_index(
    index_dir: Path,
    name: str,
    embeddings: np.ndarray,
    table: IdTable,
    *,
    corpus_splits: Sequence[str],
    clear_checkpoint_name: str,
    clear_checkpoint_sha256: Optional[str],
    view_policy: dict,
    selection_stats: dict,
    failed_dicom_ids: Sequence[str] = (),
    save_embeddings: bool = True,
    built_limit: Optional[int] = None,
    backend: str = "faiss",
) -> IndexManifest:
    """Validate, then write ids + manifest + embeddings and, for backend 'faiss', the exact IndexFlatIP."""
    if backend not in {"faiss", "numpy"}:
        raise IndexBuildError(f"unknown backend {backend!r}")
    if backend == "numpy" and not save_embeddings:
        raise IndexBuildError("backend 'numpy' searches the saved embeddings, so retrieval.save_embeddings must be true")
    if set(corpus_splits) - {"train"}:
        raise SplitLeakageError(f"only the train split may be indexed, got {list(corpus_splits)}")
    e = np.ascontiguousarray(embeddings)
    if e.dtype != np.float32 or e.ndim != 2:
        raise IndexBuildError(f"embeddings must be a float32 [N, D] array, got {e.dtype} {e.shape}")
    if e.shape[0] == 0:
        raise IndexBuildError("no embeddings to index")
    if e.shape[0] != len(table):
        raise IndexBuildError(f"{e.shape[0]} embeddings but {len(table)} ids")
    if not np.isfinite(e).all():
        raise IndexBuildError("embeddings contain NaN/Inf")
    dev = float(np.abs(np.linalg.norm(e, axis=1) - 1.0).max())
    if dev > _UNIT_TOL:
        raise IndexBuildError(f"embeddings must be L2-normalised (max |norm-1| = {dev:.2e})")

    faiss = None
    if backend == "faiss":
        faiss = _faiss()
        index = faiss.IndexFlatIP(int(e.shape[1]))
        index.add(e)
        if index.ntotal != e.shape[0]:
            raise IndexBuildError(f"FAISS holds {index.ntotal} vectors, expected {e.shape[0]}")

    index_dir.mkdir(parents=True, exist_ok=True)
    p = _paths(index_dir, name)
    ids_bytes = table.to_json().encode("utf-8")
    if backend == "faiss":
        tmp_index = p["index"].with_name(p["index"].name + ".tmp")
        faiss.write_index(index, str(tmp_index))
        os.replace(tmp_index, p["index"])
    _atomic_write_bytes(p["ids"], ids_bytes)
    if save_embeddings:
        tmp_emb = p["embeddings"].with_name(p["embeddings"].name + ".tmp.npy")
        np.save(tmp_emb, e)
        os.replace(tmp_emb, p["embeddings"])
    manifest = IndexManifest(
        name=name, corpus_splits=list(corpus_splits), n_vectors=int(e.shape[0]), embed_dim=int(e.shape[1]),
        precision="fp32", normalised=True, metric=METRIC,
        index_type=INDEX_TYPE if backend == "faiss" else NUMPY_INDEX_TYPE,
        faiss_version=faiss.__version__ if faiss is not None else None,
        clear_checkpoint_name=clear_checkpoint_name, clear_checkpoint_sha256=clear_checkpoint_sha256,
        ids_sha256=hashlib.sha256(ids_bytes).hexdigest(),
        embeddings_file=p["embeddings"].name if save_embeddings else None,
        view_policy=view_policy, selection_stats=selection_stats, n_failed=len(failed_dicom_ids),
        failed_dicom_ids=list(failed_dicom_ids), patient_disjoint_verified=True, built_limit=built_limit,
    )
    _atomic_write_bytes(p["manifest"], json.dumps(asdict(manifest), indent=2, sort_keys=True).encode("utf-8"))
    return manifest


# ---------------------------------------------------------------------------- retriever
class FaissRetriever:
    def __init__(self, index: Any, table: IdTable, manifest: IndexManifest, *, k: int,
                 similarity_floor: Optional[float], exclude_same_patient: bool):
        self.index, self.table, self.manifest = index, table, manifest
        self.k, self.similarity_floor, self.exclude_same_patient = k, similarity_floor, exclude_same_patient

    @classmethod
    def load(
        cls,
        settings: Settings,
        *,
        expected_checkpoint_sha256: Optional[str] = None,
        name: Optional[str] = None,
        index_dir: Optional[Path] = None,
        exclude_same_patient: Optional[bool] = None,
    ) -> "FaissRetriever":
        rc = settings.retrieval
        name = name or rc.index_name
        p = _paths(index_dir or settings.paths.index_dir, name)
        backend = rc.backend
        needed = ("index" if backend == "faiss" else "embeddings", "ids", "manifest")
        for key in needed:
            if not p[key].is_file():
                raise IndexVersionMismatchError(
                    f"missing {p[key]}: build the index first (python scripts/build_faiss_index.py) "
                    f"with the same retrieval.backend ({backend})"
                )
        m = IndexManifest(**json.loads(p["manifest"].read_text(encoding="utf-8")))
        ids_bytes = p["ids"].read_bytes()

        if hashlib.sha256(ids_bytes).hexdigest() != m.ids_sha256:
            raise IndexVersionMismatchError(f"{p['ids'].name} does not match the manifest hash (index and ids out of sync)")
        if set(m.corpus_splits) - {"train"}:
            raise SplitLeakageError(f"index was built from non-train splits {m.corpus_splits}")
        if m.embed_dim != settings.clear.embed_dim:
            raise IndexVersionMismatchError(f"index dim {m.embed_dim} != clear.embed_dim {settings.clear.embed_dim}")
        if m.precision != settings.clear.precision or m.precision != "fp32":
            raise IndexVersionMismatchError(f"index precision {m.precision!r} / config {settings.clear.precision!r}: fp32 only")
        want_type = INDEX_TYPE if backend == "faiss" else NUMPY_INDEX_TYPE
        if m.metric != METRIC or m.index_type != want_type:
            raise IndexVersionMismatchError(
                f"index was built as {m.index_type}/{m.metric} but retrieval.backend={backend} needs {want_type}; rebuild it"
            )
        if (expected_checkpoint_sha256 and m.clear_checkpoint_sha256
                and expected_checkpoint_sha256 != m.clear_checkpoint_sha256):
            raise IndexVersionMismatchError(
                "the index was built with a different CLEAR checkpoint than the encoder now in use "
                f"({m.clear_checkpoint_sha256[:12]}... vs {expected_checkpoint_sha256[:12]}...)"
            )

        if backend == "faiss":
            index = _faiss().read_index(str(p["index"]))
        else:
            index = _NumpyFlatIP(np.load(p["embeddings"], mmap_mode="r"))
        table = IdTable.from_json(ids_bytes.decode("utf-8"))
        if not (index.ntotal == len(table) == m.n_vectors) or index.d != m.embed_dim:
            raise IndexVersionMismatchError(
                f"index has {index.ntotal} vectors / dim {index.d}, ids {len(table)}, manifest {m.n_vectors} / {m.embed_dim}"
            )
        return cls(index, table, m, k=rc.k, similarity_floor=rc.similarity_floor,
                   exclude_same_patient=rc.exclude_same_patient if exclude_same_patient is None else exclude_same_patient)

    def __len__(self) -> int:
        return len(self.table)

    # ---- search
    def search(
        self,
        vectors: np.ndarray,
        k: Optional[int] = None,
        exclude_subject_ids: Optional[Sequence[Optional[int]]] = None,
        similarity_floor: Optional[float] = None,
    ) -> list[list[RetrievedReport]]:
        """[B, D] fp32 unit-norm queries -> B lists of up to k neighbours, best first.
        `exclude_subject_ids[q]` is the patient whose entries are removed for query q (applied only when
        exclude_same_patient is on). Ties are ordered by row index, so the output is deterministic."""
        k = self.k if k is None else k
        floor = self.similarity_floor if similarity_floor is None else similarity_floor
        q = np.asarray(vectors)
        if q.ndim == 1:
            q = q[None, :]
        if q.dtype != np.float32:
            raise EmbeddingContractError(f"queries must be float32 (fp32-only branch), got {q.dtype}")
        if q.ndim != 2 or q.shape[1] != self.manifest.embed_dim:
            raise EmbeddingContractError(f"queries must be [B, {self.manifest.embed_dim}], got {q.shape}")
        if not np.isfinite(q).all():
            raise EmbeddingContractError("query contains NaN/Inf")
        if np.abs(np.linalg.norm(q, axis=1) - 1.0).max() > _UNIT_TOL:
            raise EmbeddingContractError("queries must be L2-normalised")
        if k <= 0:
            raise EmbeddingContractError("k must be positive")
        q = np.ascontiguousarray(q)
        if exclude_subject_ids is not None and len(exclude_subject_ids) != q.shape[0]:
            raise EmbeddingContractError("exclude_subject_ids must have one entry per query")

        excl = [None] * q.shape[0]
        if self.exclude_same_patient and exclude_subject_ids is not None:
            excl = [None if s is None else int(s) for s in exclude_subject_ids]
        extra = max((self.table.subject_count(s) for s in excl if s is not None), default=0)
        k_search = min(len(self.table), k + extra)          # exact: at most `extra` rows can be removed per query
        sims, rows = self.index.search(q, k_search)

        out: list[list[RetrievedReport]] = []
        for qi in range(q.shape[0]):
            r, s = rows[qi], sims[qi]
            keep = r >= 0
            r, s = r[keep], s[keep]
            if excl[qi] is not None:
                keep = self.table.subject_id[r] != excl[qi]
                r, s = r[keep], s[keep]
            order = np.lexsort((r, -s))[:k]
            hits = []
            for rank, j in enumerate(order, start=1):
                sim = float(s[j])
                if floor is not None and sim < floor:
                    break
                row = int(r[j])
                sid = int(self.table.study_id[row])
                hits.append(RetrievedReport(rank=rank, similarity=sim, dicom_id=self.table.dicom_id[row], study_id=sid,
                                            subject_id=int(self.table.subject_id[row]), report_id=f"s{sid}"))
            out.append(hits)
        return out

    def search_embedding(self, embedding: ImageEmbedding, subject_id: Optional[int] = None) -> list[RetrievedReport]:
        """One query image. Pass the query's subject_id so its own patient is excluded."""
        if embedding.precision != self.manifest.precision:
            raise IndexVersionMismatchError(f"query precision {embedding.precision!r} != index precision {self.manifest.precision!r}")
        if embedding.checkpoint_sha256 and self.manifest.clear_checkpoint_sha256 \
                and embedding.checkpoint_sha256 != self.manifest.clear_checkpoint_sha256:
            raise IndexVersionMismatchError("query embedding was produced by a different CLEAR checkpoint than the index")
        return self.search(embedding.vector[None, :], exclude_subject_ids=[subject_id])[0]
