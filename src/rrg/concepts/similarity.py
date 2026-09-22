"""N07: image feature X . concept bank -> dense 368,294-d concept score vector (N10).

Both sides are unit-norm, so a score is a cosine similarity in [-1, 1]. It is NOT a probability.
The output stays dense and positionally aligned to ConceptID (the 68-concept CBM branch gathers
arbitrary positions from it); it is never sparsified here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np

from ..core.errors import BankDimMismatchError, EmbeddingContractError, NonFiniteScoreError
from ..core.types import ConceptHit, ImageEmbedding
from .bank import ConceptBank

if TYPE_CHECKING:
    import torch

_UNIT_TOL = 1e-3


def score_concepts_batch(
    vectors: Union[np.ndarray, "torch.Tensor"],
    bank: ConceptBank,
    device: Optional["torch.device"] = None,
    batch_size: int = 64,
) -> np.ndarray:
    """[B, D] unit-norm image features -> [B, N] float32 scores (CPU numpy)."""
    import torch

    x = torch.as_tensor(vectors).to(torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2:
        raise EmbeddingContractError(f"image features must be [B, D], got shape {tuple(x.shape)}")
    if x.shape[1] != bank.embed_dim:
        raise BankDimMismatchError(f"image feature dim {x.shape[1]} != concept embedding dim {bank.embed_dim}")
    if not torch.isfinite(x).all():
        raise NonFiniteScoreError("image feature contains NaN/Inf")
    norms = x.norm(dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=_UNIT_TOL):
        raise EmbeddingContractError(
            f"image features must be L2-normalised (norms in [{float(norms.min()):.4f}, {float(norms.max()):.4f}])"
        )

    device = device or torch.device("cpu")
    emb = bank.device_embeddings(device)
    chunks = []
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            scores = x[start : start + batch_size].to(device) @ emb.T
            if not torch.isfinite(scores).all():
                raise NonFiniteScoreError("concept scores contain NaN/Inf")
            chunks.append(scores.cpu())
    return torch.cat(chunks, dim=0).numpy()


def score_concepts(
    embedding: ImageEmbedding,
    bank: ConceptBank,
    device: Optional["torch.device"] = None,
) -> np.ndarray:
    """One image -> float32[368294], aligned to ConceptID."""
    return score_concepts_batch(embedding.vector[None, :], bank, device)[0]


def top_k(scores: np.ndarray, bank: ConceptBank, k: int) -> list[ConceptHit]:
    """Highest-scoring concepts, best first (ties broken by lower ConceptID for determinism)."""
    if scores.ndim != 1 or scores.shape[0] != len(bank):
        raise BankDimMismatchError(f"scores shape {scores.shape} does not match bank size {len(bank)}")
    k = min(k, scores.shape[0])
    part = np.argpartition(-scores, k - 1)[:k]
    order = sorted(part.tolist(), key=lambda i: (-float(scores[i]), i))
    return [ConceptHit(concept_id=i, text=bank.text(i), score=float(scores[i])) for i in order]


def gather(scores: np.ndarray, concept_ids: Sequence[int]) -> np.ndarray:
    """Positional gather, e.g. the 68 CBM concepts later."""
    return scores[np.asarray(concept_ids, dtype=np.int64)]
