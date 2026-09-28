"""N13/N17 -> N20 `Evidence C` at inference time (pipeline.md §4.7). Gathers the raw (never
grouped/refined -- pipeline.md AMBIG-3, fact 2 in §2.3) 368k scores down to the CBM's concepts
and runs the trained linear head. Output is the RAW sigmoid probability; fusion.calibrate is
what turns it into an honest, calibrated P(present) (see docs from this session's fusion design).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import CBMHeadInfo
from ..concepts.bank import ConceptBank
from ..concepts.vocabulary import clean_text
from ..core.errors import CBMIndexError


@dataclass(frozen=True)
class CBMContributingConcept:
    concept_id: int
    text: str
    contribution: float          # w_p,i * x_i -- pipeline.md §4.7's required interpretability payoff


@dataclass(frozen=True)
class CBMPrediction:
    pathology: str
    raw_logit: float
    raw_probability: float       # sigmoid(logit) -- NOT yet calibrated
    top_contributing_concepts: list[CBMContributingConcept]


class CBMPredictor:
    def __init__(self, head, info: CBMHeadInfo, concept_ids: list[int], bank: ConceptBank, top_n: int = 5):
        if info.n_concepts != len(concept_ids):
            raise CBMIndexError(f"CBM head expects {info.n_concepts} concepts, got {len(concept_ids)} concept_ids")
        self.head = head
        self.info = info
        self.concept_ids = concept_ids
        self.bank = bank
        self.top_n = top_n

    def predict(self, raw_scores: np.ndarray) -> list[CBMPrediction]:
        import torch

        needed = max(self.concept_ids, default=-1) + 1
        if raw_scores.shape[0] < needed:
            raise CBMIndexError(f"raw_scores has {raw_scores.shape[0]} entries, need at least {needed}")
        x = raw_scores[self.concept_ids].astype(np.float32, copy=False)
        with torch.inference_mode():
            xt = torch.from_numpy(x)
            logits = self.head(xt)
            probs = torch.sigmoid(logits)
            weight = self.head.weight.detach()   # [n_labels, n_concepts]

        out = []
        for j, label in enumerate(self.info.label_space):
            contrib = (weight[j] * xt).numpy()
            k = min(self.top_n, contrib.shape[0])
            order = np.argsort(-np.abs(contrib))[:k]
            top = [
                CBMContributingConcept(
                    concept_id=self.concept_ids[i], text=clean_text(self.bank.text(self.concept_ids[i])),
                    contribution=float(contrib[i]),
                )
                for i in order
            ]
            out.append(CBMPrediction(
                pathology=label, raw_logit=float(logits[j]), raw_probability=float(probs[j]),
                top_contributing_concepts=top,
            ))
        return out
