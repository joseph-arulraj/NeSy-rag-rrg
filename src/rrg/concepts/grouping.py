"""N12/N16/N19 (rev 3) -> Evidence B (N21). pipeline.md 4.6 steps 9-12.

Groups the 368k raw concept scores (N07's output, NEVER rescored here) by
(canonical_finding, anatomy, laterality), keeps present-polarity and absent-polarity evidence
strictly separate within each group (never netted -- that's N25's job, pipeline.md GAP-7), and
ranks groups by max(present_score, absent_score) so a confident negative finding is as visible
as a confident positive one. Takes NO Evidence A input, ever -- see pipeline.md's Revision Log
rev 3 and architecture.md IMPL-20.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..core.config import GroupingConfig
from ..core.errors import BankDimMismatchError
from ..core.types import AnatomyID, ConceptID, Laterality, Polarity
from .bank import ConceptBank
from .vocabulary import clean_text

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class ContributingConcept:
    concept_id: ConceptID
    text: str
    raw_score: float
    resolved_polarity: Polarity
    temporal_class: str


@dataclass(frozen=True)
class FindingGroup:
    finding: str
    anatomy: "AnatomyID | None"
    laterality: Laterality
    present_score: float
    present_count: int
    absent_score: float
    absent_count: int
    top_contributing_concepts: list[ContributingConcept]


def group_and_rank(raw_scores: np.ndarray, bank: ConceptBank, cfg: GroupingConfig) -> list[FindingGroup]:
    if bank.tags is None:
        raise ValueError("bank has no tags attached -- call ConceptBank.with_tags(...) first (see offline/build_concept_bank.py)")
    if raw_scores.shape[0] != len(bank):
        raise BankDimMismatchError(f"scores shape {raw_scores.shape} does not match bank size {len(bank)}")

    # group_key -> {present: [(idx, score)], absent: [(idx, score)]}
    buckets: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for i, tag in enumerate(bank.tags):
        if tag.group_key is None or tag.resolved_polarity is None:
            continue
        b = buckets.setdefault(tag.group_key, {"present": [], "absent": []})
        side = "present" if tag.resolved_polarity == Polarity.PRESENT else "absent" if tag.resolved_polarity == Polarity.ABSENT else None
        if side is None:   # UNCERTAIN concepts contribute to neither side's score, but stay tag-visible elsewhere
            continue
        b[side].append((i, float(raw_scores[i])))

    threshold = cfg.present_absent_count_threshold
    groups: list[FindingGroup] = []
    for key, sides in buckets.items():
        finding, anatomy_str, laterality_str = key.split("|", 2)
        anatomy = None if anatomy_str == "-" else anatomy_str
        laterality = Laterality(laterality_str)

        present_items = sorted(sides["present"], key=lambda x: -x[1])
        absent_items = sorted(sides["absent"], key=lambda x: -x[1])
        present_score = present_items[0][1] if present_items else float("nan")
        absent_score = absent_items[0][1] if absent_items else float("nan")
        present_count = sum(1 for _, s in present_items if s >= threshold)
        absent_count = sum(1 for _, s in absent_items if s >= threshold)

        top_n = cfg.contributing_concepts_per_group
        contributing = []
        for idx, score in (present_items[:top_n] + absent_items[:top_n]):
            tag = bank.tags[idx]
            contributing.append(ContributingConcept(
                concept_id=idx, text=clean_text(bank.text(idx)), raw_score=score,
                resolved_polarity=tag.resolved_polarity, temporal_class=tag.temporal_class,
            ))
        contributing.sort(key=lambda c: -c.raw_score)
        contributing = contributing[:top_n]

        groups.append(FindingGroup(
            finding=finding, anatomy=anatomy, laterality=laterality,
            present_score=present_score, present_count=present_count,
            absent_score=absent_score, absent_count=absent_count,
            top_contributing_concepts=contributing,
        ))

    def rank_key(g: FindingGroup) -> float:
        vals = [v for v in (g.present_score, g.absent_score) if v == v]  # drop NaN
        return max(vals) if vals else float("-inf")

    groups.sort(key=lambda g: (-rank_key(g), g.finding, g.anatomy or "", g.laterality.value))
    return groups[: cfg.top_k_groups]
