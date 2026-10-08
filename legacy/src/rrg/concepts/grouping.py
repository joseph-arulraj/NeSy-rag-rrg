"""N12/N16/N19 (rev 3) -> Evidence B (N21). pipeline.md 4.6 steps 9-12.

Groups the 368k raw concept scores (N07's output, NEVER rescored here) by
(canonical_finding, anatomy, laterality), keeps present-polarity and absent-polarity evidence
strictly separate within each group (never netted -- that's N25's job, pipeline.md GAP-7), and
ranks groups by max(present_score, absent_score) so a confident negative finding is as visible
as a confident positive one. Takes NO Evidence A input, ever -- see pipeline.md's Revision Log
rev 3 and architecture.md IMPL-20.

Performance note: which concepts belong to which group is a static property of the concept
bank's tags -- it does not depend on any image. `GroupingIndex` precomputes that structure
ONCE (build_grouping_index, cached on the ConceptBank); group_and_rank then does only
vectorised numpy work per image (no Python-level loop over the 368k concepts), which is what
makes scoring thousands of images practical.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.config import GroupingConfig
from ..core.errors import BankDimMismatchError
from ..core.types import AnatomyID, ConceptID, Laterality, Polarity
from .bank import ConceptBank
from .vocabulary import clean_text


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


@dataclass(frozen=True)
class GroupingIndex:
    """Precomputed, image-independent structure derived from ConceptBank.tags. Build once per
    (bank, tags) pair via build_grouping_index; reuse across every image scored against that bank."""

    n_groups: int
    group_finding: tuple[str, ...]          # group_idx -> finding
    group_anatomy: tuple["AnatomyID | None", ...]
    group_laterality: tuple[Laterality, ...]
    concept_group_idx: np.ndarray            # [N] int32; -1 = not in any group
    concept_side: np.ndarray                 # [N] int8; +1 present, -1 absent, 0 neither (incl. uncertain/excluded)
    group_present_members: tuple[np.ndarray, ...]   # group_idx -> concept indices on the present side
    group_absent_members: tuple[np.ndarray, ...]     # group_idx -> concept indices on the absent side


def build_grouping_index(bank: ConceptBank) -> GroupingIndex:
    if bank.tags is None:
        raise ValueError("bank has no tags attached -- call ConceptBank.with_tags(...) first (see offline/build_concept_bank.py)")

    n = len(bank.tags)
    group_key_to_idx: dict[str, int] = {}
    group_finding: list[str] = []
    group_anatomy: list["AnatomyID | None"] = []
    group_laterality: list[Laterality] = []
    concept_group_idx = np.full(n, -1, dtype=np.int32)
    concept_side = np.zeros(n, dtype=np.int8)

    for i, tag in enumerate(bank.tags):
        if tag.group_key is None or tag.resolved_polarity is None:
            continue
        if tag.resolved_polarity == Polarity.PRESENT:
            side = 1
        elif tag.resolved_polarity == Polarity.ABSENT:
            side = -1
        else:  # UNCERTAIN: stays out of both sides, per the original design
            continue
        gidx = group_key_to_idx.get(tag.group_key)
        if gidx is None:
            gidx = len(group_finding)
            group_key_to_idx[tag.group_key] = gidx
            finding, anatomy_str, laterality_str = tag.group_key.split("|", 2)
            group_finding.append(finding)
            group_anatomy.append(None if anatomy_str == "-" else anatomy_str)
            group_laterality.append(Laterality(laterality_str))
        concept_group_idx[i] = gidx
        concept_side[i] = side

    n_groups = len(group_finding)
    present_members: list[list[int]] = [[] for _ in range(n_groups)]
    absent_members: list[list[int]] = [[] for _ in range(n_groups)]
    present_idx = np.flatnonzero(concept_side == 1)
    absent_idx = np.flatnonzero(concept_side == -1)
    for i in present_idx:
        present_members[concept_group_idx[i]].append(int(i))
    for i in absent_idx:
        absent_members[concept_group_idx[i]].append(int(i))

    return GroupingIndex(
        n_groups=n_groups,
        group_finding=tuple(group_finding),
        group_anatomy=tuple(group_anatomy),
        group_laterality=tuple(group_laterality),
        concept_group_idx=concept_group_idx,
        concept_side=concept_side,
        group_present_members=tuple(np.asarray(m, dtype=np.int64) for m in present_members),
        group_absent_members=tuple(np.asarray(m, dtype=np.int64) for m in absent_members),
    )


def _top_n(bank: ConceptBank, raw_scores: np.ndarray, members: np.ndarray, top_n: int) -> list[ContributingConcept]:
    if members.size == 0:
        return []
    scores = raw_scores[members]
    k = min(top_n, members.size)
    order = np.argpartition(-scores, k - 1)[:k]
    order = order[np.argsort(-scores[order])]
    out = []
    for j in order:
        idx = int(members[j])
        tag = bank.tags[idx]
        out.append(ContributingConcept(
            concept_id=idx, text=clean_text(bank.text(idx)), raw_score=float(scores[j]),
            resolved_polarity=tag.resolved_polarity, temporal_class=tag.temporal_class,
        ))
    return out


def group_and_rank(
    raw_scores: np.ndarray,
    bank: ConceptBank,
    cfg: GroupingConfig,
    index: "GroupingIndex | None" = None,
) -> list[FindingGroup]:
    """Pass a precomputed `index` (build_grouping_index(bank), or bank.grouping_index()) when
    scoring many images against the same bank -- it is the same for every image and rebuilding
    it per call is the one thing in this function that would NOT be O(368k) numpy ops."""
    if raw_scores.shape[0] != len(bank):
        raise BankDimMismatchError(f"scores shape {raw_scores.shape} does not match bank size {len(bank)}")
    idx = index if index is not None else bank.grouping_index()
    if idx.n_groups == 0:
        return []

    threshold = cfg.present_absent_count_threshold
    present_score = np.full(idx.n_groups, np.nan, dtype=np.float64)
    absent_score = np.full(idx.n_groups, np.nan, dtype=np.float64)
    present_count = np.zeros(idx.n_groups, dtype=np.int64)
    absent_count = np.zeros(idx.n_groups, dtype=np.int64)

    present_mask = idx.concept_side == 1
    absent_mask = idx.concept_side == -1
    if present_mask.any():
        g = idx.concept_group_idx[present_mask]
        s = raw_scores[present_mask]
        pmax = np.full(idx.n_groups, -np.inf, dtype=np.float64)
        np.maximum.at(pmax, g, s)
        has_present = np.zeros(idx.n_groups, dtype=bool)
        has_present[np.unique(g)] = True
        present_score = np.where(has_present, pmax, np.nan)
        present_count = np.bincount(g[s >= threshold], minlength=idx.n_groups)
    if absent_mask.any():
        g = idx.concept_group_idx[absent_mask]
        s = raw_scores[absent_mask]
        amax = np.full(idx.n_groups, -np.inf, dtype=np.float64)
        np.maximum.at(amax, g, s)
        has_absent = np.zeros(idx.n_groups, dtype=bool)
        has_absent[np.unique(g)] = True
        absent_score = np.where(has_absent, amax, np.nan)
        absent_count = np.bincount(g[s >= threshold], minlength=idx.n_groups)

    rank_val = np.fmax(present_score, absent_score)   # fmax: NaN loses to a real number; NaN,NaN -> NaN
    order = np.argsort(-np.nan_to_num(rank_val, nan=-np.inf), kind="stable")
    top = order[: cfg.top_k_groups]

    top_n = cfg.contributing_concepts_per_group
    groups = []
    for gidx in top:
        gidx = int(gidx)
        present_c = _top_n(bank, raw_scores, idx.group_present_members[gidx], top_n)
        absent_c = _top_n(bank, raw_scores, idx.group_absent_members[gidx], top_n)
        contributing = sorted(present_c + absent_c, key=lambda c: -c.raw_score)[:top_n]
        groups.append(FindingGroup(
            finding=idx.group_finding[gidx], anatomy=idx.group_anatomy[gidx], laterality=idx.group_laterality[gidx],
            present_score=float(present_score[gidx]), present_count=int(present_count[gidx]),
            absent_score=float(absent_score[gidx]), absent_count=int(absent_count[gidx]),
            top_contributing_concepts=contributing,
        ))
    return groups
