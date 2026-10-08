"""N24 `Initial belief graph` (pipeline.md §5.3): assembles fuse.py's `FusedFinding` list into
the shared `BeliefGraph` contract (core/types.py). Findings N23 abstained on (band == "abstain")
are excluded from `findings` but kept in `rejected` for audit (pipeline.md §5.3's admission
threshold requirement, INV-1/INV-2) -- never silently dropped.
"""
from __future__ import annotations

from ..core.types import BeliefGraph, EvidenceRef, Finding, Laterality, StudyMeta
from .fuse import FusedFinding


def _finding_id(finding: str, anatomy: str | None, laterality: str, index: int) -> str:
    return f"{finding}|{anatomy or '-'}|{laterality}|{index}"


def _support_refs(fused: FusedFinding) -> list[EvidenceRef]:
    refs = []
    for item in fused.support:
        raw = item.get("raw_score", item.get("present_score", item.get("present_support", 0.0)))
        norm = item.get("calibrated", fused.confidence)
        refs.append(EvidenceRef(source=item["source"], raw_score=float(raw), norm_score=float(norm), provenance=item))
    return refs


def build_initial_graph(fused: list[FusedFinding], study_meta: StudyMeta) -> BeliefGraph:
    findings: list[Finding] = []
    rejected: list[Finding] = []
    for i, f in enumerate(fused):
        node = Finding(
            finding_id=_finding_id(f.finding, f.anatomy, f.laterality, i),
            label=f.finding,
            anatomy=f.anatomy,
            laterality=Laterality(f.laterality),
            polarity=f.polarity,
            confidence=f.confidence,
            support=_support_refs(f),
        )
        (findings if f.admitted else rejected).append(node)

    return BeliefGraph(
        findings=findings, relations=[], study_meta=study_meta, graph_version="initial",
        audit=[], rejected=rejected, suppressed=[],
    )
