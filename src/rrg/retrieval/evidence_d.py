"""N18 `Evidence D: Retrieved facts` (pipeline.md §4.8). Aggregates the k retrieved neighbours'
RadGraph-extracted report facts (reports/report_facts.py) into similarity-weighted, asymmetric
present/absent support per `(finding, anatomy, laterality)` key -- resolving pipeline.md's
AMBIG-4 as similarity-weighted voting, the option it recommended.

Present and absent support are kept STRICTLY SEPARATE, never netted here -- same discipline as
Evidence B's grouping (concepts/grouping.py); reconciling a present/absent conflict across
sources is N25's job, not this aggregation's.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..concepts.tagging.finding_vocab import FindingVocabulary
from ..core.config import Settings
from ..core.types import AnatomyID, Laterality, Polarity
from ..datasets.reports import ReportStore
from ..fusion.rules.radlex_client import RadLexClient
from ..reports.report_facts import get_or_extract_facts
from .faiss_index import RetrievedReport
from .radgraph_parser import RadGraphParser


@dataclass(frozen=True)
class EvidenceDItem:
    finding: str
    anatomy: Optional[AnatomyID]
    laterality: Laterality
    present_support: float     # similarity-weighted sum over neighbours asserting PRESENT
    present_count: int         # number of distinct neighbours asserting PRESENT (not sentence count)
    absent_support: float      # similarity-weighted, ASYMMETRICALLY DISCOUNTED (evidence_d.absent_weight)
    absent_count: int
    source_report_ids: list[str]


def build_evidence_d(
    neighbours: list[RetrievedReport],
    reports: ReportStore,
    radgraph: RadGraphParser,
    radlex: RadLexClient,
    vocab: FindingVocabulary,
    settings: Settings,
) -> list[EvidenceDItem]:
    groups: dict[tuple[str, Optional[str], str], dict] = {}
    for nb in neighbours:
        text = reports.get(nb.subject_id, nb.study_id)
        facts = get_or_extract_facts(nb.study_id, text, radgraph, radlex, vocab, settings)
        seen_this_neighbour: set[tuple] = set()
        for f in facts:
            if f.polarity == Polarity.UNCERTAIN:
                continue
            key = (f.canonical_finding, f.anatomy, f.laterality.value)
            side_key = (key, f.polarity)
            if side_key in seen_this_neighbour:
                # a neighbour supports a given (finding,anatomy,laterality,side) at most ONCE,
                # regardless of how many of its sentences assert it -- support_count counts
                # neighbours, not sentences (pipeline.md N18: "how many of the k neighbours asserted it")
                continue
            seen_this_neighbour.add(side_key)
            g = groups.setdefault(key, {
                "present_support": 0.0, "present_count": 0, "present_reports": [],
                "absent_support": 0.0, "absent_count": 0, "absent_reports": [],
            })
            if f.polarity == Polarity.PRESENT:
                g["present_support"] += nb.similarity
                g["present_count"] += 1
                g["present_reports"].append(nb.report_id)
            else:  # ABSENT
                g["absent_support"] += settings.evidence_d.absent_weight * nb.similarity
                g["absent_count"] += 1
                g["absent_reports"].append(nb.report_id)

    out: list[EvidenceDItem] = []
    for (finding, anatomy, laterality), g in groups.items():
        if max(g["present_count"], g["absent_count"]) < settings.evidence_d.min_support:
            continue
        out.append(EvidenceDItem(
            finding=finding, anatomy=anatomy, laterality=Laterality(laterality),
            present_support=g["present_support"], present_count=g["present_count"],
            absent_support=g["absent_support"], absent_count=g["absent_count"],
            source_report_ids=sorted(set(g["present_reports"] + g["absent_reports"])),
        ))
    return out
