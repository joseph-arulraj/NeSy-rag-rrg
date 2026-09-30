"""N23 `Score calibration + fusion` (pipeline.md §5.2), applying the log-odds design from this
session's fusion discussion:

  prior(finding)  = logit(calibrated_C)   if C has coverage for this finding (its 14 labels)
                   = logit(calibrated_B)  otherwise (B covers the full canonical vocabulary)
                   = logit(base_rate)     if neither has any data at all

  posterior_logit = prior
                   + w_b_delta_when_c_prior * asymmetric(B.present, B.absent)   [only when C was the prior]
                   + w_d_delta             * asymmetric(D.present, D.absent)    [always, when D has support]

  confidence = sigmoid(posterior_logit)

B is never summed as a second independent prior alongside C for the same finding -- C's 67
input concepts are a strict subset of B's own raw scores (pipeline.md §10.2's noisy-OR
rejection), so that would double-count the same visual signal. D is always a delta, never a
prior, because it is evidence about OTHER patients' images (corroborative), not a direct read on
this one -- see FusionConfig's docstring in core/config.py for the full reasoning.

Banding (accept/reject/uncertain/abstain) mirrors the old project's per-finding fitted
tau_hi/tau_lo design: `tau_hi >= 1.0` means this finding could not be calibrated to an
acceptable precision on the validation data at all -- never assert it (pipeline.md §5.4
AMBIG-7's "abstain" case). A REJECT band is a confident ABSENCE and is still admitted to the
graph as an explicit negative finding (pipeline.md §6.1: negative findings belong in the LLM
prompt) -- admission is `band != "abstain"`, not a confidence-magnitude cutoff, since a
confident absence has LOW `confidence` (= P(present)) by construction.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from ..cbm.infer import CBMPrediction
from ..concepts.grouping import ContributingConcept, FindingGroup
from ..core.config import Settings
from .calibrate import CalibrationArtifacts
from ..retrieval.evidence_d import EvidenceDItem


def _merge_b_groups_by_laterality(groups: list[FindingGroup]) -> dict[tuple, FindingGroup]:
    """Collapses B's `(finding, anatomy, laterality)` groups down to `(finding, laterality)`,
    merging across anatomy. Different anatomy resolutions of the same finding+laterality are
    near-synonymous PHRASING variants (e.g. some concepts describing a rib fracture resolve a
    specific RadLex anatomy term, others don't) -- not genuinely distinct clinical facts.
    Confirmed as a real, repeated problem on actual pipeline output, not hypothetical:
    pleural_effusion, edema, fracture, lung_opacity, support_devices, and no_finding all
    fragmented this way, producing multiple separate belief-graph nodes (and repetitive report
    sentences) for what was really one finding. This also removes an inconsistency: C has no
    anatomy dimension at all, so B/D fragmenting by anatomy while C doesn't was already an
    asymmetry, not a deliberate design choice.

    present_score/absent_score: MAX across the merged variants -- the same "max across group
    members" semantics concepts/grouping.py already uses *within* one group, now applied across
    the anatomy-fragmented groups too, so this doesn't inflate confidence by summing what are
    really alternative phrasings of the same evidence.
    present_count/absent_count: summed -- still meaningful as "how many concepts support this".
    top_contributing_concepts: re-ranked top-N over the union of all merged variants' concepts,
    deduplicated by concept_id (the same concept can appear in more than one anatomy-fragmented
    variant if its own anatomy resolution was itself ambiguous).
    """
    buckets: dict[tuple, list[FindingGroup]] = {}
    for g in groups:
        buckets.setdefault((g.finding, g.laterality.value), []).append(g)

    merged: dict[tuple, FindingGroup] = {}
    for (finding, laterality_val), members in buckets.items():
        present_scores = [m.present_score for m in members if np.isfinite(m.present_score)]
        absent_scores = [m.absent_score for m in members if np.isfinite(m.absent_score)]
        best_by_concept: dict[int, ContributingConcept] = {}
        for m in members:
            for c in m.top_contributing_concepts:
                prev = best_by_concept.get(c.concept_id)
                if prev is None or abs(c.raw_score) > abs(prev.raw_score):
                    best_by_concept[c.concept_id] = c
        top_n = sorted(best_by_concept.values(), key=lambda c: -abs(c.raw_score))[:5]
        merged[(finding, laterality_val)] = FindingGroup(
            finding=finding, anatomy=None, laterality=members[0].laterality,
            present_score=max(present_scores) if present_scores else float("nan"),
            present_count=sum(m.present_count for m in members),
            absent_score=max(absent_scores) if absent_scores else float("nan"),
            absent_count=sum(m.absent_count for m in members),
            top_contributing_concepts=top_n,
        )
    return merged


def _merge_d_items_by_laterality(items: list[EvidenceDItem]) -> dict[tuple, EvidenceDItem]:
    """Same merge, for D -- see _merge_b_groups_by_laterality's docstring. present_support/
    absent_support are also MAXed, not summed, for the same reason: report_facts.py resolves
    anatomy per sentence, so the same underlying neighbour evidence can fragment across anatomy
    here too, and summing it would double-count one neighbour's support across the merged
    buckets rather than treating it as one piece of evidence."""
    buckets: dict[tuple, list[EvidenceDItem]] = {}
    for d in items:
        buckets.setdefault((d.finding, d.laterality.value), []).append(d)

    merged: dict[tuple, EvidenceDItem] = {}
    for (finding, laterality_val), members in buckets.items():
        merged[(finding, laterality_val)] = EvidenceDItem(
            finding=finding, anatomy=None, laterality=members[0].laterality,
            present_support=max((m.present_support for m in members), default=0.0),
            present_count=max((m.present_count for m in members), default=0),
            absent_support=max((m.absent_support for m in members), default=0.0),
            absent_count=max((m.absent_count for m in members), default=0),
            source_report_ids=sorted({rid for m in members for rid in m.source_report_ids}),
        )
    return merged


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True)
class FusedFinding:
    finding: str
    anatomy: Optional[str]
    laterality: str
    confidence: float               # sigmoid(posterior_logit) == calibrated P(present)
    polarity: str                   # "present" | "absent" | "uncertain" -- derived from band, see module docstring
    prior_source: str                # "B" | "C" | "base_rate"
    band: str                        # "accept" | "reject" | "uncertain" | "abstain"
    admitted: bool                   # band != "abstain"
    n_supporting_sources: int
    support: list[dict]              # per-source raw contributions, for the belief graph's EvidenceRef list


_BAND_TO_POLARITY = {"accept": "present", "reject": "absent", "uncertain": "uncertain"}


def fuse_study(
    b_groups: list[FindingGroup],
    c_preds: list[CBMPrediction],
    d_items: list[EvidenceDItem],
    calib: CalibrationArtifacts,
    settings: Settings,
) -> list[FusedFinding]:
    fc = settings.fusion
    c_by_finding = {p.pathology: p for p in c_preds}

    # Merged to (finding, laterality) -- see _merge_b_groups_by_laterality's docstring for why
    # anatomy is collapsed here rather than kept as part of the node key.
    by_key = _merge_b_groups_by_laterality(b_groups)
    by_key_d = _merge_d_items_by_laterality(d_items)

    all_keys = set(by_key) | set(by_key_d)
    for finding in c_by_finding:
        if not any(k[0] == finding for k in all_keys):
            # C covers this finding but neither B nor D produced a laterality variant -- still
            # needs a node; C carries no anatomy/laterality attribute at all (pipeline.md §4.7).
            all_keys.add((finding, "unspecified"))

    out: list[FusedFinding] = []
    for finding, laterality in sorted(all_keys):
        anatomy = None  # always, post-merge -- see _merge_b_groups_by_laterality's docstring
        b_group = by_key.get((finding, laterality))
        d_item = by_key_d.get((finding, laterality))
        c_pred = c_by_finding.get(finding)

        support: list[dict] = []
        n_sources = 0

        if c_pred is not None:
            cal = calib.calibrators.get(("C", finding))
            calibrated = cal.predict_proba(c_pred.raw_probability) if cal else c_pred.raw_probability
            prior_logit = _logit(calibrated)
            prior_source = "C"
            n_sources += 1
            support.append({
                "source": "C", "raw_score": c_pred.raw_probability, "calibrated": calibrated,
                "contributing_concepts": [asdict(cc) for cc in c_pred.top_contributing_concepts],
            })
        elif b_group is not None and np.isfinite(b_group.present_score):
            cal = calib.calibrators.get(("B", finding))
            calibrated = cal.predict_proba(b_group.present_score) if cal else _sigmoid(b_group.present_score)
            prior_logit = _logit(calibrated)
            prior_source = "B"
            n_sources += 1
            support.append({
                "source": "B", "raw_score": b_group.present_score, "calibrated": calibrated,
                "contributing_concepts": [
                    {**asdict(cc), "resolved_polarity": cc.resolved_polarity.value}
                    for cc in b_group.top_contributing_concepts
                ],
            })
        else:
            cal = calib.calibrators.get(("C", finding)) or calib.calibrators.get(("B", finding))
            base_rate = cal.base_rate if cal else 0.5
            prior_logit = _logit(base_rate)
            prior_source = "base_rate"

        posterior_logit = prior_logit

        if prior_source == "C" and b_group is not None:
            present = b_group.present_score if np.isfinite(b_group.present_score) else 0.0
            absent = b_group.absent_score if np.isfinite(b_group.absent_score) else 0.0
            asym = present - fc.absent_weight_b * absent
            posterior_logit += fc.w_b_delta_when_c_prior * asym
            n_sources += 1
            support.append({
                "source": "B", "role": "delta", "present_score": present, "absent_score": absent,
                "contributing_concepts": [
                    {**asdict(cc), "resolved_polarity": cc.resolved_polarity.value}
                    for cc in b_group.top_contributing_concepts
                ],
            })

        if d_item is not None:
            # absent_support was already discounted by evidence_d.absent_weight when accumulated
            # (retrieval/evidence_d.py) -- not re-discounted here.
            asym = d_item.present_support - d_item.absent_support
            posterior_logit += fc.w_d_delta * asym
            n_sources += 1
            support.append({
                "source": "D", "role": "delta", "source_report_ids": d_item.source_report_ids,
                "present_support": d_item.present_support,
                "absent_support": d_item.absent_support,
                "n_neighbours": d_item.present_count + d_item.absent_count,
            })

        confidence = _sigmoid(posterior_logit)
        tau_hi, tau_lo = calib.thresholds.get(finding, (settings.calibration.tau_default_hi, settings.calibration.tau_default_lo))

        if tau_hi >= 1.0:
            band = "abstain"
        elif confidence >= tau_hi:
            band = "accept"
        elif confidence <= tau_lo:
            band = "reject"
        else:
            band = "uncertain"

        out.append(FusedFinding(
            finding=finding, anatomy=anatomy, laterality=laterality, confidence=confidence,
            polarity=_BAND_TO_POLARITY.get(band, "uncertain"), prior_source=prior_source, band=band,
            admitted=band != "abstain", n_supporting_sources=n_sources, support=support,
        ))
    return out
