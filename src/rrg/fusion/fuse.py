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
from ..concepts.grouping import FindingGroup
from ..core.config import Settings
from .calibrate import CalibrationArtifacts
from ..retrieval.evidence_d import EvidenceDItem


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

    # Indexed at full (finding, anatomy, laterality) granularity for the belief graph's node
    # keys; the prior/delta DECISION is made per bare finding and applied identically to every
    # (anatomy, laterality) variant of that finding B or D produced.
    by_key: dict[tuple, FindingGroup] = {(g.finding, g.anatomy, g.laterality.value): g for g in b_groups}
    by_key_d: dict[tuple, EvidenceDItem] = {(d.finding, d.anatomy, d.laterality.value): d for d in d_items}

    all_keys = set(by_key) | set(by_key_d)
    for finding in c_by_finding:
        if not any(k[0] == finding for k in all_keys):
            # C covers this finding but neither B nor D produced a spatial variant -- still
            # needs a node; C carries no anatomy/laterality attribute at all (pipeline.md §4.7).
            all_keys.add((finding, None, "unspecified"))

    out: list[FusedFinding] = []
    for finding, anatomy, laterality in sorted(all_keys, key=lambda k: (k[0], k[1] or "", k[2])):
        b_group = by_key.get((finding, anatomy, laterality))
        d_item = by_key_d.get((finding, anatomy, laterality))
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
