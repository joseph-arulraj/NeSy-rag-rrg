"""Ground-truth labels + isotonic calibration for B and C, and per-finding admission-threshold
fitting -- resolves pipeline.md §10.2's previously-open calibration-data question: labels come
from RadGraph-parsing each study's OWN report (reports/report_facts.py), on the VALIDATION split
only (train would be optimistic), not from the 14 CheXpert labels alone (too narrow -- B covers
the full canonical vocabulary, not just 14 labels).

B's raw signal is per `(finding, anatomy, laterality)` group; calibration targets are per bare
finding (that's what a report-derived ground-truth label naturally is). `per_finding_extrema`
reduces B's raw scores to one (max present, max absent) pair per finding, deliberately NOT
limited to `grouping.top_k_groups` -- the top-K view would systematically exclude negative
examples (an absent finding's own group is unlikely to be in an image's top 20), which would
bias the calibrator. C's output is already per bare finding, no reduction needed.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..cbm.concepts import load_or_resolve_concept_ids
from ..cbm.infer import CBMPredictor
from ..cbm.model import load_head
from ..concepts.bank import ConceptBank
from ..concepts.grouping import GroupingIndex
from ..concepts.similarity import score_concepts_batch
from ..concepts.tagging.finding_vocab import FindingVocabulary, load_finding_vocabulary
from ..core.config import Settings
from ..core.runtime import configure_runtime
from ..core.types import Polarity
from ..datasets.mimic_cxr import MimicCxrIndex
from ..datasets.reports import ReportStore
from .rules.radlex_client import RadLexClient
from ..ingest.image_loader import load_record_image
from ..perception.clear_encoder import ClearEncoder
from ..reports.report_facts import get_or_extract_facts
from ..retrieval.radgraph_parser import RadGraphParser


# --------------------------------------------------------------------------- ground truth
def ground_truth_polarity_for_study(
    study_id: int, text: str, radgraph: RadGraphParser, radlex: RadLexClient,
    vocab: FindingVocabulary, settings: Settings,
) -> dict[str, Polarity]:
    """A study's OWN report -> {canonical_finding: polarity}. If a finding is asserted more than
    once with different polarity in the same report, PRESENT beats ABSENT beats UNCERTAIN -- a
    report contradicting itself about one finding is rare and not this function's job to
    adjudicate (that's N25's, at graph level, not report-text level)."""
    facts = get_or_extract_facts(study_id, text, radgraph, radlex, vocab, settings)
    order = {Polarity.PRESENT: 2, Polarity.ABSENT: 1, Polarity.UNCERTAIN: 0}
    out: dict[str, Polarity] = {}
    for f in facts:
        cur = out.get(f.canonical_finding)
        if cur is None or order[f.polarity] > order[cur]:
            out[f.canonical_finding] = f.polarity
    return out


def per_finding_extrema(raw_scores: np.ndarray, index: GroupingIndex) -> dict[str, tuple[float, float]]:
    """-> {finding: (max present_score across ALL its groups, max absent_score across ALL its
    groups)}. Deliberately not limited to top_k_groups -- see module docstring."""
    present_mask = index.concept_side == 1
    absent_mask = index.concept_side == -1
    group_present = np.full(index.n_groups, -np.inf)
    group_absent = np.full(index.n_groups, -np.inf)
    if present_mask.any():
        g = index.concept_group_idx[present_mask]
        np.maximum.at(group_present, g, raw_scores[present_mask])
    if absent_mask.any():
        g = index.concept_group_idx[absent_mask]
        np.maximum.at(group_absent, g, raw_scores[absent_mask])

    out: dict[str, tuple[float, float]] = {}
    for gidx in range(index.n_groups):
        finding = index.group_finding[gidx]
        p, a = out.get(finding, (-np.inf, -np.inf))
        out[finding] = (max(p, float(group_present[gidx])), max(a, float(group_absent[gidx])))
    return out


# --------------------------------------------------------------------------- calibrator
@dataclass
class Calibrator:
    kind: str            # "isotonic" | "base_rate"
    base_rate: float
    model: Optional[object] = None   # sklearn IsotonicRegression when kind == "isotonic"

    def predict_proba(self, raw_score: float) -> float:
        if self.kind == "base_rate" or self.model is None:
            return self.base_rate
        return float(np.clip(self.model.predict([raw_score])[0], 1e-4, 1 - 1e-4))


def fit_calibrator(raw_scores: list[float], labels: list[int], min_samples: int) -> Calibrator:
    n = len(labels)
    base_rate = float(np.clip(np.mean(labels), 1e-4, 1 - 1e-4)) if n else 0.5
    if n < min_samples or len(set(labels)) < 2:
        return Calibrator(kind="base_rate", base_rate=base_rate)
    from sklearn.isotonic import IsotonicRegression
    model = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
    model.fit(np.asarray(raw_scores, dtype=np.float64), np.asarray(labels, dtype=np.float64))
    return Calibrator(kind="isotonic", base_rate=base_rate, model=model)


# --------------------------------------------------------------------------- thresholds
def fit_threshold(posteriors: list[float], labels: list[int], target_precision: float = 0.9) -> tuple[float, float]:
    """tau_hi: the lowest score threshold whose accept-set precision clears `target_precision`
    (1.01 = unreachable -> abstain, pipeline.md §5.4 AMBIG-7's "tau_hi >= 1.0 -> abstain").
    tau_lo: the symmetric threshold for the reject side (precision of predicting absent)."""
    if not labels or len(set(labels)) < 2:
        return (1.01, -0.01)   # caller should fall back to calibration.tau_default_hi/lo instead
    candidates = sorted(set(posteriors))
    tau_hi = 1.01
    for t in candidates:
        accepted = [y for p, y in zip(posteriors, labels) if p >= t]
        if accepted and sum(accepted) / len(accepted) >= target_precision:
            tau_hi = t
            break
    tau_lo = -0.01
    for t in sorted(candidates, reverse=True):
        rejected = [y for p, y in zip(posteriors, labels) if p <= t]
        if rejected and sum(1 - y for y in rejected) / len(rejected) >= target_precision:
            tau_lo = t
            break
    return tau_hi, tau_lo


# --------------------------------------------------------------------------- orchestration
@dataclass(frozen=True)
class CalibrationArtifacts:
    calibrators: dict[tuple[str, str], Calibrator]     # (source, finding) -> Calibrator; source in {"B", "C"}
    thresholds: dict[str, tuple[float, float]]          # finding -> (tau_hi, tau_lo)


def run_calibration(settings: Settings, log: Callable[[str], None] = print, limit: "int | None" = None) -> CalibrationArtifacts:
    """Builds ground truth + fits calibrators + fits thresholds over the VALIDATION split, in
    one pass (each study's image/report is only encoded/parsed once)."""
    s = settings
    device = configure_runtime(s)
    bank_raw = ConceptBank.load(s)
    from ..concepts.bank import load_tags
    tags_path = s.paths.model_weights_dir / "concept_tags.jsonl.gz"
    bank = bank_raw.with_tags(load_tags(tags_path, s.concept_bank.expected_n_concepts))
    grouping_index = bank.grouping_index()

    radlex = RadLexClient.load(s.radlex.snapshot_path, s.radlex.chest_scope_root_label, s.radlex.version,
                                tuple(s.radlex.additional_scope_roots))
    vocab = load_finding_vocabulary(s.tagging.finding_synonyms_path)
    radgraph = RadGraphParser(model_type=s.radgraph.model_type)
    reports = ReportStore.from_settings(s)
    encoder = ClearEncoder(s, device)

    concept_ids = load_or_resolve_concept_ids(s.cbm.concepts_path, s.cbm.concept_ids_cache, bank.concepts)
    head, head_info = load_head(s.cbm.head_checkpoint)
    cbm = CBMPredictor(head, head_info, concept_ids, bank)

    index = MimicCxrIndex.from_settings(s)
    records, _ = index.study_records("validate")
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError("no validate-split studies found -- check paths.mimic_root / dataset.splits")
    log(f"calibrating on {len(records)} validation studies")

    # (source, finding) -> parallel lists of (raw_score, label)
    b_data: dict[str, tuple[list[float], list[int]]] = {}
    c_data: dict[str, tuple[list[float], list[int]]] = {}

    for i, rec in enumerate(records):
        decoded, _ = load_record_image(rec, s)
        feat = encoder.encode_batch([decoded])[0].numpy()
        raw_scores = score_concepts_batch(feat[None, :], bank, device, s.concept_bank.score_batch_size)[0]

        text = reports.get(rec.subject_id, rec.study_id)
        gt = ground_truth_polarity_for_study(rec.study_id, text, radgraph, radlex, vocab, s)

        b_extrema = per_finding_extrema(raw_scores, grouping_index)
        for finding, pol in gt.items():
            if pol == Polarity.UNCERTAIN or finding not in b_extrema:
                continue
            present_max, absent_max = b_extrema[finding]
            score = present_max if np.isfinite(present_max) else -1.0
            label = 1 if pol == Polarity.PRESENT else 0
            xs, ys = b_data.setdefault(finding, ([], []))
            xs.append(score)
            ys.append(label)

        cbm_preds = {p.pathology: p.raw_probability for p in cbm.predict(raw_scores)}
        for finding, pol in gt.items():
            if pol == Polarity.UNCERTAIN or finding not in cbm_preds:
                continue
            xs, ys = c_data.setdefault(finding, ([], []))
            xs.append(cbm_preds[finding])
            ys.append(1 if pol == Polarity.PRESENT else 0)

        if (i + 1) % 20 == 0 or i + 1 == len(records):
            log(f"  {i + 1}/{len(records)} studies processed")

    calibrators: dict[tuple[str, str], Calibrator] = {}
    for finding, (xs, ys) in b_data.items():
        calibrators[("B", finding)] = fit_calibrator(xs, ys, s.calibration.min_samples_per_finding)
    for finding, (xs, ys) in c_data.items():
        calibrators[("C", finding)] = fit_calibrator(xs, ys, s.calibration.min_samples_per_finding)
    log(f"fitted {len(calibrators)} calibrators ({len(b_data)} B findings, {len(c_data)} C findings)")

    # thresholds: fit on whichever source has the most data for that finding, preferring C where both exist
    thresholds: dict[str, tuple[float, float]] = {}
    all_findings = set(b_data) | set(c_data)
    for finding in all_findings:
        source_data = c_data.get(finding) or b_data.get(finding)
        cal = calibrators.get(("C", finding)) or calibrators.get(("B", finding))
        if source_data is None or cal is None:
            continue
        xs, ys = source_data
        posteriors = [cal.predict_proba(x) for x in xs]
        hi, lo = fit_threshold(posteriors, ys)
        if hi > 1.0:  # unreachable at target precision on this data -- use the configured fallback, not abstain-forever
            hi = s.calibration.tau_default_hi
        if lo < 0.0:
            lo = s.calibration.tau_default_lo
        thresholds[finding] = (hi, lo)
    log(f"fitted thresholds for {len(thresholds)} findings")

    artifacts = CalibrationArtifacts(calibrators=calibrators, thresholds=thresholds)
    save_calibration(s, artifacts)
    return artifacts


def save_calibration(settings: Settings, artifacts: CalibrationArtifacts) -> None:
    settings.calibration.isotonic_cache.parent.mkdir(parents=True, exist_ok=True)
    with settings.calibration.isotonic_cache.open("wb") as fh:
        pickle.dump(artifacts.calibrators, fh)
    import json
    settings.calibration.threshold_cache.parent.mkdir(parents=True, exist_ok=True)
    with settings.calibration.threshold_cache.open("w", encoding="utf-8") as fh:
        json.dump({k: list(v) for k, v in artifacts.thresholds.items()}, fh, indent=2)


def load_calibration(settings: Settings) -> CalibrationArtifacts:
    with settings.calibration.isotonic_cache.open("rb") as fh:
        calibrators = pickle.load(fh)
    import json
    with settings.calibration.threshold_cache.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    thresholds = {k: (v[0], v[1]) for k, v in raw.items()}
    return CalibrationArtifacts(calibrators=calibrators, thresholds=thresholds)
