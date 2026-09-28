"""Shared RadGraph-based fact extraction from REPORT text (not concept-bank text). Two
consumers reuse this: N18 (Evidence D's aggregation of retrieved neighbours' reports) and the
calibration ground-truth pipeline (fusion.calibrate, validation-split labels for B/C).

Sentence-scoped, not whole-report-scoped: RadGraphEntity carries no usable char_span
(retrieval/radgraph_parser.py's docstring), so there is no way to attribute an entity in a
whole-report RadGraph parse back to a specific sentence for negation/laterality/temporal
context. Splitting into sentences first and parsing each independently keeps every fact
correctly scoped to its own local context, at the cost of losing cross-sentence relations
(e.g. "no consolidation. There is a small effusion, however." -- each sentence still resolves
correctly on its own). A documented scoping choice, not an oversight.

Reuses the exact same tagging primitives already built and validated for the 368k concept bank
(concepts/tagging/*) rather than reimplementing negation/temporal/laterality logic here --
report sentences and concept-bank phrases are both short, single-topic strings once split this
way, so the same machinery applies.

Known simplification, not fixed here: unlike offline/build_concept_bank.py's tag_one(), this
does not apply the pericardial/pleural disambiguation guard (_FINDING_DISAMBIGUATION) -- report
sentences carry more surrounding context than a bare concept phrase, which somewhat reduces (but
does not eliminate) that specific risk. Revisit if calibration/D's coverage report shows it
matters in practice.
"""
from __future__ import annotations

import gzip
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from ..concepts.tagging.finding_vocab import FindingVocabulary
from ..concepts.tagging.laterality import tag_laterality
from ..concepts.tagging.polarity import classify_temporal, resolve_base_polarity, resolve_polarity
from ..core.config import ReportFactsConfig, Settings, TaggingConfig
from ..core.types import AnatomyID, Laterality, Polarity
from ..fusion.rules.radlex_client import RadLexClient
from ..retrieval.radgraph_parser import RadGraphParser


@dataclass(frozen=True)
class ReportFact:
    canonical_finding: str
    anatomy: Optional[AnatomyID]
    laterality: Laterality
    polarity: Polarity
    temporal_class: str
    sentence: str


def split_sentences(text: str, pattern: str) -> list[str]:
    parts = [p.strip() for p in re.split(pattern, text) if p and p.strip()]
    return parts


def _resolve_anatomy(parse, radlex: RadLexClient) -> Optional[str]:
    for e in sorted(parse.anatomy(), key=lambda a: -len(a.text)):
        rid = radlex.resolve_anatomy_term(e.text)
        if rid is not None:
            return rid
    return None


def extract_report_facts(
    text: str,
    radgraph: RadGraphParser,
    radlex: RadLexClient,
    vocab: FindingVocabulary,
    report_facts_cfg: ReportFactsConfig,
    tagging_cfg: TaggingConfig,
) -> list[ReportFact]:
    """Every resolvable (canonical_finding, anatomy, laterality, polarity) fact in `text`,
    one report -> possibly many facts (unlike concept-bank tagging's one-fact-per-string).
    A sentence whose Observation entity doesn't map onto the canonical vocabulary contributes
    no fact for that entity (skipped, not guessed) -- resolve_observation never guesses either."""
    sentences = split_sentences(text, report_facts_cfg.sentence_split_regex)
    if not sentences:
        return []
    parses = radgraph.parse_batch(sentences)

    facts: list[ReportFact] = []
    for sentence, parse in zip(sentences, parses):
        anatomy = _resolve_anatomy(parse, radlex)
        laterality = tag_laterality(sentence)
        temporal = classify_temporal(
            sentence, tagging_cfg.temporal_implies_present, tagging_cfg.temporal_implies_absent,
            tagging_cfg.temporal_indeterminate,
        )
        for obs in parse.observations():
            canonical = vocab.resolve_observation(obs.text)
            if canonical is None:
                continue
            base_polarity = resolve_base_polarity(sentence, tagging_cfg.negation_cues, obs.certainty)
            resolved = resolve_polarity(base_polarity, temporal)
            if resolved is None:
                continue
            facts.append(ReportFact(
                canonical_finding=canonical, anatomy=anatomy, laterality=laterality,
                polarity=resolved, temporal_class=temporal, sentence=sentence,
            ))
    return facts


# ------------------------------------------------------------------------------ disk cache
def _cache_path(cache_dir: Path, study_id: int) -> Path:
    return cache_dir / f"s{study_id}.json.gz"


def load_cached_facts(cache_dir: Path, study_id: int) -> Optional[list[ReportFact]]:
    p = _cache_path(cache_dir, study_id)
    if not p.is_file():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        rows = json.load(fh)
    return [
        ReportFact(
            canonical_finding=r["canonical_finding"], anatomy=r["anatomy"],
            laterality=Laterality(r["laterality"]), polarity=Polarity(r["polarity"]),
            temporal_class=r["temporal_class"], sentence=r["sentence"],
        )
        for r in rows
    ]


def save_cached_facts(cache_dir: Path, study_id: int, facts: list[ReportFact]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = _cache_path(cache_dir, study_id)
    tmp = p.with_name(p.name + ".tmp")
    rows = [{**asdict(f), "laterality": f.laterality.value, "polarity": f.polarity.value} for f in facts]
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(rows, fh, separators=(",", ":"))
    tmp.replace(p)


def get_or_extract_facts(
    study_id: int,
    text: str,
    radgraph: RadGraphParser,
    radlex: RadLexClient,
    vocab: FindingVocabulary,
    settings: Settings,
) -> list[ReportFact]:
    """Cache-through wrapper: report facts are a pure function of (report text, RadGraph
    version, vocab version), so once computed for a study they never need recomputing."""
    cached = load_cached_facts(settings.report_facts.cache_dir, study_id)
    if cached is not None:
        return cached
    facts = extract_report_facts(text, radgraph, radlex, vocab, settings.report_facts, settings.tagging)
    save_cached_facts(settings.report_facts.cache_dir, study_id, facts)
    return facts
