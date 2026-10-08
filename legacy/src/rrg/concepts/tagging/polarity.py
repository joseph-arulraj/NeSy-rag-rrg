"""N12/N19 tagging steps 4-6 (pipeline.md 4.6, rev 3): base (negation) polarity, temporal/
comparison classification, and their combination into resolved_polarity.

Confirmed by the real RadGraph probe (pipeline.md 10.2/GAP-21):
  - RadGraph's own certainty label DOES correctly capture explicit negation
    ("no definite pneumothorax is seen" -> definitely-absent) -- reuse it when available.
  - RadGraph does NOT capture temporal/resolution semantics: "resolved" produced no entity
    and left the finding tagged definitely-present. The gazetteer below is not redundant
    with RadGraph -- it does something RadGraph provably does not do.
So: base polarity prefers RadGraph's certainty (see resolve_base_polarity), falling back to
this module's own negation-cue regex only when RadGraph produced no Observation entity for a
concept. Temporal classification is always this module's own gazetteer -- RadGraph has no
equivalent signal at all.
"""
from __future__ import annotations

import re
from typing import Optional

from ...core.types import Polarity

TemporalClass = str   # "stationary" | "implies_present" | "implies_absent" | "indeterminate"

STATIONARY: TemporalClass = "stationary"
IMPLIES_PRESENT: TemporalClass = "implies_present"
IMPLIES_ABSENT: TemporalClass = "implies_absent"
INDETERMINATE: TemporalClass = "indeterminate"


def _phrase_pattern(phrases: list[str]) -> re.Pattern:
    # Longest phrase first so "interval increase" matches before a lone "interval" would.
    parts = sorted({p.strip() for p in phrases if p.strip()}, key=len, reverse=True)
    alts = "|".join(re.escape(p).replace(r"\ ", r"\s+") for p in parts)
    return re.compile(rf"\b(?:{alts})\b", re.IGNORECASE)


def _negation_pattern(cues: list[str]) -> re.Pattern:
    return _phrase_pattern(cues)


def tag_base_polarity(text: str, negation_cues: list[str]) -> Polarity:
    """Fallback only (used when RadGraph gave no Observation entity for this concept, or
    RadGraph is unavailable for a given call). Prefer resolve_base_polarity when a RadGraph
    parse exists -- see module docstring."""
    return Polarity.ABSENT if _negation_pattern(negation_cues).search(text) else Polarity.PRESENT


def resolve_base_polarity(
    text: str,
    negation_cues: list[str],
    radgraph_observation_certainty: Optional[str],
) -> Polarity:
    """radgraph_observation_certainty: the certainty string RadGraph attached to this concept's
    (any) Observation entity, e.g. "definitely present" / "definitely absent" / "uncertain", or
    None if RadGraph produced no Observation entity at all for this concept (measured to happen
    on a meaningful fraction of short concept-bank phrases, GAP-21).

    The lexical negation regex is checked FIRST and wins whenever it fires, even over a
    RadGraph "present" certainty. Confirmed necessary, not just cautious: on the terse,
    verb-less phrase "no definitive pleural effusions", RadGraph tagged the Observation entity
    "definitely present", missing the leading "no" entirely -- it gets this right on
    full-sentence phrasing ("no definite pneumothorax IS SEEN") but not reliably on
    concept-bank-style noun phrases. The regex is simple and near-100%-precision for the same
    reason the laterality tagger is (pipeline.md 4.4): these cue words are essentially never
    used non-negatingly in this register. RadGraph's certainty is still authoritative for
    absent/uncertain calls the regex has no lexical cue for."""
    if _negation_pattern(negation_cues).search(text):
        return Polarity.ABSENT
    if radgraph_observation_certainty:
        c = radgraph_observation_certainty.lower()
        if "absent" in c:
            return Polarity.ABSENT
        if "uncertain" in c:
            return Polarity.UNCERTAIN
        if "present" in c:
            return Polarity.PRESENT
    return Polarity.PRESENT


def classify_temporal(
    text: str,
    implies_present: list[str],
    implies_absent: list[str],
    indeterminate: list[str],
) -> TemporalClass:
    """Gazetteer-based (pipeline.md 4.6 step 5). Proposed, unvalidated on real data -- GAP-20;
    QA-sample the output before trusting it in production, same as the anatomy/laterality tags."""
    # implies_absent checked first: "resolved" must win over an incidental "unchanged" elsewhere
    # in the same short phrase (rare, but the more conservative reading is not to double-count).
    if _phrase_pattern(implies_absent).search(text):
        return IMPLIES_ABSENT
    if _phrase_pattern(implies_present).search(text):
        return IMPLIES_PRESENT
    if _phrase_pattern(indeterminate).search(text):
        return INDETERMINATE
    return STATIONARY


def resolve_polarity(base: Polarity, temporal: TemporalClass) -> Optional[Polarity]:
    """pipeline.md 4.6 step 6's combination table. Returns None to mean 'exclude this concept
    from grouping entirely' (indeterminate temporal language, or a base-absent + implies-present
    combination too ambiguous to guess)."""
    if temporal == INDETERMINATE:
        return None
    if base == Polarity.PRESENT:
        if temporal in (STATIONARY, IMPLIES_PRESENT):
            return Polarity.PRESENT
        if temporal == IMPLIES_ABSENT:
            # The case a naive text-strip gets backwards: "pleural effusion resolved" without
            # this combination step would keep PRESENT (or, worse, become "pleural effusion"
            # with no polarity information at all if the qualifier were simply deleted).
            return Polarity.ABSENT
    if base == Polarity.ABSENT:
        if temporal in (STATIONARY, IMPLIES_ABSENT):
            return Polarity.ABSENT
        if temporal == IMPLIES_PRESENT:
            # Rare/ambiguous ("no longer clear, now shows...") -- flagged for manual review
            # rather than guessed, per pipeline.md 4.6 step 6.
            return None
    if base == Polarity.UNCERTAIN:
        return Polarity.UNCERTAIN if temporal in (STATIONARY, IMPLIES_PRESENT) else None
    return None
