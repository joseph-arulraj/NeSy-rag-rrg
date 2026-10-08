"""N27 prompt construction from the VERIFIED belief graph (pipeline.md §6.1). Applies the fixed,
declared confidence -> hedging-language mapping (pipeline.md GAP-9, resolved here) and enforces
INV-3 (LLM prompt contains only content derivable from the verified graph, plus fixed style
instructions -- never raw retrieved text or raw scores)."""
from __future__ import annotations

from ..core.types import BeliefGraph, Finding

# pipeline.md §6.1 GAP-9: fixed, declared confidence -> hedging language mapping. Checked in
# descending threshold order.
_HEDGE_BANDS = [(0.8, "assertive"), (0.5, "probable"), (0.0, "possible")]


def hedge_for_confidence(confidence: float) -> str:
    for threshold, hedge in _HEDGE_BANDS:
        if confidence >= threshold:
            return hedge
    return "possible"


def _render_finding(f: Finding) -> str:
    parts = [f.label]
    if f.anatomy:
        parts.append(f"anatomy={f.anatomy}")
    parts.append(f"laterality={f.laterality.value}")
    if f.attributes:
        parts.append(", ".join(f"{k}={v}" for k, v in f.attributes.items()))
    parts.append(f"polarity={f.polarity}")
    parts.append(f"confidence={f.confidence:.2f}({hedge_for_confidence(f.confidence)})")
    return " | ".join(parts)


SYSTEM_PROMPT = """You are a radiology report-writing assistant. You perform ONLY linguistic \
realisation: you turn a list of already-verified findings into standard chest X-ray report \
prose. You do NOT decide what abnormalities are present -- that has already been decided for \
you by a separate system, and every finding you are given has already been checked.

Rules, all mandatory:
- State every listed finding. State nothing that is not listed.
- Do not add hedges, causes, severities, or recommendations that are not present in the input.
- Include explicit NEGATIVE findings as ordinary negations (e.g. "No pneumothorax is seen.").
- Use the given confidence band's hedging language exactly (assertive / probable / possible) -- \
do not invent your own hedging language.
- Write in standard radiology report style: a Findings section, then an Impression section. The \
Impression is NOT a copy of the Findings section -- it is a brief, prioritised summary (most \
clinically significant items first, minor/incidental items last or omitted if truly minor). Use \
different, more concise sentence structure than Findings; do not repeat the same sentences \
verbatim between the two sections.
- Do not mention confidence scores, source names, finding_ids, or any other internal system \
detail anywhere in the output text -- translate them into ordinary clinical prose only.
- The same finding label can appear more than once, with different anatomy/laterality (e.g. \
support_devices on the right, on the left, and midline) -- these are grouped together in the \
list below. Describe every one of them in ONE combined sentence covering all the affected \
locations (e.g. "Support devices are present in the right and left lung and midline."), NOT one \
separate sentence per entry. Writing "X is present. X is also noted. X is also present." for \
repeated entries of the same finding is wrong even though each sentence is individually true -- \
combine them. If entries of the same finding differ in confidence band, keep the more cautious \
hedging for the sentence covering all of them, or split into at most two sentences (one per \
hedge level actually present), never one sentence per entry.
"""


def build_user_prompt(graph: BeliefGraph) -> str:
    """pipeline.md §6.1: INV-3 -- this is the entire clinical content the LLM ever sees."""
    if not graph.findings:
        return (
            "Verified findings: none.\n"
            "Write a normal chest X-ray report stating no acute cardiopulmonary abnormality."
        )
    lines = ["Verified findings (state every one below, state nothing else):"]
    for f in graph.findings:
        lines.append(f"- {_render_finding(f)}")
    return "\n".join(lines)
