"""N12/N19 tagging step 3 (pipeline.md 4.6, unchanged from 4.4): deterministic laterality
tagging by keyword match. No model, no RadGraph dependency -- laterality words were confirmed
(the RadGraph probe, pipeline.md 10.2/GAP-21) to never appear as RadGraph entities at all, so
this must stand alone.
"""
from __future__ import annotations

import re

from ...core.types import Laterality

_BILATERAL = {"bilateral", "bilaterally", "bibasilar", "bibasal", "both"}
_LEFT = {"left", "lt", "l"}
_RIGHT = {"right", "rt", "r"}
_MIDLINE = {"midline", "central"}


def _pattern(words: set[str]) -> re.Pattern:
    # word-boundary match; single-letter tokens (l/r) only match as a standalone hyphen-joined
    # token (e.g. "l-sided") to avoid matching inside ordinary words.
    alts = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"\b(?:{alts})\b", re.IGNORECASE)


_BILATERAL_RE = _pattern(_BILATERAL)
_LEFT_RE = re.compile(r"\b(?:left|lt)\b|\bl-(?=sided|side)", re.IGNORECASE)
_RIGHT_RE = re.compile(r"\b(?:right|rt)\b|\br-(?=sided|side)", re.IGNORECASE)
_MIDLINE_RE = _pattern(_MIDLINE)


def tag_laterality(text: str) -> Laterality:
    """First match wins by priority: bilateral > midline > (left and right both present ->
    bilateral, i.e. a sentence mentioning both sides describes a bilateral context) > left-only
    > right-only > unspecified. Never guessed from anything but explicit lexical cues."""
    if _BILATERAL_RE.search(text):
        return Laterality.BILATERAL
    if _MIDLINE_RE.search(text):
        return Laterality.MIDLINE
    left = bool(_LEFT_RE.search(text))
    right = bool(_RIGHT_RE.search(text))
    if left and right:
        return Laterality.BILATERAL
    if left:
        return Laterality.LEFT
    if right:
        return Laterality.RIGHT
    return Laterality.UNSPECIFIED
