"""N12/N19 tagging step 7 (pipeline.md 4.6): canonical finding vocabulary.

Built from a small, hand-curated synonym table (configs/finding_synonyms.yaml) anchored on the
14 CheXpert labels -- deliberately NOT built primarily from RadLex (measured too thin for common
CXR finding terms, pipeline.md 5.4) and NOT auto-clustered from RadGraph's raw observation
strings (a heavier, unproven method that was considered and deferred, pipeline.md 10.2). Growing
this table as the offline coverage report shows frequent unmapped observations is the intended
maintenance path -- same pattern as the RadLex anatomy-term mapping in pipeline.md 4.4 item 4.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from ...core.errors import ConfigError
from ...retrieval.radgraph_parser import RadGraphEntity, RadGraphRelation


@dataclass(frozen=True)
class FindingVocabulary:
    # normalised synonym surface form -> canonical finding label
    _synonym_to_canonical: dict[str, str]
    chexpert_labels: frozenset[str]

    def resolve_observation(self, text: str) -> Optional[str]:
        """Exact match, after normalisation, against the synonym table. Returns None (never
        guessed) if the observation text isn't in the table."""
        return self._synonym_to_canonical.get(_norm(text))

    def is_chexpert(self, canonical_finding: str) -> bool:
        return canonical_finding in self.chexpert_labels


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def load_finding_vocabulary(path: str | Path) -> FindingVocabulary:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"finding synonyms file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    synonym_to_canonical: dict[str, str] = {}
    chexpert: set[str] = set()
    for canonical, entry in data.items():
        if not isinstance(entry, dict) or "synonyms" not in entry:
            raise ConfigError(f"{p}: entry {canonical!r} must be a mapping with a 'synonyms' list")
        if entry.get("chexpert"):
            chexpert.add(canonical)
        # the canonical label's own name is always a valid synonym of itself
        for syn in [canonical.replace("_", " "), *entry["synonyms"]]:
            key = _norm(syn)
            if key in synonym_to_canonical and synonym_to_canonical[key] != canonical:
                raise ConfigError(
                    f"{p}: synonym {syn!r} is claimed by both {synonym_to_canonical[key]!r} and {canonical!r}"
                )
            synonym_to_canonical[key] = canonical
    return FindingVocabulary(_synonym_to_canonical=synonym_to_canonical, chexpert_labels=frozenset(chexpert))


def pick_core_observation(
    observations: list[RadGraphEntity], relations: list[RadGraphRelation]
) -> Optional[RadGraphEntity]:
    """Given one RadGraphParse's .observations() and .relations, pick the entity most likely to
    BE the finding rather than a modifier of it. Heuristic, not exact (pipeline.md 4.6's GAP-21
    notes RadGraph's relation graph is itself inconsistent): prefer an Observation that is the
    SOURCE of a 'located_at' relation (an entity located somewhere is the finding -- confirmed
    in this project's probe: "effusion"/"opacification" carry located_at edges, "small"/
    "definite"/"unchanged" never do). Falls back to the longest observation text if no
    located_at edge exists, since short modifier words are typically shorter than the finding
    word they modify. Ties broken by entity index for determinism."""
    if not observations:
        return None
    located_at_sources = {r.source_index for r in relations if r.kind == "located_at"}
    located = [o for o in observations if o.index in located_at_sources]
    if located:
        return sorted(located, key=lambda o: o.index)[0]
    return sorted(observations, key=lambda o: (-len(o.text), o.index))[0]
