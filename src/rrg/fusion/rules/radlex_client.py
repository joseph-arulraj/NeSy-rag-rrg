"""RadLex anatomy lookup, backed by a compact local snapshot (built once, offline, by
offline/build_radlex_snapshot.py from a downloaded RadLex.owl -- pipeline.md 5.4 rev 2).

No live BioPortal/network calls at runtime (per pipeline.md's determinism requirement, INV-4).
Used both offline (concepts/tagging's anatomy step, pipeline.md 4.6 step 2) and, later, by the
N25 rule engine for anatomy hierarchy checks.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ...core.errors import PipelineError


class RadLexSnapshotError(PipelineError):
    code = "E_RADLEX_SNAPSHOT"


@dataclass(frozen=True)
class RadLexClass:
    rid: str                          # e.g. "RID1327"
    label: str
    synonyms: tuple[str, ...]
    is_a_parents: tuple[str, ...]      # direct rdfs:subClassOf targets (RIDs)
    part_of: tuple[str, ...]           # this class IS PART OF these (containers)
    has_part: tuple[str, ...]          # this class HAS these AS PARTS
    anatomical_site: tuple[str, ...]   # this class's own Anatomical_Site targets, if any (for findings, not anatomy)

    @property
    def adjacent_anatomy(self) -> frozenset[str]:
        """Direct part_of UNION has_part -- for 'are A and B adjacent in the part-whole
        hierarchy' pairwise checks. NOT the scoping relation (that must stay directed, see
        RadLexClient._closure's docstring)."""
        return frozenset(self.part_of) | frozenset(self.has_part)


class RadLexClient:
    """Loaded once at startup; read-only. Term lookup is a case-insensitive exact/substring
    match over label+synonyms restricted to the chest scope (is-a + part-of union rooted at
    `chest_scope_root_label`) -- the same scope-filtering rationale established in
    pipeline.md 5.4: matching against the full ~46k-class ontology produces false positives
    (e.g. "lobe" matching thyroid lobes)."""

    def __init__(self, classes: dict[str, RadLexClass], chest_scope: frozenset[str], version: str):
        self._classes = classes
        self._chest_scope = chest_scope
        self.version = version
        self._label_index: dict[str, list[str]] = {}   # normalised text -> [rid, ...], chest-scope only
        for rid in chest_scope:
            c = classes[rid]
            for text in (c.label, *c.synonyms):
                self._label_index.setdefault(_norm(text), []).append(rid)

    @classmethod
    def load(
        cls,
        snapshot_path: str | Path,
        chest_scope_root_label: str,
        version: str,
        additional_scope_roots: tuple[str, ...] = (),
    ) -> "RadLexClient":
        path = Path(snapshot_path)
        if not path.is_file():
            raise RadLexSnapshotError(
                f"RadLex snapshot not found: {path}. Build it first: python scripts/build_radlex_snapshot.py"
            )
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
            data = json.load(fh)
        classes = {
            rid: RadLexClass(
                rid=rid, label=v["label"], synonyms=tuple(v["synonyms"]),
                is_a_parents=tuple(v["is_a_parents"]), part_of=tuple(v["part_of"]),
                has_part=tuple(v["has_part"]), anatomical_site=tuple(v.get("anatomical_site", ())),
            )
            for rid, v in data["classes"].items()
        }
        by_label = {_norm(c.label): rid for rid, c in classes.items()}
        root = by_label.get(_norm(chest_scope_root_label))
        if root is None:
            raise RadLexSnapshotError(f"chest_scope_root_label {chest_scope_root_label!r} not found in the snapshot")
        scope = _closure(root, classes)
        for extra_label in additional_scope_roots:
            extra_rid = by_label.get(_norm(extra_label))
            if extra_rid is None:
                raise RadLexSnapshotError(f"radlex.additional_scope_roots entry {extra_label!r} not found in the snapshot")
            scope |= _closure(extra_rid, classes)
        return cls(classes, frozenset(scope), data.get("version", version))

    # ---- lookups
    def get(self, rid: str) -> Optional[RadLexClass]:
        return self._classes.get(rid)

    def resolve_anatomy_term(self, text: str) -> Optional[str]:
        """Case-insensitive exact match against chest-scope labels/synonyms. Returns the RID,
        or None (never guessed). Longest-match-first is the CALLER's job when scanning a span
        of text for multiple candidate substrings (pipeline.md 4.6: 'right middle lobe' should
        match 'middle lobe of lung' + 'right', not decompose token-by-token).

        Falls back to a small hand-curated adjective->noun map (_ADJECTIVE_FORMS below) when
        the direct match fails. Needed because RadGraph's single-token Anatomy entities are
        frequently adjectival ("pleural", "cardiac", "basilar") while RadLex's labels/synonyms
        are almost entirely noun forms -- confirmed empirically: every one of "pleural",
        "basilar", "cardiac", "mediastinal", "pulmonary", "hilar", "diaphragmatic" missed
        without this fallback."""
        hits = self._label_index.get(_norm(text))
        if not hits:
            noun_form = _ADJECTIVE_FORMS.get(_norm(text))
            hits = self._label_index.get(noun_form) if noun_form else None
        if not hits:
            return None
        # Prefer a class that IS the anchor scope root's descendant most directly (fewest is-a
        # hops isn't tracked here; a tie is broken deterministically by RID for reproducibility).
        return sorted(hits)[0]

    def is_a(self, child: str, parent: str) -> bool:
        seen: set[str] = set()
        stack = [child]
        while stack:
            rid = stack.pop()
            if rid == parent:
                return True
            if rid in seen:
                continue
            seen.add(rid)
            c = self._classes.get(rid)
            if c:
                stack.extend(c.is_a_parents)
        return False

    def part_of_related(self, a: str, b: str) -> bool:
        """True if `a` and `b` are DIRECTLY part-whole adjacent (either is a direct container
        or a direct part of the other) -- a same-lobe-group tolerance check, not a transitive
        closure. See pipeline.md 4.6 step 10's anatomy-mismatch tolerance."""
        ca = self._classes.get(a)
        return ca is not None and b in ca.adjacent_anatomy

    def chest_scope_size(self) -> int:
        return len(self._chest_scope)


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


# Common CXR anatomical adjectives -> the RadLex noun-form label they refer to. Small,
# hand-curated (same "grow it as gaps show up" pattern as configs/finding_synonyms.yaml),
# not exhaustive. Values are matched through the normal chest-scope label/synonym index, so a
# value here only resolves if that noun form is itself in the scope.
_ADJECTIVE_FORMS: dict[str, str] = {
    "pleural": "pleura", "cardiac": "heart", "pulmonary": "lung", "cardiopulmonary": "lung",
    "mediastinal": "mediastinum", "hilar": "hilum", "diaphragmatic": "diaphragm",
    "tracheal": "trachea", "bronchial": "bronchus", "costal": "rib", "costophrenic": "costophrenic sulcus",
    "vascular": "blood vessel", "thoracic": "thorax", "sternal": "sternum",
    "vertebral": "vertebra", "clavicular": "clavicle", "scapular": "scapula",
    "apical": "apex of lung", "basilar": "base of lung", "basal": "base of lung",
}


def _closure(root: str, classes: dict[str, RadLexClass]) -> set[str]:
    """is-a descendants UNION Part_Of-family descendants, walked in ONE direction only:
    container -> part (never part -> container). This is the corrected rule -- an earlier
    version of this walk used the symmetric part_of/has_part union in both directions and
    produced a 27,685-class 'chest' scope (60% of the whole ontology) by ascending from
    thorax into an unrelated container and back down a sibling branch. Descend-only, this
    reproduces the value validated against the real RadLex.owl in this project's design
    discussion (253 live classes for 'thorax').

    Known limitation, confirmed against the real file, not a bug in this function: some
    clinically relevant chest terms (e.g. RadLex's 'costophrenic sulcus') sit in a wholly
    separate is-a branch ('anatomical boundary entity') with no is-a or part-of edge back to
    'thorax' at all, and are legitimately absent from this closure. The fallback keyword
    gazetteer in concepts/tagging (pipeline.md 4.6 step 2's fallback) exists partly for this."""
    descend: dict[str, list[str]] = {}
    for rid, c in classes.items():
        for parent in c.is_a_parents:
            descend.setdefault(parent, []).append(rid)          # is-a: parent -> child
        for container in c.part_of:
            descend.setdefault(container, []).append(rid)        # X part_of W  =>  W -> X
        for part in c.has_part:
            descend.setdefault(rid, []).append(part)              # W has_part X =>  W -> X
    seen = {root}
    stack = [root]
    while stack:
        rid = stack.pop()
        for n in descend.get(rid, ()):
            if n not in seen:
                seen.add(n)
                stack.append(n)
    return seen
