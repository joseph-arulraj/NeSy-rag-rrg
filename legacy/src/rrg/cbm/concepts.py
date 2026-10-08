"""N14: resolve CLEAR's predefined CBM concept texts (model_weights/cbm_concepts.md, 67 of the
original 68 -- the file itself is one short) to their `concept_id` in the 368k bank so N13's
gather (`cbm_scores = ConceptScores[idx_67]`) has something to index with (pipeline.md §4.7).

Exact, case-insensitive match against the bank's concept text, with a small hand-curated
correction table for the handful of known transcription differences between the source list and
the bank's own text -- same "flag it, don't fuzzy-match" pattern as every other lookup table in
this project (finding_synonyms.yaml, RadLex's _ADJECTIVE_FORMS). Fails loudly if anything
remains unresolved rather than guessing or silently dropping a concept.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..core.errors import CBMIndexError

_BULLET = re.compile(r"^[•\-\*]\s*")

# Known transcription differences between model_weights/cbm_concepts.md and the actual bank text
# (model_weights/mimic_concepts.csv) -- found because exact-match resolution failed on exactly
# these two of 67 lines, verified by hand against the real bank text.
_TRANSCRIPTION_FIXES: dict[str, str] = {
    "lung volumes low on the left owing to basilar atelectasis":
        "lung volumes low on the left due to basilar atelectasis",
    "transvenous pacer leads ending in the right atrium, right ventricle and left ventricle":
        "transvenous pacer leads ending in the right atrium, right ventricle, and left ventricle",
}


def load_cbm_concept_texts(path: str | Path) -> list[str]:
    """Parse the bulleted concept list, one concept per non-empty line."""
    out = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            out.append(_BULLET.sub("", raw).strip())
    return out


def resolve_cbm_concept_ids(cbm_texts: list[str], bank_concepts: tuple[str, ...]) -> list[int]:
    """Exact case-insensitive match of each CBM concept text against the bank's concept list.
    Raises CBMIndexError naming every text that couldn't be resolved, rather than resolving
    some and silently dropping the rest."""
    index = {text.strip().lower(): i for i, text in enumerate(bank_concepts)}
    ids: list[int] = []
    unresolved: list[str] = []
    for text in cbm_texts:
        key = text.strip().lower()
        if key not in index:
            key = _TRANSCRIPTION_FIXES.get(key, key)
        if key in index:
            ids.append(index[key])
        else:
            unresolved.append(text)
    if unresolved:
        raise CBMIndexError(
            f"{len(unresolved)}/{len(cbm_texts)} CBM concept(s) not found in the 368k bank "
            f"(exact match, case-insensitive): {unresolved!r}"
        )
    if len(set(ids)) != len(ids):
        raise CBMIndexError("resolved CBM concept ids contain a duplicate -- two source lines resolved to the same bank concept")
    return ids


def load_or_resolve_concept_ids(concepts_path: Path, cache_path: Path, bank_concepts: tuple[str, ...]) -> list[int]:
    """Cache the resolution (deterministic and bank-version-pinned) so repeated runs don't
    re-scan the bank's text index every time."""
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("n_bank_concepts") == len(bank_concepts):
            return cached["concept_ids"]
    texts = load_cbm_concept_texts(concepts_path)
    ids = resolve_cbm_concept_ids(texts, bank_concepts)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"n_bank_concepts": len(bank_concepts), "concept_ids": ids, "texts": texts}, indent=2),
        encoding="utf-8",
    )
    return ids
