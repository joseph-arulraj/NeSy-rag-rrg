"""The ordered 368,294-concept vocabulary (pure stdlib, no torch).

Row order defines ConceptID: concept i of this file is row i of the embedding matrix.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional

from ..core.errors import BankLoadError

_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Whitespace-collapsed form for display / matching. The bank stores the text verbatim."""
    return _WS.sub(" ", text).strip()


def load_concept_texts(csv_path: Path, expected_n: Optional[int] = None) -> tuple[str, ...]:
    """Read mimic_concepts.csv (columns: concept, concept_idx) and verify concept_idx == row order."""
    path = Path(csv_path)
    if not path.is_file():
        raise BankLoadError(f"concept vocabulary not found: {path}")
    texts: list[str] = []
    # newline="" is the csv-module convention for files that may contain quoted line breaks
    # (none in the released file, but this keeps the reader correct if one ever appears).
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "concept" not in reader.fieldnames or "concept_idx" not in reader.fieldnames:
            raise BankLoadError(
                f"{path}: expected header columns 'concept' and 'concept_idx', got {reader.fieldnames}"
            )
        for row_number, row in enumerate(reader):
            try:
                idx = int(row["concept_idx"])
            except (TypeError, ValueError) as exc:
                raise BankLoadError(f"{path}: row {row_number} has a non-integer concept_idx {row['concept_idx']!r}") from exc
            if idx != row_number:
                raise BankLoadError(
                    f"{path}: concept_idx {idx} at row {row_number}; the file must be ordered 0..N-1 "
                    "because row order defines ConceptID"
                )
            texts.append(row["concept"])
    if expected_n is not None and len(texts) != expected_n:
        raise BankLoadError(f"{path}: {len(texts)} concepts, expected {expected_n}")
    return tuple(texts)
