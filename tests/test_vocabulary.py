import csv

import pytest

from rrg.concepts.vocabulary import clean_text, load_concept_texts
from rrg.core.errors import BankLoadError


def _write(path, rows, header=("concept", "concept_idx")):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def test_loads_in_order_and_keeps_text_verbatim(tmp_path):
    p = tmp_path / "c.csv"
    _write(p, [("no pneumothorax", 0), ("line one\nline two", 1), ('says "hi", ok', 2)])
    texts = load_concept_texts(p, expected_n=3)
    assert texts == ("no pneumothorax", "line one\nline two", 'says "hi", ok')
    assert clean_text(texts[1]) == "line one line two"


def test_row_order_must_match_concept_idx(tmp_path):
    p = tmp_path / "c.csv"
    _write(p, [("a", 0), ("b", 2)])
    with pytest.raises(BankLoadError, match="ordered 0..N-1"):
        load_concept_texts(p)


@pytest.mark.parametrize("rows,kw,msg", [([("a", 0)], dict(expected_n=2), "expected 2"), ([("a", "x")], {}, "non-integer")])
def test_count_and_index_validation(tmp_path, rows, kw, msg):
    p = tmp_path / "c.csv"
    _write(p, rows)
    with pytest.raises(BankLoadError, match=msg):
        load_concept_texts(p, **kw)


def test_wrong_header_and_missing_file(tmp_path):
    p = tmp_path / "c.csv"
    _write(p, [("a", 0)], header=("text", "id"))
    with pytest.raises(BankLoadError, match="header"):
        load_concept_texts(p)
    with pytest.raises(BankLoadError, match="not found"):
        load_concept_texts(tmp_path / "missing.csv")


@pytest.mark.assets
def test_real_vocabulary(settings, need_assets):
    texts = load_concept_texts(settings.paths.concepts_csv, settings.concept_bank.expected_n_concepts)
    assert len(texts) == 368294 and len(set(texts)) == 368294
    assert texts[0].startswith("et tube and left subclavian catheter")
