"""Case-based evidence (user item B, 2026-10-07), optional. Attaches to a belief graph the k most similar
fit-split studies (same patient excluded; scripts/case_neighbours.py) and, per finding, how many of them are
positive. Supporting context only: it never changes a probability or a band (rule C1, recorded in the audit
trail). Only study IDs and labels are stored, never report text. Not sent to the language model.
"""
from __future__ import annotations

from . import kg


def attach(g, row: dict, k: int = 5):
    """row: one line of data/features/cases_<variant>_k<k>.parquet as a dict."""
    counts = {f: {"positive": int(row[f"case_pos_{f}"]), "known": int(row[f"case_known_{f}"])}
              for f in kg.finding_ids() if f"case_pos_{f}" in row}
    g.cases = {"k": k, "neighbours": [{"study_id": int(s), "similarity": float(v)} for s, v in zip(row["nb_study_ids"], row["nb_sims"])],
               "counts": counts}
    from .belief import AuditEntry
    g.audit.append(AuditEntry(rule="C1", target="graph", field="cases", before=None, after=f"{k} similar fit-split cases",
                              reason="case-based evidence attached as supporting context only; no probability or band was changed"))
    return g


def line(g, f: str) -> str | None:
    c = getattr(g, "cases", None)
    if not c or f not in c["counts"]:
        return None
    return f"(seen in {c['counts'][f]['positive']} of {c['k']} similar cases)"
