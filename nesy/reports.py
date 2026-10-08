"""MIMIC-CXR report text, read in place from mimic-cxr-reports.zip, split into sections.

Only FINDINGS and IMPRESSION describe the image. INDICATION/HISTORY/COMPARISON/TECHNIQUE are
excluded from anything used as a label or a fact (AUDIT.md defect 11). Reports with neither section
fall back to the text after the last non-clinical header, and are flagged `section_source="fallback"`.
"""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass

from . import paths as P

REPORTS_ZIP = P.DATA / "mimic-cxr-reports.zip"

# A header is an upper-case phrase at line start followed by a colon.
_HEADER = re.compile(r"^[ \t]*([A-Z][A-Z /&\-]{2,40}):", re.M)
_FINDINGS = {"FINDINGS", "FINDING", "FINDINGS AND IMPRESSION", "FINDINGS/IMPRESSION", "IMPRESSION AND FINDINGS"}
_IMPRESSION = {"IMPRESSION", "IMPRESSIONS", "CONCLUSION", "SUMMARY"}
_NON_CLINICAL = {"INDICATION", "INDICATIONS", "HISTORY", "CLINICAL HISTORY", "REASON FOR EXAMINATION", "REASON FOR EXAM",
                 "COMPARISON", "COMPARISONS", "TECHNIQUE", "EXAMINATION", "EXAM", "PROCEDURE", "CLINICAL INFORMATION",
                 "NOTIFICATION", "RECOMMENDATION", "RECOMMENDATIONS", "WET READ", "ADDENDUM", "FINAL REPORT"}
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Sections:
    findings: str
    impression: str
    source: str          # "sections" | "findings_only" | "impression_only" | "fallback" | "empty"

    @property
    def text(self) -> str:
        return " ".join(t for t in (self.findings, self.impression) if t)


def clean(t: str) -> str:
    return _WS.sub(" ", t).strip()


def split_sections(report: str) -> Sections:
    heads = [(m.start(), m.end(), m.group(1).strip()) for m in _HEADER.finditer(report)]
    find, imp = [], []
    for i, (s, e, name) in enumerate(heads):
        body = report[e: heads[i + 1][0] if i + 1 < len(heads) else len(report)]
        if name in _FINDINGS:
            find.append(body)
        elif name in _IMPRESSION:
            imp.append(body)
    f, im = clean(" ".join(find)), clean(" ".join(imp))
    if f and im:
        return Sections(f, im, "sections")
    if f:
        return Sections(f, "", "findings_only")
    if im:
        return Sections("", im, "impression_only")
    # fallback: text after the last non-clinical header (or the whole report if there is none)
    last = max((e for s, e, n in heads if n in _NON_CLINICAL), default=0)
    tail = clean(report[last:])
    tail = re.sub(r"^FINAL REPORT\s*", "", tail)
    return Sections(tail, "", "fallback") if tail else Sections("", "", "empty")


class ReportZip:
    def __init__(self, path=REPORTS_ZIP):
        self.zip = zipfile.ZipFile(path)

    def raw(self, subject_id: int, study_id: int) -> str:
        sid = str(subject_id)
        return self.zip.read(f"files/p{sid[:2]}/p{sid}/s{study_id}.txt").decode("utf-8", errors="replace")

    def sections(self, subject_id: int, study_id: int) -> Sections:
        return split_sections(self.raw(subject_id, study_id))
