"""Stage 7 (first part): presentation rules and the template report. Text only; the belief graph is
not changed here. Every sentence keeps the finding id it came from.

  P1 omit_parent        omit a parent when a more specific child is stated with at least the same
                        certainty (present child -> omit present/possible parent; possible child ->
                        omit possible parent)
  P2 pertinent_negative state an absent finding only if kg pertinent_negatives says so
  P3 silent             never mention a finding in the silent band (or one marked reportable: false)
  P4 hedge_possible     possible findings use the KG's hedged phrase
  P5 always_report      the pipeline always returns a report, even for an empty or failed graph
  P6 confident_side     state a side only for lateralisable findings whose side confidence is at least
                        kg side_min_confidence; otherwise state the finding with no side
  P7 no_acute_abnormality  impression "No acute cardiopulmonary abnormality." iff any_abnormality is absent
  P8 zone               lung-parenchyma findings only: state the zone (upper / middle / lower) only together
                        with a stated side, from a valid zone entry on that side; never a lobe
"""
from __future__ import annotations

from . import kg
from .belief import ROOT, BeliefGraph

SIDE_WORD = {"left": "left ", "right": "right ", "bilateral": "bilateral "}


def _cap(t: str) -> str:
    return t[:1].upper() + t[1:] if t else t


def _phrase(f: str, kind: str, side: str | None) -> str:
    t = kg.load_findings()["_by_id"][f]["phrase"][kind]
    s = SIDE_WORD.get(side or "", "")
    return _cap(" ".join(t.replace("{side}", s).replace("{Side}", _cap(s)).split()))


def shown_side(f: str, n) -> str | None:                                   # P6
    kgd = kg.load_findings()
    if not kgd["_by_id"][f]["lateralisable"] or n.side is None:
        return None
    if n.side_conf is None or n.side_conf < kgd.get("side_min_confidence", 0.8):
        return None
    return n.side


ZONE_FINDINGS = {"lung_opacity", "atelectasis", "consolidation", "pneumonia", "lung_lesion"}   # lung parenchyma


def shown_zone(f: str, n, side: str | None) -> str | None:                 # P8
    if f not in ZONE_FINDINGS or side not in ("left", "right"):
        return None
    for a in n.anatomy or []:
        if a.get("zone") and a.get("side") == side and a.get("valid") is not False:
            return a["zone"]
    return None


def pertinent(f: str, g: BeliefGraph) -> str | None:
    for r in kg.load_findings().get("pertinent_negatives", []):
        if r["finding"] != f:
            continue
        w = r["when"]
        if w == "always":
            return r["id"]
        if any(g.nodes.get(x) and g.nodes[x].band == "present" for x in w["any_present"]):
            return r["id"]
    return None


def _show_cases_default() -> bool:
    import yaml
    from pathlib import Path
    f = Path(__file__).resolve().parents[1] / "configs/report.yaml"
    return bool(yaml.safe_load(f.read_text()).get("show_case_evidence", False)) if f.exists() else False


def render(g: BeliefGraph, show_cases: bool | None = None) -> dict:
    """-> {findings, impression, sentences: [{text, finding, band, prob, rule}], omitted: [{finding, rule, reason}]}"""
    try:
        kgd = kg.load_findings()
        sentences, omitted = [], []
        RK = {"possible": 1, "present": 2}
        stated = {f: RK[n.band] for f, n in g.nodes.items() if n.band in RK and kgd["_by_id"][f].get("reportable", True)}
        for f in kg.finding_ids():
            n = g.nodes.get(f)
            if n is None:
                continue
            if n.band == "silent" or not kgd["_by_id"][f].get("reportable", True):
                omitted.append({"finding": f, "rule": "P3", "reason": "silent band" if n.band == "silent" else "not reportable"})
                continue
            if n.band in ("present", "possible"):
                kids = [c for c in kg.descendants(f) if stated.get(c, 0) >= RK[n.band]]
                if kids:
                    omitted.append({"finding": f, "rule": "P1", "reason": f"more specific child stated with at least the same certainty: {kids}"})
                    continue
                side = shown_side(f, n)
                if "{side}" not in kgd["_by_id"][f]["phrase"][n.band].lower():
                    side = None                                  # the KG phrase has no side slot: no side is stated
                zone = shown_zone(f, n, side)
                if zone:                                         # P8: "... in the right lower zone." (side moves into the zone phrase)
                    text = _phrase(f, n.band, None).rstrip(".") + f" in the {side} {zone} zone."
                else:
                    text = _phrase(f, n.band, side)
                sentences.append({"text": text, "finding": f, "band": n.band, "prob": n.prob, "side": side, "zone": zone,
                                  "rule": "P4" if n.band == "possible" else None,
                                  "side_withheld": n.side is not None and side is None})
            elif n.band == "absent":
                rid = pertinent(f, g)
                if rid is None:
                    omitted.append({"finding": f, "rule": "P2", "reason": "absent and not a pertinent negative"})
                    continue
                sentences.append({"text": _phrase(f, "absent", None), "finding": f, "band": "absent", "prob": n.prob,
                                  "rule": f"P2:{rid}"})
        if show_cases is None:
            show_cases = _show_cases_default()
        if show_cases:                                           # C1: presentation only, sentence text unchanged
            from .case_evidence import line
            for s in sentences:
                if s["band"] in ("present", "possible"):
                    s["case_line"] = line(g, s["finding"])
        pos = [s for s in sentences if s["band"] == "present"]
        pss = [s for s in sentences if s["band"] == "possible"]
        neg = [s for s in sentences if s["band"] == "absent"]
        findings = " ".join(s["text"] + (f" {s['case_line']}" if s.get("case_line") else "") for s in pos + pss + neg)
        if g.no_acute_abnormality:                                                       # P7
            impression = _phrase(ROOT, "absent", None)
        elif pos or pss:
            impression = " ".join(s["text"] for s in pos + pss)
        else:
            impression = "No finding meets the reporting threshold."   # not a normal call: the root is not absent
        return {"findings": findings or "No reportable findings.", "impression": impression,
                "sentences": sentences, "omitted": omitted}
    except Exception as exc:  # noqa: BLE001 -- P5: never fail to return a report
        return {"findings": "Automated report unavailable for this study.", "impression": "Please review the images directly.",
                "sentences": [], "omitted": [], "error": f"P5 fallback: {type(exc).__name__}: {exc}"}
