"""Stage 6: decision bands, belief graph and audit trail.

Bands (user decision 2026-10-06), thresholds refitted on the thresh split for each stage's model:
  present   p >= present threshold (PPV >= 0.7, at least 10 true positives)
  possible  p >= possible threshold (PPV >= 0.4) and below present
  absent    p <= absent threshold (<= 5 % of positives missed; pneumothorax <= 2 %); takes priority
            over possible where the two overlap (common findings)
  silent    everything else: not mentioned in the report
A tier that is unreachable for a finding is None, and the finding never enters it.

Decision rules (ids used in the audit):
  D1 band              apply the four bands above; no thresholds or no probability -> silent
  D2 parent_raise      a parent's band is raised to its most positive STATED child (present > possible),
                       along the is_a edge that justifies it. Silent children make no claim and never
                       change a parent (otherwise rare, mostly-silent findings would block every
                       "no acute abnormality").
  D3 no_acute_abnormality  true iff the root node any_abnormality is in its absent band AND every finding
                       on the KG critical list (normal_call.critical, provisional) is in its own absent band
  D4 band_check        fails if a finding has an absent threshold but no val study falls in its absent band
Graph rules:
  B1 one node per finding (never per laterality); side and anatomy are node attributes
  B2 edges come from the KG (is_a, associated_with); none are invented
  B3 every change to a node goes through BeliefGraph.set(), which writes an audit entry naming the rule
     and, where one applies, the KG edge
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from . import kg

RANK = {"absent": 0, "silent": 1, "possible": 2, "present": 3}
ROOT = "any_abnormality"


@dataclass
class Node:
    finding: str
    prob: float | None
    band: str
    side: str | None = None
    side_conf: float | None = None
    anatomy: list[dict] = field(default_factory=list)
    contributions: dict[str, float] = field(default_factory=dict)
    ctr: dict | None = None


@dataclass
class AuditEntry:
    rule: str
    target: str
    field: str
    before: object
    after: object
    reason: str
    edge: str | None = None


class BeliefGraph:
    def __init__(self, study_id):
        self.study_id = study_id
        self.nodes: dict[str, Node] = {}
        self.audit: list[AuditEntry] = []
        kgd = kg.load_findings()
        self.edges = ([{"type": "is_a", **{k: e[k] for k in ("id", "child", "parent")}} for e in kgd.get("is_a", [])]
                      + [{"type": "associated_with", **{k: e[k] for k in ("id", "a", "b")}} for e in kgd.get("associated_with", [])])
        self.no_acute_abnormality: bool | None = None
        self.cases: dict | None = None                          # optional case-based evidence (C1), context only

    def add(self, node: Node, rule: str = "D1", reason: str = ""):
        if node.finding in self.nodes:
            raise ValueError(f"B1: node {node.finding} already exists (one node per finding)")
        self.nodes[node.finding] = node
        self.audit.append(AuditEntry(rule, node.finding, "band", None, node.band, reason or f"p={node.prob}"))

    def set(self, finding: str, fld: str, value, rule: str, reason: str, edge: str | None = None):
        n = self.nodes[finding]
        before = getattr(n, fld)
        if before == value:
            return
        setattr(n, fld, value)
        self.audit.append(AuditEntry(rule, finding, fld, before, value, reason, edge))

    def to_dict(self) -> dict:
        return {"study_id": self.study_id, "nodes": {k: asdict(v) for k, v in self.nodes.items()},
                "edges": self.edges, "no_acute_abnormality": self.no_acute_abnormality, "cases": self.cases,
                "audit": [asdict(a) for a in self.audit]}


def band(p: float | None, thr: dict | None) -> str:                       # D1
    if thr is None or p is None or (isinstance(p, float) and math.isnan(p)):
        return "silent"
    if thr.get("present") is not None and p >= thr["present"]:
        return "present"
    if thr.get("absent") is not None and p <= thr["absent"]:      # absent has priority over possible
        return "absent"
    if thr.get("possible") is not None and p >= thr["possible"]:
        return "possible"
    return "silent"


def raise_parents(g: BeliefGraph):                                        # D2
    edge_id = {(e["child"], e["parent"]): e["id"] for e in g.edges if e["type"] == "is_a"}
    for f in kg.topo_order():                                             # children first, so raises cascade
        if f not in g.nodes:
            continue
        for p in kg.parents_of(f):
            if p not in g.nodes:
                continue
            c, pb = g.nodes[f].band, g.nodes[p].band
            if c in ("present", "possible") and RANK[c] > RANK[pb]:
                g.set(p, "band", c, "D2", f"child {f} is {c}", edge=edge_id[(f, p)])


def derive_no_acute_abnormality(g: BeliefGraph, critical: list[str] | None = None):   # D3
    before = g.no_acute_abnormality
    root = g.nodes.get(ROOT)
    nc = kg.load_findings().get("normal_call") or {}
    crit = nc.get("critical", []) if critical is None else critical
    not_absent = [f for f in crit if f in g.nodes and g.nodes[f].band != "absent"]
    g.no_acute_abnormality = root is not None and root.band == "absent" and not not_absent
    reason = f"{ROOT} band is {root.band if root else 'missing'}"
    if root is not None and root.band == "absent" and not_absent:
        reason += f"; critical finding(s) not absent: {not_absent} (rule {nc.get('id', 'n1')})"
    g.audit.append(AuditEntry("D3", "no_acute_abnormality", "value", before, g.no_acute_abnormality, reason,
                              edge=nc.get("id") if crit else None))


def check_bands(probs: dict[str, list[float]], thresholds: dict[str, dict]) -> dict[str, dict]:   # D4
    """probs: finding -> validation-set probabilities. Raises if a finding has an absent threshold but
    no validation study falls in its absent band."""
    report, bad = {}, []
    for f, ps in probs.items():
        bands = [band(p, thresholds.get(f)) for p in ps]
        counts = {b: bands.count(b) for b in ("present", "possible", "silent", "absent")}
        report[f] = counts
        if (thresholds.get(f) or {}).get("absent") is not None and counts["absent"] == 0:
            bad.append(f)
    if bad:
        raise ValueError(f"D4: empty absent band on validation data for {bad}: {[report[f] for f in bad]}")
    return report


def build(study_id, probs: dict[str, float], thresholds: dict[str, dict], contributions: dict[str, dict] | None = None,
          grounding: dict[str, dict] | None = None, ctr: dict | None = None, critical: list[str] | None = None,
          band_rules: list | None = None) -> BeliefGraph:
    """band_rules: callables rule(g) applied after grounding and before D2/D3. They may change bands
    or withhold sides (with audit entries), never probabilities."""
    g = BeliefGraph(study_id)
    for f in kg.finding_ids():
        p = probs.get(f)
        n = Node(finding=f, prob=None if p is None else float(p), band=band(p, thresholds.get(f)),
                 contributions=(contributions or {}).get(f, {}))
        g.add(n, "D1", f"p={p} thresholds={thresholds.get(f)}")
    for f, gr in (grounding or {}).items():
        if f in g.nodes:
            if gr.get("localisations"):
                g.set(f, "anatomy", [asdict(l) for l in gr["localisations"]], "G4", "grounded localisations (invalid ones flagged, kept)")
            stated = g.nodes[f].band in ("present", "possible")
            if gr.get("side") and stated and kg.load_findings()["_by_id"][f]["lateralisable"]:
                src = "side head P(side | finding)" if gr.get("side_conf") is not None else "valid lateral localisations"
                g.set(f, "side", gr["side"], "G6", f"side from {src}"
                      + (f", p={gr['side_conf']:.2f}" if gr.get("side_conf") is not None else ""))
                if gr.get("side_conf") is not None:
                    g.set(f, "side_conf", float(gr["side_conf"]), "G6", "side-head probability of the stated side")
    if ctr is not None and "cardiomegaly" in g.nodes:
        g.set("cardiomegaly", "ctr", ctr, "G5", ctr.get("note", ""))
    for rule in band_rules or []:
        rule(g)
    raise_parents(g)
    derive_no_acute_abnormality(g, critical)
    return g


def marginals(conditionals: dict[str, "object"]) -> dict[str, "object"]:
    """P(f) = P(f | parent) x P(parent), multiplied down the is_a tree (each finding has at most one
    parent). Works on floats or numpy arrays. A child can never exceed its parent."""
    out = {}
    for f in kg.topo_order()[::-1]:                                        # parents first
        ps = kg.parents_of(f)
        if len(ps) > 1:
            raise ValueError(f"{f} has several is_a parents; the factorised head assumes a tree")
        out[f] = conditionals[f] * (out[ps[0]] if ps else 1.0)
    return out
