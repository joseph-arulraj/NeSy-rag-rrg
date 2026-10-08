"""Belief graph for one study from the head's predictions plus Stage 4 region evidence (CheXmask set B).

Knowledge-based band rules (each optional, each changes bands or sides only, never probabilities):
  R1 localisation_support  a present finding whose relevant regions all score below the finding's
                           region threshold is demoted to possible
  R2 side_agreement        a side is stated only when the side head and the region-derived side agree
Zone (always on, rule G7): for lateralisable parenchymal findings with a side, the highest-scoring third on
that side is attached as the zone if it scores >= the region threshold (zone and side only, never a lobe).
"""
from __future__ import annotations

import numpy as np

from . import belief as B, kg
from . import region_support as RS

PARENCHYMAL = {"lung_opacity", "atelectasis", "consolidation", "pneumonia", "lung_lesion"}


def side_grounding(row: dict, findings: list[str]) -> dict:
    """Side head output (pside_<f>_<class> columns) -> grounding dicts."""
    out = {}
    for f in findings:
        vals = {k.rsplit("_", 1)[1]: v for k, v in row.items() if k.startswith(f"pside_{f}_") and v is not None
                and not (isinstance(v, float) and np.isnan(v))}
        if vals:
            side = max(vals, key=vals.get)
            out[f] = {"localisations": [], "side": side, "side_conf": float(vals[side])}
    return out


def make_rules(ev: dict[str, dict], region_thr: dict[str, float], use_r1: bool, use_r2: bool):
    def r1(g):
        for f, e in ev.items():
            n = g.nodes.get(f)
            thr = region_thr.get(f)
            if n is None or n.band != "present" or thr is None or e["support"] is None:
                continue
            if e["support"] < thr:
                g.set(f, "band", "possible", "R1", f"localisation support: max region score {e['support']:.2f} < {thr:.2f}")

    def r2(g):
        for f, e in ev.items():
            n = g.nodes.get(f)
            if n is None or n.side is None:
                continue
            rs = RS.region_side(e, region_thr.get(f))
            if rs != n.side:
                g.set(f, "side", None, "R2", f"side head says {n.side}, regions say {rs}: side withheld")
                g.set(f, "side_conf", None, "R2", "side withheld")

    def g7_zone(g):
        for f, e in ev.items():
            n = g.nodes.get(f)
            thr = region_thr.get(f)
            if (n is None or f not in PARENCHYMAL or n.band not in ("present", "possible") or n.side not in ("left", "right")
                    or thr is None or n.side not in e["zone"]):
                continue
            z, s = e["zone"][n.side]
            if s >= thr:
                g.set(f, "anatomy", [{"zone": z, "side": n.side, "score": round(s, 3), "source": "stage4 CheXmask regions"}],
                      "G7", f"zone from the highest-scoring {n.side} third ({s:.2f} >= {thr:.2f})")

    rules = []
    if use_r1:
        rules.append(r1)
    if use_r2:
        rules.append(r2)
    rules.append(g7_zone)
    return rules


def build(study_id, row: dict, thr: dict, ev: dict[str, dict], region_thr: dict[str, float],
          use_r1: bool = False, use_r2: bool = False, ctr: dict | None = None):
    findings = kg.finding_ids()
    probs = {f: row.get(f"p_{f}") for f in findings}
    return B.build(study_id, probs, thr, grounding=side_grounding(row, findings), ctr=ctr,
                   band_rules=make_rules(ev, region_thr, use_r1, use_r2))
