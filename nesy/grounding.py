"""Stage 6 grounding (symbolic). Rules, each with an id used in the audit trail:

  G1 region_to_anatomy   map a localised region name to an anatomy node (kg/anatomy.yaml)
  G2 resolve_part_of     walk part_of to find the lung, the patient side and the zone (upper / middle /
                         lower) of a node. Lobes are never output (user decision 2026-10-07): a single
                         frontal image cannot reliably determine the lobe, so a lobe localisation keeps
                         only its lung and side, and reports and belief graphs state zone and side.
  G3 image_to_patient    convert an image-space x position to the patient's side. Standard frontal
                         display (PA and AP) shows the patient's left on the image right.
  G4 may_occur_in        flag (never delete) a localisation outside the anatomy the KG allows
  G5 ctr                 cardiothoracic ratio from CheXmask; AP views flagged less reliable
  G6 side_from_regions   combine valid lateral localisations into left / right / bilateral
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import kg

# zone of a node, from its own name (zones and CheXmask thirds); lobes deliberately have no zone
ZONE_WORDS = (("apical", "upper"), ("upper", "upper"), ("mid", "middle"), ("middle", "middle"),
              ("lower", "lower"), ("costophrenic", "lower"))
LOBE_WORDS = ("lobe", "lingula")
LUNGS = {"right_lung": "right", "left_lung": "left"}
MIDLINE_BAND = 0.05     # |x - 0.5| below this is treated as midline in G3


@dataclass
class Localisation:
    region: str
    score: float
    node: str | None = None
    zone: str | None = None
    lung: str | None = None
    side: str | None = None
    valid: bool | None = None
    notes: list[str] = field(default_factory=list)


def region_to_anatomy(region: str) -> str | None:                       # G1
    return kg.load_anatomy()["imagenome_region_to_node"].get(region.strip().lower())


def zone_of(name: str) -> str | None:
    """upper / middle / lower from a zone or CheXmask-third name; None for lobes and anything else."""
    n = name.lower()
    if any(w in n for w in LOBE_WORDS):
        return None
    return next((z for w, z in ZONE_WORDS if w in n), None)


def resolve_part_of(node: str) -> dict:                                  # G2
    a = kg.load_anatomy()
    path = kg.anatomy_path(node)
    lung = next((n for n in path if n in LUNGS), None)
    side = LUNGS[lung] if lung else a["_node"][node]["side"]
    return {"path": path, "zone": zone_of(node), "lung": lung, "side": side}


def image_to_patient(x_center: float, view: str | None, horizontally_flipped: bool = False) -> str:   # G3
    """x_center in [0, 1] from the image's left edge. Lateral views have no left/right."""
    if view not in ("PA", "AP"):
        return "unknown"
    if abs(x_center - 0.5) < MIDLINE_BAND:
        return "midline"
    on_image_right = x_center > 0.5
    if horizontally_flipped:
        on_image_right = not on_image_right
    return "left" if on_image_right else "right"


def may_occur_in(finding: str, node: str) -> bool:                       # G4
    rule = kg.load_findings()["_moi"].get(finding)
    if rule is None:
        return True
    return any(a in kg.anatomy_path(node) for a in rule["anatomy"])


def ctr(ctr_value: float | None, view: str | None) -> dict:             # G5
    if ctr_value is None or (isinstance(ctr_value, float) and math.isnan(ctr_value)):
        return {"ctr": None, "reliable": False, "note": "no CheXmask measurement (missing or failed QC)"}
    if view == "PA":
        return {"ctr": float(ctr_value), "reliable": True, "note": "PA view"}
    return {"ctr": float(ctr_value), "reliable": False, "note": f"{view or 'unknown'} view magnifies the heart; CTR is less reliable"}


def ground(finding: str, localisations: list[tuple[str, float]], min_score: float = 0.5) -> dict:
    """G1, G2, G4 and G6 for one finding. Returns the resolved localisations (all kept, invalid ones
    flagged) and the combined side from valid lateral localisations scoring >= min_score."""
    out: list[Localisation] = []
    for region, score in localisations:
        loc = Localisation(region=region, score=float(score))
        loc.node = region_to_anatomy(region)
        if loc.node is None:
            loc.valid = False
            loc.notes.append("G1: region not in anatomy map")
        else:
            r = resolve_part_of(loc.node)
            loc.zone, loc.lung, loc.side = r["zone"], r["lung"], r["side"]
            loc.valid = may_occur_in(finding, loc.node)
            if not loc.valid:
                loc.notes.append(f"G4: {finding} is not expected in {loc.node}")
        out.append(loc)
    sides = {l.side for l in out if l.valid and l.score >= min_score and l.side in ("left", "right")}
    side = "bilateral" if sides == {"left", "right"} else (sides.pop() if sides else None)       # G6
    return {"localisations": out, "side": side}
