"""Item 12: build kg/anatomy.yaml, the part_of tree for the 36 Chest ImaGenome regions.

part_of comes from RadLex 4.3 (old pipeline's parsed snapshot) wherever RadLex has a class for the
region; ImaGenome zones, hila and sided costophrenic angles have no RadLex class and are attached by
hand to the closest RadLex structure (`source: manual`). Every node carries `side` (patient side:
left / right / midline / bilateral) so grounding can resolve lobe -> lung -> side through part_of.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from nesy.runlog import Run  # noqa: E402

SNAPSHOT = Path("/scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/data/radlex/raw/radlex_snapshot.json.gz")

# node id -> (label, RadLex label to look up or None, manual parent or None, side)
NODES = {
    "thorax": ("thorax", "thorax", None, "midline"),
    "lungs": ("lungs", "lung", "thorax", "bilateral"),
    "right_lung": ("right lung", "right lung", "lungs", "right"),
    "left_lung": ("left lung", "left lung", "lungs", "left"),
    "right_upper_lobe": ("upper lobe of right lung", "upper lobe of right lung", None, "right"),
    "right_middle_lobe": ("middle lobe of right lung", "middle lobe of lung", None, "right"),
    "right_lower_lobe": ("lower lobe of right lung", "lower lobe of right lung", None, "right"),
    "left_upper_lobe": ("upper lobe of left lung", "upper lobe of left lung", None, "left"),
    "lingula": ("lingula", "lingula", None, "left"),
    "left_lower_lobe": ("lower lobe of left lung", "lower lobe of left lung", None, "left"),
    # ImaGenome zones: radiographic zones, not lobes -> manual, part of the lung on that side
    "right_apical_zone": ("right apical zone", None, "right_lung", "right"),
    "right_upper_lung_zone": ("right upper lung zone", None, "right_lung", "right"),
    "right_mid_lung_zone": ("right mid lung zone", None, "right_lung", "right"),
    "right_lower_lung_zone": ("right lower lung zone", None, "right_lung", "right"),
    "left_apical_zone": ("left apical zone", None, "left_lung", "left"),
    "left_upper_lung_zone": ("left upper lung zone", None, "left_lung", "left"),
    "left_mid_lung_zone": ("left mid lung zone", None, "left_lung", "left"),
    "left_lower_lung_zone": ("left lower lung zone", None, "left_lung", "left"),
    "right_hilar_structures": ("right hilar structures", None, "right_lung", "right"),
    "left_hilar_structures": ("left hilar structures", None, "left_lung", "left"),
    "pleura": ("pleura", "pleura", None, "bilateral"),
    "right_costophrenic_angle": ("right costophrenic angle", None, "pleura", "right"),
    "left_costophrenic_angle": ("left costophrenic angle", None, "pleura", "left"),
    "diaphragm": ("diaphragm", "diaphragm", None, "bilateral"),
    "right_hemidiaphragm": ("right hemidiaphragm", "right hemidiaphragm", None, "right"),
    "left_hemidiaphragm": ("left hemidiaphragm", "left hemidiaphragm", None, "left"),
    "mediastinum": ("mediastinum", "mediastinum", None, "midline"),
    "upper_mediastinum": ("upper mediastinum", "superior mediastinum", "mediastinum", "midline"),
    "heart": ("heart", "heart", "mediastinum", "midline"),
    "cardiac_silhouette": ("cardiac silhouette", None, "heart", "midline"),
    "right_cardiac_silhouette": ("right cardiac silhouette", None, "cardiac_silhouette", "midline"),
    "left_cardiac_silhouette": ("left cardiac silhouette", None, "cardiac_silhouette", "midline"),
    "right_cardiophrenic_angle": ("right cardiophrenic angle", None, "cardiac_silhouette", "midline"),
    "left_cardiophrenic_angle": ("left cardiophrenic angle", None, "cardiac_silhouette", "midline"),
    "right_atrium": ("right atrium", "right atrium", "heart", "midline"),
    "cavoatrial_junction": ("cavoatrial junction", None, "mediastinum", "midline"),
    "svc": ("superior vena cava", "superior vena cava", "mediastinum", "midline"),
    "trachea": ("trachea", "trachea", "mediastinum", "midline"),
    "carina": ("carina", "carina", None, "midline"),
    "aortic_arch": ("aortic arch", "aortic arch", "mediastinum", "midline"),
    "descending_aorta": ("descending aorta", "descending aorta", "mediastinum", "midline"),
    "chest_wall": ("chest wall", "chest wall", None, "bilateral"),
    "right_clavicle": ("right clavicle", "right clavicle", "chest_wall", "right"),
    "left_clavicle": ("left clavicle", "left clavicle", "chest_wall", "left"),
    "spine": ("spine", "thoracic vertebral column", "chest_wall", "midline"),
    "abdomen": ("abdomen", "abdomen", None, "midline"),
    "right_upper_abdomen": ("right upper abdomen", None, "abdomen", "right"),
    "left_upper_abdomen": ("left upper abdomen", None, "abdomen", "left"),
}
# RadLex part_of edges rejected on review: aortic arch -> descending aorta (RadLex lists both as parts
# of each other's system); attach the arch to the mediastinum instead.
OVERRIDE = {"aortic_arch": "mediastinum"}
IMAGENOME_REGION = {v[0]: k for k, v in NODES.items()}
IMAGENOME_REGION.update({"right lung": "right_lung", "left lung": "left_lung", "svc": "svc", "spine": "spine"})


def main():
    with Run("build-anatomy", {"snapshot": str(SNAPSHOT)}) as run:
        C = json.load(gzip.open(SNAPSHOT, "rt"))["classes"]
        by_label = {}
        for rid, v in C.items():
            for s in [v["label"]] + v.get("synonyms", []):
                by_label.setdefault(s.lower(), rid)
        rid_of = {k: by_label.get(v[1]) if v[1] else None for k, v in NODES.items()}
        node_of_rid = {r: k for k, r in rid_of.items() if r}
        out, n_radlex, n_manual = [], 0, 0
        for k, (label, rl, manual_parent, side) in NODES.items():
            rid = rid_of[k]
            parent, source = None, None
            if rid:
                # first RadLex part_of ancestor that is one of our nodes (walk up to 4 levels)
                frontier, seen = list(C[rid]["part_of"]), set()
                for _ in range(4):
                    hit = next((node_of_rid[p] for p in frontier if p in node_of_rid and node_of_rid[p] != k), None)
                    if hit:
                        parent, source = hit, "radlex"
                        break
                    nxt = []
                    for p in frontier:
                        if p in C and p not in seen:
                            seen.add(p)
                            nxt += C[p]["part_of"]
                    frontier = nxt
            if k in OVERRIDE:
                parent, source = OVERRIDE[k], "manual_override"
            if parent is None and manual_parent:
                parent, source = manual_parent, "manual"
            n_radlex += source == "radlex"
            n_manual += source == "manual"
            out.append({"id": k, "label": label, "radlex_id": rid, "side": side, "part_of": parent,
                        "part_of_source": source or "root"})
            run.log(f"  {k:26s} radlex {rid or '-':9s} part_of {parent or '-':22s} ({source or 'root'})")
        doc = {"version": "0.1", "source": "RadLex 4.3 part_of where available; manual for ImaGenome-only regions",
               "nodes": out, "imagenome_region_to_node": dict(sorted(IMAGENOME_REGION.items()))}
        p = Path(__file__).resolve().parents[1] / "kg/anatomy.yaml"
        p.write_text("# GENERATED by scripts/build_anatomy.py; edit there, not here.\n" + yaml.safe_dump(doc, sort_keys=False))
        run.log(f"wrote {p}: {len(out)} nodes, part_of from RadLex {n_radlex}, manual {n_manual}")


if __name__ == "__main__":
    main()
