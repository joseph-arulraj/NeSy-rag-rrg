"""KG task 1: RadLex IDs for every finding, zone, anatomy node and device type, from reviewed choices
(candidates: runs/*_radlex-candidates). match: exact (RadLex preferred label or a synonym names the same
concept), approximate (closest RadLex class is broader, narrower or differently scoped; the note says how),
none (no suitable class; nothing forced). RadLex labels and is_a parents are looked up from the CSV, never
typed. Writes kg/radlex_map.yaml and kg/devices.yaml. Every approximate / none row needs radiologist review.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from nesy.runlog import Run  # noqa: E402

RADLEX = Path("/scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/data/radlex/raw/Radlex.csv")
KG = Path(__file__).resolve().parents[1] / "kg"

FINDINGS = {  # finding: (rid or None, match, note)
    "any_abnormality": ("RID28453", "approximate", "RadLex 'abnormal' is a normality descriptor, not a finding class; our root is 'any pathology present'."),
    "lung_opacity": ("RID28530", "approximate", "RadLex 'opacity' is not restricted to the lung; ours is lung opacity."),
    "atelectasis": ("RID28493", "exact", ""),
    "consolidation": ("RID43255", "exact", ""),
    "pneumonia": ("RID5350", "exact", ""),
    "edema": ("RID4866", "exact", "pulmonary edema"),
    "lung_lesion": ("RID50149", "approximate", "Ours = nodule OR mass; RadLex separates pulmonary nodule (RID50149) and lung mass (RID39056)."),
    "enlarged_cardiomediastinum": (None, "none", "No RadLex class for an enlarged or widened cardiomediastinal silhouette."),
    "cardiomegaly": (None, "none", "No RadLex class for cardiomegaly or an enlarged heart (RadLex 4.3)."),
    "pleural_effusion": ("RID34539", "exact", ""),
    "pleural_other": ("RID43269", "approximate", "RadLex 'pleural plaque' is narrower; ours also covers pleural thickening and scarring (no RadLex class)."),
    "pneumothorax": ("RID5352", "exact", ""),
    "fracture": ("RID4650", "exact", "Ours is any visible fracture (rib, clavicle, spine)."),
    "support_devices": ("RID29033", "approximate", "RadLex 'medical device' is broader; ours = tubes, lines and devices on a chest radiograph."),
}
ZONES = {  # generic zone used in reports (P8): (rid, match, note)
    "upper": ("RID1348", "exact", "upper lung zone"), "middle": ("RID1351", "exact", "mid lung zone"),
    "lower": ("RID1354", "exact", "lower lung zone"),
    "apical": ("RID28584", "approximate", "RadLex 'apex of lung'; we fold the apical zone into 'upper' in reports"),
}
ANATOMY_NEW = {  # anatomy nodes without a RadLex ID in anatomy.yaml
    "right_upper_lung_zone": ("RID1350", "exact", ""), "right_mid_lung_zone": ("RID1353", "exact", ""),
    "right_lower_lung_zone": ("RID1356", "exact", ""), "left_upper_lung_zone": ("RID1349", "exact", ""),
    "left_mid_lung_zone": ("RID1352", "exact", ""), "left_lower_lung_zone": ("RID1355", "exact", ""),
    "right_apical_zone": ("RID28584", "approximate", "RadLex 'apex of lung' has no side"),
    "left_apical_zone": ("RID28584", "approximate", "RadLex 'apex of lung' has no side"),
    "right_hilar_structures": ("RID34566", "approximate", "RadLex 'pulmonary hilum' has no side"),
    "left_hilar_structures": ("RID34566", "approximate", "RadLex 'pulmonary hilum' has no side"),
    "right_costophrenic_angle": ("RID1536", "approximate", "RadLex 'right costophrenic sulcus' (anatomical sulcus; the radiographic angle)"),
    "left_costophrenic_angle": ("RID1535", "approximate", "RadLex 'left costophrenic sulcus'"),
    "right_cardiophrenic_angle": ("RID1533", "approximate", "RadLex 'right cardiophrenic sulcus'"),
    "left_cardiophrenic_angle": ("RID1532", "approximate", "RadLex 'left cardiophrenic sulcus'"),
    "cardiac_silhouette": (None, "none", "radiographic projection, no RadLex class"),
    "right_cardiac_silhouette": (None, "none", "radiographic projection, no RadLex class"),
    "left_cardiac_silhouette": (None, "none", "radiographic projection, no RadLex class"),
    "cavoatrial_junction": (None, "none", "RadLex has only the descriptor 'cavoatrial'"),
    "right_upper_abdomen": ("RID29994", "approximate", "RadLex 'right upper quadrant of abdomen'"),
    "left_upper_abdomen": ("RID29995", "approximate", "RadLex 'left upper quadrant of abdomen'"),
}
ANATOMY_EXISTING_APPROX = {"lungs": "RadLex 'lung' (single organ); ours is both lungs",
                           "upper_mediastinum": "RadLex 'superior mediastinum'"}
DEVICES = {  # Chest ImaGenome device / tubesandlines label: (our id, rid, match, note)
    "chest tube": ("chest_tube", "RID5573", "exact", "RadLex 'thoracostomy tube' (synonym chest tube)"),
    "mediastinal drain": ("mediastinal_drain", "RID5571", "approximate", "RadLex 'mediastinal tube'"),
    "pigtail catheter": ("pigtail_catheter", "RID5583", "exact", ""),
    "endotracheal tube": ("endotracheal_tube", "RID5557", "exact", ""),
    "tracheostomy tube": ("tracheostomy_tube", "RID5560", "exact", ""),
    "picc": ("picc", "RID5581", "approximate", "RadLex preferred label 'peripheral intravenous central catheter' (synonym PICC)"),
    "ij line": ("internal_jugular_line", "RID5578", "approximate", "RadLex 'central venous catheter' (no access-site class)"),
    "chest port": ("chest_port", None, "none", "RadLex 'port' classes are not the implanted venous port"),
    "subclavian line": ("subclavian_line", "RID5578", "approximate", "RadLex 'central venous catheter' (no access-site class)"),
    "swan-ganz catheter": ("pulmonary_artery_catheter", "RID5584", "exact", ""),
    "intra-aortic balloon pump": ("iabp", "RID5587", "exact", ""),
    "enteric tube": ("enteric_tube", "RID5561", "approximate", "RadLex 'feeding tube' (enteric tubes include drainage tubes)"),
    "sternotomy wires": ("sternotomy_wires", None, "none", "no RadLex class"),
    "cabg grafts": ("cabg_grafts", None, "none", "no RadLex device class for coronary bypass grafts"),
    "aortic graft/repair": ("aortic_graft", None, "none", "RadLex 'graft' is too general"),
    "prosthetic valve": ("prosthetic_valve", "RID5550", "exact", "RadLex 'prosthetic cardiac valve'"),
    "cardiac pacer and wires": ("pacemaker", "RID5436", "approximate", "RadLex 'pacemaker' (leads not separate)"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    a = ap.parse_args()
    with Run("build-radlex-map", vars(a), a.run_dir) as run:
        r = pd.read_csv(RADLEX, usecols=["Class ID", "Preferred Label", "Obsolete", "Parents"], dtype=str, low_memory=False)
        r["rid"] = r["Class ID"].str.extract(r"(RID\d+)")
        r = r.drop_duplicates("rid").set_index("rid")
        r["parent"] = r.Parents.fillna("").str.extract(r"(RID\d+)")[0]

        def row(kind, key, rid, match, note):
            if rid is not None:
                if rid not in r.index:
                    raise ValueError(f"{key}: {rid} not in RadLex")
                if str(r.at[rid, "Obsolete"]).upper() == "TRUE":
                    raise ValueError(f"{key}: {rid} is obsolete")
            par = r.at[rid, "parent"] if rid else None
            return {"kind": kind, "key": key, "radlex_id": rid, "radlex_label": (r.at[rid, "Preferred Label"] if rid else None),
                    "radlex_parent": (f"{par} {r.at[par, 'Preferred Label']}" if isinstance(par, str) and par in r.index else None),
                    "match": match, "note": note, "source": (f"RadLex 4.3 {rid}" if rid else "author-curated, needs review"),
                    "review": "needed" if match != "exact" else "not needed"}
        rows = [row("finding", k, *v) for k, v in FINDINGS.items()] + [row("zone", k, *v) for k, v in ZONES.items()]
        anat = yaml.safe_load((KG / "anatomy.yaml").read_text())["nodes"]
        for n in anat:
            if n["id"] in ANATOMY_NEW:
                rows.append(row("anatomy", n["id"], *ANATOMY_NEW[n["id"]]))
            elif n.get("radlex_id"):
                rid = n["radlex_id"]
                same = str(r.at[rid, "Preferred Label"]).lower() == n["label"].lower()
                m, note = ("exact", "") if same and n["id"] not in ANATOMY_EXISTING_APPROX else ("approximate", ANATOMY_EXISTING_APPROX.get(n["id"], f"RadLex '{r.at[rid, 'Preferred Label']}'"))
                rows.append(row("anatomy", n["id"], rid, m, note))
            else:
                raise ValueError(f"anatomy node {n['id']} has no decision")
        devs = []
        for lab, (did, rid, m, note) in DEVICES.items():
            rr = row("device", did, rid, m, note)
            rows.append(rr)
            devs.append({"id": did, "imagenome_label": lab, "finding": "support_devices", "radlex_id": rid, "match": m,
                         "source": f"Chest ImaGenome semantics/attribute_relations_v1.txt ({lab}); " + rr["source"],
                         "review": rr["review"]})
        (KG / "radlex_map.yaml").write_text("# GENERATED by v2/scripts/build_radlex_map.py from reviewed choices; edit there.\n"
                                            + yaml.safe_dump({"version": "0.1", "radlex_version": "4.3", "entries": rows}, sort_keys=False, allow_unicode=True))
        (KG / "devices.yaml").write_text("# GENERATED by v2/scripts/build_radlex_map.py. Device types (Chest ImaGenome labels), all is_a support_devices.\n"
                                         + yaml.safe_dump({"version": "0.1", "device_types": devs}, sort_keys=False, allow_unicode=True))
        c = pd.DataFrame(rows).groupby(["kind", "match"]).size().unstack(fill_value=0)
        c.to_csv(run.dir / "match_counts.csv")
        run.log("match counts:\n" + c.to_string())
        for x in rows:
            run.log(f"  {x['kind']:8s} {x['key']:28s} {x['match']:11s} {x['radlex_id']} {x['radlex_label']} (parent {x['radlex_parent']})")


if __name__ == "__main__":
    main()
