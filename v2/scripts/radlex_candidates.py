"""KG task 1, step 1: RadLex candidates for every KG term (finding, zone, anatomy node, device type).
For each term: RadLex classes whose preferred label or a synonym equals the normalised term (exact
candidates), and the top word-overlap candidates otherwise. Non-obsolete classes only. The final choice
(exact / approximate / none) is made by review in kg/radlex_map.yaml; nothing is forced.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from nesy.runlog import Run  # noqa: E402

RADLEX = Path("/scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/data/radlex/raw/Radlex.csv")
KG = Path(__file__).resolve().parents[1] / "kg"
DEVICES = ["chest tube", "mediastinal drain", "pigtail catheter", "endotracheal tube", "tracheostomy tube",
           "peripherally inserted central catheter", "internal jugular central venous catheter", "chest port",
           "subclavian central venous catheter", "pulmonary artery catheter", "intra-aortic balloon pump", "enteric tube",
           "nasogastric tube", "sternotomy wires", "coronary artery bypass graft", "aortic graft", "prosthetic heart valve",
           "cardiac pacemaker", "implantable cardioverter-defibrillator", "central venous catheter", "medical device"]
FINDING_TERMS = {"any_abnormality": ["abnormal", "abnormality"], "lung_opacity": ["lung opacity", "opacity", "pulmonary opacity"],
                 "atelectasis": ["atelectasis"], "consolidation": ["consolidation", "airspace consolidation"],
                 "pneumonia": ["pneumonia"], "edema": ["pulmonary edema", "edema"],
                 "lung_lesion": ["lung nodule", "pulmonary nodule", "lung mass", "nodule", "mass"],
                 "enlarged_cardiomediastinum": ["widened mediastinum", "enlarged cardiomediastinal silhouette", "mediastinal widening"],
                 "cardiomegaly": ["cardiomegaly", "enlarged heart"], "pleural_effusion": ["pleural effusion"],
                 "pleural_other": ["pleural thickening", "pleural plaque"], "pneumothorax": ["pneumothorax"],
                 "fracture": ["fracture", "rib fracture"], "support_devices": ["medical device", "support device"]}
ZONES = ["upper lung zone", "middle lung zone", "lower lung zone", "apical zone", "upper zone", "lower zone"]


def norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(s).lower().replace("oedema", "edema")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("radlex-candidates", vars(args), args.run_dir) as run:
        r = pd.read_csv(RADLEX, usecols=["Class ID", "Preferred Label", "Synonyms", "Obsolete", "Parents"], dtype=str, low_memory=False)
        r = r[r.Obsolete.str.upper() != "TRUE"]
        r["rid"] = r["Class ID"].str.extract(r"(RID\d+)")
        r["parent"] = r.Parents.fillna("").str.extract(r"(RID\d+)")
        lab = dict(zip(r.rid, r["Preferred Label"]))
        names = []
        for rid, pl, syn in zip(r.rid, r["Preferred Label"], r.Synonyms.fillna("")):
            for n in [pl] + [x for x in re.split(r"\|", syn) if x]:
                names.append((norm(n), rid, n, n == pl))
        exact = {}
        for n, rid, raw, pref in names:
            exact.setdefault(n, []).append((rid, raw, pref))
        words = [(set(n.split()), n, rid, raw) for n, rid, raw, _ in names]
        run.log(f"RadLex classes (non-obsolete) {len(r):,}; names incl. synonyms {len(names):,}")
        anat = yaml.safe_load((KG / "anatomy.yaml").read_text())["nodes"]
        terms = [("finding", f, t) for f, ts in FINDING_TERMS.items() for t in ts]
        terms += [("zone", z, z) for z in ZONES]
        terms += [("anatomy", n["id"], n["label"]) for n in anat]
        terms += [("device", d, d) for d in DEVICES]
        out = []
        for kind, key, term in terms:
            q = norm(term)
            ex = exact.get(q, [])
            qs = set(q.split())
            cand = sorted(((len(qs & w) / len(qs | w), n, rid, raw) for w, n, rid, raw in words if qs & w), reverse=True)[:args.top]
            out.append({"kind": kind, "key": key, "term": term,
                        "exact": [{"rid": rid, "name": raw, "preferred": lab.get(rid), "parent": None} for rid, raw, _ in ex],
                        "candidates": [{"rid": rid, "name": raw, "preferred": lab.get(rid), "score": round(s, 2)} for s, n, rid, raw in cand]})
        par = dict(zip(r.rid, r.parent))
        for o in out:
            for e in o["exact"]:
                e["parent"] = f"{par.get(e['rid'])} {lab.get(par.get(e['rid']), '')}"
        (run.dir / "radlex_candidates.json").write_text(json.dumps(out, indent=1))
        for o in out:
            run.log(f"{o['kind']:8s} {o['key']:28s} '{o['term']}': exact {[(e['rid'], e['preferred'], e['parent']) for e in o['exact']]}"
                    + ("" if o["exact"] else f"; top {[(c['rid'], c['name'], c['score']) for c in o['candidates'][:4]]}"))


if __name__ == "__main__":
    main()
