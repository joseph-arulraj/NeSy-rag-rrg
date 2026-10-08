"""Explicit versus default side and region labels in Chest ImaGenome (user point 3, 2026-10-06).

ImaGenome attaches each report phrase's findings to regions. When the phrase states no side or
location, the finding is attached to default regions (e.g. both lungs), which is why edema came out
"bilateral" in 100 % of positives. Here each phrase is re-read with its attributes:

  side-explicit phrase      contains left / right / bilateral / both / bibasilar / biapical / ...
  location-explicit phrase  contains a location word (upper, mid, lower, apex/apical, base/basal/
                            basilar, hilar/perihilar, retrocardiac, lobe, lingula, costophrenic,
                            zone, lateral, medial, paratracheal, subpulmonic, ...)

Outputs (data/features/):
  imagenome_side_explicit_v1.parquet   per (image, finding) positive: side stated in the text
                                       (left / right / bilateral) or 'default' (masked for training)
  imagenome_region_explicit_v1.parquet per (image, region, finding) positive: explicit flag (the
                                       region was named by location words, and by side if the
                                       region is sided). Non-explicit positives are masked, never
                                       treated as positive.
Per-finding counts of explicit vs default go to the log and metrics.
"""
from __future__ import annotations

import json
import re
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from nesy import manifests as M, paths as P  # noqa: E402
from nesy.runlog import Progress, Run  # noqa: E402

ZIP = P.IMAGENOME / "silver_dataset/scene_graph.zip"
MAP = yaml.safe_load((P.V2 / "kg/imagenome_map.yaml").read_text())
LABEL_TO_F: dict[str, list[str]] = {}
for f, spec in MAP["map"].items():
    for lab in spec.get("labels", []):
        LABEL_TO_F.setdefault(lab, []).append(f)
CAT_TO_F = {c: [f] for f, spec in MAP["map"].items() for c in spec.get("categories", [])}
LEFT = re.compile(r"\bleft\b|\blt\b", re.I)
RIGHT = re.compile(r"\bright\b|\brt\b", re.I)
BILAT = re.compile(r"\b(bilateral|bilaterally|both|bibasilar|bibasal|biapical|bilobar|lungs)\b", re.I)
LOC = re.compile(r"\b(upper|mid|middle|lower|apex|apices|apical|base|bases|basal|basilar|bibasilar|biapical|hilar|hila|hilum|"
                 r"perihilar|retrocardiac|lobe|lobes|lobar|lingula|lingular|costophrenic|zone|zones|lateral|medial|"
                 r"paratracheal|subpulmonic|peripheral|central|infrahilar|suprahilar|paramediastinal|subsegmental)\b", re.I)
_zip = None


def _init():
    global _zip
    _zip = zipfile.ZipFile(ZIP)


def phrase_side(p: str) -> str | None:
    l, r, b = bool(LEFT.search(p)), bool(RIGHT.search(p)), bool(BILAT.search(p))
    if b or (l and r):
        return "bilateral"
    if l:
        return "left"
    if r:
        return "right"
    return None


def region_side(region: str) -> str | None:
    return "left" if region.startswith("left ") else "right" if region.startswith("right ") else None


def parse(member: str):
    d = json.loads(_zip.read(member))
    did = d["image_id"]
    side_rows, reg_rows = {}, {}
    for a in d.get("attributes", []):
        region = a.get("bbox_name")
        for attrs, phrase in zip(a.get("attributes", []), a.get("phrases", [])):
            phrase = str(phrase)
            ps, ploc = phrase_side(phrase), bool(LOC.search(phrase))
            fs = set()
            for s in attrs:
                parts = s.split("|")
                if len(parts) != 3 or parts[1] != "yes":
                    continue
                cat, _, lab = parts
                fs.update(LABEL_TO_F.get(lab, []))
                fs.update(CAT_TO_F.get(cat, []))
            for f in fs:
                # side per (image, finding): explicit if any supporting phrase states a side
                cur = side_rows.setdefault(f, set())
                cur.add(ps or "default")
                rs = region_side(region)
                explicit = ploc and (rs is None or ps in (rs, "bilateral"))
                k = (region, f)
                reg_rows[k] = reg_rows.get(k, False) or explicit
    sides = []
    for f, ss in side_rows.items():
        stated = ss - {"default"}
        side = ("bilateral" if ("bilateral" in stated or stated == {"left", "right"}) else stated.pop()) if stated else "default"
        sides.append({"dicom_id": did, "finding": f, "side_text": side})
    regs = [{"dicom_id": did, "region": r, "finding": f, "explicit": e} for (r, f), e in reg_rows.items()]
    return sides, regs


def main():
    with Run("imagenome-explicit", {"zip": str(ZIP)}) as run:
        man = M.load("imagenome_silver", columns=["dicom_id", "zip_member", "subject_id", "study_id"])
        members = man.zip_member.tolist()
        prog = Progress(run, len(members), "scene graphs", every_s=60)
        S, R = [], []
        with Pool(16, initializer=_init) as pool:
            for i, (s, r) in enumerate(pool.imap(parse, members, chunksize=200), 1):
                S += s
                R += r
                if i % 20000 == 0:
                    prog.update(i)
        prog.update(len(members), force=True)
        side = pd.DataFrame(S).merge(man[["dicom_id", "subject_id", "study_id", "split"]], on="dicom_id")
        reg = pd.DataFrame(R).merge(man[["dicom_id", "split"]], on="dicom_id")
        side.to_parquet(P.FEATURES / "imagenome_side_explicit_v1.parquet", index=False)
        reg.to_parquet(P.FEATURES / "imagenome_region_explicit_v1.parquet", index=False)
        tab = side.pivot_table(index="finding", columns="side_text", values="dicom_id", aggfunc="count", fill_value=0)
        tab["n_pos"] = tab.sum(1)
        tab["explicit_frac"] = 1 - tab.get("default", 0) / tab.n_pos
        stated = tab[[c for c in ("left", "right", "bilateral") if c in tab]].sum(1)
        tab["bilateral_frac_of_explicit"] = tab.get("bilateral", 0) / stated.clip(lower=1)
        run.log("SIDE labels per finding (positive images):\n" + tab.round(3).to_string())
        rt = reg.groupby("finding").explicit.agg(n_region_pos="size", explicit="sum")
        rt["explicit_frac"] = rt.explicit / rt.n_region_pos
        run.log("REGION labels per finding (positive image x region pairs):\n" + rt.round(3).to_string())
        tab.to_csv(run.dir / "side_explicit_counts.csv")
        rt.to_csv(run.dir / "region_explicit_counts.csv")
        for f in tab.index:
            run.metric(finding=f, side_explicit_frac=float(tab.at[f, "explicit_frac"]),
                       bilateral_frac_of_explicit=float(tab.at[f, "bilateral_frac_of_explicit"]),
                       region_explicit_frac=float(rt.explicit_frac.get(f, float("nan"))))


if __name__ == "__main__":
    main()
