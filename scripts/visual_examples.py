"""Visual examples of each data source (2 per set): MIMIC, MS-CXR, VinDr (train), PadChest-GR.
CPU only. No metrics are computed on any set.

Per example: one PNG with titled panels side by side, plus a .txt with everything known about it.
  1 original image (VinDr/PadChest shown after windowing so they are visible)
  2 the image as CLEAR would see it with the LETTERBOX preprocessing (aspect kept, longest side 448,
    zero padding; VinDr inverted if MONOCHROME1 and percentile-windowed; PadChest percentile-windowed).
    Note: the features computed so far used the STRETCH preprocessing.
  3 CheXmask lungs and heart as coloured outlines, labelled patient LEFT / RIGHT
  4 CheXmask-derived regions: each lung in upper/middle/lower thirds, plus the heart, labelled
  5 Chest ImaGenome region boxes (MIMIC and MS-CXR only)
  6 radiologist annotations: MS-CXR boxes + phrase, VinDr boxes + class (per radiologist), PadChest-GR
    boxes + finding label
  7 (MS-CXR only) highest-scoring Stage 4 region (set A, ImaGenome boxes) for the phrase's finding, next
    to the radiologist box
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from chexmask_features import features as cm_features, rle_decode  # noqa: E402
from nesy import data as D, external as X, kg, manifests as M, paths as P  # noqa: E402
from nesy.regions import CHEXMASK_REGIONS, chexmask_region_masks  # noqa: E402
from nesy.reports import ReportZip  # noqa: E402
from nesy.runlog import Run  # noqa: E402

OUT = P.V2 / "outputs/visual_examples"
H = 520                       # display height of every panel
TITLE = 34
FONT = ImageFont.load_default(size=16)
SMALL = ImageFont.load_default(size=12)
COL = {"lung_left": (255, 70, 70), "lung_right": (70, 140, 255), "heart": (255, 215, 0)}
THIRD_COL = {"upper": (255, 120, 120), "middle": (120, 220, 120), "lower": (150, 150, 255)}
CM_FILE = {"mimic": ("MIMIC-CXR-JPG.csv", "dicom_id"), "vindr": ("VinDr-CXR.csv", "image_id"), "padchest": ("Padchest.csv", "ImageID")}
MSCXR_TO_F = {"Cardiomegaly": "cardiomegaly", "Lung Opacity": "lung_opacity", "Edema": "edema", "Consolidation": "consolidation",
              "Pneumonia": "pneumonia", "Atelectasis": "atelectasis", "Pneumothorax": "pneumothorax", "Pleural Effusion": "pleural_effusion"}
PATH13 = [f for f in kg.finding_ids() if f != "any_abnormality"]


# ----------------------------------------------------------------------------------- drawing
def disp(arr8: np.ndarray):
    """uint8 grey -> RGB PIL at display height H; returns image and scale (sx, sy) from original pixels."""
    im = Image.fromarray(arr8).convert("RGB")
    w = max(1, int(round(im.width * H / im.height)))
    return im.resize((w, H), Image.LANCZOS), (w / im.width, H / im.height)


def titled(im: Image.Image, title: str) -> Image.Image:
    out = Image.new("RGB", (im.width, im.height + TITLE), (255, 255, 255))
    out.paste(im, (0, TITLE))
    d = ImageDraw.Draw(out)
    for k, line in enumerate(textwrap.wrap(title, max(12, im.width // 9))[:2]):
        d.text((4, 2 + 16 * k), line, fill=(0, 0, 0), font=SMALL if k else FONT)
    return out


def outline(mask_disp: np.ndarray) -> np.ndarray:
    m = mask_disp
    er = m.copy()
    er[1:, :] &= m[:-1, :]
    er[:-1, :] &= m[1:, :]
    er[:, 1:] &= m[:, :-1]
    er[:, :-1] &= m[:, 1:]
    edge = m & ~er
    thick = edge.copy()
    thick[1:, :] |= edge[:-1, :]
    thick[:, 1:] |= edge[:, :-1]
    return thick


def mask_to_disp(mask: np.ndarray, size) -> np.ndarray:
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127


def label_at(d: ImageDraw.ImageDraw, mask_disp: np.ndarray, text: str, color):
    ys, xs = np.nonzero(mask_disp)
    if len(xs):
        x, y = int(xs.mean()), int(ys.mean())
        d.rectangle([x - 2, y - 2, x + 8 * len(text), y + 16], fill=(0, 0, 0))
        d.text((x, y), text, fill=color, font=SMALL)


def panel_masks(base, masks, size):
    im = base.copy()
    a = np.asarray(im).copy()
    for k in ("lung_left", "lung_right", "heart"):
        a[outline(mask_to_disp(masks[k], size))] = COL[k]
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    label_at(d, mask_to_disp(masks["lung_left"], size), "patient LEFT lung", COL["lung_left"])
    label_at(d, mask_to_disp(masks["lung_right"], size), "patient RIGHT lung", COL["lung_right"])
    label_at(d, mask_to_disp(masks["heart"], size), "heart", COL["heart"])
    return im


def panel_regions(base, regions, size):
    a = np.asarray(base).astype(np.float32)
    names = [r for r in CHEXMASK_REGIONS if r.count("_") == 2] + ["heart"]
    for r in names:
        c = COL["heart"] if r == "heart" else THIRD_COL[r.split("_")[2]]
        m = mask_to_disp(regions[r], size)
        a[m] = a[m] * 0.55 + np.array(c) * 0.45
        a[outline(m)] = (255, 255, 255)
    im = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for r in names:
        side = "L" if "left" in r else "R" if "right" in r else ""
        txt = "heart" if r == "heart" else f"{side} {r.split('_')[2]}"
        label_at(d, mask_to_disp(regions[r], size), txt, (255, 255, 255))
    return im


def panel_boxes(base, boxes, sc, color_of, width=2):
    """boxes: list of (x1, y1, x2, y2, label) in original pixels."""
    im = base.copy()
    d = ImageDraw.Draw(im)
    for x1, y1, x2, y2, lab in boxes:
        c = color_of(lab)
        d.rectangle([x1 * sc[0], y1 * sc[1], x2 * sc[0], y2 * sc[1]], outline=c, width=width)
        d.rectangle([x1 * sc[0], y1 * sc[1] - 13, x1 * sc[0] + 7 * len(lab[:32]), y1 * sc[1]], fill=(0, 0, 0))
        d.text((x1 * sc[0] + 1, y1 * sc[1] - 13), lab[:32], fill=c, font=SMALL)
    return im


def hstack(panels):
    w = sum(p.width for p in panels) + 6 * (len(panels) - 1)
    out = Image.new("RGB", (w, max(p.height for p in panels)), (255, 255, 255))
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width + 6
    return out


def palette(i):
    pal = [(255, 99, 71), (60, 179, 113), (30, 144, 255), (238, 130, 238), (255, 165, 0), (0, 206, 209), (255, 255, 0), (199, 21, 133)]
    return pal[i % len(pal)]


# ----------------------------------------------------------------------------------- data
def chexmask_rows(dataset, ids, run):
    fname, idcol = CM_FILE[dataset]
    want, got = set(ids), {}
    cols = [idcol, "Dice RCA (Mean)", "Dice RCA (Max)", "Left Lung", "Right Lung", "Heart", "Height", "Width"]
    for i, ch in enumerate(pd.read_csv(P.CHEXMASK / "OriginalResolution" / fname, usecols=cols, chunksize=2000, dtype={idcol: str})):
        hit = ch[ch[idcol].isin(want)]
        for r in hit.itertuples(index=False, name=None):
            got[r[0]] = r
        if len(got) == len(want):
            break
    run.log(f"  CheXmask {dataset}: found {len(got)}/{len(want)} (scanned {(i + 1) * 2000:,} rows)")
    return got


def masks_from_row(row):
    _, rca, rca_max, l, r, h, Hh, Ww = row
    ml, mr, mh = rle_decode(l, int(Hh), int(Ww)), rle_decode(r, int(Hh), int(Ww)), rle_decode(h, int(Hh), int(Ww))
    feats = cm_features((row[0], rca, rca_max, l, r, h, Hh, Ww))
    return {"lung_left": ml, "lung_right": mr, "heart": mh}, chexmask_region_masks(ml, mr, mh), feats


def letterbox_448(arr8):
    from letterbox_test import letterbox
    return np.asarray(letterbox(Image.fromarray(arr8)))


def write_txt(path, info: dict):
    lines = []
    for k, v in info.items():
        if isinstance(v, (dict, list)):
            lines.append(f"{k}:")
            lines.append(json.dumps(v, indent=2, default=str))
        else:
            lines.append(f"{k}: {v}")
    path.write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage4-run", default=str(P.V2 / "runs/20261006-1709_stage4-regions"))
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("visual-examples", vars(args), args.run_dir) as run:
        rng = np.random.default_rng(args.seed)
        OUT.mkdir(parents=True, exist_ok=True)
        t = D.study_table()
        man = M.load("mimic", columns=["dicom_id", "image_path", "view", "source_split", "subject_id", "study_id", "rows", "cols",
                                       "has_imagenome_scene_graph"] + [c for c in pd.read_parquet(P.MANIFESTS / "mimic.parquet").columns if c.startswith("cx_")])
        cm_feat = pd.read_parquet(P.FEATURES / "chexmask_mimic.parquet", columns=["dicom_id", "qc_ok"])
        regions = pd.read_parquet(P.FEATURES / "imagenome_regions.parquet")
        reg_expl = pd.read_parquet(P.FEATURES / "imagenome_region_explicit_v1.parquet")
        side_expl = pd.read_parquet(P.FEATURES / "imagenome_side_explicit_v1.parquet")
        rz = ReportZip()
        ex = []

        # ---------- MIMIC: official validate, one PA, one AP, >= 2 positive findings
        v = t[(t.split == "val")].merge(man[["dicom_id", "image_path", "has_imagenome_scene_graph"]], on="dicom_id").merge(cm_feat, on="dicom_id")
        v["npos"] = (v[PATH13] == 1).sum(1)
        for view in ("PA", "AP"):
            c = v[(v.view == view) & (v.npos >= 2) & v.has_imagenome_scene_graph & v.qc_ok.fillna(False).astype(bool)]
            ex.append(("mimic", c.iloc[int(rng.integers(len(c)))].to_dict()))
        # ---------- MS-CXR: not official test, different findings
        ms = M.load("mscxr", columns=["dicom_id", "subject_id", "study_id", "category", "label_text", "x", "y", "w", "h",
                                      "image_width", "image_height", "view", "image_path", "phrase_id"])
        off = man.set_index("dicom_id").source_split
        ms["official_split"] = ms.dicom_id.map(off)
        cand = ms[ms.official_split != "test"]
        cats = cand.category.unique()
        rng.shuffle(cats)
        for cat in cats[:2]:
            ids = cand[cand.category == cat].dicom_id.unique()
            ex.append(("ms_cxr", {"dicom_id": ids[int(rng.integers(len(ids)))], "category": cat}))
        # ---------- VinDr train with >= 1 box
        vd = M.load("vindr")
        c = vd[(vd.source_split == "train") & (vd.n_boxes > 0)]
        for i in rng.choice(len(c), 2, replace=False):
            ex.append(("vindr", c.iloc[int(i)].to_dict()))
        # ---------- PadChest-GR with >= 1 boxed finding (all external test: no metrics)
        pg = M.load("padchest_gr")
        cmids = set(pd.read_csv(P.CHEXMASK / "OriginalResolution/Padchest.csv", usecols=["ImageID"]).ImageID)
        c = pg[(pg.n_boxes > 0) & pg.image_id.isin(cmids)]
        for i in rng.choice(len(c), 2, replace=False):
            ex.append(("padchest_gr", c.iloc[int(i)].to_dict()))
        run.log("selected: " + ", ".join(f"{s}:{(e.get('dicom_id') or e.get('image_id'))[:12]}" for s, e in ex))

        cm = {}
        cm.update(chexmask_rows("mimic", [e["dicom_id"] for s, e in ex if s in ("mimic", "ms_cxr")], run))
        cm.update(chexmask_rows("vindr", [e["image_id"] for s, e in ex if s == "vindr"], run))
        cm.update(chexmask_rows("padchest", [e["image_id"] for s, e in ex if s == "padchest_gr"], run))

        s4 = Path(args.stage4_run)
        s4_idx = pd.read_parquet(s4 / "scores_index.parquet")
        s4_sc = np.load(s4 / "scores_A_imagenome.npy", mmap_mode="r")
        from region_features import IG_REGIONS
        from stage4_regions import FIND as S4_FIND
        reports_gr = {r["ImageID"]: r for r in json.loads((P.PADCHEST_GR / "grounded_reports_20240819.json").read_text())}
        issues, readme_rows = [], []

        for set_name, e in ex:
            sub = OUT / set_name
            sub.mkdir(parents=True, exist_ok=True)
            info, panels = {"dataset": set_name}, []
            if set_name in ("mimic", "ms_cxr"):
                did = e["dicom_id"]
                r = man.set_index("dicom_id").loc[did]
                arr = np.asarray(Image.open(r.image_path).convert("L"))
                info.update(image_id=did, patient_id=int(r.subject_id), study_id=int(r.study_id), view=r.view,
                            official_split=r.source_split, project_split=t.set_index("dicom_id").split.get(did, "not in study table"))
            elif set_name == "vindr":
                im, meta = X.load_vindr(e["image_path"])
                arr = np.asarray(im)
                info.update(image_id=e["image_id"], patient_id=e["patient_id"] + " (no patient ID in VinDr; image used as patient)",
                            view="frontal (VinDr is frontal-only; no view tag)", split="VinDr train folder (role vindr_train)",
                            photometric=meta["photometric"], bits_stored=meta["bits_stored"])
            else:
                im, meta = X.load_padchest(e["archive_prefix"], e["archive_member"])
                arr = np.asarray(im)
                info.update(image_id=e["image_id"], patient_id=e["patient_id"], study_id=e["study_id"], view="frontal (PadChest-GR)",
                            split="PadChest-GR (external test for this project; NO metrics computed)", png_mode=meta["mode"])
            Hh, Ww = arr.shape
            base, sc = disp(arr)
            size = base.size
            panels.append(titled(base, f"1 original ({Ww}x{Hh}px)" + (" - windowed for display" if set_name in ("vindr", "padchest_gr") else "")))
            lb = letterbox_448(arr)
            panels.append(titled(Image.fromarray(lb).convert("RGB").resize((H, H), Image.NEAREST),
                                 "2 CLEAR input, LETTERBOX 448x448 (features so far used STRETCH)"))
            key = info["image_id"]
            if key in cm:
                masks, regs, f = masks_from_row(cm[key])
                if masks["lung_left"].shape != arr.shape:
                    issues.append(f"{set_name}/{key}: CheXmask mask size {masks['lung_left'].shape} != image {arr.shape}")
                panels.append(titled(panel_masks(base, masks, size), "3 CheXmask lungs + heart (patient sides)"))
                panels.append(titled(panel_regions(base, regs, size), "4 CheXmask regions: lung thirds + heart"))
                info.update(chexmask_quality_dice_rca_mean=round(f["rca_mean"], 4), chexmask_quality_dice_rca_max=round(f["rca_max"], 4),
                            chexmask_qc_ok=bool(f["rca_mean"] >= 0.7 and f["masks_complete"]),
                            heart_to_chest_ratio_ctr=round(f.get("ctr", float("nan")), 4),
                            heart_to_chest_ratio_widest_row=round(f.get("ctr_maxrow", float("nan")), 4),
                            left_lung_mask_on_image_right=f["left_lung_on_image_right"])
                if not f["left_lung_on_image_right"]:
                    issues.append(f"{set_name}/{key}: CheXmask 'Left Lung' is NOT on the image right")
                if f["rca_mean"] < 0.7:
                    issues.append(f"{set_name}/{key}: CheXmask quality below 0.7 ({f['rca_mean']:.3f})")
            else:
                info["chexmask"] = "no CheXmask masks for this image"
                issues.append(f"{set_name}/{key}: no CheXmask row")
            if set_name in ("mimic", "ms_cxr"):
                b = regions[regions.dicom_id == key]
                boxes = [(x.original_x1, x.original_y1, x.original_x2, x.original_y2, x.region) for x in b.itertuples()]
                panels.append(titled(panel_boxes(base, boxes, sc, lambda lab: palette(hash(lab) % 8), width=1),
                                     f"5 Chest ImaGenome boxes ({len(boxes)})"))
                if any(x2 > Ww * 1.02 or y2 > Hh * 1.02 for x1, y1, x2, y2, _ in boxes):
                    issues.append(f"{set_name}/{key}: some ImaGenome boxes extend beyond the image")
                ex_reg = reg_expl[reg_expl.dicom_id == key]
                info["imagenome_location_labels_positive"] = [
                    f"{x.finding} @ {x.region} ({'explicit' if x.explicit else 'default assignment, masked'})" for x in ex_reg.itertuples()]
                info["imagenome_side_labels"] = {x.finding: x.side_text for x in side_expl[side_expl.dicom_id == key].itertuples()}
                st = int(info["study_id"])
                lab = t.set_index("study_id")
                if st in lab.index:
                    info["labels_v2 (1 pos / 0 neg / NaN masked)"] = {f: (None if pd.isna(lab.at[st, f]) else float(lab.at[st, f])) for f in kg.finding_ids()}
                info["chexpert_raw (1 / 0 / -1 / blank)"] = {c[3:]: (None if pd.isna(man.set_index("dicom_id").at[key, c]) else float(man.set_index("dicom_id").at[key, c]))
                                                            for c in man.columns if c.startswith("cx_")}
                sec = rz.sections(int(info["patient_id"]), st)
                info["report_FINDINGS"] = sec.findings
                info["report_IMPRESSION"] = sec.impression
                info["report_section_source"] = sec.source
            if set_name == "ms_cxr":
                g = ms[ms.dicom_id == key]
                boxes = [(x.x, x.y, x.x + x.w, x.y + x.h, f"{x.category}: {x.label_text}") for x in g.itertuples()]
                panels.append(titled(panel_boxes(base, boxes, sc, lambda lab: (0, 255, 0), width=3), "6 MS-CXR radiologist boxes + phrase"))
                info["ms_cxr_phrases"] = [{"category": x.category, "phrase": x.label_text, "box_xywh": [x.x, x.y, x.w, x.h]} for x in g.itertuples()]
                cat = e["category"]
                f = MSCXR_TO_F[cat]
                row = s4_idx.index[s4_idx.dicom_id == key]
                if len(row) and f in S4_FIND:
                    scs = np.asarray(s4_sc[row[0], :, S4_FIND.index(f)], np.float32)
                    rel = [j for j, rr in enumerate(IG_REGIONS) if (("cardiac" in rr) if f == "cardiomegaly" else
                                                                    ("lung" in rr or "zone" in rr or "costophrenic" in rr or "hilar" in rr))
                           and not np.isnan(scs[j])]
                    jb = rel[int(np.argmax(scs[rel]))]
                    bb = regions[(regions.dicom_id == key) & (regions.region == IG_REGIONS[jb])].iloc[0]
                    rad = [(x1, y1, x2, y2, "radiologist") for x1, y1, x2, y2, lab in boxes if lab.startswith(cat)]
                    p7 = panel_boxes(base, rad + [(bb.original_x1, bb.original_y1, bb.original_x2, bb.original_y2, f"Stage 4 top: {IG_REGIONS[jb]}")],
                                     sc, lambda lab: (0, 255, 0) if lab == "radiologist" else (255, 0, 255), width=3)
                    panels.append(titled(p7, f"7 Stage 4 top region for {f} (magenta) vs radiologist (green)"))
                    info["stage4_top_region"] = {"finding": f, "region": IG_REGIONS[jb], "score": float(scs[jb]),
                                                 "region_scores_relevant": {IG_REGIONS[j]: round(float(scs[j]), 3) for j in rel}}
                    rx = [((x1 + x2) / 2 / Ww) for x1, y1, x2, y2, _ in rad]
                    reg_side = "left" if IG_REGIONS[jb].startswith("left") else "right" if IG_REGIONS[jb].startswith("right") else None
                    box_sides = {"left" if x > 0.5 else "right" for x in rx}
                    if reg_side and len(box_sides) == 1 and reg_side not in box_sides:
                        issues.append(f"ms_cxr/{key}: Stage 4 top region {IG_REGIONS[jb]} is on the other side from the radiologist box")
            if set_name == "vindr":
                ann = pd.read_csv(P.VINDR / "annotations/annotations_train.csv")
                g = ann[(ann.image_id == key) & ann.x_min.notna()]
                rads = sorted(g.rad_id.unique())
                boxes = [(x.x_min, x.y_min, x.x_max, x.y_max, f"{x.rad_id}: {x.class_name}") for x in g.itertuples()]
                panels.append(titled(panel_boxes(base, boxes, sc, lambda lab: palette(rads.index(lab.split(':')[0])), width=2),
                                     "6 VinDr radiologist boxes (colour = radiologist)"))
                info["vindr_boxes"] = [{"radiologist": x.rad_id, "class": x.class_name, "xyxy": [x.x_min, x.y_min, x.x_max, x.y_max]} for x in g.itertuples()]
                info["vindr_image_labels (number of 3 radiologists marking each)"] = {c[4:]: int(e[c]) for c in e if c.startswith("lbl_") and e[c] > 0}
                if any(x2 > Ww or y2 > Hh for x1, y1, x2, y2, _ in boxes):
                    issues.append(f"vindr/{key}: a box extends beyond the image")
            if set_name == "padchest_gr":
                rep = reports_gr[key]
                boxes = []
                for fnd in rep["findings"]:
                    for bx in fnd.get("boxes", []):
                        boxes.append((bx[0] * Ww, bx[1] * Hh, bx[2] * Ww, bx[3] * Hh, ", ".join(fnd.get("labels", [])) or "box"))
                panels.append(titled(panel_boxes(base, boxes, sc, lambda lab: (0, 255, 0), width=3), "6 PadChest-GR boxes + finding label"))
                info["padchest_gr_labels"] = list(e["labels"]) if e.get("labels") is not None else []
                info["padchest_gr_sentences_en"] = [{"sentence": f.get("sentence_en"), "labels": f.get("labels", []),
                                                     "locations": f.get("locations", []), "n_boxes": len(f.get("boxes", []))} for f in rep["findings"]]
                info["patient_age"] = e.get("patient_age")
                info["patient_sex"] = e.get("patient_sex")
            fig = hstack(panels)
            stem = f"{set_name}_{str(key)[:16]}"
            fig.save(sub / f"{stem}.png")
            write_txt(sub / f"{stem}.txt", info)
            readme_rows.append((set_name, f"{set_name}/{stem}.png", len(panels)))
            run.log(f"  wrote {sub / stem}.png ({len(panels)} panels) and .txt")

        readme = ["# Visual examples (2 per data source)", "",
                  f"Generated by `scripts/visual_examples.py` (run `{run.dir.name}`). CPU only; no metrics computed on any set.", "",
                  "## Files", ""] + [f"- `{p}` + `.txt`: {s}, {n} panels" for s, p, n in readme_rows] + [
                  "", "## Panels (left to right)", "",
                  "1. **Original image.** VinDr and PadChest-GR are high-bit-depth; they are shown after the percentile windowing so they are visible.",
                  "2. **CLEAR input with LETTERBOX preprocessing**: aspect kept, longest side 448, zero padding to 448×448 (VinDr inverted if MONOCHROME1, VinDr/PadChest percentile-windowed). NOTE: the features computed so far used the STRETCH preprocessing (plain resize to 448×448).",
                  "3. **CheXmask masks**: patient-left lung (red), patient-right lung (blue), heart (yellow) outlines from the OriginalResolution files.",
                  "4. **CheXmask-derived regions**: each lung split into upper / middle / lower thirds of its own height (red / green / blue tints), plus the heart (yellow); these are the 9 pooling regions minus the two whole lungs.",
                  "5. **Chest ImaGenome boxes** (MIMIC and MS-CXR only): the 36 silver-standard anatomical region boxes.",
                  "6. **Radiologist annotations**: MS-CXR boxes with category and phrase; VinDr boxes with class name, colour per radiologist (R1–R17); PadChest-GR boxes with finding labels.",
                  "7. **MS-CXR only: Stage 4** (set A, ImaGenome-box classifier, stretch features) highest-scoring relevant region for the phrase's finding (magenta) next to the radiologist box (green).",
                  "", "The `.txt` file next to each PNG holds the IDs, view, split, labels, heart-to-chest ratio, CheXmask quality, Chest ImaGenome location labels and the report text / phrases / sentences."]
        (OUT / "README.md").write_text("\n".join(readme) + "\n")
        run.log("ISSUES FOUND:\n" + ("\n".join(issues) if issues else "none by the automatic checks"))
        (OUT / "automatic_checks.txt").write_text("\n".join(issues) + "\n" if issues else "none\n")


if __name__ == "__main__":
    main()
