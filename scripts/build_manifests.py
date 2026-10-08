"""Stage 0: one Parquet manifest per dataset, built from the raw files in place.

Manifests hold image location, patient/study IDs, view, the dataset's own split, raw labels and
references to annotations. The project split (DRAFT until the user freezes it) is NOT stored here;
it lives in data/splits/ and is joined at read time by nesy.manifests.load(), so changing a split
never means rewriting a manifest.

Raw labels are stored as published (CheXpert: 1 / 0 / -1 / NaN). The project policy "not mentioned
= negative" is applied by the label loader, not baked into the manifest.

  python -u scripts/build_manifests.py [--only mimic,mscxr,imagenome,vindr,padchest_gr]
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nesy import paths as P  # noqa: E402
from nesy.multizip import open_split_zip  # noqa: E402
from nesy.runlog import Run  # noqa: E402

# Published counts (from each dataset's README / PhysioNet page as recorded in DATA.md).
PUBLISHED = {
    "mimic_images": 377_110, "mimic_studies": 227_835, "mimic_patients": 65_379,
    "vindr_train": 15_000, "vindr_test": 3_000,
    "padchest_gr_studies": 4_555,
    "imagenome_gold_images": 1_000, "imagenome_gold_patients": 500,
}

CHEXPERT = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture",
            "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
            "Pneumothorax", "Support Devices"]


def snake(s: str) -> str:
    return s.lower().replace(" ", "_").replace("/", "_")


def check(run: Run, name: str, found: int, expected: int) -> None:
    ok = found == expected
    run.log(f"  count check {name}: found {found:,}, published {expected:,} -> {'OK' if ok else 'MISMATCH'}")
    run.metric(check=name, found=found, published=expected, ok=ok)
    if not ok:
        raise ValueError(f"{name}: found {found} but published {expected}")


def write(run: Run, df: pd.DataFrame, name: str) -> None:
    P.MANIFESTS.mkdir(parents=True, exist_ok=True)
    out = P.MANIFESTS / f"{name}.parquet"
    tmp = out.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    run.log(f"  wrote {out} ({len(df):,} rows, {out.stat().st_size / 1e6:.1f} MB)")
    run.metric(manifest=name, rows=len(df), path=str(out))


# ------------------------------------------------------------------------------------- MIMIC
def build_mimic(run: Run) -> pd.DataFrame:
    run.log("MIMIC-CXR-JPG")
    meta = pd.read_csv(P.MIMIC / "mimic-cxr-2.0.0-metadata.csv.gz")
    split = pd.read_csv(P.MIMIC / "mimic-cxr-2.0.0-split.csv.gz")
    chex = pd.read_csv(P.MIMIC / "mimic-cxr-2.0.0-chexpert.csv.gz")
    run.log(f"  in: metadata {len(meta):,}, split {len(split):,}, chexpert {len(chex):,} rows")

    df = meta.merge(split[["dicom_id", "split"]], on="dicom_id", how="left", validate="one_to_one")
    run.log(f"  images without an official split: {df.split.isna().sum()}")
    df = df.rename(columns={"split": "source_split", "ViewPosition": "view", "Rows": "rows", "Columns": "cols"})
    sid = df.subject_id.astype(str)
    df["image_path"] = (str(P.MIMIC) + "/files/p" + sid.str[:2] + "/p" + sid + "/s" + df.study_id.astype(str)
                        + "/" + df.dicom_id + ".jpg")
    df["patient_id"] = df.subject_id
    df["is_frontal"] = df.view.isin(["PA", "AP"])

    chex = chex.rename(columns={c: "cx_" + snake(c) for c in CHEXPERT})
    n0 = len(df)
    df = df.merge(chex.drop(columns=["subject_id"]), on="study_id", how="left", validate="many_to_one")
    assert len(df) == n0
    df["has_chexpert_row"] = df.study_id.isin(set(chex.study_id))
    no_row = ~df.has_chexpert_row
    run.log(f"  studies with no CheXpert row (unlabelable): {df.loc[no_row, 'study_id'].nunique()}")

    # Annotation availability (references only; annotations stay in place).
    ms = pd.read_csv(P.MSCXR / "MS_CXR_Local_Alignment_v1.1.0.csv", usecols=["dicom_id"])
    gold = pd.read_csv(P.IMAGENOME / "gold_dataset/gold_bbox_coordinate_annotations_1000images.csv", usecols=["image_id"])
    gold_ids = set(gold.image_id.astype(str).str.replace(".dcm", "", regex=False))
    with zipfile.ZipFile(P.IMAGENOME / "silver_dataset/scene_graph.zip") as z:
        sg = {Path(n).name.replace("_SceneGraph.json", "") for n in z.namelist() if n.endswith("_SceneGraph.json")}
    df["in_mscxr"] = df.dicom_id.isin(set(ms.dicom_id))
    df["in_imagenome_gold"] = df.dicom_id.isin(gold_ids)
    df["has_imagenome_scene_graph"] = df.dicom_id.isin(sg)
    run.log(f"  flags: in_mscxr {df.in_mscxr.sum():,}, in_imagenome_gold {df.in_imagenome_gold.sum():,}, "
            f"scene graphs {df.has_imagenome_scene_graph.sum():,} of {len(sg):,} in zip")

    check(run, "mimic_images", len(df), PUBLISHED["mimic_images"])
    check(run, "mimic_studies", df.study_id.nunique(), PUBLISHED["mimic_studies"])
    check(run, "mimic_patients", df.patient_id.nunique(), PUBLISHED["mimic_patients"])
    if df.dicom_id.duplicated().any():
        raise ValueError("duplicate dicom_id in MIMIC manifest")

    # Spot-check that image files exist (a full stat of 377k files on cephfs is slow; the file
    # count already matches IMAGE_FILENAMES exactly, see DATA.md).
    sample = df.sample(500, random_state=0)
    missing = [p for p in sample.image_path if not Path(p).is_file()]
    run.log(f"  file existence spot check: {500 - len(missing)}/500 present")
    if missing:
        raise FileNotFoundError(f"{len(missing)} sampled MIMIC images missing, e.g. {missing[:3]}")

    df["dataset"] = "mimic"
    df["image_id"] = df.dicom_id
    cols = ["dataset", "image_id", "dicom_id", "patient_id", "subject_id", "study_id", "image_path", "view", "is_frontal",
            "source_split", "rows", "cols", "StudyDate", "StudyTime", "PerformedProcedureStepDescription",
            "ViewCodeSequence_CodeMeaning", "PatientOrientationCodeSequence_CodeMeaning", "has_chexpert_row",
            *["cx_" + snake(c) for c in CHEXPERT], "in_mscxr", "in_imagenome_gold", "has_imagenome_scene_graph"]
    out = df[cols].sort_values(["subject_id", "study_id", "dicom_id"]).reset_index(drop=True)
    write(run, out, "mimic")
    return out


# ------------------------------------------------------------------------------------ MS-CXR
def build_mscxr(run: Run, mimic: pd.DataFrame) -> None:
    run.log("MS-CXR")
    ms = pd.read_csv(P.MSCXR / "MS_CXR_Local_Alignment_v1.1.0.csv")
    run.log(f"  in: {len(ms):,} box rows")
    ms = ms.rename(columns={"split": "source_split", "category_name": "category"})
    ms["subject_id"] = ms.path.str.extract(r"/p(\d{8})/")[0].astype(int)
    ms["study_id"] = ms.path.str.extract(r"/s(\d{8})/")[0].astype(int)
    m = mimic.set_index("dicom_id")
    missing = ~ms.dicom_id.isin(m.index)
    if missing.any():
        raise ValueError(f"{missing.sum()} MS-CXR images not in MIMIC")
    ms["image_path"] = m.loc[ms.dicom_id, "image_path"].values
    ms["view"] = m.loc[ms.dicom_id, "view"].values
    if not (m.loc[ms.dicom_id, "subject_id"].values == ms.subject_id.values).all():
        raise ValueError("MS-CXR subject_id parsed from path disagrees with MIMIC metadata")
    ms["phrase_id"] = ms.groupby(["dicom_id", "label_text"], sort=False).ngroup()
    ms["dataset"] = "mscxr"
    ms["patient_id"] = ms.subject_id
    ms["image_id"] = ms.dicom_id
    run.log(f"  images {ms.dicom_id.nunique():,}, patients {ms.subject_id.nunique():,}, phrases {ms.phrase_id.nunique():,}, "
            f"boxes {len(ms):,}")
    cols = ["dataset", "image_id", "dicom_id", "patient_id", "subject_id", "study_id", "image_path", "view", "source_split",
            "phrase_id", "category", "label_text", "x", "y", "w", "h", "image_width", "image_height"]
    write(run, ms[cols], "mscxr")


# ---------------------------------------------------------------------------- Chest ImaGenome
def build_imagenome(run: Run, mimic: pd.DataFrame) -> None:
    run.log("Chest ImaGenome")
    m = mimic.set_index("dicom_id")
    # silver: one row per scene graph, referenced by zip member
    with zipfile.ZipFile(P.IMAGENOME / "silver_dataset/scene_graph.zip") as z:
        members = [n for n in z.namelist() if n.endswith("_SceneGraph.json")]
    sil = pd.DataFrame({"zip_member": members})
    sil["dicom_id"] = sil.zip_member.map(lambda n: Path(n).name.replace("_SceneGraph.json", ""))
    run.log(f"  silver scene graphs in zip: {len(sil):,}")
    inm = sil.dicom_id.isin(m.index)
    run.log(f"  silver scene graphs whose dicom_id is in MIMIC: {inm.sum():,}; dropped {(~inm).sum():,}")
    sil = sil[inm].copy()
    for c in ["subject_id", "study_id", "image_path", "view"]:
        sil[c] = m.loc[sil.dicom_id, c].values
    sil["zip_path"] = str(P.IMAGENOME / "silver_dataset/scene_graph.zip")
    sil["dataset"] = "imagenome_silver"
    sil["patient_id"] = sil.subject_id
    sil["image_id"] = sil.dicom_id
    write(run, sil[["dataset", "image_id", "dicom_id", "patient_id", "subject_id", "study_id", "image_path", "view",
                    "zip_path", "zip_member"]], "imagenome_silver")

    # gold: one row per image, with references to the annotation files
    g = pd.read_csv(P.IMAGENOME / "gold_dataset/gold_bbox_coordinate_annotations_1000images.csv")
    g["dicom_id"] = g.image_id.astype(str).str.replace(".dcm", "", regex=False)
    per = g.groupby("dicom_id").agg(n_boxes=("bbox_name", "size"), regions=("bbox_name", lambda s: sorted(set(s))))
    per = per.reset_index()
    missing = ~per.dicom_id.isin(m.index)
    if missing.any():
        raise ValueError(f"{missing.sum()} gold images not in MIMIC")
    for c in ["subject_id", "study_id", "image_path", "view"]:
        per[c] = m.loc[per.dicom_id, c].values
    per["bbox_file"] = str(P.IMAGENOME / "gold_dataset/gold_bbox_coordinate_annotations_1000images.csv")
    per["attributes_file"] = str(P.IMAGENOME / "gold_dataset/gold_attributes_relations_500pts_500studies1st.txt")
    per["dataset"] = "imagenome_gold"
    per["patient_id"] = per.subject_id
    per["image_id"] = per.dicom_id
    check(run, "imagenome_gold_images", len(per), PUBLISHED["imagenome_gold_images"])
    check(run, "imagenome_gold_patients", per.patient_id.nunique(), PUBLISHED["imagenome_gold_patients"])
    write(run, per[["dataset", "image_id", "dicom_id", "patient_id", "subject_id", "study_id", "image_path", "view",
                    "n_boxes", "regions", "bbox_file", "attributes_file"]], "imagenome_gold")


# --------------------------------------------------------------------------------- VinDr-CXR
def build_vindr(run: Run) -> None:
    run.log("VinDr-CXR")
    rows = []
    for sp in ["train", "test"]:
        files = sorted((P.VINDR / sp).glob("*.dicom"))
        rows += [{"image_id": f.stem, "image_path": str(f), "source_split": sp} for f in files]
    df = pd.DataFrame(rows)
    check(run, "vindr_train", int((df.source_split == "train").sum()), PUBLISHED["vindr_train"])
    check(run, "vindr_test", int((df.source_split == "test").sum()), PUBLISHED["vindr_test"])

    lab_tr = pd.read_csv(P.VINDR / "annotations/image_labels_train.csv")
    lab_te = pd.read_csv(P.VINDR / "annotations/image_labels_test.csv").rename(columns={"Other disease": "Other diseases"})
    label_cols = [c for c in lab_tr.columns if c not in ("image_id", "rad_id")]
    if set(label_cols) != set(c for c in lab_te.columns if c != "image_id"):
        raise ValueError("VinDr train/test label columns differ after renaming 'Other disease'")
    run.log(f"  labels in: train {len(lab_tr):,} rows ({lab_tr.image_id.nunique():,} images x radiologists), "
            f"test {len(lab_te):,} rows")
    # Train: 3 radiologists per image. Store the number who marked each label (0..3); merging policy
    # is a modelling decision, applied later. Test: one consensus label (0/1).
    votes = lab_tr.groupby("image_id")[label_cols].sum()
    n_rads = lab_tr.groupby("image_id").size().rename("n_radiologists")
    lab = pd.concat([votes.join(n_rads), lab_te.set_index("image_id")[label_cols].assign(n_radiologists=1)])
    lab.columns = ["lbl_" + snake(c) if c != "n_radiologists" else c for c in lab.columns]
    df = df.join(lab, on="image_id")
    run.log(f"  images without labels: {df.n_radiologists.isna().sum()}; radiologists per train image: "
            f"{df.loc[df.source_split == 'train', 'n_radiologists'].value_counts().to_dict()}")

    ann_tr = pd.read_csv(P.VINDR / "annotations/annotations_train.csv")
    ann_te = pd.read_csv(P.VINDR / "annotations/annotations_test.csv")
    nb = pd.concat([ann_tr.dropna(subset=["x_min"]), ann_te.dropna(subset=["x_min"])]).groupby("image_id").size()
    df["n_boxes"] = df.image_id.map(nb).fillna(0).astype(int)
    df["annotation_file"] = np.where(df.source_split == "train", str(P.VINDR / "annotations/annotations_train.csv"),
                                     str(P.VINDR / "annotations/annotations_test.csv"))
    # No patient ID is published or present in the DICOM headers (PatientID is empty in the sampled
    # files), so each image is treated as its own patient. Recorded as an assumption in DATA.md.
    df["patient_id"] = "vindr_" + df.image_id
    df["study_id"] = df.image_id
    df["view"] = pd.NA            # not in headers; VinDr-CXR is published as frontal (PA) adult CXRs
    df["dataset"] = "vindr"
    df["role"] = np.where(df.source_split == "test", "external_test", "vindr_train")
    cols = ["dataset", "image_id", "patient_id", "study_id", "image_path", "view", "source_split", "role", "n_radiologists",
            "n_boxes", "annotation_file", *[c for c in df.columns if c.startswith("lbl_")]]
    write(run, df[cols], "vindr")


# ------------------------------------------------------------------------------- PadChest-GR
def build_padchest_gr(run: Run) -> None:
    run.log("PadChest-GR")
    reports = json.loads((P.PADCHEST_GR / "grounded_reports_20240819.json").read_text())
    with zipfile.ZipFile(P.PADCHEST_GR / "master_table.csv.zip") as z:
        mt = pd.read_csv(z.open("master_table.csv"), dtype={"StudyID": str, "ImageID": str, "PatientID": str})
    run.log(f"  in: {len(reports):,} grounded reports, master table {len(mt):,} rows")
    zf = open_split_zip(str(P.PADCHEST_GR / "Padchest_GR_files/PadChest_GR.zip"))
    info = {Path(i.filename).name: i for i in zf.infolist() if i.filename.endswith(".png")}
    from collections import Counter
    run.log(f"  PNG members in split archive: {len(info):,}; compression types (0=stored, 8=deflate): "
            f"{dict(Counter(i.compress_type for i in info.values()))}")

    rows = []
    for r in reports:
        f = r["findings"]
        labels = sorted({l for x in f for l in x.get("labels", [])})
        rows.append({
            "image_id": str(r["ImageID"]), "study_id": str(r["StudyID"]),
            "prior_study_id": r["PreviousStudyID"] and str(r["PreviousStudyID"]),
            "prior_image_id": r["PreviousImageID"] and str(r["PreviousImageID"]),
            "n_sentences": len(f), "n_abnormal_sentences": sum(bool(x.get("abnormal")) for x in f),
            "n_boxes": sum(len(x.get("boxes", [])) for x in f), "labels": labels,
        })
    df = pd.DataFrame(rows)
    pat = mt.drop_duplicates("ImageID").set_index("ImageID")
    df["patient_id"] = df.image_id.map(pat.PatientID)
    df["source_split"] = df.image_id.map(pat.split)
    df["patient_age"] = df.image_id.map(pat.PatientAge)
    df["patient_sex"] = df.image_id.map(pat.PatientSex_DICOM)
    run.log(f"  images without a master-table row: {df.patient_id.isna().sum()}")
    df["in_archive"] = df.image_id.isin(info)
    df["archive_member"] = df.image_id.map(lambda k: info[k].filename if k in info else None)
    df["archive_prefix"] = str(P.PADCHEST_GR / "Padchest_GR_files/PadChest_GR.zip")
    df["image_path"] = None   # images are read from the archive, never extracted (see DATA.md)
    df["view"] = pd.NA
    df["dataset"] = "padchest_gr"
    df["role"] = "external_test"
    df["reports_file"] = str(P.PADCHEST_GR / "grounded_reports_20240819.json")
    check(run, "padchest_gr_studies", df.study_id.nunique(), PUBLISHED["padchest_gr_studies"])
    check(run, "padchest_gr_images_in_archive", int(df.in_archive.sum()), PUBLISHED["padchest_gr_studies"])
    cols = ["dataset", "image_id", "patient_id", "study_id", "image_path", "archive_prefix", "archive_member", "in_archive",
            "view", "source_split", "role", "prior_study_id", "prior_image_id", "patient_age", "patient_sex",
            "n_sentences", "n_abnormal_sentences", "n_boxes", "labels", "reports_file"]
    write(run, df[cols], "padchest_gr")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="mimic,mscxr,imagenome,vindr,padchest_gr")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    todo = args.only.split(",")
    with Run("manifests", {"only": todo, "published_counts": PUBLISHED}, args.run_dir) as run:
        mimic = None
        if {"mimic", "mscxr", "imagenome"} & set(todo):
            mimic = build_mimic(run) if "mimic" in todo else pd.read_parquet(P.MANIFESTS / "mimic.parquet")
        for name, fn in [("mscxr", build_mscxr), ("imagenome", build_imagenome)]:
            if name in todo:
                fn(run, mimic)
        if "vindr" in todo:
            build_vindr(run)
        if "padchest_gr" in todo:
            build_padchest_gr(run)


if __name__ == "__main__":
    main()
