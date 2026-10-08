"""Item 10: finding-by-region labels and side labels per image from Chest ImaGenome, on the frozen splits.

Input: data/features/imagenome_region_labels.parquet (parsed from scene_graph.zip), kg/imagenome_map.yaml.
Region label per (image, region, finding): 1 if any mapped label is asserted there, else 0 if a mapped
label is explicitly negated there, else absent from the table (not mentioned for that region). The
is_a closure (findings.yaml) is applied per region.
Side per (image, finding), from lateral regions only: left / right / bilateral / unlateralised
(positive only in non-lateral regions) / negative (no positive region). Non-lateralisable findings
get side NA.

Outputs: data/features/imagenome_finding_region_v1.parquet (long), imagenome_side_v1.parquet (wide).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from nesy import kg, manifests as M, paths as P  # noqa: E402
from nesy.runlog import Run  # noqa: E402

with Run("imagenome-labels", {"map": str(P.V2 / "kg/imagenome_map.yaml"), "split_version": M.SPLIT_VERSION}) as run:
    mp = yaml.safe_load((P.V2 / "kg/imagenome_map.yaml").read_text())
    lab = pd.read_parquet(P.FEATURES / "imagenome_region_labels.parquet")
    run.log(f"in: {len(lab):,} (image, region, label) rows, {lab.dicom_id.nunique():,} images")
    parts = []
    for f, spec in mp["map"].items():
        m = lab.label.isin(spec.get("labels", [])) | lab.category.isin(spec.get("categories", []))
        x = lab[m].groupby(["dicom_id", "region"]).agg(n_yes=("n_yes", "sum"), n_no=("n_no", "sum")).reset_index()
        x["finding"] = f
        parts.append(x)
        run.log(f"  {f:28s} mapped label rows {int(m.sum()):,}")
    reg = pd.concat(parts, ignore_index=True)
    reg["value"] = np.where(reg.n_yes > 0, 1, 0)
    # is_a closure per region: a positive child makes the parent positive at the same region
    n_before = int(reg.value.sum())
    for f in kg.topo_order():
        for p in kg.parents_of(f):
            pos = reg[(reg.finding == f) & (reg.value == 1)][["dicom_id", "region"]].assign(finding=p, n_yes=0, n_no=0, value=1)
            reg = pd.concat([reg, pos]).sort_values("value", ascending=False).drop_duplicates(["dicom_id", "region", "finding"])
    run.log(f"region positives before/after is_a closure: {n_before:,} / {int(reg.value.sum()):,}")

    sp = M.load("imagenome_silver", columns=["dicom_id", "subject_id", "study_id"])
    reg = reg.merge(sp[["dicom_id", "subject_id", "study_id", "split"]], on="dicom_id", how="left")
    reg[["dicom_id", "study_id", "subject_id", "split", "region", "finding", "value", "n_yes", "n_no"]].to_parquet(
        P.FEATURES / "imagenome_finding_region_v1.parquet", index=False)
    run.log(f"wrote finding-by-region table: {len(reg):,} rows")

    L, R = set(mp["lateral_regions"]["left"]), set(mp["lateral_regions"]["right"])
    pos = reg[reg.value == 1]
    side_rows = sp[["dicom_id", "subject_id", "study_id", "split"]].copy()
    lat = {f["id"]: f["lateralisable"] for f in kg.load_findings()["findings"]}
    for f in kg.finding_ids():
        pf = pos[pos.finding == f]
        has_l = side_rows.dicom_id.isin(set(pf[pf.region.isin(L)].dicom_id))
        has_r = side_rows.dicom_id.isin(set(pf[pf.region.isin(R)].dicom_id))
        has_any = side_rows.dicom_id.isin(set(pf.dicom_id))
        side = np.select([has_l & has_r, has_l, has_r, has_any], ["bilateral", "left", "right", "unlateralised"], "negative")
        side_rows[f"side_{f}"] = side if lat[f] else np.where(has_any, "not_lateralisable", "negative")
        vc = pd.Series(side_rows[f"side_{f}"]).value_counts().to_dict()
        run.log(f"  side {f:28s} {vc}")
        run.metric(finding=f, **{k: int(v) for k, v in vc.items()})
    side_rows.to_parquet(P.FEATURES / "imagenome_side_v1.parquet", index=False)
    run.log(f"wrote side table: {len(side_rows):,} images; by split {side_rows.split.value_counts().to_dict()}")
