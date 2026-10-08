"""Parse Chest ImaGenome silver scene graphs (read from scene_graph.zip in place) into two tables:

  data/features/imagenome_regions.parquet   one row per (dicom_id, region): box in ORIGINAL image
                                            pixels (original_x1..y2) and in the 224x224 frame (x1..y2)
  data/features/imagenome_region_labels.parquet
                                            one row per (dicom_id, region, category, label): how many
                                            report phrases assert it (n_yes) or negate it (n_no)

The region boxes are what Stage 4 needs at CLEAR extraction time, to pool patch features per region.
Labels come from the silver NLP pipeline over the report, so they are as noisy as that pipeline.
Comparison relationships between studies are not parsed here.

Resumable: members are processed in fixed shards; existing shard files are skipped.

  python -u v2/scripts/imagenome_parse.py --workers 16
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from nesy import paths as P  # noqa: E402
from nesy.runlog import Progress, Run  # noqa: E402

ZIP = P.IMAGENOME / "silver_dataset/scene_graph.zip"
SHARD = 5000
_zip = None


def _init():
    global _zip
    _zip = zipfile.ZipFile(ZIP)


def parse_one(member: str):
    d = json.loads(_zip.read(member))
    did = d["image_id"]
    regions = []
    for o in d.get("objects", []):
        regions.append({"dicom_id": did, "region": o["bbox_name"],
                        **{k: o.get(k) for k in ("x1", "y1", "x2", "y2", "original_x1", "original_y1", "original_x2",
                                                 "original_y2", "original_width", "original_height")}})
    labels: Counter = Counter()
    bad = 0
    for a in d.get("attributes", []):
        region = a.get("bbox_name")
        for phrase_attrs in a.get("attributes", []):
            for s in set(phrase_attrs):
                parts = s.split("|")
                if len(parts) != 3:
                    bad += 1
                    continue
                cat, pol, lab = parts
                labels[(region, cat, lab, pol)] += 1
    lab_rows = {}
    for (region, cat, lab, pol), n in labels.items():
        r = lab_rows.setdefault((region, cat, lab), {"dicom_id": did, "region": region, "category": cat, "label": lab,
                                                    "n_yes": 0, "n_no": 0})
        r["n_yes" if pol == "yes" else "n_no"] += n
    meta = {"dicom_id": did, "viewpoint": d.get("viewpoint"), "study_id": d.get("study_id"),
            "patient_id": d.get("patient_id"), "n_objects": len(regions), "n_attr_regions": len(d.get("attributes", [])),
            "n_relationships": len(d.get("relationships", [])), "bad_attr_strings": bad}
    return regions, list(lab_rows.values()), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("imagenome-parse", {**vars(args), "zip": str(ZIP), "shard": SHARD}, args.run_dir) as run:
        man = pd.read_parquet(P.MANIFESTS / "imagenome_silver.parquet", columns=["dicom_id", "zip_member", "subject_id", "study_id"])
        members = man.zip_member.tolist()[: args.limit]
        run.log(f"scene graphs to parse: {len(members):,}")
        out_dir = P.FEATURES / ("imagenome_shards" if args.limit is None else "imagenome_shards_smoke")
        out_dir.mkdir(parents=True, exist_ok=True)
        prog = Progress(run, len(members), "scene graphs parsed")
        with Pool(args.workers, initializer=_init) as pool:
            for si, start in enumerate(range(0, len(members), SHARD)):
                done = min(start + SHARD, len(members))
                paths = [out_dir / f"{kind}_{si:04d}.parquet" for kind in ("regions", "labels", "meta")]
                if all(p.exists() for p in paths):
                    prog.update(done)
                    continue
                res = pool.map(parse_one, members[start:done], chunksize=50)
                for p, rows in zip(paths, ([r for x in res for r in x[0]], [r for x in res for r in x[1]], [x[2] for x in res])):
                    tmp = p.with_suffix(".tmp")
                    pd.DataFrame(rows).to_parquet(tmp, index=False)
                    tmp.replace(p)
                prog.update(done)
        prog.update(len(members), force=True)

        def cat(kind):
            return pd.concat([pd.read_parquet(p) for p in sorted(out_dir.glob(f"{kind}_*.parquet"))], ignore_index=True)

        regions, labels, meta = cat("regions"), cat("labels"), cat("meta")
        # consistency with the MIMIC manifest
        chk = meta.merge(man, on="dicom_id")
        mism = ((chk.patient_id != chk.subject_id) | (chk.study_id_x != chk.study_id_y)).sum()
        run.log(f"scene graphs parsed {len(meta):,}; patient/study id mismatches vs MIMIC manifest: {mism}")
        run.log(f"region rows {len(regions):,} ({regions.region.nunique()} distinct regions); "
                f"regions per image median {meta.n_objects.median():.0f}, images with 0 regions {(meta.n_objects == 0).sum():,}")
        run.log(f"label rows {len(labels):,}; categories {labels.category.value_counts().to_dict()}; "
                f"malformed attribute strings {int(meta.bad_attr_strings.sum())}")
        top = labels[labels.category.isin(["anatomicalfinding", "disease"])].groupby("label").n_yes.apply(lambda s: (s > 0).sum())
        run.log("most frequent positive findings (images x regions): " + ", ".join(f"{k} {v:,}" for k, v in top.sort_values(ascending=False).head(15).items()))
        run.log("region list: " + ", ".join(sorted(regions.region.unique())))
        sfx = "" if args.limit is None else "_smoke"
        for name, df in [("imagenome_regions", regions), ("imagenome_region_labels", labels), ("imagenome_meta", meta)]:
            p = P.FEATURES / f"{name}{sfx}.parquet"
            df.to_parquet(p.with_suffix(".tmp"), index=False)
            p.with_suffix(".tmp").replace(p)
            run.log(f"wrote {p} ({len(df):,} rows, {p.stat().st_size / 1e6:.0f} MB)")
        run.metric(n_graphs=len(meta), n_region_rows=len(regions), n_label_rows=len(labels), id_mismatches=int(mism))


if __name__ == "__main__":
    main()
