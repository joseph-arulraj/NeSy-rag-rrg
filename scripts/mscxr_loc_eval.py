"""MS-CXR localisation evaluation of a Stage 4 run, with stricter overlap tests and chance baselines.

For each MS-CXR phrase (all its boxes together) and each region set:
  candidate regions = anatomically relevant regions present for the image (cardiomegaly: cardiac
  regions / heart; other findings: lung, zone, costophrenic and hilar regions / lungs and thirds)
  chosen region     = model: highest Stage 4 score; baselines: (a) random relevant region (expected
                      value over candidates), (b) always the largest candidate region
  tests on the 32x32 grid of the run's geometry:
    hit_any     region and box overlap at all (lenient; the earlier criterion)
    iou_0.1     fractional IoU(region, box) >= 0.1
    centre_in   a box centre falls in a patch where the region weight >= 0.5
    side        region side == box side (lateral regions, one-sided boxes only)
  "small regions only": the same with whole lungs and the whole cardiac silhouette / heart removed
  from the candidates (zones and thirds), which makes the test harder for the lenient criterion.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nesy import grounding as GR, manifests as M, paths as P  # noqa: E402
from nesy.regions import CHEXMASK_REGIONS, GRID, box_weights  # noqa: E402
from nesy.runlog import Run  # noqa: E402
from region_features import IG_REGIONS  # noqa: E402
from stage4_regions import FIND, MSCXR_TO_F  # noqa: E402

WHOLE = {"right lung", "left lung", "cardiac silhouette", "lung_left", "lung_right"}


def relevant(set_name, f):
    regions = IG_REGIONS if set_name == "A" else CHEXMASK_REGIONS
    if f == "cardiomegaly":
        return [j for j, r in enumerate(regions) if "cardiac" in r or r == "heart"]
    return [j for j, r in enumerate(regions) if ("lung" in r or "zone" in r or "costophrenic" in r or "hilar" in r)]


def side_of(r):
    return "left" if ("left" in r) else "right" if ("right" in r) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage4-run", required=True)
    ap.add_argument("--geometry", choices=["stretch", "letterbox"], required=True)
    ap.add_argument("--weights", required=True, help="comma list of CheXmask weight names (data/features/regions/<name>_*)")
    ap.add_argument("--label", default="")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("mscxr-loc-eval", vars(args), args.run_dir) as run:
        s4 = Path(args.stage4_run)
        idx = pd.read_parquet(s4 / "scores_index.parquet")
        pos = {d: i for i, d in enumerate(idx.dicom_id)}
        SC = {"A": np.load(s4 / "scores_A_imagenome.npy", mmap_mode="r"), "B": np.load(s4 / "scores_B_chexmask.npy", mmap_mode="r")}
        ms = M.load("mscxr", columns=["dicom_id", "phrase_id", "category", "x", "y", "w", "h", "image_width", "image_height"])
        ms = ms[ms.dicom_id.isin(pos)]
        regs = pd.read_parquet(P.FEATURES / "imagenome_regions.parquet", filters=[("dicom_id", "in", list(ms.dicom_id.unique()))])
        reg_by = {d: g.set_index("region") for d, g in regs.groupby("dicom_id")}
        from nesy import weights_store
        widx, wall = weights_store.load(args.weights)
        W8 = {d: np.asarray(wall[int(widx.at[d, "slot"])], np.float32) / 255.0 for d in set(ms.dicom_id) if d in widx.index}
        run.log(f"MS-CXR phrases {ms.phrase_id.nunique():,} on {ms.dicom_id.nunique():,} images; CheXmask grids for {len(W8):,} images")
        rng = np.random.default_rng(0)
        rows = []
        for (did, pid), g in ms.groupby(["dicom_id", "phrase_id"]):
            f = MSCXR_TO_F.get(g.category.iat[0])
            if f not in FIND:
                continue
            Wd, Hd = int(g.image_width.iat[0]), int(g.image_height.iat[0])
            box = np.zeros((GRID, GRID), np.float32)
            centres = []
            for b in g.itertuples():
                box = np.maximum(box, box_weights(b.x, b.y, b.x + b.w, b.y + b.h, Wd, Hd, args.geometry))
                centres.append(((b.x + b.w / 2), (b.y + b.h / 2)))
            sides = {GR.image_to_patient(cx / Wd, "PA") for cx, _ in centres} - {"midline"}
            box_side = sides.pop() if len(sides) == 1 else None
            cgrid = np.zeros((GRID, GRID), bool)
            from nesy.regions import canvas_map
            sx, sy, ox, oy = canvas_map(Wd, Hd, args.geometry)
            for cx, cy in centres:
                cgrid[min(GRID - 1, int((cy * sy + oy) / 14)), min(GRID - 1, int((cx * sx + ox) / 14))] = True
            for set_name in ("A", "B"):
                regions = IG_REGIONS if set_name == "A" else CHEXMASK_REGIONS
                sc = np.asarray(SC[set_name][pos[did], :, FIND.index(f)], np.float32)
                grids = {}
                for j in relevant(set_name, f):
                    if np.isnan(sc[j]):
                        continue
                    if set_name == "A":
                        rb = reg_by.get(did)
                        if rb is None or regions[j] not in rb.index:
                            continue
                        b = rb.loc[regions[j]]
                        b = b.iloc[0] if isinstance(b, pd.DataFrame) else b
                        grids[j] = box_weights(b.original_x1, b.original_y1, b.original_x2, b.original_y2, Wd, Hd, args.geometry)
                    else:
                        if did not in W8:
                            continue
                        grids[j] = W8[did][j]
                if not grids:
                    continue

                def tests(j):
                    rg = grids[j]
                    inter, union = np.minimum(rg, box).sum(), np.maximum(rg, box).sum()
                    rs = side_of(regions[j])
                    return {"hit_any": float((rg * box).sum() > 0), "iou_0.1": float(inter / max(union, 1e-9) >= 0.1),
                            "centre_in": float((rg[cgrid] >= 0.5).any()),
                            "side": (float(rs == box_side) if (rs and box_side) else np.nan)}
                for variant, cands in (("all", list(grids)), ("small_only", [j for j in grids if regions[j] not in WHOLE])):
                    if not cands:
                        continue
                    jm = cands[int(np.argmax(sc[cands]))]
                    jl = max(cands, key=lambda j: grids[j].sum())
                    tm, tl = tests(jm), tests(jl)
                    tr = pd.DataFrame([tests(j) for j in cands]).mean().to_dict()
                    for who, t in (("model", tm), ("largest_region", tl), ("random_region", tr)):
                        rows.append({"region_set": set_name, "variant": variant, "chooser": who, "finding": f, "dicom_id": did,
                                     "phrase_id": pid, **t})
        df = pd.DataFrame(rows)
        df.to_csv(run.dir / "mscxr_loc_detail.csv", index=False)
        pd.set_option("display.width", 250)
        pd.set_option("display.max_rows", 400)
        cols = ["hit_any", "iou_0.1", "centre_in", "side"]
        for variant in ("all", "small_only"):
            sub = df[df.variant == variant]
            t = sub.groupby(["region_set", "finding", "chooser"])[cols].mean().unstack("chooser")
            t = t.reindex(columns=pd.MultiIndex.from_product([cols, ["model", "random_region", "largest_region"]]))
            t.columns = [f"{a}|{b}" for a, b in t.columns]
            t["n"] = sub[sub.chooser == "model"].groupby(["region_set", "finding"]).size()
            run.log(f"MS-CXR localisation [{variant} candidate regions] (model vs baselines):\n" + t.round(3).to_string())
            ov = sub.groupby(["region_set", "chooser"])[cols].mean().round(3)
            run.log(f"OVERALL [{variant}]:\n" + ov.to_string())
            ov.to_csv(run.dir / f"overall_{variant}.csv")
            t.to_csv(run.dir / f"per_finding_{variant}.csv")


if __name__ == "__main__":
    main()
