"""Stage 4: region classifier (A2) on pooled CLEAR patch features, for two region sets, plus the
MS-CXR localisation check.

Region labels per (image, region, finding), explicit labels only (user decision 2026-10-06):
  1       ImaGenome positive whose report phrase names the location (imagenome_region_explicit_v1)
  0       every region of a study whose label for the finding is negative (labels_v2)
  masked  everything else (default-assigned positives, unmentioned regions of positive studies)
Region set A: the 36 ImaGenome boxes. Region set B: the 9 CheXmask regions; a CheXmask region is
positive if an explicit positive ImaGenome region maps to it (B_FROM_A below).
Model: per finding and region set, one logistic regression on [pooled vector (768), region one-hot],
fit on the train split (all explicit positives, negatives subsampled to 4x positives, max 400k rows),
on GPU (full-batch L-BFGS). Validation: region-level AUROC on val, and image-level AUROC of the
max region score vs the study label.
MS-CXR check (MS-CXR patients are never in the fit split): for each phrase-box, take the finding's
highest-scoring region among anatomically relevant regions; hit = the region overlaps a box of that
phrase; side correct = the region's patient side equals the box side (image-centre test, G3).
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nesy import grounding as GR, kg, manifests as M, paths as P  # noqa: E402
from nesy.regions import CHEXMASK_REGIONS, GRID, box_weights  # noqa: E402
from nesy.runlog import Run  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from region_features import IG_REGIONS  # noqa: E402

FIND = [f for f in kg.finding_ids() if f not in ("any_abnormality", "support_devices")]
B_FROM_A = {
    "lung_left": ["left lung", "left apical zone", "left upper lung zone", "left mid lung zone", "left lower lung zone",
                  "left hilar structures", "left costophrenic angle"],
    "lung_right": ["right lung", "right apical zone", "right upper lung zone", "right mid lung zone", "right lower lung zone",
                   "right hilar structures", "right costophrenic angle"],
    "heart": ["cardiac silhouette", "left cardiac silhouette", "right cardiac silhouette"],
    "lung_left_upper": ["left apical zone", "left upper lung zone"],
    "lung_left_middle": ["left mid lung zone", "left hilar structures"],
    "lung_left_lower": ["left lower lung zone", "left costophrenic angle", "left hemidiaphragm"],
    "lung_right_upper": ["right apical zone", "right upper lung zone"],
    "lung_right_middle": ["right mid lung zone", "right hilar structures"],
    "lung_right_lower": ["right lower lung zone", "right costophrenic angle", "right hemidiaphragm"],
}
MSCXR_TO_F = {"Cardiomegaly": "cardiomegaly", "Lung Opacity": "lung_opacity", "Edema": "edema", "Consolidation": "consolidation",
              "Pneumonia": "pneumonia", "Atelectasis": "atelectasis", "Pneumothorax": "pneumothorax", "Pleural Effusion": "pleural_effusion"}
SIDE_A = {r: ("left" if r.startswith("left ") else "right" if r.startswith("right ") else None) for r in IG_REGIONS}
SIDE_B = {r: ("left" if "left" in r else "right" if "right" in r else None) for r in CHEXMASK_REGIONS}


def load_shards(name):
    d = P.FEATURES / "regions" / f"{name}_shards"
    ks = sorted(int(Path(f).name.split("_")[1]) for f in glob.glob(str(d / "shard_*_global.npy")))
    out = {k: [] for k in ("rows", "imagenome", "chexmask", "imagenome_present", "chexmask_present")}
    for k in ks:
        for key in out:
            out[key].append(np.load(d / f"shard_{k:04d}_{key}.npy"))
    return {k: np.concatenate(v) for k, v in out.items()}


def hash_fold(subject_id, k):
    import hashlib
    return int(hashlib.sha256(f"s4fold:{subject_id}".encode()).hexdigest()[:8], 16) % k


def fit_gpu(X, y, l2=1e-3, iters=200):
    import torch
    Xt = torch.tensor(X, dtype=torch.float32, device="cuda")
    yt = torch.tensor(y, dtype=torch.float32, device="cuda")
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    Xt = (Xt - mu) / sd
    w = torch.zeros(X.shape[1], device="cuda", requires_grad=True)
    b = torch.zeros(1, device="cuda", requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt) + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return {"w": w.detach().cpu().numpy(), "b": float(b.detach().cpu()), "mu": mu.cpu().numpy(), "sd": sd.cpu().numpy()}


def score(m, X):
    return ((X - m["mu"]) / m["sd"]) @ m["w"] + m["b"]


def score_region(m, V, ok, j, onehot, chunk=65536):
    """Same as score(m, [V[ok, j], onehot[j]]) but without building the big matrix: the model is linear, so
    score = V[ok, j] @ (w_v / sd_v) + constant(j), computed on the GPU in chunks."""
    import torch
    d = V.shape[2]
    wv = torch.tensor(m["w"][:d] / m["sd"][:d], dtype=torch.float32, device="cuda")
    const = float(m["b"] - (m["mu"][:d] / m["sd"][:d]) @ m["w"][:d] + ((onehot[j] - m["mu"][d:]) / m["sd"][d:]) @ m["w"][d:])
    rows = np.nonzero(ok)[0]
    out = np.empty(len(rows), np.float32)
    for a in range(0, len(rows), chunk):
        x = torch.tensor(np.asarray(V[rows[a:a + chunk], j], np.float32), device="cuda")
        out[a:a + chunk] = (x @ wv).cpu().numpy() + const
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="mimic_full")
    ap.add_argument("--max-rows", type=int, default=400000)
    ap.add_argument("--crossfit", type=int, default=2,
                    help="K-fold by patient within the fit split: fit-split images are scored by the fold model that did "
                         "not see them, so the scores can be used as features downstream (Stage 4 ablation); other "
                         "splits use the average of the K fold models")
    ap.add_argument("--skip-mscxr", action="store_true", help="skip the built-in (lenient, stretch-geometry) MS-CXR check; "
                                                            "use scripts/mscxr_loc_eval.py instead")
    ap.add_argument("--save-models", action="store_true", help="save the fitted per-finding fold models (models.joblib)")
    ap.add_argument("--label-splits", default="all", help="comma list of splits whose labels are loaded (e.g. train,val); "
                    "other splits get no labels (fitting uses train only; val only for the logged AUROCs)")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("stage4-regions", vars(args), args.run_dir) as run:
        from sklearn.metrics import roc_auc_score
        S = load_shards(args.shards)
        ids = pd.read_parquet(P.FEATURES / "regions" / f"{args.shards}_ids.parquet").image_id.values
        img = pd.DataFrame({"dicom_id": ids[S["rows"]]})
        man = M.load("mimic", columns=["dicom_id", "subject_id", "study_id", "image_path", "rows", "cols", "view"])
        img = img.merge(man, on="dicom_id", how="left")
        assert img.subject_id.notna().all()
        if args.label_splits == "all":
            lab = pd.read_parquet(P.V2 / "data/labels/labels_v2.parquet").drop(columns=["subject_id", "split"])
        else:
            import pyarrow.parquet as pq
            keep = args.label_splits.split(",")
            lab = pq.read_table(P.V2 / "data/labels/labels_v2.parquet", filters=[("split", "in", keep)]).to_pandas().drop(columns=["subject_id", "split"])
            run.log(f"labels loaded for splits {keep} only: {len(lab):,} studies")
        img = img.merge(lab, on="study_id", how="left")
        run.log(f"images with pooled features: {len(img):,}; splits {img.split.value_counts().to_dict()}")
        ex = pd.read_parquet(P.FEATURES / "imagenome_region_explicit_v1.parquet")
        ex = ex[ex.explicit]
        # vectorised explicit-positive lookup: per finding, a boolean [n_images, region] matrix for set A names
        img_row = {d: i for i, d in enumerate(img.dicom_id)}
        A_IDX = {r: j for j, r in enumerate(IG_REGIONS)}
        posA = {}
        for f, g in ex.groupby("finding"):
            m = np.zeros((len(img), len(IG_REGIONS)), bool)
            ri = g.dicom_id.map(img_row)
            ci = g.region.map(A_IDX)
            ok = ri.notna() & ci.notna()
            m[ri[ok].astype(int).values, ci[ok].astype(int).values] = True
            posA[f] = m
        fold_of_img = np.array([hash_fold(s_, args.crossfit) for s_ in img.subject_id.values])
        rng = np.random.default_rng(0)
        scores, summary, saved = {}, [], {}
        for set_name, vec_key, pres_key, regions in (("A_imagenome", "imagenome", "imagenome_present", IG_REGIONS),
                                                     ("B_chexmask", "chexmask", "chexmask_present", CHEXMASK_REGIONS)):
            V, PR = S[vec_key], S[pres_key]
            R = len(regions)
            onehot = np.eye(R, dtype=np.float32)
            scores[set_name] = {}
            for f in FIND:
                pm = posA.get(f, np.zeros((len(img), len(IG_REGIONS)), bool))
                Y = np.full((len(img), R), np.nan, np.float32)
                neg_img = (img[f].values == 0)
                Y[neg_img] = 0
                for j, r in enumerate(regions):
                    srcs = [r] if set_name.startswith("A") else B_FROM_A[r]
                    hit = pm[:, [A_IDX[s] for s in srcs if s in A_IDX]].any(1)
                    Y[hit & (img[f].values == 1), j] = 1
                Y[~PR] = np.nan
                fitm = (img.split == "train").values[:, None] & ~np.isnan(Y)
                ii, jj = np.nonzero(fitm)
                yy = Y[ii, jj]
                pi, ni = np.nonzero(yy == 1)[0], np.nonzero(yy == 0)[0]
                if len(pi) < 50:
                    run.log(f"  {set_name} {f}: only {len(pi)} explicit positive regions in train; skipped")
                    continue
                n_pos = min(len(pi), args.max_rows // 5)
                pi = rng.choice(pi, n_pos, replace=False)
                ni = rng.choice(ni, min(len(ni), 4 * n_pos), replace=False)
                sel = np.concatenate([pi, ni])
                models_k = []
                for k in range(args.crossfit):
                    keep = (fold_of_img[ii[sel]] != k) if args.crossfit > 1 else np.ones(len(sel), bool)
                    s_k = sel[keep]
                    models_k.append(fit_gpu(np.hstack([V[ii[s_k], jj[s_k]].astype(np.float32), onehot[jj[s_k]]]), yy[s_k]))
                # score every (image, region) present: fit-split images by the fold model that did not see them
                sc = np.full((len(img), R), np.nan, np.float32)
                is_fit = (img.split == "train").values
                for j in range(R):
                    ok = PR[:, j]
                    if not ok.any():
                        continue
                    per_k = np.stack([score_region(mk, V, ok, j, onehot) for mk in models_k])   # [K, n]
                    fk = fold_of_img[ok]
                    out_j = per_k.mean(0)
                    fit_ok = is_fit[ok]
                    out_j[fit_ok] = per_k[fk[fit_ok], np.nonzero(fit_ok)[0]]
                    sc[ok, j] = out_j
                scores[set_name][f] = sc
                saved.setdefault(set_name, {})[f] = models_k
                vm = (img.split == "val").values
                vr = vm[:, None] & ~np.isnan(Y) & ~np.isnan(sc)
                auc_r = roc_auc_score(Y[vr], sc[vr]) if len(np.unique(Y[vr])) == 2 else np.nan
                imgmax = np.nanmax(np.where(np.isnan(sc), -np.inf, sc), axis=1)
                vi = vm & ~np.isnan(img[f].values) & np.isfinite(imgmax)
                auc_i = roc_auc_score(img[f].values[vi], imgmax[vi]) if len(np.unique(img[f].values[vi])) == 2 else np.nan
                summary.append({"region_set": set_name, "finding": f, "train_pos_regions": int((yy == 1).sum()),
                                "val_region_auroc": auc_r, "val_image_auroc_max_region": auc_i})
                run.log(f"  {set_name:12s} {f:28s} train explicit-positive regions {int((yy == 1).sum()):,}; "
                        f"val region AUROC {auc_r:.3f}; image AUROC (max region) {auc_i:.3f}")
                run.metric(region_set=set_name, finding=f, val_region_auroc=auc_r, val_image_auroc=auc_i)
            np.save(run.dir / f"scores_{set_name}.npy", np.stack([scores[set_name].get(f, np.full((len(img), R), np.nan, np.float32))
                                                                   for f in FIND], axis=-1).astype(np.float16))
        pd.DataFrame(summary).to_csv(run.dir / "stage4_val.csv", index=False)
        img[["dicom_id", "study_id", "split"]].to_parquet(run.dir / "scores_index.parquet", index=False)
        if args.save_models:
            import joblib
            joblib.dump({"models": saved, "regions": {"A_imagenome": IG_REGIONS, "B_chexmask": CHEXMASK_REGIONS},
                         "findings": FIND, "crossfit": args.crossfit,
                         "note": "per set and finding: list of K fold models {w, b, mu, sd} on [region vector (768), region one-hot]; "
                                 "score = ((x - mu) / sd) @ w + b"}, run.dir / "models.joblib")
            run.log(f"saved fold models in {run.dir / 'models.joblib'}")

        if args.skip_mscxr:
            run.log("built-in MS-CXR check skipped (use scripts/mscxr_loc_eval.py)")
            return
        # ---- MS-CXR localisation check
        ms = M.load("mscxr", columns=["dicom_id", "phrase_id", "category", "x", "y", "w", "h", "image_width", "image_height", "view"])
        pos_img = {d: i for i, d in enumerate(img.dicom_id)}
        rows = []
        wsets = {}
        for set_name in scores:
            for (did, pid), g in ms.groupby(["dicom_id", "phrase_id"]):
                f = MSCXR_TO_F.get(g.category.iat[0])
                if f not in scores[set_name] or did not in pos_img:
                    continue
                i = pos_img[did]
                sc = scores[set_name][f][i]
                regions = IG_REGIONS if set_name.startswith("A") else CHEXMASK_REGIONS
                if f == "cardiomegaly":
                    allowed = [j for j, r in enumerate(regions) if "cardiac" in r or r == "heart"]
                else:
                    allowed = [j for j, r in enumerate(regions) if ("lung" in r or "zone" in r or "costophrenic" in r or "hilar" in r)]
                allowed = [j for j in allowed if not np.isnan(sc[j])]
                if not allowed:
                    continue
                jbest = allowed[int(np.argmax(sc[allowed]))]
                W, H = int(g.image_width.iat[0]), int(g.image_height.iat[0])
                boxmask = np.zeros((GRID, GRID), np.float32)
                for b in g.itertuples():
                    boxmask = np.maximum(boxmask, box_weights(b.x, b.y, b.x + b.w, b.y + b.h, W, H))
                # region weight on the grid
                if set_name.startswith("A"):
                    rg = REGION_GRID_A(did, regions[jbest], W, H)
                else:
                    rg = REGION_GRID_B(did, jbest)
                hit = float((rg * boxmask).sum()) > 0 if rg is not None else np.nan
                xs = [(b.x + b.w / 2) / W for b in g.itertuples()]
                box_sides = {GR.image_to_patient(x, "PA") for x in xs} - {"midline"}
                box_side = "bilateral" if box_sides == {"left", "right"} else (box_sides.pop() if box_sides else None)
                rside = (SIDE_A if set_name.startswith("A") else SIDE_B)[regions[jbest]]
                side_ok = (rside == box_side) if (rside and box_side in ("left", "right")) else np.nan
                rows.append({"region_set": set_name, "finding": f, "dicom_id": did, "phrase_id": pid, "best_region": regions[jbest],
                             "hit": hit, "box_side": box_side, "region_side": rside, "side_correct": side_ok})
        loc = pd.DataFrame(rows)
        loc.to_csv(run.dir / "mscxr_localisation.csv", index=False)
        t = loc.groupby(["region_set", "finding"]).agg(n=("hit", "size"), hit_rate=("hit", "mean"),
                                                       side_correct=("side_correct", "mean"), n_side=("side_correct", "count"))
        run.log("MS-CXR localisation (highest-scoring relevant region overlaps the radiologist box; side correct):\n" + t.round(3).to_string())
        run.log("overall: " + loc.groupby("region_set")[["hit", "side_correct"]].mean().round(3).to_string())


_REG = None
_W = None


def REGION_GRID_A(did, region, W, H):
    global _REG
    if _REG is None:
        _REG = pd.read_parquet(P.FEATURES / "imagenome_regions.parquet").set_index(["dicom_id", "region"])
    try:
        b = _REG.loc[(did, region)]
    except KeyError:
        return None
    if isinstance(b, pd.DataFrame):
        b = b.iloc[0]
    return box_weights(b.original_x1, b.original_y1, b.original_x2, b.original_y2, W, H)


def REGION_GRID_B(did, j):
    global _W
    if _W is None:
        idx = pd.read_parquet(P.FEATURES / "regions/mimic_full_index.parquet").set_index("image_id")
        _W = (idx, np.load(P.FEATURES / "regions/mimic_full_weights.npy", mmap_mode="r"))
    idx, w = _W
    if did not in idx.index or not idx.at[did, "found"]:
        return None
    return w[int(idx.at[did, "row"]), j].astype(np.float32) / 255.0


if __name__ == "__main__":
    main()
