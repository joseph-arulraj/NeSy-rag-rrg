"""Stretch vs letterbox preprocessing for CLEAR, on the full val set, with no training.

  stretch    clear.hub.build_cxr_preprocess as released: bicubic resize to exactly 448x448 (what we use)
  letterbox  the CLEAR authors' dataset preprocessing (src/clear/data_processing.py: preprocess):
             grey, aspect-preserving LANCZOS resize so the longest side is 448, zero-padded to 448x448;
             then the same normalisation (the hub preprocess on a 448x448 image only converts/normalises)
Compared, per patient bootstrap (2,000 resamples, 95 % percentile CI), letterbox minus stretch:
  * each of the 77 approved concepts' AUROC for its own finding (cosine of image and concept text
    embedding; normal concepts scored against the absence of any_abnormality), and their mean
  * per-finding zero-shot AUROC with CLEAR's own method (clear.zero_shot: softmax over the
    "{}" / "no {}" prompt pair, the authors' benchmark templates), and the macro mean
Decision rule (user, 2026-10-06): switch to letterbox only if it is better on average with a CI
excluding zero.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from nesy import clear_model, data as D, manifests as M, paths as P  # noqa: E402
from nesy.runlog import Progress, Run  # noqa: E402

ZS_NAMES = {"atelectasis": "atelectasis", "cardiomegaly": "cardiomegaly", "consolidation": "consolidation", "edema": "edema",
            "enlarged_cardiomediastinum": "enlarged cardiomediastinum", "fracture": "fracture", "lung_lesion": "lung lesion",
            "lung_opacity": "lung opacity", "pleural_effusion": "pleural effusion", "pleural_other": "pleural other",
            "pneumonia": "pneumonia", "pneumothorax": "pneumothorax", "support_devices": "support devices"}


def letterbox(im, size=448):
    from PIL import Image
    im = im.convert("L")
    r = size / max(im.size)
    new = tuple(int(x * r) for x in im.size)
    im = im.resize(new, Image.LANCZOS)
    out = Image.new("L", (size, size))
    out.paste(im, ((size - new[0]) // 2, (size - new[1]) // 2))
    return out


class DS:
    def __init__(self, paths, pre, mode):
        self.paths, self.pre, self.mode = paths, pre, mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        im = Image.open(self.paths[i])
        im.load()
        if self.mode == "letterbox":
            im = letterbox(im)
        return i, self.pre(im)


def auroc(y, s):
    m = ~np.isnan(y)
    y, s = y[m], s[m]
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("letterbox-test", vars(args), args.run_dir) as run:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        from clear.zero_shot import zeroshot_classifier
        t = D.study_table()
        v = t[t.split == "val"].reset_index(drop=True)
        man = M.load("mimic", columns=["dicom_id", "image_path"])
        v = v.merge(man[["dicom_id", "image_path"]], on="dicom_id")
        run.log(f"val studies: {len(v):,}, patients {v.subject_id.nunique():,}")
        model, pre = clear_model.load("cuda")
        emb = {}
        for mode in ("stretch", "letterbox"):
            dl = DataLoader(DS(v.image_path.tolist(), pre, mode), batch_size=64, num_workers=args.workers)
            out = np.zeros((len(v), 768), np.float32)
            prog = Progress(run, len(v), f"{mode} images", every_s=60)
            n = 0
            with torch.inference_mode():
                for idx, x in dl:
                    out[idx.numpy()] = F.normalize(model.encode_image(x.cuda()).float(), dim=-1).cpu().numpy()
                    n += len(idx)
                    prog.update(n)
            emb[mode] = out
            np.save(run.dir / f"val_{mode}.npy", out.astype(np.float16))
        cached = D.embeddings(v.row.values)
        cached /= np.linalg.norm(cached, axis=1, keepdims=True)
        run.log(f"stretch vs cached store: cosine min {(emb['stretch'] * cached).sum(1).min():.5f}; "
                f"letterbox vs stretch: cosine median {np.median((emb['letterbox'] * emb['stretch']).sum(1)):.4f}, "
                f"min {(emb['letterbox'] * emb['stretch']).sum(1).min():.4f}")

        # concept scores
        con = pd.read_csv(P.V2 / "data/concepts/concepts_final_k6_pruned.csv")
        CE = torch.load(clear_model.OLD / "model_weights/concept_embeddings_368294.pt", map_location="cpu", weights_only=False, mmap=True)
        C = CE[torch.as_tensor(con.concept_id.values)].float().numpy()
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        lab = v.copy()
        lab["no_finding"] = 1 - lab["any_abnormality"]
        Ycon = np.stack([lab[f].values.astype(float) for f in con.finding], 1)
        # zero-shot (CLEAR's own method)
        with torch.no_grad():
            Wp = zeroshot_classifier([ZS_NAMES[f] for f in ZS_NAMES], ["{}"], model).float().cpu().numpy()
            Wn = zeroshot_classifier([ZS_NAMES[f] for f in ZS_NAMES], ["no {}"], model).float().cpu().numpy()
        Yzs = np.stack([v[f].values.astype(float) for f in ZS_NAMES], 1)
        scores = {}
        for mode, E in emb.items():
            lp, ln = E @ Wp, E @ Wn
            zs = np.exp(lp) / (np.exp(lp) + np.exp(ln))
            scores[mode] = {"concept": E @ C.T, "zeroshot": zs}

        def all_aucs(sel):
            r = {}
            for mode in emb:
                r[(mode, "concept")] = np.array([auroc(Ycon[sel, j], scores[mode]["concept"][sel, j]) for j in range(Ycon.shape[1])])
                r[(mode, "zeroshot")] = np.array([auroc(Yzs[sel, j], scores[mode]["zeroshot"][sel, j]) for j in range(Yzs.shape[1])])
            return r
        point = all_aucs(np.arange(len(v)))
        pats = v.subject_id.values
        up, inv = np.unique(pats, return_inverse=True)
        groups = [np.nonzero(inv == i)[0] for i in range(len(up))]
        rng = np.random.default_rng(0)
        boot = {k: [] for k in ("concept", "zeroshot")}
        prog = Progress(run, args.n_boot, "bootstrap", every_s=60)
        for b in range(args.n_boot):
            sel = np.concatenate([groups[i] for i in rng.integers(0, len(up), len(up))])
            r = all_aucs(sel)
            for k in boot:
                boot[k].append(r[("letterbox", k)] - r[("stretch", k)])
            prog.update(b + 1)
        rows = []
        for k, names in (("concept", [f"{f}: {txt}" for f, txt in zip(con.finding, con.text)]), ("zeroshot", list(ZS_NAMES))):
            B = np.array(boot[k])
            d = point[("letterbox", k)] - point[("stretch", k)]
            for j, nm in enumerate(names):
                lo, hi = np.nanpercentile(B[:, j], [2.5, 97.5])
                rows.append({"test": k, "item": nm, "stretch": round(point[("stretch", k)][j], 4), "letterbox": round(point[("letterbox", k)][j], 4),
                             "diff": round(d[j], 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4), "excludes_zero": bool(lo > 0 or hi < 0)})
            mlo, mhi = np.nanpercentile(np.nanmean(B, 1), [2.5, 97.5])
            rows.append({"test": k, "item": "MEAN", "stretch": round(np.nanmean(point[("stretch", k)]), 4),
                         "letterbox": round(np.nanmean(point[("letterbox", k)]), 4), "diff": round(np.nanmean(d), 4),
                         "ci_low": round(mlo, 4), "ci_high": round(mhi, 4), "excludes_zero": bool(mlo > 0 or mhi < 0)})
        out = pd.DataFrame(rows)
        out.to_csv(run.dir / "letterbox_vs_stretch.csv", index=False)
        pd.set_option("display.width", 220)
        pd.set_option("display.max_rows", 200)
        pd.set_option("display.max_colwidth", 70)
        run.log("LETTERBOX minus STRETCH (val, patient bootstrap):\n" + out.to_string(index=False))
        for k in ("concept", "zeroshot"):
            s = out[(out.test == k) & (out.item != "MEAN")]
            run.log(f"{k}: letterbox better (CI > 0) {int(((s.ci_low > 0)).sum())}, worse (CI < 0) {int((s.ci_high < 0).sum())}, "
                    f"unclear {int((~s.excludes_zero).sum())} of {len(s)}")
        m = out[out.item == "MEAN"]
        run.log("DECISION INPUT (means):\n" + m.to_string(index=False))


if __name__ == "__main__":
    main()
