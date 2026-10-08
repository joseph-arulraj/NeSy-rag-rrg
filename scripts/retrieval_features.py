"""Item 4: FAISS index on the fit (train) split only, and neighbour label features for every study.

For each study with a frontal embedding, in every split: the k nearest fit-split studies by cosine,
EXCLUDING every study of the query's own patient (so train queries never see themselves or their
other studies), and per finding the fraction of those neighbours that are positive, among neighbours
whose label for that finding is not masked. k in {5, 10, 25}; k is chosen on val in Stage 5.
Columns: nb{k}_frac_<finding>, nb{k}_n_known_<finding>, nb{k}_mean_sim.

Output: data/features/retrieval_v1.parquet, index data/index/fit_v1.faiss (+ ids parquet).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nesy import data as D, paths as P  # noqa: E402
from nesy.runlog import Progress, Run  # noqa: E402

KS = (5, 10, 25)
OUT = D.RETRIEVAL                                    # depends on NESY_EMB (stretch / letterbox)
IDX_DIR = P.V2 / ("data/index" if D.VARIANT == "stretch" else "data/index_lb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("retrieval-features", {**vars(args), "ks": KS, "embedding_variant": D.VARIANT}, args.run_dir) as run:
        import faiss
        faiss.omp_set_num_threads(args.threads)
        t = D.study_table()
        X = D.embeddings(t.row.values)
        X /= np.linalg.norm(X, axis=1, keepdims=True)           # renormalise after the fp16 cast
        fit = t[t.split == "train"].reset_index(drop=True)
        Xf = X[t.split.values == "train"]
        run.log(f"studies with an embedding: {len(t):,}; fit (train) corpus: {len(fit):,}; by split "
                f"{t.split.value_counts().to_dict()}")
        index = faiss.IndexFlatIP(Xf.shape[1])
        index.add(Xf)
        IDX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(IDX_DIR / "fit_v1.faiss"))
        fit[["row", "dicom_id", "study_id", "subject_id"]].to_parquet(IDX_DIR / "fit_v1_ids.parquet", index=False)
        run.log(f"wrote FAISS IndexFlatIP with {index.ntotal:,} vectors to {IDX_DIR / 'fit_v1.faiss'}")

        # patients' study counts in the corpus decide how far to over-fetch
        per_pat = fit.subject_id.value_counts()
        q_pat_count = t.subject_id.map(per_pat).fillna(0).astype(int).values
        kmax = max(KS)
        Y = fit[D.FINDINGS].values                                   # 1 / 0 / NaN
        known = ~np.isnan(Y)
        pos = np.nan_to_num(Y) == 1
        fit_pat = fit.subject_id.values

        res = {f"nb{k}_{s}_{f}": np.full(len(t), np.nan, np.float32) for k in KS for s in ("frac", "n_known") for f in D.FINDINGS}
        res.update({f"nb{k}_mean_sim": np.full(len(t), np.nan, np.float32) for k in KS})
        prog = Progress(run, len(t), "queries", every_s=60)
        leaks = 0
        for s in range(0, len(t), args.chunk):
            e = min(s + args.chunk, len(t))
            fetch = int(kmax + q_pat_count[s:e].max())
            sims, nn = index.search(X[s:e], fetch)
            own = fit_pat[nn] == t.subject_id.values[s:e, None]
            sims = np.where(own, -np.inf, sims)
            order = np.argsort(-sims, axis=1, kind="stable")[:, :kmax]
            nn = np.take_along_axis(nn, order, 1)
            sims = np.take_along_axis(sims, order, 1)
            if not np.isfinite(sims).all():
                raise RuntimeError("fewer than kmax neighbours after patient exclusion")
            leaks += int((fit_pat[nn] == t.subject_id.values[s:e, None]).sum())
            for k in KS:
                kn = known[nn[:, :k]]                                # [q, k, F]
                ps = pos[nn[:, :k]]
                nk = kn.sum(1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    fr = np.where(nk > 0, ps.sum(1) / nk, np.nan)
                for j, f in enumerate(D.FINDINGS):
                    res[f"nb{k}_frac_{f}"][s:e] = fr[:, j]
                    res[f"nb{k}_n_known_{f}"][s:e] = nk[:, j]
                res[f"nb{k}_mean_sim"][s:e] = sims[:, :k].mean(1)
            prog.update(e)
        prog.update(len(t), force=True)
        run.log(f"own-patient neighbours after exclusion: {leaks} (must be 0)")
        if leaks:
            raise RuntimeError("patient leak in neighbours")
        out = pd.concat([t[["study_id", "subject_id", "split"]].reset_index(drop=True), pd.DataFrame(res)], axis=1)
        out.to_parquet(OUT, index=False)
        run.log(f"wrote {OUT} ({len(out):,} studies, {out.shape[1]} columns)")
        # sanity: neighbour fraction vs own label, val split, k=10
        v = out[out.split == "val"].merge(t[["study_id"] + D.FINDINGS], on="study_id")
        from sklearn.metrics import roc_auc_score
        for f in D.FINDINGS:
            m = v[f].notna()
            if v.loc[m, f].nunique() == 2:
                a = roc_auc_score(v.loc[m, f], v.loc[m, f"nb10_frac_{f}"].fillna(0))
                run.log(f"  val AUROC of nb10_frac_{f} alone: {a:.3f}")
                run.metric(finding=f, val_auroc_nb10_frac=a)


if __name__ == "__main__":
    main()
