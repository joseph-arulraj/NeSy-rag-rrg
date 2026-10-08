"""Case-based evidence (user item B, 2026-10-07): for every study, the k most similar FIT-split studies by
CLEAR embedding cosine (letterbox FAISS index on the fit split, data/index_lb), excluding every study of
the query's own patient, with their study IDs, similarities and labels only (no report text). Per finding:
how many of the k neighbours are positive and how many have a known label.

Output: data/features/cases_<variant>_k<k>.parquet with columns study_id, nb_study_ids (list), nb_sims
(list), case_pos_<finding>, case_known_<finding>.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("case-neighbours", {**vars(args), "variant": D.VARIANT}, args.run_dir) as run:
        import faiss
        faiss.omp_set_num_threads(args.threads)
        idx_dir = P.V2 / ("data/index" if D.VARIANT == "stretch" else "data/index_lb")
        index = faiss.read_index(str(idx_dir / "fit_v1.faiss"))
        fit = pd.read_parquet(idx_dir / "fit_v1_ids.parquet")
        t = D.study_table()
        lab = t.set_index("study_id")[D.FINDINGS]
        Y = lab.loc[fit.study_id].values                         # fit-split labels in index order (1 / 0 / NaN)
        X = D.embeddings(t.row.values)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        per_pat = fit.subject_id.value_counts()
        qn = t.subject_id.map(per_pat).fillna(0).astype(int).values
        fit_pat, fit_sid = fit.subject_id.values, fit.study_id.values
        run.log(f"index {index.ntotal:,} fit studies ({idx_dir.name}); queries {len(t):,}; k {args.k}")
        nb_ids, nb_sims = [None] * len(t), [None] * len(t)
        pos = np.zeros((len(t), len(D.FINDINGS)), np.int16)
        known = np.zeros((len(t), len(D.FINDINGS)), np.int16)
        prog = Progress(run, len(t), "queries", every_s=60)
        leaks = 0
        for s in range(0, len(t), args.chunk):
            e = min(s + args.chunk, len(t))
            sims, nn = index.search(X[s:e], int(args.k + qn[s:e].max()))
            own = fit_pat[nn] == t.subject_id.values[s:e, None]
            sims = np.where(own, -np.inf, sims)
            order = np.argsort(-sims, axis=1, kind="stable")[:, :args.k]
            nn = np.take_along_axis(nn, order, 1)
            sims = np.take_along_axis(sims, order, 1)
            assert np.isfinite(sims).all()
            leaks += int((fit_pat[nn] == t.subject_id.values[s:e, None]).sum())
            yk = Y[nn]                                            # [q, k, F]
            pos[s:e] = (np.nan_to_num(yk) == 1).sum(1)
            known[s:e] = (~np.isnan(yk)).sum(1)
            for i in range(e - s):
                nb_ids[s + i] = fit_sid[nn[i]].tolist()
                nb_sims[s + i] = np.round(sims[i], 4).tolist()
            prog.update(e)
        run.log(f"own-patient neighbours after exclusion: {leaks} (must be 0)")
        if leaks:
            raise RuntimeError("patient leak")
        out = pd.DataFrame({"study_id": t.study_id.values, "split": t.split.values, "nb_study_ids": nb_ids, "nb_sims": nb_sims,
                            **{f"case_pos_{f}": pos[:, j] for j, f in enumerate(D.FINDINGS)},
                            **{f"case_known_{f}": known[:, j] for j, f in enumerate(D.FINDINGS)}})
        o = P.FEATURES / f"cases_{D.VARIANT}_k{args.k}.parquet"
        out.to_parquet(o, index=False)
        run.log(f"wrote {o} ({len(out):,} studies)")


if __name__ == "__main__":
    main()
