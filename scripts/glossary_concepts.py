"""KG task 7 (ablation only): glossary-derived concept list.

Phrases: data/concepts/glossary_phrases.csv (short phrases written from the terms the Fleischner glossary uses for
each finding; findings without glossary wording get none). Each phrase is encoded with CLEAR's text encoder
(clip tokenize -> model.encode_text -> L2 norm, as clear.zero_shot does). Check first: the same encoding is
applied to concept-bank sentences and must reproduce the stored bank embeddings (cosine >= --min-check-cos),
so the phrases are scored exactly as the 77 data-driven concepts are (cosine of unit image and text embeddings).
Univariate AUROC of each phrase for its own finding on the fit split (all fit studies with a known label),
0.60 floor as for the 77. Survivors -> data/concepts/concepts_glossary.csv (concept_id = row of
data/concepts/glossary_embeddings.pt), for stage3_factorised.py --concepts ... --concept-emb ....
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nesy import clear_model, data as D, paths as P  # noqa: E402
from nesy.runlog import Run  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_concepts import BANK_CSV, BANK_EMB, auroc_columns  # noqa: E402

OUT = P.V2 / "data/concepts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrases", default=str(OUT / "glossary_phrases.csv"))
    ap.add_argument("--reference", default=str(OUT / "concepts_final_k6_pruned.csv"), help="the 77 (for the encoder check)")
    ap.add_argument("--min-auroc", type=float, default=0.6)
    ap.add_argument("--min-check-cos", type=float, default=0.999)
    ap.add_argument("--run-dir", default=None)
    a = ap.parse_args()
    with Run("glossary-concepts", vars(a), a.run_dir) as run:
        import torch
        from clear import clip as clip_module
        model, _ = clear_model.load("cpu")

        def encode(texts):
            with torch.no_grad():
                e = model.encode_text(clip_module.tokenize(list(texts), context_length=77)).float()
            return (e / e.norm(dim=-1, keepdim=True)).numpy()

        # 1. encoder check against the stored bank embeddings (the 77 + 200 random bank sentences)
        bank = pd.read_csv(BANK_CSV)
        ref = pd.read_csv(a.reference).drop_duplicates("concept_id")
        ids = np.unique(np.concatenate([ref.concept_id.values, np.random.default_rng(0).choice(len(bank), 200, replace=False)]))
        stored = torch.load(BANK_EMB, map_location="cpu", weights_only=False, mmap=True)
        if isinstance(stored, dict):
            stored = next(v for v in stored.values() if hasattr(v, "shape"))
        S = stored[torch.as_tensor(ids)].float().numpy()
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        mine = np.vstack([encode(bank.concept.astype(str).iloc[ids[i:i + 64]]) for i in range(0, len(ids), 64)])
        cos = (S * mine).sum(1)
        run.log(f"encoder check on {len(ids)} bank sentences: cosine min {cos.min():.6f} median {np.median(cos):.6f}")
        if cos.min() < a.min_check_cos:
            raise RuntimeError(f"text encoding does not reproduce the bank embeddings (min cosine {cos.min():.4f})")

        # 2. encode the glossary phrases
        ph = pd.read_csv(a.phrases)
        ph["phrase"] = ph.phrase.str.strip()
        assert not ph.phrase.duplicated().any(), "duplicate phrases"
        E = encode(ph.phrase)
        torch.save(torch.from_numpy(E), OUT / "glossary_embeddings.pt")
        run.log(f"{len(ph)} phrases for {ph.finding.nunique()} findings encoded -> {OUT / 'glossary_embeddings.pt'}")

        # 3. fit-split AUROC for each phrase's own finding (and the best other finding, as for the 77)
        t = D.study_table()
        fit = t[t.split == "train"]
        X = D.embeddings(fit.row.values)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        sc = (X @ E.T).astype(np.float32)
        run.log(f"scored {len(fit):,} fit-split studies (NESY_EMB={D.VARIANT}) x {len(ph)} phrases")
        targets = [f for f in D.FINDINGS if f != "any_abnormality"]
        auc = {}
        for f in targets:
            y = fit[f].values
            m = ~np.isnan(y)
            auc[f] = auroc_columns(sc[m], y[m].astype(int))
        rows = []
        for j, r in ph.iterrows():
            others = {g: auc[g][j] for g in targets if g != r.finding}
            bo = max(others, key=others.get)
            rows.append({"finding": r.finding, "rank": 0, "source": f"Fleischner glossary: {r.glossary_term} p{r.glossary_page}",
                         "concept_id": j, "text": r.phrase, "fit_auroc_own": round(float(auc[r.finding][j]), 4),
                         "best_other_finding": bo, "fit_auroc_best_other": round(float(others[bo]), 4)})
        allp = pd.DataFrame(rows)
        allp["kept"] = allp.fit_auroc_own >= a.min_auroc
        allp = allp.sort_values(["finding", "fit_auroc_own"], ascending=[True, False])
        allp["rank"] = allp.groupby("finding").cumcount() + 1
        allp.to_csv(run.dir / "glossary_phrases_scored.csv", index=False)
        kept = allp[allp.kept].drop(columns="kept")
        kept.to_csv(OUT / "concepts_glossary.csv", index=False)
        kept.to_csv(run.dir / "concepts_glossary.csv", index=False)
        pd.set_option("display.width", 220)
        run.log("all phrases (fit-split AUROC, own finding):\n" + allp.to_string(index=False))
        summ = allp.groupby("finding").agg(n=("kept", "size"), kept=("kept", "sum"), best=("fit_auroc_own", "max"))
        run.log("per finding:\n" + summ.to_string())
        run.log(f"kept {len(kept)} of {len(allp)} phrases (floor {a.min_auroc}) -> {OUT / 'concepts_glossary.csv'}")


if __name__ == "__main__":
    main()
