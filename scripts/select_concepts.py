"""Item 7: choose named concepts for the Stage 3 bottleneck, using the fit (train) split only.

1. Text filters on the 368,294-sentence CLEAR concept bank (no image data involved):
   - drop temporal / comparison sentences (expanded lexicon: unchanged, again, new, interval,
     compared, prior, improved, worsened, increased, decreased, resolved, remains, persistent, now, ...)
   - negation fix (AUDIT defect 7 and 12): drop every sentence containing a negation or hedge cue,
     including bare "not", "without", "may", "possible", "likely", "or", "vs", rather than trying to
     tag polarity. A concept is an affirmative observation, so "no effusion" is never a concept and
     is never read as "effusion".
   - short and generic: 2-6 words, no digits, no "___", no units.
2. Lexicon: a concept is a candidate for finding f only if its text names f (kg-style regexes below),
   so a concept can only explain the finding it is about. A separate "normal" group (lungs clear,
   heart size normal, ...) is a candidate for the derived no_finding.
3. Image-text scores (cosine of CLEAR image and text embeddings) on a fixed 40,000-study sample of
   the fit split; univariate AUROC vs the item 1 labels; top K per finding with AUROC >= 0.6, greedy
   de-duplication at text-embedding cosine > 0.92.
Output: data/concepts/concepts_v1_candidates.csv (for review; not used for training until approved).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from nesy import clear_model, data as D, paths as P  # noqa: E402
from nesy.runlog import Run  # noqa: E402

BANK_CSV = clear_model.OLD / "model_weights/mimic_concepts.csv"
BANK_EMB = clear_model.OLD / "model_weights/concept_embeddings_368294.pt"
OUT = P.V2 / "data/concepts"

W = r"\b"
TEMPORAL = re.compile(W + r"(unchanged|stable|again|new|newly|interval|compared|comparison|prior|previous|previously|since|"
                      r"improv\w*|worsen\w*|increas\w*|decreas\w*|resolv\w*|resolution|progress\w*|re-?demonstrat\w*|remain\w*|"
                      r"persist\w*|now|recurr\w*|continu\w*|similar|than|before|earlier|recent|recently|residual|still|"
                      r"longer|less|more|larger|smaller|developing|evolving|clearing|cleared|interim|today|chronic\w*|"
                      r"old|healed|healing|status|post|removed|removal|placement|placed|advanced|withdrawn|repositioned|"
                      # v2 additions found in the v1 candidate list (2026-10-06 review):
                      r"change|changes|changed|changing|constant|constantly|development|develop\w*|stabl\w*|stability|"
                      r"longstanding|long-standing|enlarging|grown|growing|re-?visualized|remnant|evacuated|drained|known|"
                      r"unclear|postoperative|ards|hemothorax|asbestos-related|metastatic|"
                      # v3 additions (2026-10-06 review of v2):
                      r"regression|regressed|worse|better|monitored|monitor\w*|closely|needs|successful|advancement|advanced|"
                      r"insignificant|reduction|reduced|re-|redemonstrated|demonstrated|concurrent|supervening|"
                      r"hemodynamically|metastasis|metastases|metastatic|asbestos\w*|known|significant|prior|"
                      r"probable|questionable|suspected|presumed|likely|appearance)" + W)
SPURIOUS = re.compile(W + r"(simulat\w*|mimic\w*|obscur\w*|artifact\w*|pseudo\w*|projecting)" + W)
NEGHEDGE = re.compile(W + r"(no|not|without|absent|absence|negative|free|none|neither|nor|may|might|possibl\w*|probabl\w*|"
                      r"likely|suspect\w*|question\w*|cannot|could|suggest\w*|concern\w*|equivocal|vs|versus|or|differential|"
                      r"consider\w*|presum\w*|suspicious|rule|exclude\w*|unlikely|perhaps|minimal|subtle|borderline|"
                      r"apparent|appears?|seen|evidence|definite)" + W)
JUNK = re.compile(r"\d|___|\bcm\b|\bmm\b|\btip\b|\bterminat\w*|\bproject\w*|\bpositioned\b|\blevel\b|\bpatient\b")
LEX = {
    "atelectasis": r"\b(atelectasis|atelectatic|collapsed?)\b",
    "consolidation": r"\b(consolidation|consolidations|consolidative)\b",
    "pneumonia": r"\b(pneumonia|pneumonias)\b",
    "edema": r"\b(edema|oedema|congestion|congested)\b",
    "lung_lesion": r"\b(nodule|nodules|nodular|mass|masses|lesion|lesions|tumou?r|neoplasm)\b",
    "lung_opacity": r"\b(opacity|opacities|opacification|infiltrate|infiltrates|haziness|hazy)\b",
    "enlarged_cardiomediastinum": r"\bmediastin\w*\b.*\b(widen\w*|enlarg\w*|promin\w*)\b|\b(widen\w*|enlarg\w*)\b.*\bmediastin\w*",
    "cardiomegaly": r"\bcardiomegaly\b|\b(heart|cardiac)\b.*\b(enlarg\w*|large)\b|\b(enlarg\w*|large)\b.*\b(heart|cardiac)\b",
    "pleural_effusion": r"\b(effusion|effusions)\b|\bblunt\w*\b",
    "pleural_other": r"\bpleural\b.*\b(thicken\w*|scar\w*|plaques?|calcif\w*)\b|\b(thicken\w*|scar\w*|plaques?|calcif\w*)\b.*\bpleura",
    "pneumothorax": r"\b(pneumothorax|pneumothoraces)\b",
    "fracture": r"\b(fracture|fractures|fractured)\b",
    "support_devices": r"\b(tube|catheter|line|pacemaker|pacer|leads?|wires?|picc|port|drain|stent|valve|device|sternotomy|clips)\b",
    "no_finding": r"\b(normal|clear|unremarkable|well[- ]expanded|sharp)\b",
}


PLAIN = {   # plain terms, tried in order
    "lung_opacity": ["lung opacity", "opacity", "opacities", "pulmonary opacity", "parenchymal opacity"],
    "atelectasis": ["atelectasis"],
    "consolidation": ["consolidation"],
    "pneumonia": ["pneumonia"],
    "edema": ["pulmonary edema", "edema"],
    "lung_lesion": ["lung nodule", "pulmonary nodule", "nodule", "lung mass", "pulmonary mass", "mass"],
    "enlarged_cardiomediastinum": ["enlarged cardiomediastinum", "widened mediastinum", "mediastinal widening",
                                   "enlarged cardiomediastinal silhouette"],
    "cardiomegaly": ["cardiomegaly"],
    "pleural_effusion": ["pleural effusion", "effusion"],
    "pleural_other": ["pleural thickening", "pleural plaques", "pleural plaque"],
    "pneumothorax": ["pneumothorax"],
    "fracture": ["rib fracture", "fracture", "rib fractures"],
    "support_devices": ["support devices", "support device", "lines and tubes", "tubes and lines"],
    "no_finding": ["normal chest radiograph", "normal chest x-ray", "normal chest", "normal study"],
}


def auroc_columns(S: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUROC of each column of S against binary y (Mann-Whitney)."""
    r = rankdata(S, axis=0)
    npos = y.sum()
    nneg = len(y) - npos
    return (r[y == 1].sum(0) - npos * (npos + 1) / 2) / (npos * nneg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n-sample", type=int, default=12000)
    ap.add_argument("--min-auroc", type=float, default=0.6)
    ap.add_argument("--max-words", type=int, default=4, help="v3: 2-4 words (short, generic observations)")
    ap.add_argument("--ks", default="6,12", help="list sizes to build (final lists, 2026-10-06)")
    ap.add_argument("--review", default=str(OUT / "concepts_v3_review.csv"),
                    help="review file: rows with keep_proposed == False are excluded unless in --keep-texts")
    ap.add_argument("--keep-texts", default="lower lobes collapse")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("select-concepts", vars(args), args.run_dir) as run:
        import torch
        bank = pd.read_csv(BANK_CSV)
        txt = bank.concept.astype(str).str.lower().str.strip()
        nw = txt.str.split().str.len()
        f_temporal = txt.str.contains(TEMPORAL)
        f_neg = txt.str.contains(NEGHEDGE)
        f_junk = txt.str.contains(JUNK)
        f_len = ~nw.between(2, args.max_words)
        f_spur = txt.str.contains(SPURIOUS)
        keep = ~(f_temporal | f_neg | f_junk | f_len | f_spur)
        run.log(f"bank {len(txt):,}; drop temporal/comparison {int(f_temporal.sum()):,}; negated/hedged {int(f_neg.sum()):,}; "
                f"digits/positions {int(f_junk.sum()):,}; length outside 2-{args.max_words} words {int(f_len.sum()):,}; "
                f"simulates/mimics/obscured {int(f_spur.sum()):,}; kept {int(keep.sum()):,}")
        from nesy import kg
        hits = {f: txt.str.contains(pat, regex=True) for f, pat in LEX.items()}
        cand = {}
        for f in LEX:
            allowed = {f} | (set(kg.ancestors(f)) if f in kg.finding_ids() else set())
            others = np.zeros(len(txt), bool)
            for g, h in hits.items():
                if g not in allowed and g != "no_finding":
                    others |= h.values
            m = keep.values & hits[f].values & ~others
            cand[f] = np.nonzero(m)[0]
            run.log(f"  lexicon candidates {f:28s} {len(cand[f]):,} (after dropping compounds naming another finding)")
        # plain terms: the finding's own term with no severity/extent/side word. They must pass every text
        # filter except the 2-word minimum (single-word terms such as "cardiomegaly" are allowed).
        keep_but_len = ~(f_temporal | f_neg | f_junk | f_spur | (nw > args.max_words))
        norm = txt.str.replace(r"[.\s]+$", "", regex=True)
        plain_id = {}
        for f, terms in PLAIN.items():
            for term in terms:
                hit = np.nonzero((norm == term).values & keep_but_len.values)[0]
                if len(hit):
                    plain_id[f] = int(hit[0])
                    break
        run.log("plain terms found: " + ", ".join(f"{f}='{txt.iloc[i]}'" for f, i in plain_id.items()))
        run.log("findings with NO plain term in the bank passing the filters: " + (", ".join(f for f in PLAIN if f not in plain_id) or "none"))
        allc = np.unique(np.concatenate(list(cand.values()) + [np.array(list(plain_id.values()), dtype=int)]))
        emb = torch.load(BANK_EMB, map_location="cpu", weights_only=False, mmap=True)
        if isinstance(emb, dict):
            emb = next(v for v in emb.values() if hasattr(v, "shape"))
        C = np.asarray(emb[torch.as_tensor(allc)].float().numpy())
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        pos_of = {c: i for i, c in enumerate(allc)}
        run.log(f"text embeddings for {len(allc):,} candidate concepts loaded from {BANK_EMB.name}")

        t = D.study_table()
        fit = t[t.split == "train"].sample(args.n_sample, random_state=0)
        X = D.embeddings(fit.row.values)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        S = X @ C.T
        run.log(f"scored {len(fit):,} fit-split studies x {len(allc):,} concepts")

        rows = []
        fit["no_finding"] = 1 - fit["any_abnormality"]          # normal concepts: absence of the any_abnormality root
        targets = [f for f in D.FINDINGS if f != "any_abnormality"] + ["no_finding"]
        # AUROC of every candidate concept against every target (needed for the specificity column);
        # one rank pass per target over the 12k-study sample, float32
        S = S.astype(np.float32)
        all_auc = {}
        for f in targets:
            y = fit[f].values
            m = ~np.isnan(y)
            all_auc[f] = auroc_columns(S[m], y[m].astype(int))
            run.log(f"  AUROC computed for target {f} ({int(m.sum()):,} studies with a known label)")
        rev = pd.read_csv(args.review)
        keep_texts = {x.strip() for x in args.keep_texts.split(";") if x.strip()}
        excluded = set(rev.loc[~rev.keep_proposed & ~rev.text.isin(keep_texts), "concept_id"].astype(int))
        run.log(f"excluded after review: {len(excluded)} concepts; kept despite flag: {sorted(keep_texts)}")

        def row(f, j, rank, source):
            others = {g: all_auc[g][j] for g in targets if g != f}
            best_other = max(others, key=others.get)
            return {"finding": f, "rank": rank, "source": source, "concept_id": int(allc[j]), "text": txt.iloc[allc[j]],
                    "fit_auroc_own": round(float(all_auc[f][j]), 4), "best_other_finding": best_other,
                    "fit_auroc_best_other": round(float(others[best_other]), 4)}
        OUT.mkdir(parents=True, exist_ok=True)
        for K in [int(k) for k in args.ks.split(",")]:
            rows = []
            for f in targets:
                idx = np.array([pos_of[c] for c in cand[f] if c not in excluded], dtype=int)
                order = idx[np.argsort(-all_auc[f][idx])] if len(idx) else idx
                chosen, src = [], []
                if f in plain_id:
                    chosen.append(pos_of[plain_id[f]])
                    src.append("plain_term")
                for j in order:
                    if len(chosen) >= K or all_auc[f][j] < args.min_auroc:
                        break
                    if j in chosen or any(C[j] @ C[k] > 0.92 for k in chosen):
                        continue
                    chosen.append(j)
                    src.append("ranked")
                rows += [row(f, j, r, sc) for r, (j, sc) in enumerate(zip(chosen, src), 1)]
                if len(chosen) < K:
                    run.log(f"  WARNING k={K}: {f} has only {len(chosen)} concepts passing filters and AUROC >= {args.min_auroc}")
            sel = pd.DataFrame(rows)
            sel.to_csv(OUT / f"concepts_final_k{K}.csv", index=False)
            sel.to_csv(run.dir / f"concepts_final_k{K}.csv", index=False)
            run.log(f"k={K}: {len(sel)} concepts ({sel.concept_id.nunique()} unique) -> {OUT / f'concepts_final_k{K}.csv'}")
            run.log("\n" + sel.to_string(index=False))


if __name__ == "__main__":
    main()
