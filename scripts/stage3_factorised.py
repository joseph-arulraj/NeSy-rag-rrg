"""Stage 3: factorised concept-bottleneck head on NAMED features only.

Features (all human-readable, standardised on the fit split; missing never encoded as zero):
  concepts  cosine score of the study's frontal image with each approved concept sentence
            (data/concepts/concepts_v1_approved.csv; refuses to run on the unapproved candidates file)
  chexmask  CheXmask side/CTR features + missing indicator + AP flag (as in Stage 2)
  nb<k>     optional neighbour label fractions (Stage 5 ablation)

Model (stage one):
  * root findings (no is_a parent): P(f) — logistic, fit split, labels known
  * child findings: P(child | parent=1) — logistic on fit-split studies whose parent is positive
  * side: P(side | finding) over {left, right, bilateral} — multinomial logistic, fit-split images
    positive for the finding with an ImaGenome side label (caveat: ImaGenome "bilateral" includes
    "side not stated"; see RESULTS item 10)
Calibration: each conditional Platt-scaled on calib (children on calib studies with parent positive),
then marginals are multiplied down the hierarchy: P(child) = P(child | parent) * P(parent). A child
can therefore never exceed its parent (checked and reported; must be 0 violations).
Reports, on val: AUROC / AUPRC / ECE per finding, hierarchy violations, and the AUROC gap to the
Stage 1 embedding probe (--baseline). Per-feature-group contributions (weight x value) are saved.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from nesy import data as D, kg, paths as P  # noqa: E402
from nesy.evaluate import binary_metrics, threshold_candidates  # noqa: E402
from nesy.runlog import Run  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe import CS, build_features, inner_holdout  # noqa: E402

APPROVED = P.V2 / "data/concepts/concepts_final_k6.csv"      # final lists (user-approved procedure, 2026-10-06)
CONCEPT_EMB = Path("/scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/model_weights/concept_embeddings_368294.pt")


def concept_features(t: pd.DataFrame, fit_mask: np.ndarray, run: Run, path: Path = APPROVED, emb_path: Path = CONCEPT_EMB):
    import torch
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} missing: the concept list must be approved before Stage 3 trains on it")
    sel = pd.read_csv(path).drop_duplicates("concept_id")
    emb = torch.load(emb_path, map_location="cpu", weights_only=False, mmap=True)
    C = emb[torch.as_tensor(sel.concept_id.values)].float().numpy()
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    X = D.embeddings(t.row.values)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    S = X @ C.T
    mu, sd = S[fit_mask].mean(0), S[fit_mask].std(0) + 1e-6
    names = [f"concept:{c}:{txt}" for c, txt in zip(sel.concept_id, sel.text)]
    run.log(f"concept features: {S.shape[1]} approved concepts")
    return ((S - mu) / sd).astype(np.float32), names


def aw_partners(f: str) -> set[str]:
    return {e["b"] if e["a"] == f else e["a"] for e in kg.load_findings().get("associated_with", []) if f in (e["a"], e["b"])}


def aw_mask(X, names, f: str | None):
    """Zero the association-link columns (aw_s1_<g>) except f's KG partners; f=None zeroes all of them."""
    cols = [j for j, n in enumerate(names) if n.startswith("aw_s1_") and (f is None or n[6:] not in aw_partners(f))]
    if not cols:
        return X
    X = X.copy()
    X[:, cols] = 0.0
    return X


def fit_logistic(X, y, core, inner, run, label):
    best = None
    for C in CS:
        m = LogisticRegression(C=C, max_iter=3000).fit(X[core], y[core])
        a = roc_auc_score(y[inner], m.decision_function(X[inner])) if len(np.unique(y[inner])) == 2 else np.nan
        if best is None or (a > best[1]):
            best = (C, a)
    run.log(f"  {label:40s} C={best[0]} inner AUROC {best[1]:.4f}")
    return LogisticRegression(C=best[0], max_iter=3000).fit(X[core | inner], y[core | inner]), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="stage3")
    ap.add_argument("--extra", default="chexmask", help="extra named feature groups, comma separated")
    ap.add_argument("--baseline", required=True, help="Stage 1 run dir (val_metrics.csv)")
    ap.add_argument("--concepts", default=str(APPROVED))
    ap.add_argument("--concept-emb", default=str(CONCEPT_EMB), help="text embeddings indexed by concept_id (KG task 7: glossary)")
    ap.add_argument("--no-concepts", action="store_true", help="input-group ablation: no concept features (only --extra)")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run(f"stage3-{args.name}", vars(args), args.run_dir) as run:
        t = D.study_table()
        fit = (t.split == "train").values
        if args.no_concepts:
            blocks, groups, names = [], [], []
            run.log("input-group ablation: NO concept features")
        else:
            Xc, names = concept_features(t, fit, run, Path(args.concepts), Path(args.concept_emb))
            blocks, groups = [Xc], ["concepts"] * Xc.shape[1]
        for g in [x for x in args.extra.split(",") if x]:
            Xg, ng = build_features(t, [g], fit, run)
            blocks.append(Xg)
            names += ng
            groups += [g] * Xg.shape[1]
        X = np.hstack(blocks)
        groups = np.array(groups)
        run.log(f"named design: {X.shape[1]} features ({pd.Series(groups).value_counts().to_dict()})")
        inner = inner_holdout(t.subject_id.values) & fit
        core = fit & ~inner
        cal = (t.split == "calib").values
        cond, models = {}, {}
        for f in kg.topo_order()[::-1]:                    # parents first
            y = t[f].values
            parents = kg.parents_of(f)
            known = ~np.isnan(y)
            if parents:
                p = parents[0]
                gate = t[p].values == 1
                label = f"P({f} | {p}=1)"
            else:
                gate = np.ones(len(t), bool)
                label = f"P({f})"
            m_fit = known & gate
            Xf = aw_mask(X, names, f)
            m, best = fit_logistic(Xf, np.nan_to_num(y).astype(int),
                                   core & m_fit, inner & m_fit, run, label)
            s = m.decision_function(Xf)
            cm = cal & m_fit
            platt = LogisticRegression(C=1e6).fit(s[cm, None], np.nan_to_num(y[cm]).astype(int))
            cond[f] = platt.predict_proba(s[:, None])[:, 1]
            contrib = {gname: (Xf[:, groups == gname] * m.coef_[0][groups == gname]).sum(1) for gname in np.unique(groups)}
            models[f] = {"model": m, "platt": platt, "parent": parents[0] if parents else None, "C": best[0],
                         "n_fit": int((fit & m_fit).sum()), "n_calib": int(cm.sum()), "contrib_mean_val": {
                             k: float(v[(t.split == 'val').values].mean()) for k, v in contrib.items()}}
        marg = {}
        for f in kg.topo_order()[::-1]:
            ps = kg.parents_of(f)
            marg[f] = cond[f] * (marg[ps[0]] if ps else 1.0)

        vm = (t.split == "val").values
        viol = {e["id"]: int((marg[e["child"]][vm] > marg[e["parent"]][vm] + 1e-12).sum()) for e in kg.load_findings()["is_a"]}
        run.log(f"hierarchy violations on val (P(child) > P(parent)): {viol} -> total {sum(viol.values())} (must be 0)")
        if sum(viol.values()):
            raise AssertionError("hierarchy violated")
        rows, thr_rows = [], []
        for f in D.FINDINGS:
            for sp in ("calib", "thresh", "val"):
                msk = (t.split == sp).values
                rows.append({"finding": f, "split": sp, **binary_metrics(t[f].values[msk], marg[f][msk])})
            th = (t.split == "thresh").values
            thr_rows += [{"finding": f, **r} for r in threshold_candidates(t[f].values[th], marg[f][th])]
        met = pd.DataFrame(rows)
        val = met[met.split == "val"].set_index("finding")
        val.to_csv(run.dir / "val_metrics.csv")
        pd.DataFrame(thr_rows).to_csv(run.dir / "threshold_candidates_thresh_split.csv", index=False)
        import json
        from nesy import thresholds as TH
        th = (t.split == "thresh").values
        (run.dir / "thresholds_4band.json").write_text(json.dumps({f: TH.fit(t[f].values[th], marg[f][th], f) for f in D.FINDINGS},
                                                                  indent=2, default=float))
        b = pd.read_csv(Path(args.baseline) / "val_metrics.csv").set_index("finding")
        cmp = pd.DataFrame({"auroc_probe": b.auroc, "auroc_stage3": val.auroc, "gap": val.auroc - b.auroc,
                            "ece_probe": b.ece, "ece_stage3": val.ece, "d_ece": val.ece - b.ece,
                            "auprc_probe": b.auprc, "auprc_stage3": val.auprc})
        cmp.to_csv(run.dir / "val_vs_probe.csv")
        run.log("VAL stage 3 vs embedding probe:\n" + cmp.round(4).to_string())
        run.log("macro: AUROC gap {:.4f}, ECE probe {:.4f} -> stage3 {:.4f}".format(cmp.gap.mean(), cmp.ece_probe.mean(), cmp.ece_stage3.mean()))
        run.metric(summary="val_macro", auroc=float(val.auroc.mean()), ece=float(val.ece.mean()), auroc_gap=float(cmp.gap.mean()),
                   hierarchy_violations=int(sum(viol.values())))

        # side head
        # explicit side labels only (side stated in the report phrase); default assignments are masked
        side = pd.read_parquet(P.FEATURES / "imagenome_side_explicit_v1.parquet")
        side = side[side.side_text != "default"].pivot_table(index="dicom_id", columns="finding", values="side_text", aggfunc="first")
        side.columns = [f"side_{c}" for c in side.columns]
        tt = t[["dicom_id"]].merge(side.reset_index(), on="dicom_id", how="left")
        side_models, side_rows = {}, []
        Xside = aw_mask(X, names, None)                          # the side head never uses association links
        for f in D.FINDINGS:
            if not kg.load_findings()["_by_id"][f]["lateralisable"]:
                continue
            if f"side_{f}" not in tt:
                continue
            s = tt[f"side_{f}"].values
            ok = np.isin(s, ["left", "right", "bilateral"])
            if (ok & fit).sum() < 200:
                continue
            m = LogisticRegression(C=1.0, max_iter=3000).fit(Xside[ok & fit], s[ok & fit])
            pv = m.predict(Xside[ok & vm])
            acc = float((pv == s[ok & vm]).mean()) if (ok & vm).any() else np.nan
            maj = pd.Series(s[ok & fit]).value_counts(normalize=True)
            side_models[f] = m
            side_rows.append({"finding": f, "n_fit": int((ok & fit).sum()), "n_val": int((ok & vm).sum()), "val_accuracy": acc,
                              "majority_class": maj.index[0], "majority_rate_fit": float(maj.iloc[0])})
            run.log(f"  side {f:20s} val accuracy {acc:.3f} (n {int((ok & vm).sum())}; majority '{maj.index[0]}' {maj.iloc[0]:.2f})")
        pd.DataFrame(side_rows).to_csv(run.dir / "side_val.csv", index=False)
        out = pd.DataFrame({"study_id": t.study_id, "split": t.split, **{f"p_{f}": marg[f] for f in D.FINDINGS}})
        for f, m in side_models.items():                         # side-head probabilities, P(side | finding)
            pr_side = m.predict_proba(Xside)
            for ci, cls in enumerate(m.classes_):
                out[f"pside_{f}_{cls}"] = pr_side[:, ci]
        out[out.split != "test"].to_parquet(run.dir / "predictions_non_test.parquet", index=False)
        joblib.dump({"models": models, "side_models": side_models, "feature_names": names, "groups": groups.tolist()},
                    run.dir / "model.joblib")


if __name__ == "__main__":
    main()
