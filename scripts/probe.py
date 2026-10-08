"""Stages 1, 2 and 5: per-finding logistic probe on study-level features, fit on the train ("fit")
split, Platt-calibrated on calib, threshold candidates on thresh, metrics on val. Test is never
scored here.

Feature groups (--features, comma separated):
  emb       the 768-d CLEAR embedding of the study's frontal image (Stage 1)
  chexmask  CheXmask side and CTR features + view (Stage 2). Missing or failed-QC rows are imputed
            with the fit-split mean AND flagged by an explicit indicator (missing is never zero).
  nb<k>     neighbour label fractions from item 4 for k in {5,10,25} (Stage 5), with indicators
            where no neighbour had a known label.
Regularisation C is chosen per finding on an internal 5 % patient holdout of the fit split (val is
not used for fitting or selection inside a stage), then refit on the whole fit split.

Labels: masked (uncertain) entries are excluded from fitting and scoring.

  python -u scripts/probe.py --name stage1 --features emb
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from nesy import data as D, paths as P  # noqa: E402
from nesy.evaluate import binary_metrics, threshold_candidates  # noqa: E402
from nesy.runlog import Run  # noqa: E402

CHEXMASK_COLS = ["ctr", "ctr_maxrow", "heart_width", "thorax_width", "lung_area_frac_left", "lung_height_frac_left",
                 "lung_base_diff_left_minus_right", "lung_apex_diff_left_minus_right", "heart_shift_to_left",
                 "lung_l_area", "lung_r_area", "heart_area", "rca_mean"]
CS = (0.01, 0.1, 1.0, 10.0)
EVAL_SPLITS = ("calib", "thresh", "val")


def inner_holdout(subject_ids: np.ndarray) -> np.ndarray:
    return np.array([int(hashlib.sha256(f"inner:{s}".encode()).hexdigest()[:8], 16) % 20 == 0 for s in subject_ids])


def build_features(t: pd.DataFrame, groups: list[str], fit_mask: np.ndarray, run: Run) -> tuple[np.ndarray, list[str]]:
    blocks, names = [], []
    for g in groups:
        if g == "emb":
            X = D.embeddings(t.row.values)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            blocks.append(X)
            names += [f"emb_{i}" for i in range(X.shape[1])]
            continue
        if g in ("chexmask", "chexmask_pa"):
            cm = pd.read_parquet(P.FEATURES / "chexmask_mimic.parquet")
            df = t[["dicom_id", "view"]].merge(cm[["dicom_id", "qc_ok", *CHEXMASK_COLS]], on="dicom_id", how="left")
            miss = ~(df.qc_ok.fillna(False).astype(bool))
            vals = df[CHEXMASK_COLS].where(~miss, np.nan)
            ap = (df.view == "AP").astype(float).values[:, None]
            if g == "chexmask":
                interact = vals[["ctr", "ctr_maxrow"]].values * ap        # CTR reads differently on AP
                raw = np.hstack([vals.values, interact])
                cols = CHEXMASK_COLS + ["ctr_x_ap", "ctr_maxrow_x_ap"]
            else:
                # chexmask_pa (2026-10-06): the heart-ratio terms are used ONLY on PA images. ctr and
                # ctr_maxrow enter as ctr x PA; on AP (magnified heart) they contribute nothing. The
                # other CheXmask features are unchanged. (The old ctr + ctr x AP pair already allowed
                # view-specific slopes; this variant removes the AP heart-ratio signal altogether.)
                pa = (df.view == "PA").astype(float).values[:, None]
                other = [c for c in CHEXMASK_COLS if c not in ("ctr", "ctr_maxrow", "heart_width")]
                raw = np.hstack([vals[other].values, vals[["ctr", "ctr_maxrow", "heart_width"]].values * pa])
                cols = other + ["ctr_x_pa", "ctr_maxrow_x_pa", "heart_width_x_pa"]
            run.log(f"chexmask: {int(miss.sum()):,} of {len(df):,} studies missing or failed QC -> imputed + indicator")
            mu = np.nanmean(raw[fit_mask], 0)
            sd = np.nanstd(raw[fit_mask], 0) + 1e-6
            z = (np.where(np.isnan(raw), mu, raw) - mu) / sd
            blocks.append(np.hstack([z, miss.values[:, None].astype(float), ap]))
            names += [f"cm_{c}" for c in cols] + ["cm_missing", "view_ap"]
            continue
        if g.startswith("nb"):
            k = int(g[2:])
            nb = pd.read_parquet(D.RETRIEVAL)
            df = t[["study_id"]].merge(nb, on="study_id", how="left")
            cols = [f"nb{k}_frac_{f}" for f in D.FINDINGS if f"nb{k}_frac_{f}" in nb.columns]
            raw = df[cols].values.astype(float)
            miss = np.isnan(raw)
            mu = np.nanmean(raw[fit_mask], 0)
            sd = np.nanstd(raw[fit_mask], 0) + 1e-6
            z = (np.where(miss, mu, raw) - mu) / sd
            sim = df[[f"nb{k}_mean_sim"]].values
            sim = (sim - np.nanmean(sim[fit_mask])) / (np.nanstd(sim[fit_mask]) + 1e-6)
            blocks.append(np.hstack([z, miss.astype(float), sim]))
            names += cols + [f"{c}_missing" for c in cols] + [f"nb{k}_mean_sim"]
            run.log(f"nb{k}: missing fractions {int(miss.sum()):,} entries")
            continue
        if g == "aw":
            # association-links stage: out-of-fold Stage 1 logits of every finding that appears in a KG
            # associated_with edge (scripts/stage1_oof.py). Stage 3 restricts each finding's model to the
            # logits of its own KG partners (stage3_factorised.aw_mask); the KG decides the links.
            from nesy import kg as _kg
            aw = pd.read_parquet(P.FEATURES / f"stage1_oof_{D.VARIANT}.parquet")
            fs = sorted({e[k] for e in _kg.load_findings().get("associated_with", []) for k in ("a", "b")})
            raw = t[["study_id"]].merge(aw, on="study_id", how="left")[[f"s1_{f}" for f in fs]].values.astype(float)
            assert not np.isnan(raw).any(), "missing out-of-fold Stage 1 logits"
            mu, sd = raw[fit_mask].mean(0), raw[fit_mask].std(0) + 1e-6
            blocks.append((raw - mu) / sd)
            names += [f"aw_s1_{f}" for f in fs]
            run.log(f"aw: out-of-fold Stage 1 logits of {fs}")
            continue
        if g == "s4":
            # Stage 4 region-classifier outputs (A2), from the run in $NESY_S4_RUN. Per finding and region
            # set: max score over relevant regions, and the max over patient-left and patient-right
            # regions. Missing (no region features) -> fit-split mean + indicator.
            import os
            s4 = Path(os.environ["NESY_S4_RUN"])
            sidx = pd.read_parquet(s4 / "scores_index.parquet")[["dicom_id"]].reset_index().rename(columns={"index": "s4row"})
            from stage4_regions import FIND as S4F
            from region_features import IG_REGIONS
            from nesy.regions import CHEXMASK_REGIONS
            m = t[["dicom_id"]].merge(sidx, on="dicom_id", how="left")
            ok = m.s4row.notna().values
            rows_ = m.s4row.fillna(0).astype(int).values
            cols, mats = [], []
            s4_sets = os.environ.get("NESY_S4_SETS", "A,B").split(",")      # B = CheXmask-derived regions (primary set)
            for set_name, regs in (("A", IG_REGIONS), ("B", CHEXMASK_REGIONS)):
                if set_name not in s4_sets:
                    continue
                sc = np.load(s4 / f"scores_{set_name}_{'imagenome' if set_name == 'A' else 'chexmask'}.npy", mmap_mode="r")
                L = [j for j, r in enumerate(regs) if r.startswith("left") or r.startswith("lung_left")]
                R = [j for j, r in enumerate(regs) if r.startswith("right") or r.startswith("lung_right")]
                for fi, f in enumerate(S4F):
                    a = np.asarray(sc[rows_, :, fi], np.float32)
                    a[~ok] = np.nan
                    if np.isnan(a).all():
                        continue
                    with np.errstate(all="ignore"):
                        for nm, sel in (("max", slice(None)), ("left", L), ("right", R)):
                            v = np.nanmax(a[:, sel], axis=1) if not isinstance(sel, slice) else np.nanmax(a, axis=1)
                            mats.append(v)
                            cols.append(f"s4{set_name}_{f}_{nm}")
            raw = np.stack(mats, 1)
            miss = np.isnan(raw)
            mu = np.nanmean(raw[fit_mask], 0)
            sd = np.nanstd(raw[fit_mask], 0) + 1e-6
            z = (np.where(miss, mu, raw) - mu) / sd
            anymiss = miss.any(1, keepdims=True).astype(float)
            blocks.append(np.hstack([z, anymiss]))
            names += cols + ["s4_missing"]
            run.log(f"s4: {len(cols)} region-classifier features (sets {s4_sets}) from {s4.name}; studies with any missing {int(anymiss.sum()):,}")
            continue
        raise ValueError(f"unknown feature group {g}")
    return np.hstack(blocks).astype(np.float32), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--features", default="emb")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--baseline", default=None, help="run dir of the stage to compare against (val_metrics.csv)")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    groups = args.features.split(",")
    with Run(f"probe-{args.name}", {**vars(args), "Cs": CS}, args.run_dir) as run:
        import os
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        t = D.study_table()
        fit_mask = (t.split == "train").values
        X, names = build_features(t, groups, fit_mask, run)
        run.log(f"design: {X.shape[0]:,} studies x {X.shape[1]} features ({', '.join(groups)}); "
                f"splits {t.split.value_counts().to_dict()}")
        inner = inner_holdout(t.subject_id.values) & fit_mask
        core = fit_mask & ~inner
        models, rows, preds, thr_rows = {}, [], {}, []
        for f in D.FINDINGS:
            y = t[f].values
            known = ~np.isnan(y)
            best = None
            for C in CS:
                m = LogisticRegression(C=C, max_iter=2000, tol=1e-4)
                m.fit(X[core & known], y[core & known].astype(int))
                a = roc_auc_score(y[inner & known].astype(int), m.decision_function(X[inner & known]))
                run.log(f"  {f:28s} C={C:<5} inner-holdout AUROC {a:.4f}")
                if best is None or a > best[1]:
                    best = (C, a)
            m = LogisticRegression(C=best[0], max_iter=3000, tol=1e-4).fit(X[fit_mask & known], y[fit_mask & known].astype(int))
            s = m.decision_function(X)
            cal_m = (t.split == "calib").values & known
            platt = LogisticRegression(C=1e6, max_iter=1000).fit(s[cal_m, None], y[cal_m].astype(int))
            p = platt.predict_proba(s[:, None])[:, 1]
            models[f] = {"probe": m, "platt": platt, "C": best[0], "inner_auroc": best[1]}
            preds[f] = p
            for sp in EVAL_SPLITS:
                msk = (t.split == sp).values
                rows.append({"finding": f, "split": sp, "C": best[0], **binary_metrics(y[msk], p[msk])})
            th = (t.split == "thresh").values
            for r in threshold_candidates(y[th], p[th]):
                thr_rows.append({"finding": f, **r})
            v = rows[-1]
            run.log(f"{f:28s} C={best[0]} val AUROC {v.get('auroc', np.nan):.4f} AUPRC {v.get('auprc', np.nan):.4f} "
                    f"ECE {v.get('ece', np.nan):.4f} (n {v['n']}, pos {v['pos']})")
            run.metric(finding=f, **{k: v[k] for k in v if k not in ("finding", "split")})

        met = pd.DataFrame(rows)
        met.to_csv(run.dir / "metrics_all_eval_splits.csv", index=False)
        val = met[met.split == "val"].set_index("finding")
        val.to_csv(run.dir / "val_metrics.csv")
        macro = val[["auroc", "auprc", "ece", "brier"]].mean()
        run.log("VAL (macro over 13 findings): " + ", ".join(f"{k} {v:.4f}" for k, v in macro.items()))
        run.metric(summary="val_macro", **{k: float(v) for k, v in macro.items()})
        thr = pd.DataFrame(thr_rows)
        thr.to_csv(run.dir / "threshold_candidates_thresh_split.csv", index=False)
        from nesy import thresholds as TH
        th = (t.split == "thresh").values
        bands = {f: TH.fit(t[f].values[th], preds[f][th], f) for f in D.FINDINGS}
        (run.dir / "thresholds_4band.json").write_text(json.dumps(bands, indent=2, default=float))
        run.log("4-band thresholds (thresh split): " + "; ".join(
            f"{f}: present {b['present'] if b['present'] is None else round(b['present'], 3)}, possible "
            f"{b['possible'] if b['possible'] is None else round(b['possible'], 3)}, absent {b['absent'] if b['absent'] is None else round(b['absent'], 3)}"
            for f, b in bands.items()))

        # hierarchy violations of the (independent) marginals, for comparison with Stage 3
        import yaml
        kgd = yaml.safe_load((P.V2 / "kg/findings.yaml").read_text())
        vm = (t.split == "val").values
        viol = {e["id"]: int((preds[e["child"]][vm] > preds[e["parent"]][vm]).sum()) for e in kgd["is_a"]}
        run.log(f"val hierarchy violations of independent marginals (P(child) > P(parent)), of {int(vm.sum())} studies: {viol}")
        run.metric(summary="hierarchy_violations_val", **viol)

        if args.baseline:
            b = pd.read_csv(Path(args.baseline) / "val_metrics.csv").set_index("finding")
            cmp = pd.DataFrame({"auroc_base": b.auroc, "auroc_new": val.auroc, "d_auroc": val.auroc - b.auroc,
                                "auprc_base": b.auprc, "auprc_new": val.auprc, "d_auprc": val.auprc - b.auprc,
                                "ece_base": b.ece, "ece_new": val.ece, "d_ece": val.ece - b.ece})
            cmp.to_csv(run.dir / "val_vs_baseline.csv")
            run.log("VAL change vs baseline " + args.baseline + ":\n" + cmp.round(4).to_string())
            run.log("macro d_auroc {:.4f}, d_auprc {:.4f}, d_ece {:.4f}".format(cmp.d_auroc.mean(), cmp.d_auprc.mean(), cmp.d_ece.mean()))

        out = pd.DataFrame({"study_id": t.study_id, "split": t.split, **{f"p_{f}": preds[f] for f in D.FINDINGS}})
        out[out.split != "test"].to_parquet(run.dir / "predictions_non_test.parquet", index=False)
        joblib.dump({"models": models, "feature_names": names, "groups": groups}, run.dir / "model.joblib")
        run.log(f"saved model and non-test predictions in {run.dir}")


if __name__ == "__main__":
    main()
