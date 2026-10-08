"""Evaluation of one end-to-end pipeline run (scripts/run_pipeline.py output), as fixed in EVAL_PLAN.md.

Guards: --split test needs --allow-test. External sets are not handled here yet. Labels are read only for the
studies in the run: CheXpert-derived labels via a parquet row filter, radiologist labels by reading the study_id
column first and then only the matching rows, so no other split's label values are loaded.

Computed (all with a patient-level bootstrap, 95 % percentile intervals):
  1. per finding and macro: AUROC, AUPRC, ECE, Brier; hierarchy violations
  2. bands: present precision / sensitivity; present-or-possible precision / sensitivity; absent-band miss rate
  3. normal call: rate; precision (labelled normal among called normal); sensitivity (called normal among
     labelled normal), against the any_abnormality root label
  4. report-to-graph agreement from Stage 8: first-pass match, final LLM match, fallback rate, template round trip
  5. H3 (--comparator): mean AUROC of the head minus the comparator, non-inferior if the lower bound > -margin
  6. report level (--refs, RadGraph parses of the reference reports): per finding, positive-mention precision /
     recall / F1 of our final report against the reference (rg_match.claims v2 on both), and a RadGraph entity F1
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
sys.path.insert(0, str(V2 / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

RAD = Path("/scratch/prj/bhi_zihe_imaging/mimic_cxr_full/mimic-cxr-jpg/mimic-cxr-jpg-2.1.0.physionet.org/mimic-cxr-2.1.0-test-set-labeled.csv")


def load_labels(source: str, study_ids: np.ndarray, run) -> pd.DataFrame:
    from nesy import labels as LB
    if source == "chexpert":
        import pyarrow.parquet as pq
        tab = pq.read_table(V2 / "data/labels/labels_v2.parquet", filters=[("study_id", "in", [int(x) for x in study_ids])])
        lab = tab.to_pandas().set_index("study_id")
        from nesy import kg
        return lab[kg.finding_ids()]
    ids = pd.read_csv(RAD, usecols=["study_id"]).study_id
    rows = np.nonzero(ids.isin(study_ids).values)[0]
    skip = sorted(set(range(1, len(ids) + 1)) - {r + 1 for r in rows})         # file line numbers (header = 0)
    raw = pd.read_csv(RAD, skiprows=skip)
    run.log(f"radiologist labels: {len(raw)} of {len(ids)} labelled studies are in this run; values per column: "
            + "; ".join(f"{c} {raw[c].value_counts(dropna=False).to_dict()}" for c in LB.RADIOLOGIST_COLUMNS))
    return LB.from_codes(LB.radiologist_codes(raw))


def boot_index(subjects: np.ndarray, n_boot: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(subjects, return_inverse=True)
    members = [np.nonzero(inv == i)[0] for i in range(len(uniq))]
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([members[i] for i in pick])


def ece(y, p, bins=15):
    from nesy.evaluate import ece as _e
    return _e(y, p, bins)


def metr(y, p):
    m = ~np.isnan(y)
    y, p = y[m].astype(int), p[m]
    if len(y) == 0 or y.min() == y.max():
        return {"auroc": np.nan, "auprc": np.nan, "ece": np.nan, "brier": np.nan, "n": int(len(y)), "pos": int(y.sum()) if len(y) else 0}
    return {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p), "ece": ece(y, p),
            "brier": float(np.mean((p - y) ** 2)), "n": int(len(y)), "pos": int(y.sum())}


def ci(vals):
    v = np.asarray([x for x in vals if not np.isnan(x)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [np.nan, np.nan]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-run", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--labels", default="chexpert", choices=["chexpert", "radiologist"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--comparator", default=None, help="NAME=predictions parquet (p_<finding> columns) for H3")
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--refs", default=None, help="dir of RadGraph reference parses (jsonl.gz) for report-level scoring")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--run-dir", default=None)
    a = ap.parse_args()
    if a.split == "test" and not a.allow_test:
        sys.exit("REFUSED: test evaluation needs --allow-test (user approval required)")
    os.environ.setdefault("NESY_EMB", "letterbox")
    from nesy import data as D, kg, rg_match as M
    from nesy.runlog import Progress, Run
    with Run("evaluate", vars(a), a.run_dir) as run:
        pr = Path(a.pipeline_run)
        pred = pd.read_parquet(pr / "predictions.parquet")
        if set(pred.split) != {a.split}:
            raise ValueError(f"run holds splits {set(pred.split)}, not {a.split}")
        lab = load_labels(a.labels, pred.study_id.values, run)
        pred = pred[pred.study_id.isin(lab.index)].reset_index(drop=True)
        idx = pd.read_parquet(D.EMB_INDEX)[["study_id", "subject_id"]]
        subj = pred[["study_id"]].merge(idx, on="study_id", how="left").subject_id.values
        F = D.FINDINGS
        Y = lab.loc[pred.study_id, F].values.astype(float)
        P_ = pred[[f"p_{f}" for f in F]].values
        run.log(f"{a.split}: {len(pred):,} studies with {a.labels} labels, {len(np.unique(subj)):,} patients")
        out = {"split": a.split, "labels": a.labels, "n_studies": len(pred), "n_patients": int(len(np.unique(subj)))}
        # 1. per finding
        point = {f: metr(Y[:, j], P_[:, j]) for j, f in enumerate(F)}
        boots = {f: {"auroc": [], "ece": []} for f in F}
        macro_b = {"auroc": [], "ece": [], "auprc": []}
        comp = None
        if a.comparator:
            cname, cpath = a.comparator.split("=", 1)
            cp = pd.read_parquet(cpath).set_index("study_id").loc[pred.study_id]
            C_ = cp[[f"p_{f}" for f in F]].values
            cpoint = {f: metr(Y[:, j], C_[:, j]) for j, f in enumerate(F)}
            comp = {"name": cname, "diff_boot": []}
        prog = Progress(run, a.n_boot, "bootstrap resamples", every_s=60)
        for b, ix in enumerate(boot_index(subj, a.n_boot)):
            ms = {f: metr(Y[ix, j], P_[ix, j]) for j, f in enumerate(F)}
            for f in F:
                boots[f]["auroc"].append(ms[f]["auroc"])
                boots[f]["ece"].append(ms[f]["ece"])
            for k in macro_b:
                macro_b[k].append(np.nanmean([ms[f][k] for f in F]))
            if comp:
                cs = [metr(Y[ix, j], C_[ix, j])["auroc"] for j, f in enumerate(F)]
                comp["diff_boot"].append(np.nanmean([ms[f]["auroc"] for f in F]) - np.nanmean(cs))
            prog.update(b + 1)
        per = pd.DataFrame({f: {**point[f], "auroc_ci": ci(boots[f]["auroc"]), "ece_ci": ci(boots[f]["ece"])} for f in F}).T
        per.to_csv(run.dir / "per_finding.csv")
        out["macro"] = {k: float(np.nanmean([point[f][k] for f in F])) for k in ("auroc", "auprc", "ece")}
        out["macro_ci"] = {k: ci(v) for k, v in macro_b.items()}
        viol = sum(int((pred[f"p_{e['child']}"] > pred[f"p_{e['parent']}"] + 1e-12).sum()) for e in kg.load_findings()["is_a"])
        out["hierarchy_violations"] = viol
        pd.set_option("display.width", 220)
        run.log("1. per finding:\n" + per[["n", "pos", "auroc", "auroc_ci", "auprc", "ece"]].to_string())
        run.log(f"   macro {out['macro']} CI {out['macro_ci']}; hierarchy violations {viol}")
        # 2-3. bands and normal call from the belief graphs
        G = {json.loads(l)["study_id"]: json.loads(l)["graph"] for l in open(pr / "graphs.jsonl")}
        band = pd.DataFrame([{f: G[s]["nodes"][f]["band"] for f in F} for s in pred.study_id], index=pred.study_id)
        rows = []
        for j, f in enumerate(F):
            y = Y[:, j]
            k = ~np.isnan(y)
            bb = band[f].values
            r = {"finding": f}
            for name, sel in (("present", bb == "present"), ("stated", np.isin(bb, ["present", "possible"]))):
                tp = int((sel & k & (y == 1)).sum())
                r[f"{name}_n"] = int((sel & k).sum())
                r[f"{name}_precision"] = tp / max(int((sel & k).sum()), 1)
                r[f"{name}_sensitivity"] = tp / max(int((k & (y == 1)).sum()), 1)
            ab = bb == "absent"
            r["absent_n"] = int((ab & k).sum())
            r["absent_miss_rate"] = int((ab & k & (y == 1)).sum()) / max(int((k & (y == 1)).sum()), 1)
            rows.append(r)
        bands = pd.DataFrame(rows).set_index("finding")
        bands.to_csv(run.dir / "bands.csv")
        run.log("2. bands:\n" + bands.round(3).to_string())
        nc = np.array([bool(G[s]["no_acute_abnormality"]) for s in pred.study_id])
        yr = Y[:, F.index("any_abnormality")]
        kk = ~np.isnan(yr)
        out["normal_call"] = {"rate": float(nc.mean()), "n_called": int(nc.sum()),
                              "precision": float(((yr == 0) & nc & kk).sum() / max((nc & kk).sum(), 1)),
                              "sensitivity": float(((yr == 0) & nc & kk).sum() / max(((yr == 0) & kk).sum(), 1))}
        run.log(f"3. normal call: {out['normal_call']}")
        # 4. Stage 8
        s8p = pr / "stage8.jsonl"
        if s8p.exists():
            rs = [json.loads(l) for l in open(s8p)]
            rs = [r for r in rs if r["study_id"] in set(pred.study_id)]
            N = len(rs)
            out["stage8"] = {"n": N, "first_pass_match_rate": sum(r["first_pass_match"] for r in rs) / N,
                             "final_llm_match_rate": 1 - sum(r["fallback"] for r in rs) / N,
                             "fallback_rate": sum(r["fallback"] for r in rs) / N,
                             "template_roundtrip_match_rate": sum(not r["template_mismatch"] for r in rs) / N}
            run.log(f"4. Stage 8: {out['stage8']}")
        # 5. H3
        if comp:
            d0 = out["macro"]["auroc"] - float(np.nanmean([cpoint[f]["auroc"] for f in F]))
            lo, hi = ci(comp["diff_boot"])
            out["H3"] = {"comparator": comp["name"], "comparator_macro_auroc": float(np.nanmean([cpoint[f]["auroc"] for f in F])),
                         "head_minus_comparator": d0, "ci": [lo, hi], "margin": a.margin, "non_inferior": bool(lo > -a.margin)}
            run.log(f"5. H3: {out['H3']}")
        # 6. report level
        if a.refs and s8p.exists():
            refs = {}
            for fp in sorted(Path(a.refs).glob("*.jsonl.gz")):
                for l in gzip.open(fp, "rt"):
                    r = json.loads(l)
                    if r["study_id"] in set(pred.study_id):
                        refs[r["study_id"]] = r
            rs = {r["study_id"]: r for r in rs}
            common = [s for s in pred.study_id if s in refs and s in rs and "final_claims" in rs[s]]
            run.log(f"6. report level: {len(common):,} studies with a reference parse and our report's claims")
            if common:
                def pos(cl):
                    return {c["finding"] for c in cl if c["cert"] in ("present", "possible") and c["finding"] != "any_abnormality"}
                tp = {f: 0 for f in F}
                fp_ = dict(tp)
                fn = dict(tp)
                ef1 = []
                for s in common:
                    ours = pos(rs[s]["final_claims"])
                    ref = pos(M.claims(refs[s], version=2))
                    for f in F:
                        tp[f] += (f in ours) and (f in ref)
                        fp_[f] += (f in ours) and (f not in ref)
                        fn[f] += (f not in ours) and (f in ref)
                    e1 = {(e["tokens"].lower(), e["label"]) for e in (rs[s].get("final_entities") or {}).values()}
                    e2 = {(e["tokens"].lower(), e["label"]) for e in (refs[s].get("entities") or {}).values()}
                    ef1.append(2 * len(e1 & e2) / max(len(e1) + len(e2), 1))
                rl = pd.DataFrame({f: {"tp": tp[f], "fp": fp_[f], "fn": fn[f],
                                       "precision": tp[f] / max(tp[f] + fp_[f], 1), "recall": tp[f] / max(tp[f] + fn[f], 1)}
                                   for f in F if f != "any_abnormality"}).T
                rl["f1"] = 2 * rl.precision * rl.recall / (rl.precision + rl.recall).replace(0, np.nan)
                rl.to_csv(run.dir / "report_level.csv")
                T, FP, FN = rl.tp.sum(), rl.fp.sum(), rl.fn.sum()
                out["report_level"] = {"n": len(common), "micro_f1": float(2 * T / max(2 * T + FP + FN, 1)),
                                       "macro_f1": float(rl.f1.fillna(0).mean()), "radgraph_entity_f1_mean": float(np.mean(ef1))}
                run.log("6. report level (positive mentions, our 13 findings):\n" + rl.round(3).to_string()
                        + f"\n   {out['report_level']}")
        (run.dir / "eval.json").write_text(json.dumps(out, indent=1, default=float))
        run.metric(summary="eval", **{k: v for k, v in out["macro"].items()})


if __name__ == "__main__":
    main()
