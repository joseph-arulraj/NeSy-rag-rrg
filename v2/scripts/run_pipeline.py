"""End-to-end pipeline runner: cached image features -> C+A+R head -> calibrated marginals -> four bands ->
belief graph (grounding, R1/R2 as configured, zone G7, D2/D3) -> case evidence (context only) -> template
report -> Stage 8 LLM phrasing with RadGraph round-trip guard and template fallback.

All choices come from configs/pipeline.yaml. config.json records the git commit and a manifest: SHA-256 of the
head model, calibrators and thresholds, region scores and thresholds, concept list and embeddings, CheXmask
features, case table, every KG file, the configs and the code files used, so a final freeze is one manifest.

Guards: the MIMIC test split is refused unless --allow-test; external sets (vindr_test, padchest_gr) are refused
unless --allow-external. No labels are loaded by this script: label columns are dropped from the study table
before any row selection, and only the fit split (feature standardisation) and the requested split are used.

Outputs (run dir or --out, resumable): predictions.parquet, graphs.jsonl (graph + template per study),
stage8.jsonl (one line per study), reports.jsonl (final report and its source), summary.json.

  python -u v2/scripts/run_pipeline.py --split val [--no-stage8] [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
sys.path.insert(0, str(V2 / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

EXTERNAL = ("vindr_test", "padchest_gr")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def manifest(cfg: dict, cfg_path: Path) -> dict:
    from nesy import data as D, paths as P
    i = cfg["inputs"]
    head = V2 / i["head_run"]
    hcfg = json.loads((head / "config.json").read_text())
    files = {
        "head_model_and_calibrators": head / "model.joblib",
        "head_thresholds_4band": head / "thresholds_4band.json",
        "head_config": head / "config.json",
        "concept_list": V2 / hcfg["concepts"],
        "concept_text_embeddings": Path("/scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/model_weights/concept_embeddings_368294.pt"),
        "splits": V2 / "data/splits/mimic_splits_v1.parquet",
        "image_embeddings": D.EMB, "image_embeddings_index": D.EMB_INDEX,
        "chexmask_features": P.FEATURES / "chexmask_mimic.parquet",
        "s4_scores_B": V2 / i["s4_run"] / "scores_B_chexmask.npy", "s4_scores_index": V2 / i["s4_run"] / "scores_index.parquet",
        "region_thresholds_and_rules": V2 / i["region_thresholds_from"] / "rules_eval.json",
        "case_table": V2 / i["cases"],
        "pipeline_config": cfg_path, "report_config": V2 / "configs/report.yaml",
        "llm_config": (V2 / cfg["stage8"]["llm_model_from"]).resolve(),
        "imagenome_side_labels_unused_at_inference": P.FEATURES / "imagenome_side_explicit_v1.parquet",
    }
    for p in sorted((V2 / "kg").rglob("*.yaml")):
        files[f"kg/{p.relative_to(V2 / 'kg')}"] = p
    for p in sorted((V2 / "nesy").glob("*.py")):
        files[f"code/nesy/{p.name}"] = p
    for s in ("run_pipeline.py", "stage8_phrase.py", "stage3_factorised.py", "probe.py", "stage4_regions.py", "region_features.py"):
        files[f"code/scripts/{s}"] = V2 / "scripts" / s
    files.pop("imagenome_side_labels_unused_at_inference")
    return {k: {"path": str(p), "sha256": sha256(Path(p)), "bytes": Path(p).stat().st_size} for k, p in files.items()}


def label_free_study_table() -> pd.DataFrame:
    """D.study_table() without the label file: embedding index (row order) + frozen patient split + view."""
    from nesy import data as D, manifests as M
    idx = pd.read_parquet(D.EMB_INDEX)
    sp = pd.read_parquet(V2 / "data/splits/mimic_splits_v1.parquet", columns=["subject_id", "split"])
    view = M.load("mimic", columns=["dicom_id", "view"])[["dicom_id", "view"]]
    t = idx.merge(sp, on="subject_id", how="left", validate="many_to_one").merge(view, on="dicom_id", how="left")
    if t.split.isna().any():
        raise ValueError("studies without a split")
    return t[["row", "study_id", "dicom_id", "subject_id", "split", "view"]]


def head_predictions(cfg: dict, split: str, run) -> pd.DataFrame:
    """C+A+R head on the requested split. Features are standardised with fit-split statistics exactly as in
    training, so only the fit split and the requested split are assembled."""
    import joblib
    from nesy import data as D, kg, paths as P
    from probe import build_features
    from stage3_factorised import aw_mask, concept_features
    head = V2 / cfg["inputs"]["head_run"]
    hcfg = json.loads((head / "config.json").read_text())
    bundle = joblib.load(head / "model.joblib")
    t = label_free_study_table()
    t = t[t.split.isin(["train", split])].reset_index(drop=True)          # other splits never assembled
    fit = (t.split == "train").values
    run.log(f"study rows assembled: fit {int(fit.sum()):,} (standardisation only), {split} {int((~fit).sum()):,}")
    Xc, names = concept_features(t, fit, run, V2 / hcfg["concepts"])
    blocks = [Xc]
    for g in [x for x in hcfg["extra"].split(",") if x]:
        Xg, ng = build_features(t, [g], fit, run)
        blocks.append(Xg)
        names += ng
    X = np.hstack(blocks)
    assert names == bundle["feature_names"], "feature names differ from the trained head"
    sel = ~fit
    meta = pd.DataFrame({"study_id": t.study_id.values[sel], "dicom_id": t.dicom_id.values[sel], "split": split})
    return apply_head(bundle, X[sel], names, meta, run)


def apply_head(bundle: dict, X: np.ndarray, names: list[str], meta: pd.DataFrame, run) -> pd.DataFrame:
    from nesy import data as D, kg
    from stage3_factorised import aw_mask
    cond, marg = {}, {}
    for f in kg.topo_order()[::-1]:
        m = bundle["models"][f]
        cond[f] = m["platt"].predict_proba(m["model"].decision_function(aw_mask(X, names, f))[:, None])[:, 1]
    for f in kg.topo_order()[::-1]:
        ps = kg.parents_of(f)
        marg[f] = cond[f] * (marg[ps[0]] if ps else 1.0)
    res = pd.concat([meta.reset_index(drop=True), pd.DataFrame({f"p_{f}": marg[f] for f in D.FINDINGS})], axis=1)
    for f, m in bundle["side_models"].items():
        pr = m.predict_proba(aw_mask(X, names, None))
        for ci, cls in enumerate(m.classes_):
            res[f"pside_{f}_{cls}"] = pr[:, ci]
    viol = sum(int((marg[e["child"]] > marg[e["parent"]] + 1e-12).sum()) for e in kg.load_findings()["is_a"])
    run.log(f"head: {len(res):,} studies ({meta.split.iloc[0] if len(meta) else ''}); hierarchy violations {viol} (must be 0)")
    if viol:
        raise AssertionError("hierarchy violated")
    return res


EXT_SOURCES = {  # dataset: (region cache, CheXmask feature file)
    "vindr_test": ("ext_vindr_test_lb", "chexmask_vindr_test.parquet"),
    "padchest_gr": ("ext_padchest_gr_lb", "chexmask_padchest_gr.parquet"),
}


def external_path_predictions(cfg: dict, dataset: str, split: str, s4_source: str, run):
    """Head predictions from the image-set feature path (nesy/ext_features.py): CLEAR global embedding from the
    region cache, CheXmask anatomy features, Stage 4 region scores from the saved Stage 4 models
    (s4_source='models') or from the cached MIMIC scores (s4_source='cache', MIMIC development check only).
    dataset 'mimic' = development check on a MIMIC split; returns (predictions, (findings, {dicom: row}, scores))."""
    import joblib
    import torch
    from nesy import data as D, ext_features as XF, paths as P
    from stage4_regions import FIND
    i = cfg["inputs"]
    head = V2 / i["head_run"]
    hcfg = json.loads((head / "config.json").read_text())
    bundle = joblib.load(head / "model.joblib")
    names = bundle["feature_names"]
    sel = pd.read_csv(V2 / hcfg["concepts"]).drop_duplicates("concept_id")
    emb = torch.load(Path("/scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/model_weights/concept_embeddings_368294.pt"),
                     map_location="cpu", weights_only=False, mmap=True)
    C = emb[torch.as_tensor(sel.concept_id.values)].float().numpy()
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    t = label_free_study_table()
    fit = t[t.split == "train"]
    st = XF.fit_stats(hcfg, fit.row.values, fit.dicom_id.values, fit.view.values, V2 / i["s4_run"], names, C, D.embeddings)
    run.log(f"fit-split statistics from {len(fit):,} MIMIC fit studies (concepts, CheXmask, Stage 4 scores of {i['s4_run']})")
    if dataset == "mimic":
        tt = t[t.split == split]
        ids, study, view = tt.dicom_id.values, tt.study_id.values, tt.view.values
        cache, cmfile = "mimic_full_lb", "chexmask_mimic.parquet"
    else:
        cache, cmfile = EXT_SOURCES[dataset]
        ids = pd.read_parquet(P.FEATURES / "regions" / f"{cache.replace('_lb', '')}_ids.parquet").image_id.values
        study = ids
        vd = cfg["external"]["view_default"]
        view = np.array([vd] * len(ids), dtype=object)
        run.log(f"{dataset}: {len(ids):,} images; view unknown for every image -> treated as {vd!r} (config external.view_default)")
    rc = XF.load_region_cache(cache, list(ids))
    if len(rc["image_id"]) != len(ids):
        raise ValueError(f"region cache {cache} has {len(rc['image_id'])} of {len(ids)} images")
    n_noreg = int((~rc["present"].any(1)).sum())
    if s4_source == "models":
        mdl = joblib.load(V2 / i["s4_models_run"] / "models.joblib")
        sc = XF.score_regions(mdl["models"]["B_chexmask"], rc["chexmask"], rc["present"], list(FIND))
        run.log(f"Stage 4 scores from saved models {i['s4_models_run']} ({len(mdl['models']['B_chexmask'])} findings); "
                f"images without any region {n_noreg}")
    else:
        if dataset != "mimic":
            raise ValueError("cached Stage 4 scores exist only for MIMIC")
        sidx = pd.read_parquet(V2 / i["s4_run"] / "scores_index.parquet")
        pos = {d_: k for k, d_ in enumerate(sidx.dicom_id)}
        cached = np.load(V2 / i["s4_run"] / "scores_B_chexmask.npy", mmap_mode="r")
        sc = np.stack([np.asarray(cached[pos[d_]], np.float32) for d_ in ids])
        run.log("Stage 4 scores from the cached MIMIC run (development check of the feature path)")
    cm = pd.read_parquet(P.FEATURES / cmfile)
    cmr = XF.chexmask_raw(cm, ids, view)
    run.log(f"CheXmask anatomy: {int(cmr[1].sum()):,} of {len(ids):,} images missing or failed QC (indicator, never zero)")
    X = XF.design(st, names, rc["global"], C, cmr, sc, list(FIND))
    meta = pd.DataFrame({"study_id": study, "dicom_id": ids, "split": split if dataset == "mimic" else dataset})
    res = apply_head(bundle, X, names, meta, run)
    np.save(run.dir / "design_matrix.npy", X) if dataset == "mimic" else None
    return res, (list(FIND), {d_: k for k, d_ in enumerate(ids)}, sc)


def build_graphs(cfg: dict, preds: pd.DataFrame, run, region=None) -> list[dict]:
    from nesy import case_evidence as CE, data as D, pipeline_graph as PG, region_support as RS, report as R
    i, rules = cfg["inputs"], cfg["rules"]
    if cfg["report"]["layout"] != "current":
        raise NotImplementedError(f"report layout {cfg['report']['layout']!r}: only 'current' exists (RadReport proposals not approved)")
    thr = json.loads((V2 / i["head_run"] / "thresholds_4band.json").read_text())
    rthr = json.loads((V2 / i["region_thresholds_from"] / "rules_eval.json").read_text())["region_thresholds"]
    if region is None:
        FIND, idx, sc = RS.load(V2 / i["s4_run"])
        pos = {d: k for k, d in enumerate(idx.dicom_id)}
    else:
        FIND, pos, sc = region
    cases = pd.read_parquet(V2 / i["cases"])
    cases = cases[cases.study_id.isin(preds.study_id)].set_index("study_id")
    k_cases = int(Path(i["cases"]).stem.rsplit("_k", 1)[1])
    out, n_ev, n_case = [], 0, 0
    for row in preds.to_dict("records"):
        j = pos.get(row["dicom_id"])
        ev = {} if j is None else {f: RS.evidence(np.asarray(sc[j], np.float32), fi, f) for fi, f in enumerate(FIND) if f in D.FINDINGS}
        n_ev += j is not None
        g = PG.build(row["study_id"], row, thr, ev, rthr, use_r1=bool(rules["R1_localisation_support"]),
                     use_r2=bool(rules["R2_side_agreement"]))
        if row["study_id"] in cases.index:
            CE.attach(g, cases.loc[row["study_id"]].to_dict(), k=k_cases)
            n_case += 1
        tpl = R.render(g, show_cases=bool(cfg["report"]["show_case_evidence"]))
        sid = row["study_id"]
        out.append({"study_id": int(sid) if isinstance(sid, (int, np.integer)) else sid, "graph": g, "template": tpl})
    run.log(f"graphs: {len(out):,}; with region evidence {n_ev:,}; with case evidence {n_case:,}; "
            f"rules R1={rules['R1_localisation_support']} R2={rules['R2_side_agreement']}; layout {cfg['report']['layout']}; "
            f"show_case_evidence {cfg['report']['show_case_evidence']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(V2 / "configs/pipeline.yaml"))
    ap.add_argument("--dataset", default="mimic", choices=["mimic", *EXTERNAL])
    ap.add_argument("--split", default="val")
    ap.add_argument("--allow-test", action="store_true", help="required to run on the MIMIC test split")
    ap.add_argument("--allow-external", action="store_true", help="required to run on an external test set")
    ap.add_argument("--no-stage8", action="store_true", help="stop after the template reports")
    ap.add_argument("--feature-path", default="standard", choices=["standard", "external"],
                    help="external = the image-set feature path used for external sets (on MIMIC: development check)")
    ap.add_argument("--s4-models-run", default=None, help="override inputs.s4_models_run (Stage 4 run with models.joblib)")
    ap.add_argument("--s4-source", default="models", choices=["models", "cache"],
                    help="external feature path: region scores from saved Stage 4 models, or cached MIMIC scores")
    ap.add_argument("--max-attempts", type=int, default=None, help="override stage8.max_attempts")
    ap.add_argument("--comparator", default=None, help="override stage8.comparator (v1 | v2)")
    ap.add_argument("--retry", default=None, help="override stage8.retry (feedback | none)")
    ap.add_argument("--study-ids", default=None, help="file with one study_id per line: run only these (development)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--out", default=None, help="output dir (default: run dir); a fixed one resumes Stage 8")
    ap.add_argument("--run-dir", default=None)
    a = ap.parse_args()
    cfg_path = Path(a.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    s8 = cfg["stage8"]
    if a.s4_models_run:
        cfg["inputs"]["s4_models_run"] = a.s4_models_run
    for k in ("max_attempts", "comparator", "retry"):
        if getattr(a, k) is not None:
            s8[k] = getattr(a, k)
    # ---- guards (before anything is loaded)
    if a.dataset != "mimic":
        if not a.allow_external:
            sys.exit(f"REFUSED: {a.dataset} is an external test set; pass --allow-external only when the user has approved external runs")
        a.feature_path = "external"
    if a.split == "test" and not a.allow_test:
        sys.exit("REFUSED: the MIMIC test split needs --allow-test (user approval required)")
    if a.split not in cfg["guards"]["allowed_splits"] + (["test"] if a.allow_test else []):
        sys.exit(f"REFUSED: split {a.split!r} not allowed")
    os.environ["NESY_EMB"] = cfg["inputs"]["embedding"]
    os.environ["NESY_S4_RUN"] = cfg["inputs"]["s4_run"]
    os.environ["NESY_S4_SETS"] = cfg["inputs"]["s4_sets"]
    os.chdir(V2)
    from nesy.runlog import Progress, Run
    with Run("pipeline", {**vars(a), "pipeline_config": cfg}, a.run_dir) as run:
        man = manifest(cfg, cfg_path)
        (run.dir / "manifest.json").write_text(json.dumps(man, indent=1))
        cj = json.loads((run.dir / "config.json").read_text())
        cj["manifest"] = man
        (run.dir / "config.json").write_text(json.dumps(cj, indent=2, default=str))
        run.log(f"manifest: {len(man)} files hashed -> manifest.json (also in config.json); git {cj.get('git_commit')}")
        out = Path(a.out) if a.out else run.dir
        out.mkdir(parents=True, exist_ok=True)
        region = None
        if a.feature_path == "external":
            preds, region = external_path_predictions(cfg, a.dataset, a.split, a.s4_source, run)
        else:
            preds = head_predictions(cfg, a.split, run)
        if a.study_ids:
            keep = {int(x) for x in Path(a.study_ids).read_text().split()}
            preds = preds[preds.study_id.isin(keep)]
        if a.limit:
            preds = preds.sample(min(a.limit, len(preds)), random_state=0)
        preds.to_parquet(out / "predictions.parquet", index=False)
        items = build_graphs(cfg, preds, run, region)
        with open(out / "graphs.jsonl", "w") as fh:
            for it in items:
                fh.write(json.dumps({"study_id": it["study_id"], "graph": it["graph"].to_dict(), "template": it["template"]}, default=str) + "\n")
        bands = Counter((f, n.band) for it in items for f, n in it["graph"].nodes.items())
        normal = sum(bool(it["graph"].no_acute_abnormality) for it in items)
        summ = {"split": a.split, "n": len(items), "normal_calls": normal, "normal_call_rate": normal / max(len(items), 1),
                "bands": {f"{f}:{b}": c for (f, b), c in sorted(bands.items())}}
        if a.no_stage8 or not s8["enabled"]:
            run.log("Stage 8 skipped; reports = templates")
            reports = [{"study_id": it["study_id"], "source": "template", **{k: it["template"][k] for k in ("findings", "impression")}} for it in items]
        else:
            from radgraph import RadGraph
            from nesy import stage8 as S8
            llm_cfg = yaml.safe_load((V2 / s8["llm_model_from"]).read_text())["llm"]
            llm = S8.LLM(os.path.expanduser(s8["key_file"]), llm_cfg)
            rg = RadGraph(model_type=s8["radgraph_model"], batch_size=32, cuda=0)
            res_path = out / "stage8.jsonl"
            done = {json.loads(l)["study_id"] for l in res_path.read_text().splitlines() if l.strip()} if res_path.exists() else set()
            todo = [it for it in items if it["study_id"] not in done]
            run.log(f"Stage 8: {len(items):,} studies, {len(done):,} done, {len(todo):,} to do; LLM {llm_cfg['model']}; "
                    f"max attempts {s8['max_attempts']}, retry {s8['retry']}, comparator {s8['comparator']}, threads {s8['threads']}")
            prog = Progress(run, len(todo), "studies phrased", every_s=120)
            for c in range(0, len(todo), a.chunk):
                part = [{"study_id": it["study_id"], "template": it["template"], "no_acute": it["graph"].no_acute_abnormality}
                        for it in todo[c:c + a.chunk]]
                res = S8.phrase(part, llm, rg, int(s8["max_attempts"]), int(s8["threads"]), s8["retry"], s8["comparator"])
                with open(res_path, "a") as fh:
                    for r in res:
                        fh.write(json.dumps(r, default=str) + "\n")
                prog.update(min(c + a.chunk, len(todo)))
            ids = {it["study_id"] for it in items}
            rs = [json.loads(l) for l in res_path.read_text().splitlines() if l.strip()]
            rs = [r for r in rs if r["study_id"] in ids]
            N = len(rs)
            fk = Counter((m["kind"], m.get("finding")) for r in rs if r["attempts"] for m in r["attempts"][0]["mismatch"])
            summ["stage8"] = {"n": N, "first_pass_match_rate": sum(r["first_pass_match"] for r in rs) / N,
                              "final_llm_match_rate": 1 - sum(r["fallback"] for r in rs) / N,
                              "fallback_rate": sum(r["fallback"] for r in rs) / N,
                              "rescued_by_retry": sum((not r["fallback"]) and not r["first_pass_match"] for r in rs),
                              "template_roundtrip_match_rate": sum(not r["template_mismatch"] for r in rs) / N,
                              "template_mismatches": dict(Counter(f"{m['kind']}:{m.get('finding')}" for r in rs for m in r["template_mismatch"]).most_common()),
                              "top_first_attempt_mismatches": [f"{k}:{f}={n}" for (k, f), n in fk.most_common(15)],
                              "llm_calls": llm.calls, "llm_errors": llm.errors, "llm_status": {str(k): v for k, v in llm.status.items()}}
            reports = [{"study_id": r["study_id"], "source": r["source"], **r["final_text"]} for r in rs]
        with open(out / "reports.jsonl", "w") as fh:
            for r in reports:
                fh.write(json.dumps(r) + "\n")
        (run.dir / "summary.json").write_text(json.dumps(summ, indent=1))
        run.log("SUMMARY " + json.dumps({k: v for k, v in summ.items() if k != "bands"}, indent=1))
        run.metric(summary="pipeline", n=summ["n"], normal_call_rate=summ["normal_call_rate"],
                   **({f"stage8_{k}": v for k, v in summ["stage8"].items() if isinstance(v, (int, float))} if "stage8" in summ else {}))


if __name__ == "__main__":
    main()
