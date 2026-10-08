"""Archive a trained probe / Stage 3 run: predicted probabilities for EVERY study in every split (test included)
from the run's saved model.joblib, plus a copy of the model and its config. No labels are compared and no
metric is computed on any split; the only check is that the regenerated NON-test predictions equal the run's
own predictions_non_test.parquet (so the features were rebuilt exactly).

  --kind probe   : scripts/probe.py runs (Stages 1, 2, 5); features from config "features"
  --kind stage3  : scripts/stage3_factorised.py runs; concepts + config "extra"; marginals multiplied down the
                   is_a tree; side-head probabilities included
Embedding variant follows NESY_EMB (stretch by default) and must match the run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nesy import data as D, kg, paths as P  # noqa: E402
from nesy.runlog import Run  # noqa: E402
from probe import build_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="trained run directory")
    ap.add_argument("--kind", choices=["probe", "stage3"], required=True)
    ap.add_argument("--out", required=True, help="archive directory for this stage")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    with Run("export-predictions", vars(args), args.run_dir) as run:
        src, out = Path(args.src), Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        cfg = json.loads((src / "config.json").read_text())
        a = cfg.get("args", cfg)
        bundle = joblib.load(src / "model.joblib")
        t = D.study_table()
        fit = (t.split == "train").values
        run.log(f"studies {len(t):,}; splits {t.split.value_counts().to_dict()}; embedding variant {D.VARIANT}")
        if args.kind == "probe":
            groups = a["features"].split(",")
            X, names = build_features(t, groups, fit, run)
            assert names == bundle["feature_names"], "feature names differ from the trained model"
            preds = {}
            for f in D.FINDINGS:
                m = bundle["models"][f]
                preds[f] = m["platt"].predict_proba(m["probe"].decision_function(X)[:, None])[:, 1]
            res = pd.DataFrame({"study_id": t.study_id, "split": t.split, **{f"p_{f}": preds[f] for f in D.FINDINGS}})
        else:
            from stage3_factorised import concept_features
            Xc, names = concept_features(t, fit, run, Path(a["concepts"]) if Path(a["concepts"]).is_absolute()
                                         else P.V2 / a["concepts"])
            blocks = [Xc]
            for g in [x for x in a["extra"].split(",") if x]:
                Xg, ng = build_features(t, [g], fit, run)
                blocks.append(Xg)
                names += ng
            X = np.hstack(blocks)
            assert names == bundle["feature_names"], "feature names differ from the trained model"
            cond, marg = {}, {}
            from stage3_factorised import aw_mask
            for f in kg.topo_order()[::-1]:
                m = bundle["models"][f]
                cond[f] = m["platt"].predict_proba(m["model"].decision_function(aw_mask(X, names, f))[:, None])[:, 1]
            for f in kg.topo_order()[::-1]:
                ps = kg.parents_of(f)
                marg[f] = cond[f] * (marg[ps[0]] if ps else 1.0)
            res = pd.DataFrame({"study_id": t.study_id, "split": t.split, **{f"p_{f}": marg[f] for f in D.FINDINGS}})
            for f, m in bundle["side_models"].items():
                pr = m.predict_proba(aw_mask(X, names, None))
                for ci, cls in enumerate(m.classes_):
                    res[f"pside_{f}_{cls}"] = pr[:, ci]
        # reproduction check on NON-test rows only
        old = pd.read_parquet(src / "predictions_non_test.parquet").set_index("study_id")
        new = res[res.split != "test"].set_index("study_id").loc[old.index]
        cols = [c for c in old.columns if c.startswith("p")]
        diff = float(np.nanmax(np.abs(new[cols].values - old[cols].values)))
        run.log(f"non-test reproduction: {len(old):,} studies, {len(cols)} columns, max |diff| {diff:.2e}")
        if diff > 1e-4:   # float32 rounding in concept scores gives ~1e-6
            raise AssertionError(f"regenerated non-test predictions differ from the run's own (max {diff})")
        res.to_parquet(out / "predictions_all_splits.parquet", index=False)
        shutil.copy2(src / "model.joblib", out / "model.joblib")
        for fn in ("config.json", "thresholds_4band.json", "val_metrics.csv"):
            if (src / fn).exists():
                shutil.copy2(src / fn, out / fn)
        (out / "SOURCE.txt").write_text(f"source run: {src}\nkind: {args.kind}\nembedding variant: {D.VARIANT}\n"
                                        f"non-test reproduction max |diff|: {diff:.2e}\n"
                                        "test-split rows: predictions only; no test metric was computed.\n")
        run.log(f"wrote {out} (rows {len(res):,}; test rows {int((res.split == 'test').sum()):,})")


if __name__ == "__main__":
    main()
