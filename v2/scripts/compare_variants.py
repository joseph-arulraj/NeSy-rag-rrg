"""Stretch vs letterbox, side by side, for every stage (user request 2026-10-06). Reads the stretch reference
runs (fixed below) and the letterbox runs recorded by the overnight orchestrator ($ORCH/last_<step>).
Writes comparison.md in the run dir.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from nesy import paths as P  # noqa: E402
from nesy.runlog import Run  # noqa: E402

R = P.V2 / "runs"
STRETCH = {
    "Stage 1 (embedding probe)": R / "20261006-1218_stage1-v2",
    "Stage 2 (+ CheXmask)": R / "20261006-1236_stage2",
    "Stage 3 (factorised, pruned k6)": R / "20261006-1327_stage3-k6-pruned",
    "Stage 5 (embedding + nb10)": R / "20261006-1331_stage5-nb10",
}
LB_STEP = {"Stage 1 (embedding probe)": "lb-stage1", "Stage 2 (+ CheXmask)": "lb-stage2",
           "Stage 3 (factorised, pruned k6)": "lb-stage3", "Stage 5 (embedding + nb10)": "lb-stage5-emb-nb10",
           "Stage 3 + nb10 (Stage 5 on the Stage 3 base)": "lb-stage5-s3-nb10", "Stage 3 + Stage 4 regions": "lb-stage3-s4"}


def md(df: pd.DataFrame) -> str:
    return df.to_markdown() if hasattr(df, "to_markdown") else df.to_string()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orch", required=True)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    orch = Path(args.orch)

    def lb(step):
        f = orch / f"last_{step}"
        return P.V2 / f.read_text().strip() if f.exists() else None
    with Run("compare-variants", vars(args), args.run_dir) as run:
        out = ["# Stretch vs letterbox (val; seen by backbone)", ""]
        rows, per = [], {}
        for stage, step in LB_STEP.items():
            s_dir, l_dir = STRETCH.get(stage), lb(step)
            for variant, d in (("stretch", s_dir), ("letterbox", l_dir)):
                if d is None or not (d / "val_metrics.csv").exists():
                    continue
                v = pd.read_csv(d / "val_metrics.csv").set_index("finding")
                rows.append({"stage": stage, "variant": variant, "macro_auroc": v.auroc.mean(), "macro_auprc": v.auprc.mean(),
                             "macro_ece": v.ece.mean(), "run": d.name})
                per[(stage, variant)] = v.auroc
        t = pd.DataFrame(rows)
        out += ["## Macro metrics", "", t.round(4).to_string(index=False), ""]
        pf = pd.DataFrame({f"{s} | {v}": a for (s, v), a in per.items()})
        out += ["## Per-finding AUROC", "", pf.round(3).to_string(), ""]
        # Stage 4 image-level AUROC and MS-CXR localisation
        s4s = R / "20261006-1709_stage4-regions" / "stage4_val.csv"
        s4l = lb("lb-stage4")
        if s4s.exists() and s4l is not None and (s4l / "stage4_val.csv").exists():
            a, b = pd.read_csv(s4s), pd.read_csv(s4l / "stage4_val.csv")
            m = a.merge(b, on=["region_set", "finding"], suffixes=("_stretch", "_letterbox"))
            out += ["## Stage 4 image-level AUROC (max region)", "",
                    m[["region_set", "finding", "val_image_auroc_max_region_stretch", "val_image_auroc_max_region_letterbox"]].round(3).to_string(index=False), ""]
        locs = sorted(R.glob("*_mscxr-loc-stretch"))
        locl = lb("lb-mscxr-loc")
        for variant_name in ("all", "small_only"):
            parts = []
            for variant, d in (("stretch", locs[-1] if locs else None), ("letterbox", locl)):
                if d is not None and (d / f"overall_{variant_name}.csv").exists():
                    parts.append(pd.read_csv(d / f"overall_{variant_name}.csv").assign(variant=variant))
            if parts:
                out += [f"## MS-CXR localisation, {variant_name} candidate regions (model vs baselines)", "",
                        pd.concat(parts).round(3).to_string(index=False), ""]
        # normal call
        ncs = R / "20261006-1330_normal-call-stage3-pruned" / "normal_call_table.csv"
        ncl = lb("lb-normal-call")
        parts = [pd.read_csv(ncs).assign(variant="stretch")] if ncs.exists() else []
        if ncl is not None and (ncl / "normal_call_table.csv").exists():
            parts.append(pd.read_csv(ncl / "normal_call_table.csv").assign(variant="letterbox"))
        if parts:
            out += ["## Normal call (Stage 3)", "", pd.concat(parts).to_string(index=False), ""]
        text = "\n".join(out)
        (Path(run.dir) / "comparison.md").write_text(text)
        run.log("\n" + text)


if __name__ == "__main__":
    main()
