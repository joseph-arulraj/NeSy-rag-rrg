"""Append the 2026-10-07 runner / Stage 8 / external-prep results to RESULTS.md and update QUEUE.md, from the
outputs of the lane-close chain (written automatically; numbers read from the run files)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent


def rd(orch: Path, step: str) -> Path | None:
    f = orch / f"last_{step}"
    return V2 / f.read_text().strip() if f.exists() else None


def js(p: Path | None, name: str):
    try:
        return json.loads((p / name).read_text())
    except Exception:                                   # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orch", required=True)
    ap.add_argument("--stage8-run", default="runs/20261007-120046_pipeline-val-v2")
    ap.add_argument("--run-dir", default=None)
    a = ap.parse_args()
    import sys
    sys.path.insert(0, str(V2))
    from nesy.runlog import Run
    with Run("results-runner", vars(a), a.run_dir) as run:
        orch = V2 / a.orch if not Path(a.orch).is_absolute() else Path(a.orch)
        L = [(orch / "results_static.md").read_text().rstrip(), ""]
        s8 = js(V2 / a.stage8_run, "summary.json")
        L.append(f"### Stage 8 rerun on validate, comparator v2 and feedback v2, up to 3 attempts (`v2/{a.stage8_run}`; written automatically)")
        if s8 and "stage8" in s8:
            x = s8["stage8"]
            L += [f"- First-pass match **{x['first_pass_match_rate']:.3f}** (was 0.875 with v1); final LLM match **{x['final_llm_match_rate']:.3f}** "
                  f"(was 0.875); fallback **{x['fallback_rate']:.3f}** (was 0.125); rescued by a retry {x['rescued_by_retry']}; "
                  f"template round trip **{x['template_roundtrip_match_rate']:.3f}** (was 0.962).",
                  f"- Remaining first-attempt mismatches: {x['top_first_attempt_mismatches']}; template mismatches: {x['template_mismatches']}.",
                  f"- LLM calls {x['llm_calls']}, errors {x['llm_errors']}, HTTP status counts {x['llm_status']}. Normal-call rate {s8['normal_call_rate']:.3f}.",
                  "- Retry policy: kept at 3 attempts if retries rescue most first-attempt failures (see 'rescued by a retry' vs fallback); "
                  "the user's rule is to cut to 1 attempt + template if they rescue few. Decision recorded in STATE.md by Claude after reading this."]
        else:
            L.append("- NOT AVAILABLE: the Stage 8 run did not finish (see its log).")
        ev = js(rd(orch, "eval-val-v2"), "eval.json")
        L += ["", f"### Evaluation of the runner output on validate (`v2/{rd(orch, 'eval-val-v2').relative_to(V2) if rd(orch, 'eval-val-v2') else '?'}`; CheXpert-derived labels)"]
        if ev:
            L.append(f"- Macro AUROC {ev['macro']['auroc']:.4f} [{ev['macro_ci']['auroc'][0]:.4f}, {ev['macro_ci']['auroc'][1]:.4f}], AUPRC {ev['macro']['auprc']:.4f}, "
                     f"ECE {ev['macro']['ece']:.4f}; hierarchy violations {ev['hierarchy_violations']}; normal call {ev['normal_call']}.")
            if "H3" in ev:
                h = ev["H3"]
                L.append(f"- H3 (dev, val): head − {h['comparator']} macro AUROC {h['head_minus_comparator']:+.4f} [{h['ci'][0]:+.4f}, {h['ci'][1]:+.4f}]; "
                         f"margin {h['margin']}; non-inferior: {h['non_inferior']}.")
            if "report_level" in ev:
                r = ev["report_level"]
                L.append(f"- Report level vs the cached validate reference parses (positive mentions of our 13 findings, comparator v2 on both): "
                         f"micro F1 {r['micro_f1']:.3f}, macro F1 {r['macro_f1']:.3f}, RadGraph entity F1 (exact tokens + label) {r['radgraph_entity_f1_mean']:.3f}; "
                         f"n {r['n']}. Per finding: `report_level.csv` in the eval run.")
        else:
            L.append("- NOT AVAILABLE (eval step failed or did not run).")
        L += ["", "### External-run prep (user request 2026-10-07; nothing run on external images, no external labels)"]
        s4n = (orch.parent / "20261007-1215_lane-extprep" / "last_s4-saved")
        L.append(f"- Stage 4 region classifiers retrained with saved models: `v2/{s4n.read_text().strip() if s4n.exists() else '?'}` "
                 "(same settings as `20261006-224243_lb-stage4`; labels of train/val only).")
        c = js(rd(orch, "s4-compare"), "s4_compare.json")
        if c:
            L.append(f"- Retrained vs cached set-B scores: val max |diff| {c['val']['max_abs_diff']:.4f}, corr {c['val']['corr']:.6f}; "
                     f"all non-fit max |diff| {c['all_non_fit']['max_abs_diff']:.4f}; fit (out-of-fold) max |diff| {c['fit']['max_abs_diff']:.4f}. "
                     f"Scores from the saved models vs the retrained run's own val scores: max |diff| {c['models_vs_new_run_val']['max_abs_diff']:.4f} (float16 storage).")
        for step, what in (("eq-extpath-cache", "External feature path on MIMIC val with the cached region scores (tests the duplicated transforms)"),
                           ("eq-extpath-models", "External feature path on MIMIC val with region scores from the saved models (the path an external run uses)")):
            e = js(rd(orch, step), "equivalence.json")
            if e:
                L.append(f"- {what}: probabilities max |diff| {e['probabilities']['max_abs_diff']:.2e}; band differences {e['graphs']['band']}, "
                         f"side {e['graphs']['side']}, zone {e['graphs']['zone']}, normal call {e['graphs']['normal_call']}; template differences "
                         f"{e['templates']['rerendered_from_reference_differs']}; ALL PASS {e['all_pass']}.")
            else:
                L.append(f"- {what}: NOT AVAILABLE (step failed).")
        L.append("- To use: set `inputs.s4_models_run` in `v2/configs/pipeline.yaml` to the retrained run (or pass `--s4-models-run`). "
                 "External runs still need `--allow-external`, the mapping decisions and a decision on `external.view_default`.")
        with open(ROOT / "RESULTS.md", "a") as fh:
            fh.write("\n" + "\n".join(L) + "\n")
        q = ROOT / "QUEUE.md"
        s = q.read_text()
        s = s.replace("**Running: lane-close chain", "**Finished: lane-close chain")
        q.write_text(s)
        run.log("RESULTS.md appended; QUEUE.md updated")


if __name__ == "__main__":
    main()
