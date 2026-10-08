"""Appends the C+A+R final-head results to RESULTS.md (user decision 2026-10-07: final head = C+A+R; retrieval
only as case evidence; R2 kept, R1 and association links dropped). Reads the step run dirs recorded by the
lane-final orchestrator ($ORCH/last_<step>). Every section is independent: a missing input is reported as
missing, never guessed. Also summarises Stage 8 for the earlier head with retrieval (--stage8-retrieval).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from nesy.runlog import Run  # noqa: E402


def boot_lines(d: Path, keep=lambda c: True, findings=("MACRO",)) -> list[str]:
    b = pd.read_csv(d / "bootstrap_differences.csv")
    b = b[b.finding.isin(findings) & b.comparison.map(keep)]
    return [f"  {r.comparison} {r.finding} {r.metric}: {r.diff:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]{' (CI excludes 0)' if r.excludes_zero else ''}"
            for r in b.itertuples()]


def stage8(label: str, d: Path) -> list[str]:
    s = json.loads((d / "summary.json").read_text())
    out = [f"- **Stage 8, {label}** (`{d}`): n {s['n']}; first-pass match {s['first_pass_match_rate']:.3f}; regeneration rate "
           f"{s['regeneration_rate']:.3f}; template fallback {s['fallback_rate']:.3f}; final LLM match {s['final_llm_match_rate']:.3f}; "
           f"template round-trip (comparator check) {s['template_roundtrip_match_rate']:.3f}; LLM calls {s['llm_calls']}, errors {s['llm_errors']} "
           f"(status counts {s.get('llm_status_counts')}).",
           f"  Mismatch kinds, first attempt: {s['mismatch_kinds_first_attempt']}; all attempts: {s['mismatch_kinds_all_attempts']}; "
           f"template: {s['template_mismatch_kinds']}. Most common first-attempt mismatches: {s['top_first_attempt_mismatches'][:10]}."]
    if (d / "samples.md").exists():
        out.append(f"  Three sample reports with belief graphs: `{d / 'samples.md'}`.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orch", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--stage8-retrieval", default="")
    ap.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "RESULTS.md"))
    ap.add_argument("--run-dir", default=None)
    a = ap.parse_args()
    with Run("results-final", vars(a), a.run_dir) as run:
        O = Path(a.orch)

        def last(step):
            f = O / f"last_{step}"
            return Path(f.read_text().strip()) if f.exists() else None
        head = Path(a.head)
        out = [f"\n### Final head = C+A+R (user decision 2026-10-07; written automatically by `scripts/results_final.py`, run `{run.dir.name}`; val, seen by backbone)",
               f"Head run `{head}`: 77 concepts + 17 CheXmask anatomy features + region scores (CheXmask set B); calibration (Platt) on calib and four-band thresholds on thresh were fitted inside that run. Retrieval is used only as case-based evidence (k = 5, context only). R2 kept, R1 and association links dropped (user decision)."]
        sections = []

        def sec(name, fn):
            try:
                sections.extend(fn())
            except Exception as exc:                          # noqa: BLE001
                sections.append(f"- {name}: NOT WRITTEN ({type(exc).__name__}: {exc})")
                run.log(f"{name} failed: {exc}")

        def head_metrics():
            v = pd.read_csv(last("view-metrics") / "view_metrics.csv")
            v = v[v.model == "HEAD"]
            mac = v.groupby("view")[["auroc", "auprc", "ece"]].mean()
            pf = v.pivot(index="finding", columns="view", values="auroc")[["all", "PA", "AP"]].round(3)
            hv = pd.read_csv(last("violations") / "hierarchy_violations.csv", index_col=0)
            return [f"- **Head metrics:** macro AUROC {mac.loc['all', 'auroc']:.4f}, AUPRC {mac.loc['all', 'auprc']:.4f}, ECE {mac.loc['all', 'ece']:.4f}; "
                    f"AUROC PA {mac.loc['PA', 'auroc']:.4f} / AP {mac.loc['AP', 'auroc']:.4f}. Hierarchy violations: total "
                    f"{hv.loc['total_over_edges'].to_dict()}, studies with ≥ 1 {hv.loc['studies_with_any_violation'].to_dict()}.",
                    "  Per finding AUROC (all / PA / AP):\n```\n" + pf.to_string() + "\n```"]

        def rules():
            r = json.loads((last("rules") / "rules_eval.json").read_text())
            r1, r2 = r["R1"], r["R2"]
            return [f"- **Rules on this head** (`{last('rules')}`; region thresholds on thresh): R1 precision {r1['prec0']:.3f} → {r1['prec1']:.3f}, "
                    f"CI {[round(x, 4) for x in r1['ci']]}, sensitivity {r1['sens0']:.3f} → {r1['sens1']:.3f}, demoted {r1['demoted']}; "
                    f"R2 side accuracy {r2['acc0']:.3f} → {r2['acc2']:.3f}, CI {[round(x, 4) for x in r2['ci']]}, share keeping a side "
                    f"{r2['keep_side0']:.3f} → {r2['keep_side2']:.3f}. Applied: R2 on, R1 off (user decision)."]

        def mlp():
            return ["- **MLP head vs linear C+A+R head** (main = early stopping on the inner holdout; `es-val` = optimistic upper bound):"] + \
                boot_lines(last("mlp-bootstrap"), lambda c: "LIN" in c)

        def densenet():
            lines = ["- **DenseNet-121 vs C+A+R head and Stage 1** (main = inner-holdout epoch):"] + boot_lines(last("c-bootstrap"))
            lines += ["  Per finding, DenseNet − HEAD AUROC:"] + boot_lines(last("c-bootstrap"), lambda c: c in ("DenseNet - HEAD", "HEAD - DenseNet"),
                                                                           findings=tuple(pd.read_csv(last("c-bootstrap") / "bootstrap_differences.csv").finding.unique()))
            if last("c-bootstrap-optimistic"):
                lines += ["  Optimistic DenseNet (epoch chosen on validate):"] + boot_lines(last("c-bootstrap-optimistic"))
            return lines

        def cases():
            t = pd.read_csv(last("case-eval") / "present_error_by_case_support.csv", index_col=0)
            ag = pd.read_csv(last("case-eval") / "case_agreement.csv", index_col=0)
            a_ = t.loc["ALL"]
            return [f"- **Case evidence on this head:** agreement {ag.loc['ALL', 'agreement']:.3f}; present calls with ≤ 1 of 5 similar positive: "
                    f"error {a_.error_low_support:.3f} (n {int(a_.n_low_support)}) vs {a_.error_other:.3f} (n {int(a_.n_other)}), "
                    f"diff {a_['diff']:+.3f} [{a_.ci_low:+.3f}, {a_.ci_high:+.3f}]."]

        sec("head metrics", head_metrics)
        sec("rules", rules)
        sec("MLP", mlp)
        sec("DenseNet", densenet)
        sec("case evidence", cases)
        sec("Stage 8 C+A+R", lambda: stage8("C+A+R head (final)", last("stage8-car")))
        if a.stage8_retrieval:
            sec("Stage 8 with retrieval", lambda: stage8("head with retrieval (earlier, kept for the record)", Path(a.stage8_retrieval)))
        text = "\n".join(out + sections) + "\n"
        (run.dir / "appended.md").write_text(text)
        with open(a.results, "a") as fh:
            fh.write(text)
        run.log("appended:\n" + text)


if __name__ == "__main__":
    main()
