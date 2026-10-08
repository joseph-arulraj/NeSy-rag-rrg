# Review guide (2026-10-07, about 1 hour)

Terms, rule IDs (D1, G6, R2, P1 …) and what each stage takes in and puts out: `GLOSSARY.md`.

All numbers are on MIMIC validate (1,733 studies, 487 patients) and are **seen by backbone**. Nothing has been run on test or external data.

## 1. Orientation (10 min)
| Read | Why |
|---|---|
| `STATE.md` | Every decision so far with its reason, the conventions, and what is open. The section "End-to-end runner and test readiness" is the newest. |
| `QUEUE.md` | What is running (nothing), and the list of decisions waiting on you. |
| `PIPELINE_BRIEF.md`, top block "Changes since this brief" | Where the build departs from the original plan. |

## 2. Results, stage by stage (20 min)
The ladder below is copied from `RESULTS.md` (letterbox preprocessing; macro over 14 outputs; CIs are 95% patient bootstrap, 2,000 resamples).

| Step | Macro AUROC | Other | RESULTS.md section (line) |
|---|---|---|---|
| Stage 1: linear probe on CLEAR embedding | 0.8305 | ECE 0.024 | "Letterbox full pass" (~405) |
| Stage 2: + CheXmask anatomy (A) | 0.8350 | +0.0044 vs S1, n.s. | same |
| Stage 3: factorised head, 77 concepts + A | 0.8306 | **0 hierarchy violations** (S1: 392) | same |
| Stage 4: region classifier (A2) | MS-CXR: region hit 0.838, side 0.867 | | same, and "MS-CXR localisation" (~386) |
| **Final head C+A+R** (concepts + anatomy + regions) | **0.8344** [0.813, 0.855] | AUPRC 0.491, ECE 0.024, 0 violations | "Final head = C+A+R" (~566) |
| + retrieval (dropped) | +0.0018 [−0.0006, +0.0043] n.s. | kept only as case evidence | "Lanes A2–A5" (~547) |
| + association links (dropped) | −0.0002 n.s. | | same |
| MLP instead of linear (dropped) | n.s. | | same |
| Glossary concepts instead of the 77 (ablation) | −0.0004 n.s. | | "KG task 7" (~666) |
| DenseNet-121 baseline | 0.8175; −0.019 vs head (significant) | 164 violations | "Lane C" (~443) |
| H3 comparator: probe on embedding + A + R | 0.8354; head − probe −0.0010 [−0.0052, +0.0031] | within the 0.02 margin | "H3 comparator" (~723) |
| Which inputs matter (C+A+R minus one group) | without A −0.17; without R −0.016; without C −0.005 n.s. | | "Input-group table" in "Lanes A2–A5" |

**Full pipeline on validate** (from "2026-10-07 midday", ~674 to the end of RESULTS.md):
- **Normal call:** 13.9% of studies, precision 0.938, sensitivity 0.366.
- **Stage 8 (LLM phrasing with the RadGraph guard):** first-pass match 0.998, final match 1.000, fallback 0, template round trip 0.999. Before the comparator fix these were 0.875 / 0.875 / 0.125 / 0.962.
- **Report level against the radiologist reports** (positive mentions of our 13 findings): micro F1 0.563, macro F1 0.366, RadGraph entity F1 0.165. These are the first numbers and there is no baseline yet.
- **Per finding:** AUROC per finding is in `runs/20261007-123526_eval-val-v2/per_finding.csv`; band precision and sensitivity in `bands.csv` (same folder). Fracture (17 positives) and pleural other (14) are too rare to read.

## 3. Generated reports with full traces (10 min)
**`outputs/trace_samples_val.md`** contains 8 validate studies, one per situation:
- a normal call;
- one finding;
- several findings with side and zone;
- a side withheld by R2;
- a parent raised by D2;
- only "possible" findings;
- "nothing reportable";
- an LLM retry that rescued the report.

Each trace shows, in order:
1. the reference report;
2. every probability, band, side and zone;
3. the full audit trail of rules that fired;
4. the case evidence;
5. the template;
6. the exact LLM prompt;
7. each LLM attempt with the comparator's verdict;
8. the final report.

The raw files for all 1,733 studies are in `runs/20261007-120046_pipeline-val-v2/`:
- `graphs.jsonl`: the belief graph and audit trail for every study;
- `stage8.jsonl`: the prompt, the attempts and the RadGraph claims;
- `reports.jsonl`: the final reports.

One thing to look at, in the trace "Several findings with side and zone" (study 50127791):
- Consolidation gets a "left lower zone" (G7).
- Consolidation is then omitted from the text because its child, pneumonia, is stated (P1).
- Pneumonia's side is withheld by R2.

So the zone never reaches the report. This is consistent with the rules, but you may want a different P1 behaviour.

## 4. Approvals (20 min): what to read and the reference to judge it by
| Decision | File to review | Reference |
|---|---|---|
| External label mappings (VinDr test, PadChest-GR) | Review sheet https://claude.ai/artifact/C1mcphtwGXA1MPJjVwavV9 (or `kg/external_maps/*.yaml`) | Label counts in `DATA.md` (end) |
| Missing view on external images | `configs/pipeline.yaml` → `external.view_default` (PA assumed) | `EVAL_PLAN.md` "External sets" |
| Retrained Stage 4 models for external runs | RESULTS.md "External-run prep": 4 band changes in 24,262, max prob diff 0.0025 | (recommend accept) |
| Test evaluation plan, H3 direction, primary labels (596 radiologist-labelled studies) | `EVAL_PLAN.md` | `PIPELINE_BRIEF.md` build-stage table |
| Go-ahead for the MIMIC test run; permission to RadGraph-parse the test reports | `EVAL_PLAN.md` | |
| Report layout and pertinent negatives (RadReport proposals) | `kg/proposals/radreport_comparison.md` | `kg/sources/Rad Chest 2 Views.html` |
| R1 off / R2 on | `configs/pipeline.yaml` `rules`; numbers in RESULTS.md "Final head = C+A+R" | |
| Knowledge base content marked "needs review" | `KG.md` (what each table is, where it comes from, what is unreviewed); then `kg/findings.yaml` (is_a, may_occur_in, pertinent negatives, report phrases, glossary definitions) and `kg/radlex_map.yaml` | `kg/sources/Fleischner Society Glossary of Terms for Thoracic Imaging.pdf`; RadLex IDs; `RESULTS.md` "Knowledge-base tasks 1–6" |
| Commit the restructured repo before the test run (clean commit in the manifest) | none | `EVAL_PLAN.md` "What is frozen" |

Reply with your decisions in any form, for example "1 yes, 2 PA, …". Claude applies them and updates QUEUE.md and STATE.md.
