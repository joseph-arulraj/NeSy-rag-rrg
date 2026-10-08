# Queue — as of 2026-10-07 14:50

**Running: nothing.** Lane-close chain finished 12:38 (`runs/20261007-1225_lane-close`); results in RESULTS.md.

**Failed: nothing that matters.** pipeline-val-v2 shows STATUS=FAILED only from a logging bug after all outputs were written (fixed). The strict equivalence check of the external path with the *retrained* Stage 4 models did not pass (4 band changes of 24,262 finding-bands, max probability diff 0.0025), because retraining is not bit-identical (region-score max diff 0.0078). With the cached scores the path is exact.

## Waiting on the user: new today (full list below)
- Accept the retrained Stage 4 models for external runs (tiny drift above) and set `inputs.s4_models_run: runs/20261007-121534_s4-saved`? (Claude recommends yes.)

## Finished (all results in RESULTS.md; paths relative to the repo root)
| Lane | Orchestrator dir | What |
|---|---|---|
| A1 | `runs/20261007-0635_lane-a1` | head variants with/without CheXmask, regions, retrieval k=5/10/25; bootstraps |
| A2 | `runs/20261007-0645_lane-a2` | decision step, head with retrieval, out-of-fold check, rules, association links |
| A3 | `runs/20261007-0817_lane-a3-h100` | MLP heads + case evidence on the head with retrieval |
| A4 | `runs/20261007-0753_lane-a4-l40` | C/A/R input-group table |
| A5 | `runs/20261007-0815_lane-a5` | retrieval re-check vs C+A+R (led to dropping retrieval) |
| B | `runs/20261007-0700_lane-b` | Stage 8 on the head with retrieval (kept, labelled) |
| C | `runs/20261007-0721_lane-c2`, `runs/20261007-0817_lane-c-post` | DenseNet-121 (inner-holdout epoch), bootstraps |
| R | `runs/20261007-0840_lane-r` | report points 7–8 for the head with retrieval |
| **final** | `runs/20261007-0925_lane-final` | **C+A+R final head**: rules, case evidence, view metrics, violations, DenseNet and MLP bootstraps, Stage 8; results appended to RESULTS.md by `results_final.py` |
| — | `runs/20261007-081007_densenet-external` | DenseNet predictions on VinDr test (3,000) and PadChest-GR (4,555); no labels read, no metrics |
| KG 1–6 | `runs/*_build-radlex-map`, `*_compare-hierarchy`, `20261007-1056_add-sources` | RadLex map, hierarchy comparison, glossary, RadReport proposal, sources; `KG.md` |

## Queued: nothing beyond the chain.

## Waiting on the user (reading order and references: `REVIEW_GUIDE.md`; sample traces: `outputs/trace_samples_val.md`)
- Approve or change the RadReport proposals (`kg/proposals/radreport_comparison.md`).
- R1/R2: on the C+A+R head the CI rule favours R1 (+0.008 [+0.003, +0.013]) and not R2 (+0.009 [−0.002, +0.021]); the user's decision (R2 on, R1 off) is applied. Revisit?
- External tests (VinDr-CXR test, PadChest-GR): on hold by user decision. Needs: mapping decisions (review sheet https://claude.ai/artifact/C1mcphtwGXA1MPJjVwavV9, label counts in DATA.md), a decision on `external.view_default` (no view field for any external image; PA assumed), and the go-ahead. **Prep already done** (see STATE.md "External-run prep").
- MIMIC test run: needs the go-ahead (`--allow-test`); plan in `EVAL_PLAN.md`. Report-level scoring on test also needs permission to RadGraph-parse the MIMIC test reports.
- H3 direction: confirm "head non-inferior to the probe if the lower 95 % bound of (head − probe) mean AUROC > −0.02" (EVAL_PLAN.md).
- Radiologist review of everything marked "needs review" (see `KG.md`).
- 2026-10-07 12:35: pipeline-val-v2 STATUS=FAILED only at the final metrics line (duplicate keyword 'n' in run.metric); every output (predictions, graphs, stage8.jsonl, reports, summary.json) was written before it. Fixed in run_pipeline.py.
