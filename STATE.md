# STATE (2026-10-07 14:50) — what a fresh reader needs that is not in the other files

Read with: `CLAUDE.md` (rules), `PIPELINE_BRIEF.md` (plan), `QUEUE.md` (what runs / waits), `RESULTS.md` (all
numbers), `DATA.md` (data), `COMPUTE.md` (allocations), `v2/KG.md` (knowledge base). All new work is in `v2/`.

## Decisions so far, with reasons
- **Letterbox, not stretch** (user, 2026-10-06): CLEAR's own data preprocessing keeps the aspect ratio and zero-pads (see DATA.md "CLEAR preprocessing"); letterbox beat stretch on zero-shot (+0.0088 [0.0015, 0.0158]) and on Stages 2–3 (+0.006, CI excludes 0); the stretch distorted the typical image by a factor of 1.20. Stretch region features deleted (user confirmed); stretch predictions and models archived in `v2/archive/stretch_v1/`.
- **One frontal image per study** (the image the CLEAR embeddings use); laterals never used. 218,139 studies; frozen patient-level splits `v2/data/splits/mimic_splits_v1.parquet` (train/fit, heldout_loc, calib, thresh, val, test). Inner holdout = 5% of fit patients by hash (`probe.inner_holdout`): chooses C for linear heads and the epoch for DenseNet/MLP.
- **Final head = C+A+R** (user, 2026-10-07 ~09:20): 77 data-driven concepts (`v2/data/concepts/concepts_final_k6_pruned.csv`) + 17 CheXmask anatomy features + Stage 4 region scores on the CheXmask-derived regions (set B), in the factorised Stage 3 head (P(child | parent) multiplied down is_a; 0 hierarchy violations by construction). Run `v2/runs/20261007-065311_a1-s3-s4b`. Platt on calib, four-band thresholds on thresh, metrics on val.
- **Retrieval only as case-based evidence**: with region features present it added +0.0018 [−0.0006, +0.0043] (n.s.). Case evidence (k = 5 similar fit-split studies, IDs and labels only, own patient excluded) is attached to the belief graph as context; never changes probabilities or bands; report line behind `v2/configs/report.yaml: show_case_evidence` (off); never sent to the LLM.
- **Two decision rules** (user): (1) accuracy-only components (retrieval, association links) are kept only if the 95% patient-bootstrap CI of the gain excludes 0; (2) CheXmask anatomy (A) and region features (R) follow a no-harm rule (dropped only if removing them is significantly better), because they also serve measurement, side and location. Knowledge rules R1/R2 were judged with the CI rule, then set by user decision.
- **Kept / dropped**: kept A, R, R2 (side agreement: state a side only if the side head and region-derived side agree); dropped retrieval from the head, association links (item 7, −0.0002 n.s.), R1 (localisation support; user decision, although on C+A+R its CI excludes 0 — flagged), MLP head (no gain), critical lists A/B (negative result 2026-10-06).
- **Bands v2**: present p ≥ 0.70; possible 0.40 ≤ p < 0.70; absent = threshold missing ≤ 5% of positives on thresh (pneumothorax and the root 2%); absent takes priority over possible; otherwise silent. Root `any_abnormality` is non-reportable and only gets an absent tier; D2 raises a parent only from stated (present/possible) children.
- **Normal call**: "No acute cardiopulmonary abnormality." iff the root is in its absent band (2% miss). Critical lists inactive. When the root is not absent and nothing is stated, the impression is "No finding meets the reporting threshold." (changed 2026-10-07: the old wording read like a normal call).
- **Report wording**: zone and side, never lobe (G2/P8; zone only for parenchymal findings with a stated side: "… in the right lower zone"); side stated only for lateralisable findings with side confidence ≥ 0.8 (P6) and R2 agreement; a side is never attached to a KG phrase without a {side} slot.
- **Stage 8 (LLM)**: KCL endpoint `arc:nano` (Qwen 27B) at `https://ai.create.kcl.ac.uk/api/v1/chat/completions`; key in `~/.rrg_llm_key` (chmod 600, never logged). Prompt = only the template's statements (no IDs, no report text, no case evidence). Generator constraints: one statement per sentence, possible statements start with "Possible", devices as "Support devices are in place.". Guard: RadGraph (modern-radgraph-xl) parse WITHOUT section headers, compare (`nesy/rg_match.py`), up to 3 attempts, then template fallback; 2 concurrent calls (4 hit HTTP 429).
- **External tests**: not yet (user). Only caching and DenseNet predictions done. External test labels read once for counts only (user request 2026-10-07, mapping review; DATA.md); no metrics.
- **Task 3**: hand-written may_occur_in kept, marked needs review; no empirical table.
- **Task 7** (2026-10-07): glossary-derived concepts (31 of 52 phrases pass 0.60) give the same head accuracy as the 77 (macro AUROC −0.0004 [−0.0037, +0.0029]); ablation only, 77 stay main. `stage3_factorised.py --concept-emb` selects the text-embedding file.

## Working conventions
- Python: `/scratch/users/k23031260/.conda/envs/rrg/bin/python` always (also inside srun).
- Launch a step: `bash v2/scripts/launch.sh <jobid> <name> <script.py> [args]` (writes `runs/<ts>_<name>/run.sh`, starts `setsid nohup srun --jobid=<id> --overlap run.sh` detached; prints the run dir). Never nested `bash -c` (held for approval).
- Orchestrators: `v2/scripts/lane_*.sh` sourcing `v2/scripts/orch_lib.sh` (`step NAME JOB [ENV=val…] -- script.py args`, foreground srun per step, `orchestrator.log` START/END lines, `last_<step>` files, STATUS DONE at the end). Start with `setsid nohup bash scripts/lane_x.sh <orch_dir> … > <orch_dir>/orch.out 2>&1 < /dev/null &` from `v2/`. Never edit a lane script while bash is running it; copy it under a new name instead.
- Run folders: `v2/runs/<YYYYMMDD-HHMMSS>_<name>/` with config.json, log.txt, metrics.jsonl, STATUS. Check STATUS + last log lines only.
- Env switches: `NESY_EMB=letterbox` (else stretch); `NESY_S4_RUN=runs/20261006-224243_lb-stage4` and `NESY_S4_SETS=B` for region features.
- Allocations (3 held): H100 512G 37831235 (comp231, to 2026-10-09 11:36); L40S 37831634 (to 2026-10-09 ~06:06); CPU 37811782 (to 2026-10-08 ~09:40). See COMPUTE.md.
- Free space ~42 GB; ask before writing > 20 GB.


## End-to-end runner and test readiness (2026-10-07 midday)
- **Runner:** `v2/scripts/run_pipeline.py` + `v2/configs/pipeline.yaml` (every undecided choice is a switch: R1/R2, report layout, show_case_evidence, Stage 8 comparator / retry / attempts, external.view_default). Each run writes `manifest.json` (SHA-256 of 52+ files: head, calibrators, thresholds, region scores, KG, configs, code). Refuses `--split test` without `--allow-test` and external sets without `--allow-external`; never loads the label file. Equivalent to the existing C+A+R results on all 1,733 val studies (`runs/20261007-114734_equivalence`).
- **Evaluation:** `v2/scripts/evaluate.py` (refuses test without `--allow-test`; reads only the labels of the run's studies). `EVAL_PLAN.md` (repo root) fixes what is computed on test: primary labels = the 596 radiologist-labelled test studies with a frontal image (of 687; 9 are val, 82 have no frontal). `nesy/labels.py` applies the training label rules to them.
- **Stage 8:** comparator v2 (`rg_match.claims(version=2)`) and feedback v2 (`nesy/stage8.py`). Diagnosis: the old 0 rescues were comparator false alarms (RadGraph drops short sentences; "in place" unlinked; "cardiomediastinum"; list-sentence side spill), not a broken retry: the retry repairs 150/150 genuine errors. v2 keeps guard sensitivity at 100 % (dropped / negated / added side).
- **Stage 8 attempts stay at 3** (decided 14:50 by the user's rule): with comparator v2, 4 of 1,733 first attempts failed and retries rescued all 4; fallback 0. Retries cost calls only when needed.
- **Report-level on val (first numbers, no baseline yet):** positive-mention micro F1 0.563, macro F1 0.366, RadGraph entity F1 0.165 (`runs/20261007-123526_eval-val-v2`).
- **H3 comparator:** `runs/20261007-114130_h3-probe-ear` (linear probe on CLEAR embedding + A + R): val macro AUROC 0.8354 vs head 0.8344, n.s.
- **External-run prep (done 2026-10-07, user request; so a later "run the full test" does NOT need it again):**
  - CheXmask anatomy features for VinDr test (3,000/3,000) and PadChest-GR (4,310/4,555; 245 missing → indicator): `v2/data/features/chexmask_{vindr_test,padchest_gr}.parquet`.
  - Stage 4 region classifiers retrained with saved models (`runs/20261007-1215_lane-extprep` → its `last_s4-saved` run, `models.joblib`), so region scores can be computed for external images.
  - External feature path `v2/nesy/ext_features.py` + `run_pipeline.py --dataset vindr_test|padchest_gr --allow-external` (or `--feature-path external` on MIMIC for checks). Checked on MIMIC val by the lane-close chain (results in RESULTS.md).
  - Models check (`runs/20261007-122720_eq-extpath-models`): retraining is not bit-identical; 4 band changes in 24,262, max prob diff 0.0025; awaiting user acceptance before setting `inputs.s4_models_run: runs/20261007-121534_s4-saved`.
  - Still needed before external runs: that acceptance, the mapping decisions, `external.view_default` decision, the go-ahead. Case evidence is MIMIC-only (external graphs get none).
- **To run the MIMIC test later:** `run_pipeline.py --split test --allow-test` on the H100 (Stage 8 ~55 min for 3,041 studies), export the H3 probe's test predictions (`export_predictions.py --kind probe --src runs/20261007-114130_h3-probe-ear`), then `evaluate.py --split test --allow-test --labels radiologist` (primary) and `--labels chexpert` (secondary), as in EVAL_PLAN.md.

## Pitfalls found and their fixes
- cephfs `open_memmap(w+)` zeroes header pages → write sequential shards or use `nesy/safe_memmap.py`; compressed npz chunks for weights.
- `pkill -f <pattern>` kills the calling shell when the pattern is in its own command line → find PIDs with `ps … | grep "[x]…"` and kill those.
- The scratchpad `/tmp/claude-*` is on the login node only; compute steps cannot read it → put inputs in a run dir.
- Login node is slow for torch imports and kills big-memory jobs (OOM) → run checks as srun steps.
- `manifests.load()` joins a `split` column; select columns explicitly to avoid name clashes and to avoid reading external labels.
- RadGraph drops entities when "FINDINGS:" / "IMPRESSION:" headers are in the text; it misses hedges in list sentences and "in the lower zone" (use "in the right lower zone").
- KCL LLM endpoint rate-limits at 4 concurrent calls (HTTP 429); use 2.
- `to_markdown` needs tabulate (not installed) → use `to_string`.
- SLURM step creation can take 1–5 min ("step creation temporarily disabled"); time limits of a PENDING job can be raised with `scontrol update`.
- **Never create a module without checking the name is free**: the Write tool overwrote the existing `v2/nesy/external.py` (DICOM/PNG loaders) on 2026-10-07; restored from the session transcript, the new code went to `nesy/ext_features.py`. v2 is not in git, so there is no other copy.
- `pkill`/`kill` via `ps | grep` on a pattern that appears in the current command line kills the current shell (happened again 2026-10-07).
- Feature names / DataFrame merges: `D.study_table()` order is the cache and embedding order; assert it.

## Asked for but not done
- RadReport proposals: awaiting approval; not applied.
- External tests: on hold.
- Radiologist review of all "needs review" items.

## What was next
KG task 7 done (no difference vs the 77; 77 stay main). Proposed next (awaiting user): end-to-end runner built and checked on val; then, on user go-ahead, MIMIC test + external tests (needs: mapping tables accepted or reviewed, go on test sets, permission to RadGraph-parse MIMIC test references). Also pending: RadReport proposals, R1/R2.
