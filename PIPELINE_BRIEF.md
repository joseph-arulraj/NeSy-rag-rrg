# Brief: rebuild the chest X-ray neurosymbolic report pipeline

## Changes since this brief (2026-10-07; details and reasons in `STATE.md`, numbers in `RESULTS.md`)

The brief below is the original plan. Where it conflicts with this list, this list holds.

1. **Grounding:** zone and side, never lobe (rules G2/P8: "... in the right lower zone").
2. **Fused head = C+A+R:** 77 data-driven concepts + 17 CheXmask anatomy features + Stage 4 region scores (CheXmask regions). Retrieval (D) is not a head input; it is used only as case-based evidence (k = 5, context only). Stage-two `associated-with` inputs are dropped (no significant gain).
3. **Decisions:** bands v2 = present / possible / absent / silent (Platt on calib, thresholds on thresh). The normal call is made when the root `any_abnormality` is in its absent band (2 % miss), not from every pathology being absent.
4. **Preprocessing:** letterbox (the CLEAR authors' aspect-preserving resize + zero padding), not stretch.
5. **Data roles:** the VinDr-CXR train detector (second A2) was not trained. The data has since been explored; `DATA.md` is the reference, not the section "The data on disk is unexplored".
6. **Compute:** four allocations allowed on 2026-10-07 (user). Steps launch via `scripts/launch.sh` / `orch_lib.sh` with a `run.sh` per run dir, not `bash -c`. Free space is about 42 GB (not ~100 GB).
7. **Labels:** still the 14 CheXpert labels (13 findings + root); the 19-finding extension has not been done.

## Fill in before starting (user)

- Repo root: `/scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg`
- Data root: `/scratch/prj/bhi_zihe_imaging/mimic_cxr_full` (MIMIC-CXR-JPG, VinDr-CXR, PadChest-GR, CheXmask, Chest ImaGenome, MS-CXR)
- Old pipeline outputs: `/scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/outputs`

## What this project is

The goal is a pipeline that reads a chest X-ray study and writes a Findings and Impression report in which every statement traces back to checkable evidence. It is meant to be neurosymbolic: neural models perceive, an explicit knowledge graph constrains and explains the predictions, and a language model only phrases what the graph already contains.

A first version exists in this repo. It used MIMIC-CXR-JPG only, with three evidence sources built on the frozen CLEAR model (DINOv2 ViT-B/14 image encoder, 368,294 concept sentences):

- B: concept similarity scores grouped per finding.
- C: a linear concept bottleneck head on 67 concepts, predicting 14 CheXpert labels.
- D: 5 nearest training images, with their reports parsed by RadGraph.

These were calibrated separately, summed in log-odds, cleaned by rules, and phrased by an LLM. Its output is poor, and we are replacing the design. The old code is still useful for data loading, CLEAR inference, RadGraph parsing and the FAISS index, so reuse what works.

## Known defects in the old pipeline

Verify each of these in the code and outputs during the audit. They were diagnosed from one full trace, not from reading the code.

1. Calibration labels treat "not mentioned in the report" as unlabeled. Calibrators are therefore fitted only on studies that explicitly discuss a finding, which inflates every probability.
2. Isotonic calibration on about 1,733 studies. Calibrated values are exact small fractions (5/7, 1/8, 15/17), so each rests on a handful of cases.
3. Additive log-odds fusion of B, C and D. All three derive from the same CLEAR embedding, so evidence is double counted and D only ever pushes confidence up.
4. The reject band is effectively empty. Findings at 0.09 are reported as "possible", and reports contain no negatives.
5. `no_finding` scored 0.995 alongside lung lesion at 0.96. The outputs are not coherent probabilities.
6. One node per (finding, laterality) with a shared prior. This produces duplicates such as "bilateral and unspecified atelectasis", and laterality carries no real information.
7. B matches whole sentences from other patients' reports, including negated and temporal ones ("not clearly depicted", "worsened").
8. C's 67 "concepts" are arbitrary report sentences. One sentence about a calcified nodule is a top contributor to eight labels, so the explanations are not faithful.
9. CLEAR was pretrained on all of MIMIC-CXR, including the official validation and test splits, plus CheXpert-Plus and ReXGradient. Any MIMIC evaluation is on images the backbone has seen.

## Reuse from the old pipeline

Reuse cached artifacts wherever they are valid, so time is not spent recomputing them. CLEAR is frozen, so its output for an image is fixed for a given checkpoint and preprocessing.

Likely reusable, after the checks below:

- CLEAR image embeddings for MIMIC images. Concept scores can be recomputed from embeddings and the concept text embeddings in seconds, so stored embeddings are enough.
- The FAISS index, if it was built from those same embeddings.
- RadGraph parses of MIMIC reports. Parsing is slow and the parses themselves were not the problem.
- The RadGraph tags on the 368k concept sentences and the synonym table.
- Data loaders and CLEAR inference code.

Not reusable, because they carry the defects: the calibrators, thresholds, the 67-concept head, the fusion code, the labels used for calibration, and the old splits.

Before trusting a cached artifact:

- Confirm which CLEAR checkpoint and image preprocessing produced it, and that it is keyed by a stable ID such as `dicom_id`.
- Recompute about 100 random images and compare. Embeddings should match to cosine similarity above 0.999. If they do not, find out why before using either version.
- Work out coverage. The old run covered about 213k training images and 1,733 validation studies. Compute only what is missing (other frontal images, the test split), not everything again.
- For the FAISS index, check that it contains only training-split patients under the new frozen splits. If the splits changed, rebuild it from the cached embeddings, which is quick.

List what was found, what passed and what must be recomputed in `AUDIT.md`.

## Target design

Inference for one study:

1. **Perception** (neural, per image).
   - A1: left lung, right lung and heart masks.
   - A2: findings localised to anatomical regions.
   - CLEAR: embedding and concept scores.
   - D: k nearest training studies as pre-parsed facts, summarised as the fraction of neighbours with each finding.
2. **Grounding** (symbolic). Map each localised finding to an anatomy ID, resolve lobe, lung and side through `part-of`, apply the image-to-patient flip, check `may-occur-in` and set a validity flag without deleting anything, and compute the cardiothoracic ratio (flag AP views as less reliable).
3. **Feature assembly.** One vector of named, human-readable features per finding at study level, including an explicit "A2 does not cover this finding" indicator. Missing must never be encoded as zero.
4. **Fused head** (learned, linear, named features only, so it remains a concept bottleneck).
   - Stage one outputs conditionals: P(parent), P(child | parent) for each `is-a` edge, and P(side | finding) over left, right, bilateral.
   - Stage two adds the stage-one logits of findings linked by `associated-with` edges. The knowledge graph decides which links exist, and training decides their weights.
5. **Calibrate each conditional, then multiply down the hierarchy.** Calibrating marginals directly can push a child above its parent.
6. **Decisions.** Per-finding thresholds give present, absent, uncertain or abstain. A parent's band is raised to its most positive child. `no_finding` is derived: true only when every pathology is absent. It is never predicted.
7. **Belief graph.** One node per finding (never one per laterality) with probability, band, side, anatomy, and per-feature-group contributions (weight times value). Edges come from the knowledge graph. Every change writes an audit entry naming the rule or edge.
8. **Presentation rules** (text only). Omit a parent when a specific child is present, state only pertinent negatives, withhold abstained findings.
9. **Phrasing and guard.** Start with a template. Add the LLM later, with a round-trip check: parse the generated text with RadGraph, compare to the graph, regenerate on mismatch, and fall back to the template after a fixed number of failures. The pipeline must always return a report.

The knowledge graph lives in reviewable YAML (`is_a`, `may_occur_in`, `associated_with`, definitions, pertinent negatives), each entry with an ID and rationale, validated against a schema on load. `part_of` is extracted from RadLex. Interpret the YAML with plain Python. Do not add a triple store, Prolog or Logic Tensor Networks: for a graph this size they add dependencies without changing what is computed, and an LTN would turn the hard hierarchy guarantee into a soft one.

## Data and roles

| Dataset | Seen by CLEAR | Role |
|---|---|---|
| MIMIC-CXR-JPG | Yes | Head training, retrieval index, in-domain calibration and test (always reported as "seen by backbone") |
| CheXmask | n/a | A1 masks, side, cardiothoracic ratio. Filter on its per-mask quality index. |
| Chest ImaGenome | Yes (MIMIC) | First A2: per-region finding classifier. Side and location labels. |
| VinDr-CXR train | No | Second A2 (box detector), only if the first localises poorly |
| VinDr-CXR test | No | External test only |
| PadChest-GR | No | External test only |
| MS-CXR | Yes (MIMIC) | Localisation check for A2 on MIMIC images |

No other data can be downloaded. Scratch quota is 1000 GB with roughly 900 GB used.

## The data on disk is unexplored

Only MIMIC-CXR-JPG has been used so far. VinDr-CXR, PadChest-GR, CheXmask, Chest ImaGenome and MS-CXR were downloaded and never opened. Nothing has been extracted, checked, cleaned or split. Treat every statement about their layout in this brief as an expectation to verify, not a fact.

For each dataset, find out the real structure yourself:

- Walk the folders, read any README, licence and checksum files, and open a few sample files of each type.
- Where the layout is unclear, read the dataset's official page or repository (PhysioNet project pages, the CheXmask and Chest ImaGenome GitHub repos, the BIMCV PadChest-GR page) if this machine has internet access. If it does not, say so and work from the files.
- Verify downloads are complete: run checksums where a checksum file is provided, and compare file and row counts against the published numbers.
- If something is still archived, check the compressed and expected extracted sizes against free space before extracting. Extract only what a stage needs.

Record what you find in `DATA.md`: layout, file formats, identifier columns, how each dataset joins to the others, counts found versus counts published, and any problems.

What I expect, unverified:

- **MIMIC-CXR-JPG:** JPG images plus CSVs for metadata (including view position), the official split, and CheXpert and NegBio labels.
- **CheXmask:** one CSV per source dataset with run-length-encoded lung and heart masks and a quality score per image, keyed by the source image ID (`dicom_id` for MIMIC). Decode on the fly and cache only derived features, not masks.
- **Chest ImaGenome:** per-image scene graph JSON files in a silver set, plus a separate gold set for 500 patients. Keyed by MIMIC `dicom_id`.
- **MS-CXR:** one COCO-format JSON of boxes and phrases. Images come from MIMIC-CXR-JPG.
- **VinDr-CXR:** DICOM images with CSVs for boxes and image-level labels. Training images have three radiologists' annotations each, which need merging. The test set has one consensus annotation.
- **PadChest-GR:** images plus grounded reports with English and Spanish finding sentences, boxes, and labels for finding type and location. Layout unknown.

Preprocessing produces one manifest table per dataset (Parquet): image path, patient ID, study ID, view, split, labels, and annotation references. All later code reads manifests, never raw folder layouts. If converting VinDr DICOMs to downsized PNG would free meaningful space, propose it with numbers, and do not delete originals without confirmation.

## Compute

The scheduler is SLURM. Claude Code runs on the login node. Check whether compute nodes have internet access before assuming anything can be downloaded or any API reached from inside a job.

GPU work runs inside long-lived allocations that are held open and reused, so a failed run never means queueing again. The user's exact commands:

```bash
# H100 (preferred, aim for two)
sbatch --partition=slam_gpu --gres=gpu:1 --cpus-per-task=16 --mem=512G --constraint=h100 --signal=USR2 --time=2-00:00:00 --wrap="sleep infinity"

# A100 (one, possibly two)
sbatch --partition=biomed_a100_gpu --gres=gpu:1 --cpus-per-task=16 --mem=512G --constraint=a100 --signal=USR2 --time=2-00:00:00 --wrap="sleep infinity"
```

How to use them:

- Check `squeue -u $USER` first and reuse any allocation that is already running.
- Aim for two running allocations, which the user is confident of getting. A third is optional: submit it, and if it is still pending 10 minutes later, cancel that one job and continue with two. If the H100 requests do not start, fall back to A100.
- Never hold more than three allocations. Do not cancel a running allocation without asking, apart from the surplus third.
- Run work as steps inside an allocation, for example `srun --jobid=<id> --overlap <command>`. Confirm the exact form that works on this cluster once and note it in `COMPUTE.md` along with job IDs, nodes and expiry times.
- A step's stdout and stderr come back to the `srun` process on the login node, not to the allocation's `slurm-<id>.out` file, and the step dies if that `srun` process dies. So launch every long step detached, with output redirected inside the step to the run directory, and use unbuffered Python so the log updates live:

  ```bash
  RUN=runs/20261006-0900_clear-embed; mkdir -p $RUN
  nohup srun --jobid=<id> --overlap bash -c "python -u train.py --run-dir $RUN > $RUN/log.txt 2>&1" > $RUN/srun.out 2>&1 &
  ```

  Test this once with a ten-second script before relying on it. Check running steps with `squeue -s -j <id>` and follow one with `tail -f $RUN/log.txt`. A failed step leaves the allocation running, so fix and relaunch in the same allocation.
- Allocations last two days. Every long run must checkpoint regularly and be resumable, because a step can be killed at the limit. Do not rely on the USR2 signal alone. When an allocation expires, resubmit with the same command.
- Use the allocations in parallel for independent work: one extracting CLEAR embeddings while another runs preprocessing or a different ablation. Each has 16 CPUs and 512 GB of RAM, so cached features fit in memory.
- Log every run to a file under the new project directory, so a failure can be diagnosed without rerunning.

## Rules that must hold throughout

- **Splits are patient-level and frozen.** Write one split file early and treat it as the single source of truth. Start from the official MIMIC split.
- **Exclusions from all MIMIC training splits:** every patient with an MS-CXR image, and every patient in the Chest ImaGenome gold-standard set. Keep both for evaluation.
- **External test sets are not touched** for training, calibration, threshold fitting or model selection.
- **Labels:** a finding not mentioned in the report counts as negative. Start with the 14 CheXpert labels already shipped with MIMIC-CXR-JPG, and extend to the 19-finding table only after the baseline works. Default for uncertain labels: mask them out of the loss. Report this choice so it can be revisited.
- **Calibration:** Platt or beta scaling, not isotonic. Calibrators, thresholds and test metrics each use a different split.
- **Thresholds:** set by a target operating point per finding. The absent band must be reachable, and a check should fail if any finding has an empty band on validation data.
- **Retrieval features for training studies** must exclude the study itself and all other studies from the same patient.
- **Concepts:** select on the training split only. Drop temporal and comparison sentences. Prefer short, generic observations over patient-specific sentences.
- **Storage:** never copy a dataset. Check free space before writing anything large. Store embeddings as float16 memory-mapped arrays.
- **Old work:** build in a new directory. Do not delete or overwrite the old pipeline, its outputs or any data.

## How to proceed

### Phase 0: audit (read-only, then stop and report)

Read the code and the existing outputs and write `AUDIT.md` covering:

- What each stage of the old pipeline does and which files implement it.
- Which of the nine known defects you can confirm, with file and line, and any others you find.
- Which components are reusable as they are.
- Which MIMIC splits the old 213k training images and 1,733 validation studies came from.
- The data inventory described under "The data on disk is unexplored", written to `DATA.md`, with free space.
- Compute: which allocations are running, the step command that works, and whether compute nodes have internet, written to `COMPUTE.md`.

The audit reads and documents. It does not extract, convert, move or delete anything. Stop after it and wait for confirmation before building.

### Build stages

Each stage is kept only if it improves a named metric over the previous stage on MIMIC validation data. Record every result in `RESULTS.md`, including negative ones.

| Stage | Build | Must show |
|---|---|---|
| 0 | Preprocessing: manifests for every dataset, frozen patient-level splits with the exclusions applied, cached CLEAR embeddings and concept scores for MIMIC | Counts match published numbers, no patient appears in two splits, exclusions verified by test |
| 1 | Fixed labels, linear probe on CLEAR embeddings, Platt calibration, thresholds, template report | Per-finding AUROC, AUPRC and calibration error. This is the baseline everything else must beat. |
| 2 | CheXmask side and cardiothoracic ratio features | Better cardiomegaly discrimination, usable side signal |
| 3 | Factorised head: hierarchy and side conditionals, named features only | Zero hierarchy violations, calibration no worse than stage 1, accuracy cost versus the embedding probe reported |
| 4 | Chest ImaGenome region classifier as A2, checked on MS-CXR | Localisation accuracy on MS-CXR, gain on lateralised findings |
| 5 | Retrieval features and stage-two `associated-with` inputs | Ablation gain for each, separately |
| 6 | Knowledge graph YAML, grounding, decisions, belief graph, audit trail | Schema validation, one unit test per rule, coherent bands |
| 7 | Presentation rules, template, then LLM with RadGraph round-trip guard | Report-to-graph agreement rate, fallback rate |

### Running stages in parallel

The stage numbers give the order of dependence, not a strict sequence. Only two things are GPU-heavy: CLEAR embedding extraction and training the A2 models. The heads are linear models on cached features and train in minutes. So use the allocations as lanes:

| Lane | Work |
|---|---|
| First, on CPU | Manifests and frozen splits. Everything else depends on these, so nothing trains until they are written and tested. |
| GPU lane 1 | CLEAR embeddings for any MIMIC images not already cached. Then the quick head experiments: stages 1, 2, 3 and 5. |
| GPU lane 2 | The Chest ImaGenome region classifier (stage 4). It needs only the manifests and splits, so start it as soon as they are frozen, alongside lane 1. |
| GPU lane 3, only if a third allocation is running | The VinDr-CXR detector, trained speculatively so both A2 options can be compared on MS-CXR. This overrides "only if the first localises poorly", because the GPU would otherwise sit idle. Also cache CLEAR embeddings for VinDr-CXR test and PadChest-GR here. Caching is allowed. Evaluating on them is not, until the end. |
| Spare CPUs in any allocation | CheXmask decoding into side and cardiothoracic-ratio features, Chest ImaGenome parsing, VinDr DICOM conversion. |

For the region classifier, try the cheap version first: take frozen CLEAR patch features, average them inside each of the 29 Chest ImaGenome region boxes, and train a linear classifier per finding on the pooled vectors. This needs one GPU pass over the MIMIC frontal images and no backbone training. Pool at extraction time and store only the 29 region vectors per image. Do not store raw patch tokens, which would not fit in the quota. Check first that the CLEAR code exposes patch tokens. Fine-tune a backbone only if this version localises poorly on MS-CXR.

With two allocations, skip lane 3 and train the VinDr detector later only if the region classifier localises poorly.

Stages 6 and 7 are mostly code, not compute. Write and unit-test them on a small sample while the GPU lanes run.

Keep one lane per allocation and record which run is on which job ID in `COMPUTE.md`. Do not start two GPU runs on the same allocation unless memory use has been measured and both fit.

After stage 7, run the external tests once on VinDr-CXR test and PadChest-GR. Both need a class mapping table to our findings. Flag that table for radiologist review.

## What to report back after each stage

- What was built and where.
- The numbers against the stage 1 baseline, per finding.
- What you verified by running it, and what you are assuming.
- Anything surprising, and any decision you made that the user might want to reverse.

## Ask before

- Deleting or moving any file you did not create.
- Changing the frozen splits.
- Using an external test set for anything.
- Writing more than about 20 GB, or extracting any archive that would leave less than 30 GB free.
- Holding a fourth allocation, or cancelling a running one.

Submitting the allocations above and running steps inside them does not need confirmation.