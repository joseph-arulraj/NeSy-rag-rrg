# NeSy-RAG-RRG: neurosymbolic chest X-ray report generation

This project reads one frontal chest X-ray and writes a FINDINGS + IMPRESSION report in which every statement traces back to checkable evidence:
- **neural models perceive**: a frozen CLEAR encoder, CheXmask anatomy and a region classifier;
- **an explicit knowledge graph constrains and explains** the predictions: the hierarchy, decision rules and report rules, with an audit trail;
- **a language model only phrases** what the graph already contains. A RadGraph round-trip check confirms it did, and falls back to the template if not.

The original design and plan are in `PIPELINE_BRIEF.md`. Every term, rule ID (D1, G6, R2, P1, …) and stage is explained in **`GLOSSARY.md`**.

## How a study becomes a report

```
frontal image
  ├─ CLEAR (DINOv2 ViT-B/14) embedding · 77 concept text embeddings → 77 concept scores           C
  ├─ CheXmask lung / heart masks → 17 anatomy features (CTR, lung areas, heart shift, view, …)   A
  └─ CLEAR patch tokens pooled in 9 CheXmask regions → Stage 4 region classifier → region scores R
        ▼
C+A+R head: linear concept bottleneck, factorised along the is_a hierarchy
            (P(child | parent) × P(parent): 0 hierarchy violations by construction), Platt-calibrated, + side head
        ▼
belief graph: bands present / possible / absent / silent (D1), side (G6, R2), zone (G7), parent raise (D2),
              normal call from the root (D3), case evidence as context (C1); every change in an audit trail
        ▼
template report (rules P1–P8) → LLM rewrite of the template's statements only → RadGraph check →
accept / retry (≤ 3) / template fallback
```

**Validate results** (MIMIC-CXR, 1,733 studies; seen by the backbone):
- **Head:** macro AUROC 0.834 over 14 outputs, ECE 0.024, 0 hierarchy violations.
- **Normal call:** 13.9% of studies, precision 0.94.
- **LLM report:** matches the graph on the first attempt 99.8% of the time; 0 fallbacks.

Every experiment, including the negative ones, is in `RESULTS.md`. The test sets have not been run yet; `EVAL_PLAN.md` fixes what will be computed.

## Repository layout

| Path | Contents |
|---|---|
| `nesy/` | Library: data and paths, knowledge graph loader, grounding, belief graph and rules, thresholds, report template, Stage 8 phrasing and RadGraph comparator, external-image features, label rules |
| `scripts/` | Pipeline steps and experiments, one per stage; `run_pipeline.py` (end-to-end runner), `evaluate.py`, lane orchestrators `lane_*.sh` + `orch_lib.sh`, `launch.sh` |
| `configs/` | `pipeline.yaml` (every runner switch: rules, report layout, Stage 8, external options), `report.yaml`, `llm.yaml` (endpoint; no key) |
| `kg/` | Knowledge base in reviewable YAML: `findings.yaml` (findings, is_a, may_occur_in, pertinent negatives, report phrases, definitions), `anatomy.yaml` (part_of), `radlex_map.yaml`, `devices.yaml`, `imagenome_map.yaml`, `external_maps/`, `proposals/`. Described in `KG.md` |
| `tests/` | Unit tests: one per rule, the label rules, the comparator, the runner and evaluation guards |
| `legacy/` | The previous pipeline (code, configs, logs, outputs). Kept for reference; not used by the current code |
| `data/`, `runs/`, `outputs/`, `archive/`, `models/`, `model_weights/` | Git-ignored. Derived data, run directories, review outputs, archived models, DenseNet weights, CLEAR weights. Contain or derive from credentialed MIMIC data |

Project records: `STATE.md` (decisions and conventions), `QUEUE.md` (what runs and what waits), `RESULTS.md`, `DATA.md`, `COMPUTE.md`, `AUDIT.md` (the old pipeline's defects), `EVAL_PLAN.md`, `REVIEW_GUIDE.md`, `GLOSSARY.md`.

## Setup

- **Python:** the conda env `rrg`, called by full path: `/scratch/users/k23031260/.conda/envs/rrg/bin/python`.
- **CLEAR:** cloned next to this repo and installed with `pip install -e ../CLEAR`.
- **Raw datasets:** read in place from `/scratch/prj/bhi_zihe_imaging/mimic_cxr_full` (`nesy/paths.py`) and never copied.
- **CLEAR weights and concept bank:** in `model_weights/`.
- **LLM key:** read at run time from `~/.rrg_llm_key` (chmod 600) and never logged or committed.
- **GPU work:** runs as steps inside held SLURM allocations (`COMPUTE.md`). `scripts/launch.sh <jobid> <name> <script.py> [args]` starts one logged run; `scripts/lane_*.sh` chain several.

## Running

```bash
PY=/scratch/users/k23031260/.conda/envs/rrg/bin/python
$PY -m pytest -q tests                                    # unit tests
$PY -u scripts/run_pipeline.py --split val                # end to end on validate (Stage 8 needs a GPU for RadGraph)
$PY -u scripts/run_pipeline.py --split val --no-stage8    # template reports only (CPU, ~1 min)
$PY -u scripts/evaluate.py --pipeline-run runs/<run> --split val --labels chexpert
$PY scripts/trace_samples.py --run runs/<run>            # readable traces -> outputs/trace_samples_val.md
```

Each run gets `runs/<timestamp>_<name>/` with `config.json` (all arguments, git commit, SLURM job, and for the runner a `manifest.json` of SHA-256 hashes of every model, threshold, KG and code file), `log.txt`, `metrics.jsonl` and `STATUS`.

**Guards:**
- The MIMIC test split needs `--allow-test`.
- The external sets (VinDr-CXR test, PadChest-GR) need `--allow-external`.
- Both are refused by default and run only with the project owner's approval.

## Data rules

- Patient-level splits are frozen in `data/splits/mimic_splits_v1.parquet`.
- Calibration, thresholds and metrics each use a different split.
- MS-CXR and Chest ImaGenome gold-standard patients never enter training.
- External test sets are never used for training, calibration, thresholds or model selection.
- MIMIC-CXR, and everything derived from it (predictions, graphs, generated reports, traces that quote reference reports), is covered by the PhysioNet credentialed-data agreement and must not be pushed to any remote. That is why those directories are git-ignored.
