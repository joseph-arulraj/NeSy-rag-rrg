# Audit of the old pipeline (Phase 0)

Written 2026-10-06. Read-only audit: nothing in the old pipeline, its outputs or the data was changed.
Repo root `/scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg` is the same directory as
`/cephfs/volumes/hpc_data_prj/bhi_zihe_imaging/91a35342-.../NeSy-rag-rrg` (symlinked path).

Legend: **V** = verified by running code or reading the file/output; **A** = assumed / inferred.

## 0. Housekeeping found on arrival

- Working tree already had `docs/architecture.md`, `docs/evidenceA.md`, `docs/pipeline.md` and `pipeline.pdf` deleted (not by me). They are still in git at `HEAD` (`git show HEAD:docs/pipeline.md`). `configs/hpc.yaml` has uncommitted real paths. **V**
- **The full MIMIC-CXR report text was not on disk** (resolved 2026-10-06 09:35: the user added `mimic_cxr_full/mimic-cxr-reports.zip`, 227,835 reports). `configs/hpc.yaml` points at `/scratch/prj/bhi_zihe_imaging/data/raw/mimic-cxr-reports.zip`, which does not exist. A search of the whole project space found only `mimic_cxr_sample/mimic-cxr-reports.zip` (100 reports). The old calibration and Evidence D could not be rerun as configured. **V**
  - Substitute on disk: Chest ImaGenome `silver_dataset/cxr-mimic-v2.0.0-processed-sentences_all.txt` holds sentence-split report text with a section tag (`history` / `finalreport`), keyed by `subject_id`, `rad_id` (= study_id). It covers **all 227,835 MIMIC studies** (2,340,797 sentences). **V**
  - CXRGraph `inference.json` holds model-predicted RadGraph-style entities/relations for all 227,835 MIMIC reports. **V** (README; file not opened)
  - The 1,757 cached RadGraph parses in `outputs/report_facts/` survive (1,733 validation studies plus neighbours).
- Python env: `rrg` (`/scratch/users/k23031260/.conda/envs/rrg`) has torch 2.14.0+cu130, sklearn 1.9.1, faiss 1.15.1, radgraph 0.1.18, CLEAR (`clear-med`, editable from `.../91a35342-.../CLEAR`), pandas 3.0.6, PIL, scipy. **Missing: pyarrow (Parquet), pydicom (VinDr DICOM), pycocotools, cv2.** **V**

## 1. What each stage of the old pipeline does

| Stage | Files | What it does |
|---|---|---|
| Data index | `src/rrg/datasets/mimic_cxr.py` | Reads official split + metadata, one frontal image per study (PA before AP, ties smallest dicom_id), drops lateral-only studies. Checks patient-disjointness of the official split. |
| Labels | `src/rrg/datasets/chexpert_labels.py` | Official CheXpert CSV → {finding: present/absent/uncertain}; blank = "unknown". |
| Reports | `src/rrg/datasets/reports.py` | Reads report .txt from a zip or a directory. |
| Image load | `src/rrg/ingest/image_loader.py`, `ingest/bulk_encode.py` | PIL decode with checks, DataLoader-parallel decode, CLEAR encode. |
| CLEAR | `src/rrg/perception/clear_encoder.py` | `clear.load_pretrained(best_model.pt)`, `encode_image`, L2-normalised fp32 768-d. Uses CLEAR's own `preprocess`. |
| Concept bank / B | `src/rrg/concepts/{bank,similarity,grouping}.py`, `offline/build_concept_bank.py`, `concepts/tagging/*` | 368,294 report sentences embedded by CLEAR's text tower; image·concept cosine; concepts RadGraph-tagged into (finding, anatomy, laterality, polarity, temporal) groups; per group max present / max absent score. |
| C (CBM) | `src/rrg/cbm/*`, `scripts/train_cbm.py`, `model_weights/cbm_concepts.md` | Linear layer on 67 concept scores → 14 CheXpert labels. |
| Retrieval / D | `src/rrg/offline/build_faiss_index.py`, `retrieval/faiss_index.py`, `retrieval/evidence_d.py`, `reports/report_facts.py` | Exact IndexFlatIP over train embeddings, same-patient exclusion, k=5 neighbours; neighbours' reports RadGraph-parsed into facts; similarity-weighted present/absent votes. |
| Calibration | `src/rrg/fusion/calibrate.py`, `scripts/run_calibration.py` | Isotonic per (source, finding) on validation; thresholds tau_hi/tau_lo at 0.9 precision. |
| Fusion | `src/rrg/fusion/fuse.py` | Log-odds: prior = calibrated C (or B), + 0.5·B delta + 1.0·D delta, sigmoid, band. |
| Belief graph | `src/rrg/fusion/graph.py` | One node per (finding, laterality); abstained → `rejected`. |
| Rules | `src/rrg/fusion/rules/rule_engine.py`, `configs/n25_rules.yaml` | Laterality attribute validity, no_finding mutual exclusivity, left/right margin rule. |
| Generation | `src/rrg/generation/{prompt,llm_client,report}.py` | Prompt → KCL hosted LLM (`arc:nano`). No guard. |
| End to end | `scripts/run_full_pipeline.py` | Ran on 5 studies of the **official test split** (`outputs/full_pipeline/test.jsonl`, all patient 10046166). |

## 2. The nine known defects

| # | Status | Evidence |
|---|---|---|
| 1 | **Confirmed V**, and worse than stated | Calibration: `fusion/calibrate.py:193-209` iterates only over `gt.items()`, the findings the report mentions, so unmentioned findings never become negatives. Result in `outputs/calibrators.pkl`: base rates atelectasis 0.939, no_finding 0.936, support_devices 0.956, lung_lesion 0.938, pleural_other 1.000 (38/38). **The C head has the same defect in training:** `cbm/train.py:168-182` masks blanks. `train_cbm.log`: atelectasis 43,179 pos / 1,420 neg, lung_opacity 48,097 / 2,693, **no_finding 71,202 / 0**. The no_finding output is the bias alone (its weight on the calcified-nodule concept is exactly 0.0), which explains defect 5. |
| 2 | **Confirmed V** | `calibrate.py:103-106` IsotonicRegression. Fitted step levels are small fractions: C cardiomegaly {0, 5/7, 8/9, 12/13, 1}, C atelectasis 15/17, C consolidation 1/8, B support 5/7. Report confidences equal these (cardiomegaly 0.7143 = 5/7 in 3 of 5 test studies). B lung_lesion has 2 steps from 48 samples. |
| 3 | **Confirmed V** | `fuse.py:200-229`: posterior = logit(C) + 0.5·(B_present − 0.3·B_absent) + 1.0·(D_present − D_absent). C's 67 inputs are a subset of B's scores, and D uses the same embedding. B's raw cosine (≈0.1–0.6) is added as a logit with a placeholder weight (`default.yaml:174-176`, "not yet fit"). Because absent support is discounted ×0.3 (`default.yaml:164`, `evidence_d.py:70`), D is mostly a positive push. In all 5 outputs, D only ever raised confidence. |
| 4 | **Confirmed V** | `outputs/thresholds.json`: tau_lo is −0.01 (unreachable) for 9 of 15 findings, and 0.0001 for 5. Cardiomegaly has tau_hi = tau_lo = 0.0001, so it is always "accept". Pneumonia at 0.0003 is banded "uncertain" and printed "possible". `fit_threshold` (`calibrate.py:110-129`) needs 0.9 NPV on validation data where negatives are rare (defect 1). It also uses the same validation studies as the calibrators (`calibrate.py:224-231`), and on the C/B posterior rather than the fused confidence that it is applied to. |
| 5 | **Confirmed V** | test.jsonl rule audit: no_finding 0.995 / 0.969 / 0.997 / 0.990 / 0.997 alongside lung_lesion 0.96, atelectasis 0.95. Cause: defect 1 in C training (no negatives for no_finding). |
| 6 | **Confirmed V** | `fuse.py:64-66` keys nodes by (finding, laterality). C has no laterality, so `fuse.py:155-159` adds an "unspecified" node as well. Output: "Bilateral and unspecified atelectasis", "fractures in the right and unspecified locations", support devices at left+right+midline+unspecified. All laterality variants share C's prior (`fuse.py:171-175`). |
| 7 | **Confirmed V** | Of the 104,271 concepts that enter a B group, 31,805 (30.5 %) are temporal (22,414 implies_present, 9,391 implies_absent). "Worsened" sentences count as present (`tagging/polarity.py:106`). Negation cues (`default.yaml:135`) lack bare "not", so 1,479 present-tagged grouped concepts contain "not", e.g. "basilar pneumothorax not clearly identified" → pneumothorax present, "small right lateral pneumothorax not apparent" → pneumothorax right present. |
| 8 | **Confirmed V** | `model_weights/cbm_concepts.md`: the 67 concepts are full report sentences, most of them temporal ("…has worsened", "…improved"). "Calcified nodule in the left upper lobe" (bank id 5939) is in the top 6 by \|weight\| for 9 of 14 labels. Its weight is negative for every label, including lung_lesion (−0.04). In the 5 test outputs it is a top-5 contributor for 10 labels. |
| 9 | **Confirmed by the brief, not re-verifiable from code (A)** | The CLEAR checkpoint `best_model.pt` has sha256 `da1617a7…6a03` (index manifest). The code does nothing to separate it from MIMIC val/test. |

### Other defects found

10. **Threshold, calibrator and evaluation share one split.** Calibrators and thresholds are both fit on the 1,733 official-validate studies (`calibrate.py:162, 224-231`). No held-out split was used for metrics. **V**
11. **Ground truth for calibration comes from RadGraph on the whole report**, including INDICATION/HISTORY and COMPARISON (`report_facts.py:77`, no section filter). A finding in the indication ("eval for pneumonia") can become a label. **V** (code), **A** (size of effect)
12. **Report-sentence negation is applied to every observation in the sentence.** "No consolidation, but a small effusion" makes both absent (`tagging/polarity.py:67`, a lexical cue wins over RadGraph certainty). **V**
13. **The old pipeline was run on the official test split** (`run_full_pipeline.py:61`, default `--split test`). The split has been looked at. Only 5 studies of one patient, but it should be recorded. **V**
14. **B and D enter fusion only if they are in B's top-20 groups or D's 5 neighbours.** A missing source is treated as zero contribution, not as missing. **V** (`fuse.py:202, 216`)
15. **Probabilities near 1 are reported for findings with no evidence at all:** pleural_other and scoliosis get 0.9999 from a base rate fit on 38/38 and 8/8 positives (`calibrate.py:100`), and are only kept out of reports by the tau_hi = 1.01 "abstain". **V**
16. **Uncertain CheXpert labels (−1) are masked in C training, as the brief's default says.** This is fine, but it is undocumented in outputs. **V**
17. **No logging to files and no run directories.** Logs are ad-hoc `nohup` captures at repo root (`calibration.log`, `train_cbm.log`, …) with no config or commit recorded. **V**

## 3. Reusable components

| Component | Verdict | Notes |
|---|---|---|
| `datasets/mimic_cxr.py` index + one-image policy | Reuse logic | Correct; will be replaced by the manifest reader but the policy is sound. |
| `datasets/chexpert_labels.py` | Reuse parser only | Must change blank → negative (rule). |
| `perception/clear_encoder.py`, `ingest/bulk_encode.py`, `ingest/image_loader.py` | Reuse | Parallel decode is the right design (~300 img/s on H100 per docstring). Patch-token exposure still to be checked in CLEAR (`model.visual`). |
| `outputs/index/train_embeddings.npy` (213,365 × 768 fp32, 655 MB) + `train_ids.json` | **Likely reusable, pending the 100-image recompute** | Keyed by dicom_id; checkpoint sha256 recorded; covers exactly official-train one-frontal-per-study (verified count match: 213,365 = official train PA/AP studies). Must be re-checked against the new splits, which remove MS-CXR and ImaGenome-gold patients (752 + 494 train patients, 16 overlap). |
| `outputs/index/train.faiss` | Rebuild | Contains the excluded patients; rebuilding from the cached embeddings takes about 1 minute. |
| `model_weights/concept_embeddings_368294.pt`, `mimic_concepts.csv` | Reuse | CLEAR text-tower outputs; frozen. |
| `model_weights/concept_tags.jsonl.gz` (RadGraph tags on 368k concepts), `configs/finding_synonyms.yaml` | Reuse with caution | Tags are usable, but defect 7 means the polarity/temporal fields must be re-derived or filtered (drop temporal, fix negation). |
| `outputs/report_facts/*.json.gz` (1,757 studies) | Partly reusable | Valid parses, but they cover only val + neighbours, and are whole-report (defect 11). New labels come from CheXpert CSV anyway. |
| `retrieval/radgraph_parser.py` | Reuse | |
| `data/radlex/` (RadLex OWL + snapshot), `offline/build_radlex_snapshot.py`, `fusion/rules/radlex_client.py` | Reuse | Needed for `part_of`. |
| Calibrators, thresholds, `cbm_head.pt`, 67-concept list, fusion, rule engine, prompt | **Do not reuse** | Carry defects 1–8. |

### Cached-embedding checks still to do (need a GPU, which hasn't been allocated yet)

- Recompute 100 random cached images and require cosine > 0.999. **Not done:** no GPU allocation has started (see COMPUTE.md), and the user has said not to extract CLEAR embeddings yet.
- Coverage: cached = 213,365 official-train studies. Missing for the new design: other frontal images of train studies (237,972 − 213,365 = 24,607 images), all of validate (1,959 frontal images) and test (3,403 frontal images). **V** (counts)

## 4. Which splits the old numbers came from (V)

- **213,365 training images** = official MIMIC `train` split, one PA/AP image per study (222,758 train studies, 9,393 with no PA/AP dropped; 84,328 PA + 129,037 AP). This matches `outputs/index/train_manifest.json`.
- **1,733 validation studies** = official `validate` split, one PA/AP image per study (1,808 validate studies, 1,733 with a PA/AP).
- **The 5 "test" studies** in `outputs/full_pipeline/test.jsonl` = official `test` split, patient 10046166.

## 5. Phase 0 items not completed

- CLEAR 100-image consistency check: needs a GPU allocation (pending).
- `srun --overlap` step test and GPU-node internet check: needs a running allocation (pending). See COMPUTE.md.
- CheXmask and VinDr checksums: running in the background at the time of writing; results go into DATA.md.
