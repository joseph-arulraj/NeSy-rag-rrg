# Results

Every experiment result, including negative ones. Newest at the bottom.

## Old pipeline (reference, from the audit, not rerun)

- No quantitative evaluation was ever recorded for the old pipeline. Its outputs: 5 official-test studies (patient 10046166), `outputs/full_pipeline/test.jsonl`. Every report asserted 13–16 findings, none negative except 1–2 per report. See AUDIT.md.
- Old CBM head training: final loss 0.2416 (200 epochs, 213,365 train studies; labels with blanks masked). Seen by backbone.

## Stage 0 — preprocessing (2026-10-06)

| What | Run dir | Headline |
|---|---|---|
| Manifests | `runs/20261006-0918_manifests` (failed at PadChest-GR: 39-digit IDs overflowed int64; fixed), `runs/20261006-0919_manifests` | 6 manifests; all count checks against published numbers pass |
| Splits v0-DRAFT | `runs/20261006-0920_splits` | 65,379 patients: train 59,588 / calib 1,897 / thresh 1,869 / val 500 / test 293 / heldout_loc 1,232 |
| Split tests | `runs/20261006-0920_split-tests` | 11/11 pass |
| CheXmask smoke | `runs/20261006-0922_chexmask-smoke` | 2,000 images: qc_ok 97.7 %, side convention 100 %, CTR median PA 0.492, AP 0.542; preprocessed vs original \|ΔCTR\| median 0.0004 |
| CheXmask features (MIMIC frontal) | `runs/20261006-0940_chexmask-features` (slam_cpu job 37811782, 16 workers, 11.8 min) | 243,334 / 243,334 frontal images (100 %); qc_ok 237,498 (97.6 %); side convention 100 %. CTR (qc_ok): PA median 0.491 (IQR 0.453–0.534, n 95,585), AP median 0.542 (IQR 0.501–0.582, n 141,913). Left-lung area fraction mean 0.459; heart shifted to patient's left by 0.074 thorax widths on average. Output `data/features/chexmask_mimic.parquet` (42 named columns) |
| Splits **frozen as v1** | `runs/20261006-1003_split-tests-v1` | User confirmed 2026-10-06. `data/splits/mimic_splits_v1.parquet` (read-only, sha256 a1e72e40…8340) has the same assignment as v0-DRAFT; 12/12 split tests pass |
| ImaGenome scene-graph parse | `runs/20261006-1004_imagenome-parse` (slam_cpu, 1.0 min) | 243,310 graphs, 0 ID mismatches vs MIMIC; 8,695,606 region boxes (36 regions/image); 7,306,391 (image, region, label) rows. Outputs `data/features/imagenome_{regions,region_labels,meta}.parquet` |
| Report sections | (interactive check) | Of 227,835 reports: FINDINGS+IMPRESSION 122,745; impression only 66,679; findings only 27,028; fallback 11,382 (flagged); empty 1 |
| RadGraph speed test | `runs/20261006-1006_radgraph-speedtest` | modern-radgraph-xl on CPU, 16 threads, batch 32: 12.0 reports/s |
| RadGraph on all reports | `runs/20261006-1010_radgraph-reports` | RUNNING; ETA ~5.3 h; first shard 1,998/2,000 reports with entities |

## Stage 0+ (items from the 2026-10-06 work list)

### Item 1 — labels (`runs/20261006-1022_labels`, output `data/labels/labels_v1.parquet`)
Rule changes (labels): R1 blank→negative 2,380,509; R2 uncertain→masked 75,018; R0 no-CheXpert-row masked 104 (8 studies × 13);
R3 positive child→parent positive 114,385 (from blank 107,560, **overriding an explicit negative 2,860**, from uncertain 3,965) — lung_opacity 56,946, consolidation 14,288, enlarged_cardiomediastinum 43,151;
R4 uncertain child + blank parent→masked 24,442 (lung_opacity 7,379, consolidation 14,505, enlarged_cardiomediastinum 2,558). Hierarchy violations after closure: 0.
Derived no_finding: 1 = 81,443, 0 = 138,707, masked 7,685; agrees with CheXpert "No Finding"=1 on 97.3 % where defined.
Final positives (all studies): lung_opacity 108,471; support_devices 66,558; pleural_effusion 54,300; enlarged_cardiomediastinum 50,330; atelectasis 45,808; cardiomegaly 44,845; edema 27,018; consolidation 25,066; pneumonia 16,556; pneumothorax 10,358; lung_lesion 6,284; fracture 4,390; pleural_other 2,011.

### Item 2 — split check (`runs/20261006-1023_split-check`)
Positives per finding, studies with a frontal image (calib / thresh / val): atelectasis 1,248/1,133/347; cardiomegaly 1,167/1,065/363; consolidation 685/582/178; edema 737/582/233; enlarged_cardiomediastinum 1,322/1,209/405; fracture 107/105/17; lung_lesion 141/192/64; lung_opacity 2,895/2,622/851; pleural_effusion 1,406/1,313/452; **pleural_other 47/47/14 (FLAGGED <50 in calib and thresh)**; pneumonia 438/360/115; pneumothorax 359/259/73; support_devices 1,892/1,575/550. Val is also thin for fracture (17) and pleural_other (14). Not re-split.

### Item 10 — Chest ImaGenome finding-by-region and side labels (`runs/20261006-1033_imagenome-labels`; done out of order because it needs no embeddings)
Mapping `kg/imagenome_map.yaml` (DRAFT, for review). Outputs `data/features/imagenome_finding_region_v1.parquet` (4,101,665 image×region×finding rows; region positives 2,192,931 → 2,337,645 after is_a closure) and `imagenome_side_v1.parquet` (243,310 images on the frozen splits: train 209,559, heldout_loc 14,837, calib 7,006, thresh 6,546, val 1,959, test 3,403). Region boxes were already parsed (`imagenome_regions.parquet`), so Stage 4 needs only the GPU feature pass.
**Caveat:** ImaGenome assigns a mention with no stated side to both lungs. Edema is "bilateral" in 100 % of positives and lung_opacity in 76 %, so "bilateral" here cannot be told apart from "side not stated". Before side labels are used for training or evaluation, check them against laterality words in the report sentence.

### Item 11 — external sets, loading only (`runs/20261006-1037_external-contact-sheets`; done early, needs no embeddings)
Loaders `nesy/external.py`. Proposed preprocessing, the same for both sets, **to confirm**: invert MONOCHROME1, window to the 0.5–99.5 intensity percentiles, 8-bit, then CLEAR's own 448² resize as for MIMIC. DICOM WindowCenter/Width are ignored because they span the full bit range in the files checked. VinDr files are uncompressed DICOM (12–14 bit); PadChest-GR is 16-bit PNG read from the split zip in memory.
Contact sheets (24 random images each: VinDr train, VinDr test, PadChest-GR, plus a MIMIC reference) look correct: inversion right, heart on the image right, normal contrast. Some PadChest images have black padding inside the frame.
Mapping drafts **for radiologist review**: `kg/external_maps/vindr_to_findings.yaml`, `padchest_gr_to_findings.yaml`. Not evaluable: VinDr support_devices and enlarged_cardiomediastinum; PadChest-GR pneumonia and edema. No labels were scored and no metric was computed.

### Item 3 — CPU embeddings (DONE: `runs/20261006-1100_clear-embed-cpu`)
- 4,774 one-frontal-per-study images (official validate 1,733 + test 3,041) embedded on CPU in 28.6 min, 0 decode failures. Unified store `data/embeddings/clear_frontal_v1.npy`: float16 [218,139 × 768] (335 MB), max \|norm−1\| after fp16 cast 0.0001, with index parquet. Test embeddings are cached only; nothing is scored on test.
- 100-image consistency check vs the cached train embeddings (`runs/20261006-1024_clear-embed-cpu`): cosine min 0.999996, median 1.000000; all > 0.999. The cached ids exactly equal the one-frontal-per-study policy on official train (213,365). **The cached embeddings are verified and reused.**
- That run then FAILED while assembling the store (slice bug on the last cached chunk, after 29 min of val/test encoding, unsaved). Fixed, with a checkpoint now written before assembly; rerun `runs/20261006-1100_clear-embed-cpu`. CPU throughput is about 2.4 images/s (12 threads).
- SLURM: step creation on comp214 intermittently takes 1–3 min ("step creation temporarily disabled… Socket timed out"); launches must be verified.

### Item 12 — Stage 6/7 code (code done; sample run pending Stage 1 predictions)
- `kg/findings.yaml` v0.2: 13 findings with definitions and phrases; derived no_finding; 6 is_a, 13 may_occur_in, 6 associated_with, 5 pertinent-negative rules, each with an id and rationale.
- `kg/anatomy.yaml` (generated by `scripts/build_anatomy.py`, run `20261006-1041_build-anatomy`): 48 nodes covering all 36 ImaGenome regions. part_of comes from RadLex 4.3 for 15 nodes; 30 are manual (ImaGenome zones, hila and costophrenic angles have no RadLex class); 1 manual override (RadLex aortic arch → descending aorta replaced by mediastinum).
- `nesy/kg.py`: schema validation (keys, unique ids, references, acyclic is_a and part_of). Rule modules: `grounding.py` (G1–G6), `belief.py` (D1–D4, B1–B3), `report.py` (P1–P5 template).
- `tests/test_stage6_rules.py`: **24/24 pass** (one test per rule, plus KG validation failure cases and region coverage).

### Item 4 — retrieval (`runs/20261006-1148_retrieval`, 2 min)
FAISS IndexFlatIP on the fit split only: 187,754 studies (`data/index/fit_v1.faiss`). Neighbour features k ∈ {5, 10, 25} for all 218,139 studies with a frontal embedding (train 187,754, heldout_loc 13,498, calib 6,283, thresh 5,830, val 1,733, test 3,041), each excluding every study of the query's own patient; own-patient neighbours after exclusion: 0. Output `data/features/retrieval_v1.parquet`.
Val AUROC of nb10_frac_<finding> alone: support_devices 0.908, pleural_effusion 0.892, edema 0.852, lung_opacity 0.833, pneumothorax 0.801, atelectasis 0.785, cardiomegaly 0.776, enlarged_cardiomediastinum 0.776, consolidation 0.719, pneumonia 0.654, pleural_other 0.647, fracture 0.625, lung_lesion 0.557.

### pleural_other — calibration LOW-CONFIDENCE (user decision 2026-10-06)
Kept as a finding, but its Platt calibrator and thresholds rest on only 47 positives each in calib and thresh (14 in val). Treat every pleural_other probability, band and metric as low-confidence.

### Point 3 — ImaGenome side/region labels: explicit vs default (`runs/20261006-1150_imagenome-explicit`)
A label is explicit if the report phrase itself states the side (left/right/bilateral/both/bibasilar…), or, for a region, the location (upper/lower/base/apex/hilar/lobe/costophrenic/… plus the side for a sided region). Otherwise it is ImaGenome's default assignment, which is now **masked**, never read as bilateral or all-regions-positive.

| finding | positive images | side explicit | of explicit: bilateral | region pairs | region explicit |
|---|---|---|---|---|---|
| atelectasis | 82,459 | 87.8 % | 49.6 % | 266,140 | 81.2 % |
| cardiomegaly | 60,397 | 12.9 % | 49.3 % | 60,397 | 10.2 % |
| consolidation | 18,588 | 88.7 % | 35.2 % | 59,005 | 80.5 % |
| edema | 37,202 | 44.5 % | 61.5 % | 174,784 | 22.9 % |
| enlarged_cardiomediastinum | 7,113 | 32.0 % | 21.4 % | 14,599 | 28.0 % |
| fracture | 10,712 | 63.7 % | 13.0 % | 13,750 | 28.6 % |
| lung_lesion | 14,812 | 79.7 % | 24.1 % | 40,549 | 67.6 % |
| lung_opacity | 161,708 | 85.9 % | 60.3 % | 773,406 | 56.4 % |
| pleural_effusion | 75,407 | 90.3 % | 50.6 % | 283,097 | 45.2 % |
| pleural_other | 14,796 | 82.2 % | 39.6 % | 44,085 | 72.0 % |
| pneumonia | 35,755 | 69.7 % | 35.5 % | 106,047 | 66.8 % |
| pneumothorax | 10,924 | 89.3 % | 9.6 % | 26,048 | 61.3 % |
| support_devices | 87,103 | 65.9 % | 18.3 % | 331,024 | 28.0 % |

KG change (`kg/findings.yaml` v0.3): `lateralisable: false` for edema, cardiomegaly and enlarged_cardiomediastinum, so they get no side prediction. No other finding is near 100 % bilateral among explicit labels. New rule **P6**: the report states a side only for lateralisable findings with side-head confidence ≥ `side_min_confidence` (0.8); otherwise the finding is stated without a side. Tests: 37/37 pass (`tests/`). The Stage 3 side head and the Stage 4 region classifier train on explicit labels only (`imagenome_side_explicit_v1.parquet`, `imagenome_region_explicit_v1.parquet`).
Note: the side-word lexicon counts "lungs" as bilateral, which may inflate edema's explicit-bilateral share; edema is non-lateralisable anyway.

### GPU lane — checks and pilot (H100 job 37814480)
- Step test and internet: OK (`runs/*_gpu-step-test`); GPU node has internet.
- CLEAR exposes patch tokens via `visual.backbone.forward_features()["x_norm_patchtokens"]`: DINOv2 ViT-B/14-reg, a 32×32 grid at 448 px, 4 register tokens excluded. **Preprocess = plain resize to 448×448 (bicubic): no crop and no padding, so 0 % of the image area is removed, but the aspect ratio is not preserved** (x and y scaled independently). Patch (i, j) covers original x ∈ [j·W/32, (j+1)·W/32), y ∈ [i·H/32, (i+1)·H/32).
- CheXmask regions use the OriginalResolution masks, because in 2 of 8 images checked the Preprocessed masks disagree with them beyond the resize/pad mapping. Weights: `data/features/regions/mimic_pilot_weights.npy` (`runs/20261006-1146_chexmask-weights-pilot`); full MIMIC weights running on CPU (`runs/20261006-1147_chexmask-weights-full`).
- Pilot, 1,000 fit-split images (`runs/20261006-1147_region-pilot`): global vector = cached embedding (cosine 1.00000 on 500 images); image sizes match metadata and CheXmask (0 mismatches). **Left/right check: same-side IoU > cross-side IoU in 100 % of images; cross-side IoU 0.000.** Grid IoU of ImaGenome lung box vs CheXmask lung: left median 0.38, right median 0.47, because the boxes are rectangles much larger than the lungs; 98.7 % of each CheXmask lung lies inside its ImaGenome box. Overlays: 10 PNGs in `overlays/`.
- Throughput: about 100 img/s after startup (the logged 12 img/s includes a 1.2 min model/worker start). Full pass 218,187 images ≈ 40 min. **Storage 70.7 kB/image = 15.4 GB for the full pass (> 15 GB limit), so this needs a decision before the full pass.**

### Item 5 — Stage 1 baseline: linear probe on CLEAR embeddings (`runs/20261006-1154_stage1`, 2.8 min). Seen by backbone.
Fit = train split; Platt on calib; threshold candidates on thresh; metrics on **val** (n ≈ 1,580–1,730 per finding). C chosen on a 5 % inner patient holdout of train.

| finding | val AUROC | AUPRC | ECE | val pos |
|---|---|---|---|---|
| support_devices | 0.927 | 0.817 | 0.028 | 550 |
| pleural_effusion | 0.917 | 0.767 | 0.029 | 452 |
| edema | 0.906 | 0.598 | 0.023 | 233 |
| pneumothorax | 0.883 | 0.350 | 0.013 | 73 |
| lung_opacity | 0.854 | 0.839 | 0.038 | 851 |
| atelectasis | 0.827 | 0.544 | 0.018 | 347 |
| cardiomegaly | 0.816 | 0.508 | 0.033 | 363 |
| enlarged_cardiomediastinum | 0.814 | 0.530 | 0.039 | 405 |
| consolidation | 0.798 | 0.406 | 0.021 | 178 |
| pneumonia | 0.763 | 0.295 | 0.016 | 115 |
| pleural_other (low-confidence) | 0.744 | 0.022 | 0.004 | 14 |
| lung_lesion | 0.711 | 0.111 | 0.019 | 64 |
| fracture | 0.694 | 0.023 | 0.007 | 17 |
| **macro** | **0.820** | **0.447** | **0.022** | |

Hierarchy violations of the independent marginals on val (P(child) > P(parent), of 1,733): cardiomegaly > enlarged_cardiomediastinum 240, pneumonia > consolidation 189, atelectasis > lung_opacity 25, consolidation > lung_opacity 15, edema > lung_opacity 10, lung_lesion > lung_opacity 0. Stage 3 must bring these to 0.
The first run (`20261006-1150_stage1`, same numbers) produced degenerate PPV candidates (a target "met" by flagging one study). Fixed: a precision/NPV target now needs ≥ 10 true positives/negatives. Threshold table: `threshold_candidates_thresh_split.csv` in the run dir. **Awaiting the user's choice.**

### Item 12 — sample run of the Stage 6/7 template (`runs/20261006-1157_sample-reports`, PROVISIONAL thresholds)
Provisional rule: present = PPV ≥ 0.8 (else 0.7), absent = NPV ≥ 0.98. D4 empty-band check: the first attempt FAILED correctly (atelectasis had an empty present band under the degenerate candidate); it PASSES after the fix. Abstained under the provisional rule: fracture, lung_lesion, pleural_other (no PPV ≥ 0.7 with ≥ 10 TPs). 20 val belief graphs + reports written (`sample_reports.json`).
Observed: the uncertain band is very wide (e.g. consolidation 1,261/1,733 val studies uncertain, atelectasis 1,271, cardiomegaly 1,195), so reports read as long lists of "possible …". With three findings abstaining, no_finding can never be true. Cosmetic bug: "{Side}" phrases start lower-case when no side is stated ("pneumonia is possible…"); to fix.

## 2026-10-06 user decisions: four bands, any_abnormality root, 36 regions

### Four bands (`nesy/thresholds.py`, `nesy/belief.py`); refitted on thresh for each stage's model, ≥ 10 TPs per PPV tier
present PPV ≥ 0.7 · possible PPV ≥ 0.4 and below present · absent ≤ 5 % of positives missed (pneumothorax ≤ 2 %) · silent otherwise (not mentioned). Absent findings are stated only if on the pertinent-negatives list.
Three implementation decisions, found by running the samples (each with a test):
1. **Absent has priority over possible where they overlap.** For common findings (prevalence > 0.4: lung_opacity, pleural_effusion, support_devices), "PPV ≥ 0.4" is met by flagging everything, so the possible threshold falls to ≈ 0.01–0.04, below the absent threshold. Dropping the absent tier there left every study with at least "possible lung opacity".
2. **The any_abnormality root gets only an absent tier.** It is never stated as a finding. Its "≤ 5 % missed" threshold (0.26) lies above its PPV ≥ 0.7 threshold (0.16, because prevalence is 63 %), which would otherwise remove the absent tier and with it every "no acute abnormality".
3. **D2 parent-raise uses only stated children (present or possible); silent children never change a parent.** Otherwise mostly-silent rare findings (fracture silent in 79 % of val) would block every "no acute abnormality".
Also: P1 now omits a parent when a child is stated with at least the same certainty (possible pneumonia no longer prints "Possible lung opacity. Possible consolidation." as well), and sentence capitalisation is fixed. Tests: **42/42 pass**.

### any_abnormality root (labels v2, KG v0.4)
`data/labels/labels_v2.parquet` (`runs/20261006-1218_labels-v2`): any_abnormality = 1 for 138,707 studies, 0 for 81,912, masked 7,216; (any_abnormality = 0) agrees with CheXpert "No Finding" = 1 on 97.1 % where defined. Hierarchy violations after closure: 0. Compared with the old derived no_finding, 469 more studies are negative instead of masked: an uncertain sub-finding under an explicitly negated parent does not mask the root, following the user's rule that only a *blank* parent is masked. The R1 count for this KG-only node (227,827) is an artefact: every root label starts blank.
KG: is_a edges isa_101–106 make the root the parent of every top-level pathology; support_devices stays outside. "No acute cardiopulmonary abnormality." is reported iff the root is in its absent band (rule D3/P7).

### Stage 1 v2 (`runs/20261006-1218_stage1-v2`; same probe, now also predicting any_abnormality)
any_abnormality val AUROC 0.882, AUPRC 0.908, ECE 0.032. The 13 findings are unchanged from Stage 1 (macro AUROC 0.820).

### Sample reports with the four bands (`runs/20261006-1230_sample-reports-v5`, Stage 1 v2 model, 20 val studies)
7 of 20 are reported "No acute cardiopulmonary abnormality": 5 are labelled normal, 2 are not (one labelled cardiomegaly + enlarged cardiomediastinum, one labelled lung_opacity).
Val band shares: any_abnormality absent 22 %; lung_opacity present 67 % / possible 7 % / absent 26 %; pleural_effusion 33/19/0 silent/48 absent; atelectasis 2/46/21/31; cardiomegaly 1/41/23/35; consolidation 0/16/59/25; fracture, lung_lesion and pleural_other never present or possible (silent 60–79 %, absent 21–40 %).
Observed: present/possible are generous for common findings (lung_opacity present in 67 % of val, label prevalence 49 %) because PPV ≥ 0.7 sits at a low threshold when prevalence is high.

### Region feature pass (GPU)
- First full-pass attempt (`20261006-1215_region-full-part1`) ran at 17 img/s. On restart, the three large output memmaps (320 MB–12 GB) had **zeroed .npy headers** after the step was killed: multi-GB np.memmap files on cephfs are not safe. The partial outputs (my files, about 7 k images) were deleted and outputs switched to sequential shard files (`<name>_shards/shard_NNNN_*.npy`, global written last as the completion marker).
- Shard version validated on the pilot (`20261006-1232_region-pilot-shards`): flip check 100 %, global = cache (cosine 1.00000), about 30–40 img/s, limited by JPEG decoding of the full-size images.
- PIL JPEG draft decoding was tested and **rejected**: only 1.35× faster, and the global embedding moves away from the cache (min cosine 0.970 < 0.999).
- Full pass (all 36 ImaGenome regions, 15.4 GB, as decided) starts when the CheXmask region weights finish; projected about 2 h at about 30 img/s.

### Concept selection (item 7), not yet approved
- v1 (`20261006-1157_concepts`) and v2 (`20261006-1219_concepts-v2`) were rejected on review: temporal wording, "simulates widening" (meaning NOT widened), compound concepts, patient-specific phrasing, long sentences. v3 (`20261006-1236_concepts-v3`) is running: 2–4 words, extended exclusions, single-finding concepts only, normal concepts scored against the absence of any_abnormality.

### any_abnormality absent band on the full val set (`runs/20261006-1239_abnormal-absent-check`, Stage 1 v2 model, 1,733 val studies)
Root absent threshold refitted on thresh for each rule. "Final" = the "no acute abnormality" call after D2 (stated children lift the root), with the four-band thresholds of the other findings.

| rule (abnormal studies missed on thresh) | threshold | view | val studies | share of val | with ≥ 1 positive pathology label |
|---|---|---|---|---|---|
| ≤ 5 % | 0.260 | root band | 375 | 21.6 % | 10.1 % (38) |
| ≤ 5 % | 0.260 | final call | 356 | 20.5 % | 9.0 % (32) |
| ≤ 2 % | 0.137 | root band = final | 248 | 14.3 % | 8.5 % (21) |
| ≤ 1 % | 0.094 | root band = final | 169 | 9.8 % | 5.9 % (10) |

Positive labels among final-call studies (studies per finding; not exclusive, parents include their children):
- ≤ 5 %: lung_opacity 21, enlarged_cardiomediastinum 6, pneumonia 6, consolidation 6, fracture 5, cardiomegaly 4, atelectasis 3, lung_lesion 2, pleural_effusion 2, pneumothorax 1, edema 1, pleural_other 1.
- ≤ 2 %: lung_opacity 17, fracture 3, pneumonia 3, consolidation 3, atelectasis 3, enlarged_cardiomediastinum 2, lung_lesion 2, cardiomegaly 1, edema 1, pleural_other 1; no effusion or pneumothorax.
- ≤ 1 %: lung_opacity 6, enlarged_cardiomediastinum 2, fracture 2, lung_lesion 2, pneumonia 1, consolidation 1, atelectasis 1, cardiomegaly 1, pleural_other 1.
Two different rates: the rule controls the share of abnormal studies missed (val, final call: 32 / 1,050 = 3.0 % at the 5 % rule, 2.0 % at 2 %, 1.0 % at 1 %). The "with ≥ 1 positive" column is the share of normal calls that are abnormal (1 − NPV): 9.0 %, 8.5 %, 5.9 %.

### Normal-call rule with critical lists — Stage 1 v2, PROVISIONAL (`runs/20261006-1250_normal-call-stage1`)
KG v0.5: rule N1 `normal_call` (status needs-radiologist-review; active list A provisionally). "No acute cardiopulmonary abnormality" is written only if the root is absent AND every critical finding is in its own absent band. Tests 43/43. Root thresholds refitted on thresh: 5 % → 0.260, 2 % → 0.137. Critical findings' own absent bands: pneumothorax ≤ 2 % missed, others ≤ 5 %. **Platt scaling was fitted on calib, so calib probabilities are in-sample for calibration; the thresholds come from thresh.**

| rule | split | normal calls | share of split | calls with ≥ 1 positive | abnormal missed / all abnormal |
|---|---|---|---|---|---|
| root 5 % | val | 356 | 20.5 % | 9.0 % (32) | 3.0 % |
| root 5 % | calib | 1,378 | 21.9 % | 11.2 % (155) | 4.1 % |
| root 5 % | pooled | 1,734 | 21.6 % | 10.8 % (187) | 3.9 % |
| root 2 % | val | 248 | 14.3 % | 8.5 % (21) | 2.0 % |
| root 2 % | calib | 859 | 13.7 % | 8.3 % (71) | 1.9 % |
| root 2 % | pooled | 1,107 | 13.8 % | 8.3 % (92) | 1.9 % |
| root 5 % + A | val | 280 | 16.2 % | 10.4 % (29) | 2.8 % |
| root 5 % + A | calib | 1,080 | 17.2 % | 11.5 % (124) | 3.3 % |
| root 5 % + A | pooled | 1,360 | 17.0 % | 11.3 % (153) | 3.2 % |
| root 5 % + B | val | 232 | 13.4 % | 9.9 % (23) | 2.2 % |
| root 5 % + B | calib | 885 | 14.1 % | 10.6 % (94) | 2.5 % |
| root 5 % + B | pooled | 1,117 | 13.9 % | 10.5 % (117) | 2.4 % |

Missed findings by type, pooled val+calib (studies per finding; parents include children):

| rule | lung_opacity | atel. | consol. | pneumonia | edema | lesion | ECM | cardiomeg. | effusion | pl. other | PTX | fracture |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| root 5 % | 106 | 27 | 35 | 33 | 2 | 19 | 52 | 39 | 7 | 5 | 3 | 23 |
| root 2 % | 57 | 12 | 19 | 18 | 2 | 12 | 16 | 10 | 3 | 3 | 2 | 13 |
| root 5 % + A | 84 | 23 | 30 | 29 | 1 | 15 | 41 | 31 | 6 | 5 | 2 | 20 |
| root 5 % + B | 60 | 16 | 25 | 24 | 1 | 11 | 36 | 27 | 5 | 4 | 2 | 14 |

Reading: the critical lists cost 22 % (A) and 36 % (B) of the root-5 % normal calls but barely change the effusion/pneumothorax misses they target (7→6→5 and 3→2→2 pooled). Those missed studies already sit in the critical finding's own absent band, so requiring that band cannot catch them; the lists mostly remove lung-opacity/consolidation-type misses. List B reaches the 2 % rule's call rate (13.9 % vs 13.8 %) with a slightly higher share of abnormal calls (10.5 % vs 8.3 %). Calib shows the same ordering as val. To repeat for the Stage 3 model.

### NEGATIVE RESULT — critical lists for the normal call (decision 2026-10-06)
Lists A (pneumothorax, effusion) and B (+ edema, consolidation) cost 22 % / 36 % of normal calls but barely changed the misses they target (pooled effusion 7→6→5, pneumothorax 3→2→2), because those studies already sit in the critical finding's own absent band. Rule N1 is now **inactive** (`critical: []` in `kg/findings.yaml` v0.5; code, candidates and tests kept, 44/44 pass). The root alone decides "No acute cardiopulmonary abnormality". 5 % and 2 % root rules to be compared on the Stage 3 model.

### Infrastructure: np.memmap header loss on cephfs (2026-10-06)
`np.lib.format.open_memmap(mode="w+")` writes the 128-byte header with a buffered write, then maps the file from offset 0. On cephfs the mapping read the header page before the write landed, and the final flush zeroed the header. This happened to `mimic_full_weights.npy` (2.0 GB) after a normal finish, and to three region-feature files. Weights repaired by rewriting the deterministic header; the data verified exactly against the independent pilot run (1,000/1,000 images identical). New helper `nesy/safe_memmap.py` (create → close → fsync header → reopen; re-assert the header on close), used by the weights script; region features now use sequential shard files.

### Concept lists — final procedure (`runs/20261006-1307_concepts-final`)
v3 filters (2–4 words, temporal/negation/hedge/spurious exclusions, single-finding concepts, fit split only) minus the 17 flagged concepts ("lower lobes collapse" kept), refilled by rank, near-duplicates (text cosine > 0.92) removed, AUROC ≥ 0.6; each finding led by its plain term where the bank has one passing the filters (1-word terms allowed). Saved with finding and individual AUROC (fit split, 12k sample): `data/concepts/concepts_final_k6.csv` (84) and `concepts_final_k12.csv` (166; enlarged_cardiomediastinum has only 10 above AUROC 0.6).
- Plain terms found: opacity (AUROC 0.66), atelectasis (0.71), consolidation (0.65), pneumonia (0.60), pulmonary edema (0.87), lung nodule (0.54), **widened mediastinum (0.41, below chance)**, cardiomegaly (0.72), pleural effusion (0.88), pleural thickening (0.60), pneumothorax (0.84), rib fracture (0.59), normal chest radiograph (0.85, normal group). **No plain term: support_devices** ("support devices", "tubes and lines" etc. are not in the bank or fail the filters).
- Refill picked a few concepts with the same problems as the flagged ones (not excluded, because they were not on the reviewed list): "associated substantial atelectasis", "multiple pleural masses" (lung_lesion), "ongoing slight mediastinal enlargement" (temporal), "some effusion". For the next review.

### Stage 2 — + CheXmask side/CTR features (`runs/20261006-1236_stage2`, baseline Stage 1 v2). Seen by backbone.
Val AUROC up on all 14: macro +0.0053, AUPRC +0.011, ECE +0.0005. Biggest: lung_lesion +0.017, consolidation +0.010, pneumonia +0.008, edema +0.007 (AUPRC +0.043), fracture +0.007 (AUPRC 0.023→0.075). **Cardiomegaly only +0.003 (0.816→0.819)**, so the stage's "better cardiomegaly discrimination" criterion is met only marginally. The side signal is not yet evaluated separately; it enters Stage 3's side head. (An earlier run, `20261006-1209_stage2`, FAILED because the KG changed mid-run.)

### Stage 3 — factorised head on named features (`runs/20261006-1310_stage3-k6`, `runs/20261006-1314_stage3-k12`). Seen by backbone.
Features: concept scores (k = 6: 84, k = 12: 166) + 17 CheXmask features. P(any_abnormality) × P(child | parent) down the is_a tree, each conditional Platt-calibrated on calib.
- **Hierarchy violations on val: 0 for both** (all 12 is_a edges), against 0–240 per edge for the independent Stage 1 probe.
- Macro val AUROC: probe 0.824 → k6 0.825 (gap +0.0006), k12 0.826 (gap +0.0021). **Named features match the 768-d embedding probe.**
- Macro ECE: probe 0.0229 → k6 0.0257, k12 0.0246, i.e. slightly worse (consolidation 0.021→0.032, pneumonia 0.016→0.029 in both).
- Per finding (probe → k6 / k12 AUROC): any_abnormality 0.882→0.878/0.879; lung_opacity 0.854→0.850/0.852; atelectasis 0.827→0.818/0.821; consolidation 0.798→0.795/0.797; pneumonia 0.763→0.763/0.764; edema 0.906→0.908/0.909; lung_lesion 0.711→0.723/0.734; ECM 0.814→0.817/0.817; cardiomegaly 0.816→0.821/0.821; effusion 0.917→0.917/0.917; pleural_other 0.744→0.751/0.761; pneumothorax 0.883→0.889/0.887; fracture 0.694→0.695/0.682; support_devices 0.927→0.921/0.925.
- Side head (explicit labels only; val accuracy vs majority-class rate): pneumothorax 0.78 vs 0.53, support_devices 0.74–0.75 vs 0.54, lung_opacity 0.72–0.73 vs 0.60, effusion 0.73 vs 0.51, consolidation 0.68–0.70 vs 0.35, atelectasis 0.68 vs 0.50, pneumonia 0.60–0.63 vs 0.36; lung_lesion 0.52 and pleural_other 0.48–0.49 are near chance.

### Normal-call rule on Stage 3, PROVISIONAL (`runs/20261006-1314_normal-call-stage3-k6`, `…1320_normal-call-stage3-k12`)
| model | rule | split | normal calls | share | calls with ≥ 1 positive | abnormal missed |
|---|---|---|---|---|---|---|
| k6 | root 5 % | val / calib / pooled | 361 / 1,415 / 1,776 | 20.8 / 22.5 / 22.2 % | 9.1 / 12.2 / 11.6 % | 3.1 / 4.6 / 4.3 % |
| k6 | root 2 % | val / calib / pooled | 261 / 913 / 1,174 | 15.1 / 14.5 / 14.6 % | 7.3 / 8.0 / 7.8 % | 1.8 / 1.9 / 1.9 % |
| k12 | root 5 % | val / calib / pooled | 363 / 1,409 / 1,772 | 20.9 / 22.4 / 22.1 % | 9.1 / 12.0 / 11.4 % | 3.1 / 4.5 / 4.2 % |
| k12 | root 2 % | val / calib / pooled | 254 / 902 / 1,156 | 14.7 / 14.4 / 14.4 % | 7.9 / 7.8 / 7.8 % | 1.9 / 1.9 / 1.9 % |
Missed findings by type are in each run's `normal_call_missed_by_type.csv`. Calib probabilities are in-sample for the Platt calibrators.

## 2026-10-06 decisions: root normal rule at 2 %; concept list k = 6 pruned

### Concept list pruned (`data/concepts/concepts_final_k6_pruned.csv`, 77 concepts)
From k = 6, removed own-AUROC < 0.60 (plain terms "lung nodule" 0.54, "widened mediastinum" 0.41, "rib fracture" 0.59) and the weak refills ("associated substantial atelectasis", "multiple pleural masses", "ongoing slight mediastinal enlargement", "some effusion"). Not refilled. Per finding: 6 for 9 findings, 5 for atelectasis, pleural_effusion and fracture, 4 for lung_lesion and enlarged_cardiomediastinum. **No finding has fewer than 3.**

### Stage 3 on the pruned list (`runs/20261006-1327_stage3-k6-pruned`; 94 named features: 77 concepts + 17 CheXmask)
Hierarchy violations on val: **0**. Macro val AUROC 0.824 (S1 0.824, S2 0.829); macro ECE 0.025 (S1 0.023, S2 0.023).

| finding | S1 emb | S2 emb+CheXmask | S3 pruned | S3−S2 | S3−S1 | ECE S1 / S2 / S3 |
|---|---|---|---|---|---|---|
| any_abnormality | 0.882 | 0.884 | 0.878 | −0.006 | −0.004 | .032/.032/.033 |
| lung_opacity | 0.854 | 0.858 | 0.849 | −0.008 | −0.005 | .038/.037/.041 |
| atelectasis | 0.827 | 0.829 | 0.818 | −0.011 | −0.009 | .018/.019/.021 |
| consolidation | 0.798 | 0.808 | 0.795 | −0.013 | −0.003 | .021/.023/.034 |
| pneumonia | 0.763 | 0.771 | 0.763 | −0.008 | 0.000 | .016/.017/.028 |
| edema | 0.906 | 0.913 | 0.908 | −0.004 | +0.003 | .023/.032/.031 |
| lung_lesion | 0.711 | 0.727 | 0.721 | −0.007 | +0.010 | .019/.017/.015 |
| enlarged_cardiomediastinum | 0.814 | 0.817 | 0.817 | 0.000 | +0.003 | .039/.037/.028 |
| cardiomegaly | 0.816 | 0.819 | 0.821 | +0.002 | +0.005 | .033/.029/.026 |
| pleural_effusion | 0.917 | 0.919 | 0.917 | −0.002 | 0.000 | .029/.034/.031 |
| pleural_other (low-conf.) | 0.744 | 0.749 | 0.750 | +0.001 | +0.006 | .004/.005/.006 |
| pneumothorax | 0.883 | 0.887 | 0.889 | +0.002 | +0.006 | .013/.014/.014 |
| fracture | 0.694 | 0.701 | 0.692 | −0.008 | −0.001 | .007/.008/.008 |
| support_devices | 0.927 | 0.928 | 0.921 | −0.007 | −0.006 | .028/.024/.040 |

### Patient-level bootstrap on val, 2,000 resamples (`runs/20261006-1330_bootstrap-s123`; 95 % percentile CIs)
- **S2 − S1 macro AUROC +0.0053 [+0.0002, +0.0100]: excludes zero.** Per finding excludes zero: lung_opacity, consolidation, pneumonia, edema, pleural_effusion (all positive). Macro ECE +0.0005 [−0.0028, +0.0028]: n.s.
- S3 − S1 macro AUROC +0.0003 [−0.0082, +0.0073]: n.s.; macro ECE +0.0026 [−0.0005, +0.0060]: n.s. Per finding excludes zero (all worse for S3): any_abnormality −0.004, lung_opacity −0.005, atelectasis −0.009, support_devices −0.006; ECE pleural_other +0.002.
- S3 − S2 macro AUROC −0.0049 [−0.0114, +0.0007]: n.s.; macro ECE +0.0021 [−0.0010, +0.0062]: n.s. Per finding excludes zero (all worse for S3): any_abnormality −0.006, lung_opacity −0.008, atelectasis −0.011, consolidation −0.013, support_devices −0.007; ECE consolidation +0.010, pleural_other +0.001.
- **Reading:** Stage 2 beats Stage 1 (small but real). Stage 3 is not distinguishable from Stage 1 overall. Against the like-for-like Stage 2 it is lower on 5 findings with CIs excluding zero, but not on the macro mean. Stage 3's named metric, hierarchy violations, goes from 240 (cardiomegaly > ECM) and 189 (pneumonia > consolidation) to 0. The cost of interpretability versus Stage 2 is about −0.005 macro AUROC, up to −0.013 for consolidation.

### Cardiomegaly: CTR alone (`runs/20261006-1327_ctr-auroc`; no fitting, QC-passed CheXmask only)
| split | view | n | positives | AUROC ctr | AUROC ctr_maxrow | median CTR pos / neg |
|---|---|---|---|---|---|---|
| val | PA | 641 | 71 | **0.850** | 0.839 | 0.560 / 0.482 |
| val | AP | 998 | 280 | **0.660** | 0.647 | 0.572 / 0.537 |
| val | all | 1,639 | 351 | 0.745 | 0.734 | |
| train | PA | 75,151 | 7,636 | 0.822 | 0.823 | 0.556 / 0.484 |
| train | AP | 104,202 | 27,198 | 0.680 | 0.678 | 0.569 / 0.532 |
CTR alone (0.850 val PA) beats the embedding probe on PA; on AP it is weak (0.66, magnification). AP is 60 % of studies and 80 % of val cardiomegaly positives, so the pooled gain from adding CTR is small (+0.003). It suggests a view-specific CTR term (already partly present as ctr×AP) or a PA-only CTR rule.

### Normal call, root 2 % (chosen) — Stage 3 pruned (`runs/20261006-1330_normal-call-stage3-pruned`)
| rule | val calls / share / with positive / abnormal missed | calib | pooled |
|---|---|---|---|
| root 2 % | 252 / 14.5 % / 7.5 % (19) / 1.8 % | 902 / 14.4 % / 8.0 % / 1.9 % | 1,154 / 14.4 % / 7.9 % / 1.9 % |
| root 5 % (reference) | 363 / 20.9 % / 9.1 % / 3.1 % | 1,416 / 22.5 % / 11.9 % / 4.5 % | 1,779 / 22.2 % / 11.4 % / 4.2 % |
Missed at 2 %, pooled: lung_opacity 54, ECM 19, consolidation 19, pneumonia 18, cardiomegaly 13, lung_lesion 13, fracture 12, atelectasis 11, effusion 3, pleural_other 3, edema 2, pneumothorax 2. Root threshold now 2 % in `nesy/thresholds.py`.

### Sample reports with belief graphs — Stage 3 pruned (`runs/*_sample-reports-stage3-pruned-v2`, 80 val studies)
Three examples (normal, single, several) with full node tables and audit in the run log. Fixes made on review: the side head is attached only to stated (present/possible) findings, the audit names it ("side head P(side|finding), p=…"), and the inactive N1 rule is no longer cited on D3. Tests 45/45. In the "several" example, effusion (p 0.46) and support devices (p 0.39) are stated present but not labelled positive.

## 2026-10-06 (afternoon): Stage 3 kept as base; bands v2; CLEAR preprocessing check

### STOPPED before external caching: CLEAR authors' preprocessing differs from ours
The CLEAR repo (`scripts/run_preprocess.py` → `src/clear/data_processing.py: img_to_hdf5/preprocess`) shows how its authors built their image sets, VinDr and PadChest included:
- **Geometry:** aspect-preserving resize (LANCZOS, longest side 448) plus **zero padding to 448×448** (letterbox). We, like the old pipeline, use the released `clear.hub.build_cxr_preprocess`: a plain bicubic **stretch** to 448×448, whose docstring says it resizes "to an exact square rather than aspect-ratio resized". This affects MIMIC as well: the cached embeddings and the running region pass use the stretch, so they are consistent with each other but maybe not with how CLEAR was trained.
- **VinDr intensity:** they read pre-converted PNGs (`train_png/`, `test_png/`); the DICOM→PNG step is not in the repo, so their windowing/inversion is unknown.
- **PadChest intensity:** 16-bit PNGs read with `cv2.imread` (colour mode), which converts to 8-bit by dropping the low byte (divide by 256), with no percentile windowing. Ours: 0.5–99.5 percentile window.
External caching waits for the user's decision.

### Bands v2 (calibrated probability; `nesy/thresholds.py`)
present p ≥ 0.70 · possible 0.40 ≤ p < 0.70 · absent as before (≤ 5 % positives missed; pneumothorax ≤ 2 %; root ≤ 2 %) with priority over possible · silent otherwise.
Band report on Stage 3 pruned (`runs/20261006-1351_band-report-s3`; each finding's own band, before D2):

| finding (val) | pos | max p | present / possible / silent / absent | present: n, precision, sensitivity | possible: n, precision, sensitivity |
|---|---|---|---|---|---|
| lung_opacity | 851 | 0.99 | 40 / 22 / 12 / 26 % | 663, 0.82, 0.64 | 362, 0.57, 0.24 |
| atelectasis | 347 | 0.87 | 2 / 21 / 44 / 33 % | 32, 0.72, 0.07 | 337, 0.47, 0.46 |
| consolidation | 178 | 0.80 | 0.3 / 7 / 65 / 28 % | 5, 0.80, 0.02 | 95, 0.48, 0.26 |
| pneumonia | 115 | 0.67 | 0 / 3 / 73 / 24 % | – | 32, 0.50, 0.14 |
| edema | 233 | 0.92 | 6 / 13 / 27 / 54 % | 100, 0.64, 0.28 | 185, 0.53, 0.42 |
| lung_lesion | 64 | 0.43 | 0 / 0.1 / 68 / 32 % | – | 1, 0.00, 0.00 |
| enlarged_cardiomediastinum | 405 | 0.84 | 2 / 22 / 45 / 31 % | 25, 0.60, 0.04 | 353, 0.53, 0.46 |
| cardiomegaly | 363 | 0.82 | 1 / 19 / 46 / 34 % | 19, 0.63, 0.03 | 321, 0.51, 0.45 |
| pleural_effusion | 452 | 0.96 | 17 / 16 / 18 / 49 % | 289, 0.79, 0.50 | 259, 0.56, 0.32 |
| pleural_other | 14 | 0.42 | 0 / 0.1 / 56 / 44 % | – | 1, 0.00, 0.00 |
| pneumothorax | 73 | 0.78 | 0.2 / 4 / 61 / 35 % | 4, 0.25, 0.01 | 59, 0.53, 0.43 |
| fracture | 17 | 0.25 | 0 / 0 / 77 / 23 % | – | – |
| support_devices | 550 | 0.98 | 26 / 11 / 12 / 51 % | 447, 0.80, 0.65 | 187, 0.64, 0.22 |
| any_abnormality (root, absent tier only) | 1,050 | | absent 14.5 % (19 positives) | | |
Pooled val+calib: present precision lung_opacity 0.83, effusion 0.81, devices 0.81, pneumothorax 0.71 (n 105), edema 0.73, atelectasis 0.73; possible precision 0.47–0.63. **Present/possible precision is close to the calibrated probability range**, as intended. Fracture, lung_lesion and pleural_other never reach 0.40, so they are only ever absent or silent. Pneumonia, consolidation, cardiomegaly and enlarged_cardiomediastinum rarely reach 0.70 (present sensitivity 0–4 %).

### Sample reports, bands v2 (`runs/20261006-1351_sample-reports-bands-v2`, Stage 3 pruned)
- Normal (57231052, PA): all absent; "No acute cardiopulmonary abnormality."; labels agree.
- Single (50698281, AP): "Possible support device" (p 0.57). Labels: lung_opacity positive (p 0.26, silent: a miss), devices negative.
- Several (58052191, AP): present lung opacity (0.89), effusion (0.71), devices (0.80); possible atelectasis, edema, borderline heart. Labels: lung_opacity, atelectasis, edema, effusion, devices positive, so every present call is correct; possible cardiomegaly/ECM are labelled negative.

### NEGATIVE RESULT — heart ratio used only on PA images (`runs/20261006-1351_stage3-pa-ctr`, bootstrap `…1438_bootstrap-s3-pa`)
Stage 3 pruned with CheXmask CTR / CTR-maxrow / heart width entering only as ×PA (feature group `chexmask_pa`), against the current Stage 3 (CTR + CTR×AP), val:
- cardiomegaly 0.821 → 0.819, Δ −0.0018 [−0.0034, −0.0000]; enlarged_cardiomediastinum 0.817 → 0.815, Δ −0.0019 [−0.0036, −0.0002]. Both CIs exclude zero, both slightly worse. ECE differences n.s. Macro AUROC Δ −0.0002 (n.s.). Hierarchy violations 0.
- Reason: the existing pair (ctr, ctr×AP) already lets PA and AP have different slopes. Removing the AP heart-ratio signal loses a little information instead of adding any. Stage 3 keeps the current CheXmask features.

### Stages 1–3 by view (val, `runs/20261006-1440_view-metrics`) — S2 is the accuracy reference
Val: AP about 1,030 studies, PA about 645. Macro AUROC AP: S1 0.799, S2 0.808, S3 0.804 · PA: S1 0.852, S2 0.856, S3 0.850. Macro ECE AP: 0.031 / 0.031 / 0.035 · PA: 0.024 / 0.024 / 0.025.
| finding | AP S1 / S2 / S3 | PA S1 / S2 / S3 | pos AP / PA |
|---|---|---|---|
| any_abnormality | 0.813 / 0.816 / 0.809 | 0.902 / 0.903 / 0.898 | 779 / 271 |
| lung_opacity | 0.793 / 0.802 / 0.786 | 0.873 / 0.871 / 0.868 | 655 / 196 |
| atelectasis | 0.778 / 0.779 / 0.772 | 0.885 / 0.887 / 0.872 | 265 / 82 |
| consolidation | 0.777 / 0.786 / 0.771 | 0.820 / 0.829 / 0.820 | 130 / 48 |
| pneumonia | 0.738 / 0.744 / 0.731 | 0.803 / 0.811 / 0.807 | 77 / 38 |
| edema | 0.852 / 0.863 / 0.855 | 0.971 / 0.972 / 0.975 | 213 / 20 |
| lung_lesion | 0.673 / 0.687 / 0.691 | 0.803 / 0.828 / 0.811 | 44 / 20 |
| enlarged_cardiomediastinum | 0.746 / 0.746 / 0.746 | 0.883 / 0.888 / 0.887 | 324 / 81 |
| cardiomegaly | 0.751 / 0.750 / 0.753 | 0.891 / 0.897 / 0.899 | 292 / 71 |
| pleural_effusion | 0.875 / 0.879 / 0.876 | 0.967 / 0.968 / 0.965 | 358 / 94 |
| pleural_other | 0.739 / 0.747 / 0.775 | 0.733 / 0.741 / 0.707 | 9 / 5 |
| pneumothorax | 0.860 / 0.873 / 0.866 | 0.931 / 0.921 / 0.932 | 48 / 25 |
| fracture | 0.897 / 0.942 / 0.941 | 0.576 / 0.591 / 0.592 | 4 / 13 |
| support_devices | 0.890 / 0.892 / 0.883 | 0.887 / 0.881 / 0.873 | 502 / 48 |
Every model is much better on PA than AP: macro gap about 0.05, any_abnormality 0.90 vs 0.81, cardiomegaly about 0.90 vs 0.75. On PA the Stage 3 cardiomegaly AUROC (0.899) already exceeds CTR alone (0.850); AP cardiomegaly (0.75) is the weak spot. Fracture, pleural_other and the PA edema/devices cells rest on ≤ 25 positives and should not be read.

### GPU region-feature pass — DONE (`runs/20261006-1304_region-full`, H100, stretch geometry)
218,187 MIMIC images (every one-frontal-per-study image in all splits, plus 48 extra MS-CXR/gold images) in 124.8 min (29.1 img/s, limited by JPEG decoding: 5,243 s waiting for data vs 1,473 s GPU). Per image, float16: global embedding + 36 ImaGenome-box vectors + 9 CheXmask-region vectors. 27 shards, 15 GB, in `data/features/regions/mimic_full_shards/`. Image size vs metadata/CheXmask mismatches: 0. Global vector = cached store (cosine 1.00000 on 500). **Free space after the pass: 30.3 GB.**
Left/right check (ImaGenome lung box vs CheXmask lung, same side > cross side for both lungs): 99.9 % of 217,041 images. Of the 242 failures, 7 are fully swapped (likely genuinely flipped images or mislabelled masks); 235 have poor alignment on both sides. List: `flip_check_failures.csv` in the run dir, for review.

### Stretch vs letterbox (CLEAR authors' preprocessing), no training — MIXED, awaiting user (`runs/20261006-1709_letterbox-test`)
Full val (1,733 studies, 500 patients); letterbox = grey, aspect-preserving LANCZOS to longest side 448, zero-padded to 448². Embedding agreement letterbox vs stretch: cosine median 0.964, min 0.487 (stretch vs cached store 0.99999). Patient bootstrap, 2,000 resamples, letterbox − stretch:
- **77 approved concepts, own-finding AUROC: mean 0.7765 → 0.7792, Δ +0.0027 [−0.0018, +0.0072]; does NOT exclude zero.** Per concept: 13 better (CI > 0), 3 worse, 61 unclear. Cardiomegaly concepts all 6 better (mean +0.021), pneumonia 3 better (+0.015); enlarged_cardiomediastinum 3 of 4 worse (mean −0.017).
- **Zero-shot (CLEAR's own softmax over "{}" / "no {}"): macro 0.7369 → 0.7457, Δ +0.0088 [+0.0015, +0.0158]; excludes zero.** Per finding: cardiomegaly +0.027 [+0.017, +0.036], enlarged_cardiomediastinum +0.021 [+0.010, +0.032], support_devices +0.014 [+0.008, +0.022] better; pleural_other −0.038 [−0.055, −0.017] worse; the other 9 unclear.
- Decision rule ("better on average with an interval excluding zero"): met for zero-shot, not for the concept test, so **not clearly met**. No switch started; waiting for the user. The stretch features stay in use. Heart-size findings gain the most, consistent with stretching distorting the cardiothoracic proportions.

### Stage 4 — region classifier (A2) on stretch features (`runs/20261006-1709_stage4-regions`). Seen by backbone.
Explicit ImaGenome region labels only (default assignments masked); negatives = all regions of studies negative for the finding. Per finding and region set, one logistic regression on [pooled vector, region one-hot], fit split; GPU L-BFGS.
- Region-level val AUROC is 0.91–0.995 (set A) and 0.75–0.98 (set B). These are **flattering**, because every region of a negative study is an easy negative.
- Image-level val AUROC (max over regions), against Stage 1 / Stage 2: lung_opacity A 0.849 / B 0.846 (S1 0.854, S2 0.858); atelectasis 0.822 / 0.822 (0.827, 0.829); consolidation 0.786 / 0.789 (0.798, 0.808); pneumonia 0.759 / 0.761; edema 0.902 / 0.903; lung_lesion 0.735 / 0.727 (0.711, 0.727); ECM 0.746 / – ; cardiomegaly 0.794 / 0.771 (0.816, 0.819); effusion 0.915 / 0.914; pleural_other 0.693 / 0.720; pneumothorax 0.881 / 0.879; fracture A 0.721 (280 positive regions; val region AUROC undefined) / –.
- Set B has no model for enlarged_cardiomediastinum or fracture: no explicit ImaGenome region maps to a CheXmask region for them.

### MS-CXR localisation check (first result; MS-CXR patients never in the fit split)
Highest-scoring anatomically relevant region for the phrase's finding; hit = that region overlaps a radiologist box (> 0 overlap on the 32×32 grid); side correct = region side equals box side (image-centre test), for lateral regions and one-sided boxes.

| finding | phrases | hit A / B | side correct A / B (n) |
|---|---|---|---|
| atelectasis | 61 | 0.98 / 0.98 | 0.96 / 0.96 (24) |
| cardiomegaly | 333 / 326 | 1.00 / 1.00 | – |
| consolidation | 117 | 0.95 / 0.93 | 0.92 / 0.89 (61) |
| edema | 44 | 0.95 / 0.98 | 0.80 / 0.80 (5) |
| lung_opacity | 82 | 0.83 / 0.82 | 0.77 / 0.74 (57) |
| pleural_effusion | 96 | 0.93 / 0.93 | 0.90 / 0.90 (51) |
| pneumonia | 182 | 0.81 / 0.86 | 0.74 / 0.80 (133) |
| pneumothorax | 245 | 0.88 / 0.88 | 0.88 / 0.88 (239) |
| **overall** | | **0.920 / 0.924** | **0.846 / 0.853** |

**Caveats, so these are not over-read:** the overlap rule is lenient. Large regions (a whole lung) overlap almost any box on their side, and cardiomegaly's heart regions always overlap the heart box (1.00 is trivial). Side correctness is the more informative number (chance about 0.5 for one-sided boxes). Still to do: chance baselines (random relevant region; "always the larger lung"), a stricter overlap criterion (IoU ≥ 0.1 or box centre inside the region), and restricting to the smaller regions (zones/thirds). The two region sets localise equally well.

### Stage 5 (provisional) — embedding probe + neighbour label fractions, against Stage 1 (`runs/20261006-1304_stage5-nb5`, `…1331_stage5-nb10`, `…1354_stage5-nb25`)
Macro val ΔAUROC vs Stage 1: k = 5 +0.0048, **k = 10 +0.0049**, k = 25 +0.0036 (ΔAUPRC +0.009 / +0.008 / +0.009; ΔECE +0.001 or less). k = 10, per finding: pneumonia +0.020, lung_lesion +0.018, pleural_other +0.009, consolidation +0.007, edema +0.005, lung_opacity +0.004; fracture −0.006. These runs add the neighbours to the **embedding probe** (Stage 1). The ablation on the Stage 3 base (as the user decided) and with bootstrap CIs is still to do.

### Visual examples (`outputs/visual_examples/`, run `runs/20261006-1805_visual-examples`, CPU, no metrics)
2 examples each: MIMIC (val, one PA, one AP, ≥ 2 positive findings), MS-CXR (not official test, different findings), VinDr (train folder, ≥ 1 box), PadChest-GR (≥ 1 boxed finding). Each has one PNG (5–7 panels) and a .txt; README.md lists the files and panels.
Visual review (by eye, all 8): left/right correct everywhere (patient-left lung on the image right; VinDr "L" markers agree). CheXmask masks fit the lungs and heart in 6 of 8. Problems:
- `ms_cxr_c9638d78…` (AP, low volumes, rotated): CheXmask quality 0.696 (< 0.7, fails QC). The right-lung outline is shrunken and the heart outline overlaps the right lung, so the masks do not fit. Its CTR and region features should not be trusted (QC already flags it).
- `vindr_8111591b…`: large left pleural effusion with an opaque left hemithorax. The CheXmask left-lung mask covers the whole hemithorax (anatomical lung, not aerated lung), and the heart border is obscured, so the heart mask is a guess. CTR is unreliable when a lung or heart border is opacified, a general limitation to note for effusion and collapse cases. The three VinDr radiologists' effusion boxes differ in extent; one also marks a small right effusion.
- Chest ImaGenome boxes for trachea, spine and abdomen extend past the image edge in 2 of 4 MIMIC/MS-CXR images (automatic check). These are silver-standard boxes; harmless for pooling (clipped to the grid).
- MS-CXR Stage 4 panel: for the bibasilar atelectasis example, the top region (right lower lung zone) overlaps one of the two radiologist boxes; for the cardiomegaly example, the cardiac-silhouette box contains the radiologist's heart box but is larger.

## 2026-10-06 evening — switch to letterbox (user decision); overnight plan running (`runs/20261006-2009_overnight-letterbox/orchestrator.log`)

### MS-CXR localisation REFERENCE on stretch features, stricter tests and baselines (`runs/20261006-201442_mscxr-loc-stretch`)
Stage 4 run `20261006-1709_stage4-regions` (no cross-fitting; MS-CXR patients never in the fit split). Tests on the 32×32 grid: hit_any (any overlap, lenient), IoU ≥ 0.1, box centre in region (weight ≥ 0.5), side correct. Baselines: a random relevant region (expected value) and always the largest relevant region.

| set | chooser | hit_any | IoU ≥ 0.1 | centre in | side |
|---|---|---|---|---|---|
| A (ImaGenome boxes) | **model** | **0.920** | **0.795** | **0.822** | **0.846** |
| A | random region | 0.585 | 0.452 | 0.331 | 0.500 |
| A | largest region | 0.716 | 0.593 | 0.685 | 0.387 |
| B (CheXmask regions) | **model** | **0.924** | **0.826** | **0.767** | **0.853** |
| B | random region | 0.619 | 0.510 | 0.457 | 0.500 |
| B | largest region | 0.735 | 0.629 | 0.651 | 0.468 |
Small regions only (zones/thirds; whole lungs and whole cardiac silhouette removed): model IoU ≥ 0.1 A 0.790 / B 0.790 vs random 0.416 / 0.469 and largest 0.524 / 0.459; centre-in 0.625 / 0.641 vs 0.229 / 0.401; side 0.844 / 0.842.
Per finding, model IoU ≥ 0.1 (A / B; random region): atelectasis 0.93 / 0.89 (0.25 / 0.34), consolidation 0.83 / 0.86 (0.29 / 0.41), pneumonia 0.80 / 0.79 (0.27 / 0.36), lung_opacity 0.67 / 0.76 (0.23 / 0.35), pneumothorax 0.62 / 0.70 (0.20 / 0.24), effusion 0.63 / 0.62 (0.21 / 0.18), edema 0.55 / 0.80 (0.38 / 0.48); cardiomegaly is uninformative (every candidate overlaps the heart box). Weak spots: effusion centre-in 0.37 / 0.33 (boxes at the costophrenic angle vs lower-zone regions) and edema on set A.
**Reading: A2 localises well above chance on every test**: about +0.35 IoU ≥ 0.1 over a random region and +0.35 side accuracy over chance. This is the reference for the letterbox rerun.
(First attempt `20261006-200506_mscxr-loc-stretch` FAILED on a table-formatting bug after computing; fixed.)

### Letterbox pilot (1,000 images) — GATE PASSED (`runs/20261006-201635_region-pilot-lb`, gate `…202054_pilot-gate`)
Region maps recomputed for the padded geometry (`nesy/regions.py`; CheXmask weights `mimic_pilot_lb`, compressed chunks, 0.4 MB). Checks vs the stretch pilot on the same images: left/right flip check 100 %; median same-side grid IoU (ImaGenome lung box vs CheXmask lung) left 0.375 (stretch 0.382), right 0.460 (0.470); left-lung mask share inside its box 1.00; image size mismatches 0; letterbox global vs an independent `encode_image` of the letterboxed image cosine min 1.00000. 23 / 1,000 images have CheXmask Dice RCA < 0.7, so their CheXmask regions are set missing. 10 overlays in the run's `overlays/` (one checked by eye: masks and boxes sit on the anatomy in the padded frame). Full pass continuing automatically.

### Letterbox full pass and retraining — results as they arrive (all val, seen by backbone)
- **Full letterbox region pass** (`runs/20261006-214338_region-full-lb`): 218,187 images in 56.1 min (64.9 img/s), 27 shards `mimic_full_lb_shards` (15 GB); size mismatches 0; flip check 99.9 %; global vs independent encode cosine 1.00000. Region features set missing for the 242 left/right failures (both sets) and for the 4,485 images with CheXmask Dice RCA < 0.7 (CheXmask set). Letterbox CheXmask weights stored as compressed chunks (`mimic_full_lb_p0/p1_chunks`, < 0.1 GB). Free space after the pass 20.8 GB.
- **Stage 1 letterbox** (`…224426_lb-stage1`): macro AUROC 0.8305 vs stretch 0.8240 (+0.0065); AUPRC 0.490 vs 0.480; ECE 0.024 vs 0.023. Gains: lung_lesion +0.043, pneumonia +0.016, cardiomegaly +0.013, consolidation +0.010, ECM +0.008; fracture −0.033 (17 positives).
- **Stage 2 letterbox** (`…224712_lb-stage2`): macro AUROC 0.8350 (stretch 0.8293); vs letterbox Stage 1 +0.0044 AUROC, −0.0032 ECE (ECE 0.021).
- **Stage 3 letterbox** (`…231036_lb-stage3`, pruned k6): hierarchy violations 0; macro AUROC about 0.831 (stretch 0.824); gap to letterbox Stage 2 −0.0044 [−0.0098, +0.0007]; ECE 0.024. CIs exclude zero (S3 lower than S2) for any_abnormality, lung_opacity, atelectasis, consolidation, edema and support_devices; pneumonia ECE +0.017.
- **Stage 4 letterbox** (`…224243_lb-stage4`, 2-fold patient cross-fitting so fit-split scores are out-of-fold; vectorised labels and linear-decomposed GPU scoring, identical results to the old code on all 7 findings compared). (First attempt `…224123_lb-stage4` FAILED: missing `mimic_full_lb_ids.parquet`; fixed by copying the shared id list.)
- **MS-CXR localisation, letterbox** (`…225323_lb-mscxr-loc`) vs stretch: all candidates, IoU ≥ 0.1 A 0.797 (0.795) / B 0.838 (0.826); side A 0.846 (0.846) / B 0.867 (0.853); small regions only, IoU ≥ 0.1 A 0.781 (0.790) / B 0.795 (0.790), side A 0.788 (0.844) / B 0.845 (0.842). Baselines unchanged (random IoU ≥ 0.1 about 0.45–0.51). Unchanged within about 0.01 except small-region side accuracy on set A (−0.056).
- **Stage 4 ablation on the Stage 3 base** (`…232438_lb-stage3-s4`, bootstrap `…234919_lb-bootstrap-s4`): 66 region-classifier features (per finding: max, max over left, max over right, for both region sets) + missing indicator, cross-fitted. Hierarchy violations 0. Macro AUROC +0.0035 over Stage 3 [−0.0029, +0.0113], n.s.; ECE −0.0005 n.s.; vs Stage 2 −0.0009 [−0.0071, +0.0061], n.s. Per finding vs Stage 3: pleural_other +0.028, lung_lesion +0.012, pneumothorax +0.008 (pneumothorax vs Stage 2: +0.013 [+0.002, +0.024], CI excludes zero), lung_opacity +0.004; fracture −0.012. **Lateralised findings**, side-head accuracy S3 → S3+s4: pneumothorax 0.797 → 0.831, support_devices 0.743 → 0.761, atelectasis 0.679 → 0.688, effusion 0.719 → 0.726, lung_opacity 0.725 → 0.727; pneumonia 0.630 → 0.598, consolidation 0.691 → 0.683, fracture 0.696 → 0.609 (n 23). **Reading:** the region features close most of the Stage 3 vs Stage 2 accuracy gap and help pneumothorax (discrimination and side), but the overall gain is not significant.
- **Stage 5 letterbox** (`…232412_lb-stage5-emb-nb10`): macro AUROC 0.8315, +0.0009 vs letterbox Stage 1 [−0.0061, +0.0069] n.s. (stretch: +0.0049); ECE −0.0029 n.s. **Stage 3 + nb10** (`…235600_lb-stage5-s3-nb10`): 0.8331, +0.0026 vs Stage 3 [−0.0004, +0.0059] n.s.; 0 hierarchy violations; ECE 0.025.
- **Letterbox build-stage rule (bootstrap `…000917_lb-bootstrap`):** Stage 2 − Stage 1 = +0.0044 AUROC [−0.0019, +0.0096], ECE −0.0032 [−0.0048, +0.0011]: **on letterbox, Stage 2 no longer beats Stage 1 significantly on either metric** (stretch: +0.0053, CI excluded 0). Stage 3 − Stage 2 −0.0044 [−0.0098, +0.0007]; Stage 5 − Stage 2 −0.0035 [−0.0065, −0.0002] (significantly lower). For the user to decide whether Stage 2 stays.
- **Stretch vs letterbox, side by side** (`runs/20261007-001447_compare-variants/comparison.md`; paired patient bootstrap `runs/20261007-001536_bootstrap-stretch-vs-lb`, 2,000 resamples, same val studies):

  | stage | stretch macro AUROC | letterbox | Δ [95 % CI] | ECE stretch → letterbox |
  |---|---|---|---|---|
  | Stage 1 | 0.8240 | 0.8305 | +0.0065 [−0.0003, +0.0125] | 0.0229 → 0.0242 (n.s.) |
  | Stage 2 | 0.8293 | 0.8350 | **+0.0057 [+0.0002, +0.0109]** | 0.0233 → 0.0211 (n.s.) |
  | Stage 3 (pruned k6) | 0.8244 | 0.8306 | **+0.0062 [+0.0002, +0.0127]** | 0.0254 → 0.0244 (n.s.) |
  | Stage 5 (emb + nb10) | 0.8290 | 0.8315 | +0.0025 [−0.0033, +0.0079] | 0.0239 → 0.0214 (n.s.) |
  | Stage 3 + nb10 | — | 0.8331 | | 0.0251 |
  | Stage 3 + Stage 4 regions | — | 0.8341 | | 0.0239 |
  | Stage 4 MS-CXR IoU≥0.1 (A / B) | 0.795 / 0.826 | 0.797 / 0.838 | | |
  | Stage 4 MS-CXR side (A / B) | 0.846 / 0.853 | 0.846 / 0.867 | | |

  Per-finding differences whose CI excludes zero are consistent across stages: cardiomegaly +0.008 to +0.013, enlarged cardiomediastinum +0.007 to +0.011, any_abnormality +0.004 to +0.006, support_devices +0.004 to +0.006 (all stages); atelectasis +0.007 (S2, S3); effusion +0.005 (S1); edema +0.006 (S5). Only loss: fracture −0.037 (S2) and −0.041 (S5), 17 positives. **Reading:** letterbox helps most where aspect ratio matters (heart/mediastinal size), as expected from the stretch distortion.
- **External feature cache** (`runs/20261007-002319_ext-features`, caching only: no labels read, no metrics): letterbox, percentile windowing 0.5–99.5. VinDr-CXR test 3,000 images (CheXmask 3,000; RCA ≥ 0.7: 2,985; 15 with region features missing); PadChest-GR 4,555 images (CheXmask 4,310; RCA ≥ 0.7: 4,307; 248 missing = 245 without masks + 3 low RCA). No images dropped, size mismatches 0. Outputs `data/features/regions/ext_{vindr_test,padchest_gr}_lb_shards` (45 MB + 67 MB). Tested first on 40 images per set (`…001135_ext-features-test`).
- **Overnight orchestrator** (`runs/20261006-2009_overnight-letterbox`) finished 00:31. Its "skipped: Stage 4 or MS-CXR" line refers to the first lb-stage4 attempt; both were rerun successfully by hand (`gpu_chain_rerun.sh`), and the Stage 4 ablation and its bootstrap were run by `ablation_s4.sh`. Nothing was skipped in the end. Free space at the end 19.2 GB; nothing deleted. Stretch region features (`mimic_full_shards`, 15 GB) kept pending the user's decision.

### Stretch archive before deletion (2026-10-07)
`archive/stretch_v1/` (0.615 GB, 38 files, sha256 in `MANIFEST.txt`; check run `runs/20261007-*_verify-stretch-archive`). Predictions for every study in every split, test included (predictions only, no test metric computed or viewed), plus trained models:
- stage1, stage2, stage3 (pruned k6, incl. side-head probabilities), stage5 (emb + nb10): `predictions_all_splits.parquet` (218,139 studies; test 3,041) regenerated from each run's `model.joblib` by `scripts/export_predictions.py`; non-test rows reproduce the original predictions (max |diff| 1.8e-7 to 5.7e-6).
- stage4_k1 (original Stage 4, `20261006-1709`) and stage4_crossfit2 (`20261006-202854`): per image × region × finding scores (218,187 images; test 3,042) and fold models. The original runs did not save weights; reran with `--save-models` (`20261007-055844_stage4-stretch-{k1,crossfit}-save`), scores identical to the originals (same index and NaN pattern, mean |diff| 0, max one float16 step).
- **Deleted 2026-10-07 (user confirmed "yes, delete")**, 33 GB: `data/features/regions/mimic_full_shards/`, `mimic_full_weights.npy`, `data/index/` (stretch retrieval index), pilot outputs (`mimic_pilot_*` except `mimic_pilot_ids.parquet`), test-mode caches (`ext_*_lb_test_shards`, `exttest_*`), CheXmask `OriginalResolution/{CheXpert,ChestX-Ray8}.csv` and `Preprocessed/{CheXpert,Padchest,VinDr-CXR}.csv`. Kept: stretch embeddings, both CheXmask MIMIC files, CheXmask OriginalResolution VinDr and PadChest, external caches, PadChest-GR prior studies, VinDr train, all runs and results. Stretch Stage 4 can no longer be recomputed from region features; its scores and models are in `archive/stretch_v1/`. Free space after deletion: 53.5 GB (was 18.1 GB).

### Leakage check: Stage 4 and retrieval features for fit-split studies (2026-10-07)
- **Stage 4 → Stage 3 ablation** (`lb-stage3-s4` read `NESY_S4_RUN=runs/20261006-224243_lb-stage4`, config `crossfit: 2`): train-split images are scored by the fold model that did not see them (2 folds by hash of subject_id, so patient-level); calib/thresh/val/test images get the mean of the two fold models, neither trained on them. Out-of-fold by construction.
- **Retrieval nb features** (`retrieval_features.py`, letterbox run `20261006-224123_lb-retrieval`): FAISS index on train only; for every query (train included) all studies of the query's own patient are excluded from its neighbours, so a train study's own label never enters its feature (leave-one-patient-out). Logged own-patient neighbours after exclusion: 0 (both stretch and letterbox runs).
- **Empirical check** (`runs/20261007-oof-check/`, train vs val only): AUROC of each feature alone against the study label, train minus val. nb10: mean +0.007 (range −0.029 to +0.079). s4 max features: mean +0.027, but ≤ 0.018 for every finding except lung_lesion (+0.058/+0.069; 64 val positives) and pleural_other (+0.184/+0.155; 14 val positives), where val AUROC is very noisy. No sign of in-sample leakage for the common findings; pleural_other is unresolved at this val size. Minor mismatch: train scores come from one half-data fold model, other splits from the average of two (val spread 0.974 vs train 0.992 in standardised units).

### Lane C: DenseNet-121 baseline (2026-10-07; val; ImageNet init, fine-tuned on the MIMIC fit split; same one-frontal-per-study images, splits and labels as the pipeline, see DATA.md)
- Image cache `data/image_cache_256/` (letterbox 256 px, compressed uint8 shards): 218,139 studies, 8.9 GB (`runs/20261007-063539_image-cache-p{0,1}`).
- Run `runs/20261007-071846_densenet121` (L40S; ImageNet weights `models/densenet121-a639ec97.pth`; 14 sigmoid outputs; masked BCE; 224 random crop, no flip; bf16; 8 epochs, OneCycle lr 1e-4, batch 96; ~2.9 min/epoch). Trained on the fit split minus its 5% inner holdout (178,839 studies); **epoch chosen on the inner holdout (8,915 studies): epoch 4** (inner macro AUROC 0.8365). Platt on calib. Predictions for all splits incl. test saved (no test metric).
  (A first run that trained on the inner holdout, `…070920_densenet121`, was stopped after 2 epochs when the selection rule changed; not used.)
- **Main (inner-holdout epoch 4), val:** macro AUROC 0.8175, AUPRC 0.480, ECE 0.023; PA 0.843 / AP 0.798; hierarchy violations 164 (independent sigmoids; largest: isa_006 cardiomegaly > enlarged_cardiomediastinum 90, isa_101 lung_opacity > any_abnormality 33, isa_005 pneumonia > consolidation 23; per-edge table below. Correction 2026-10-07: a chat message at 07:53 mislabelled the first two edges as pneumonia > consolidation (90) and cardiomegaly > enlarged cardiomediastinum (33); recounted from the predictions). Letterbox Stage 1 for reference: 0.8305 (PA 0.855 / AP 0.808).
- Per finding vs letterbox Stage 1: lower on 13 of 14 (lung_lesion −0.056, consolidation −0.034, fracture −0.020, pneumothorax −0.018), higher on pleural_other (+0.021, 14 positives).
- **Optimistic upper bound (epoch chosen on validate, epoch 3; labelled optimistic):** macro AUROC 0.8199, ECE 0.022, 142 violations.
- Patient bootstrap against Stage 1 and the final head: pending (lane C waits for the final head).

### Lane A items 1-2: final head decision (2026-10-07 08:13; `runs/20261007-081332_decide-head`; val, seen by backbone; patient bootstrap 2,000)
Rules (user, revised 07:50 before the step ran): CheXmask anatomy (A) and region features (R, CheXmask set B) no-harm (dropped only if removing them is significantly better); retrieval and association links must have a CI excluding 0. Bootstraps: `runs/20261007-080304_a1-bootstrap`, `runs/20261007-080839_a2-bootstrap`. Head runs: C `…063123_a1-s3-nocm`, C+A+R `…065311_a1-s3-s4b`, C+R `…071353_a1-s3-s4b-nocm`, C+A+nb5 `…072433_a2-s3-nb5`, C+A+nb25 `…074437_a2-s3-nb25`.
- **CheXmask anatomy: KEEP.** With region features present, with − without: macro AUROC **+0.0023 [+0.0004, +0.0044]** (significantly better with), ECE +0.0002 [−0.0015, +0.0020], cardiomegaly +0.0015 [−0.0003, +0.0037], ECM −0.0002 [−0.0018, +0.0015]. Without region features: +0.0016 [−0.0005, +0.0040] n.s.
- **Region features (set B): KEEP** (no harm): macro AUROC +0.0038 [−0.0019, +0.0108], ECE −0.0008 [−0.0040, +0.0016].
- **Retrieval: KEEP, k = 25.** Stage 3 + nbK vs Stage 3 (macro AUROC): k=5 +0.0028 [+0.0004, +0.0051]; k=10 +0.0026 [−0.0004, +0.0059]; k=25 **+0.0038 [+0.0005, +0.0080]**; ECE n.s. for all. Caveats: k was chosen on validate among three values (mild optimism); retrieval was tested on the base without region features, so the final head is also compared against C+A+R (`runs/*_lane-a5`, pending).
- **Final head: concepts (77) + CheXmask anatomy + region scores (set B) + retrieval k=25**, trained as `runs/20261007-081339_final-head` (lane A2).
- **DenseNet evaluation crop (confirmed in code):** evaluation uses the **centre** 224×224 crop of the 256-px letterbox, offsets (16, 16) (`scripts/densenet_train.py: batches_to_tensor(train=False)`; external predictions `scripts/densenet_external.py` slice `[16:240, 16:240]`); training uses random 224 crops (offsets 0–32), no flip.

### Hierarchy violations, validate (same counting for every model; `runs/*_hierarchy-violations`)
Per is_a edge, studies with P(child) > P(parent); total over edges; studies with ≥ 1 violation (1,733 val studies):

| edge | DenseNet-121 (main) | linear probe (Stage 1, letterbox) | interpretable head (Stage 3 factorised, letterbox) |
|---|---|---|---|
| isa_001 atelectasis < lung_opacity | 1 | 34 | 0 |
| isa_002 consolidation < lung_opacity | 0 | 1 | 0 |
| isa_003 edema < lung_opacity | 6 | 11 | 0 |
| isa_004 lung_lesion < lung_opacity | 0 | 0 | 0 |
| isa_005 pneumonia < consolidation | 23 | 72 | 0 |
| isa_006 cardiomegaly < enlarged_cardiomediastinum | 90 | 166 | 0 |
| isa_101 lung_opacity < any_abnormality | 33 | 80 | 0 |
| isa_102 enlarged_cardiomediastinum < any_abnormality | 2 | 14 | 0 |
| isa_103 pleural_effusion < any_abnormality | 8 | 14 | 0 |
| isa_104 pleural_other < any_abnormality | 0 | 0 | 0 |
| isa_105 pneumothorax < any_abnormality | 1 | 0 | 0 |
| isa_106 fracture < any_abnormality | 0 | 0 | 0 |
| **total over edges** | **164** | **392** | **0** |
| **studies with ≥ 1 violation** | **157** | **362** | **0** |
The factorised head multiplies P(child | parent) down the tree, so 0 holds by construction; the final head is the same construction (added when its run is done).

### Report-draft points 7 and 8 (written automatically by `scripts/results_7_8.py`, run `20261007-084333_results-7-8`; val, seen by backbone)
Final head run: `runs/20261007-081339_final-head` (concepts + CheXmask anatomy + region set B + retrieval k=25).
- **7. Final head:** macro AUROC 0.8362, **AUPRC 0.4946**, ECE 0.0242; **AUROC PA 0.8652 / AP 0.8161** (n PA 661, AP 1072).
  Per finding (AUROC all / PA / AP, AUPRC all):

```
view                          all     PA     AP  auprc_all
finding                                                   
any_abnormality             0.888  0.904  0.827      0.913
atelectasis                 0.827  0.880  0.783      0.534
cardiomegaly                0.828  0.907  0.757      0.522
consolidation               0.810  0.838  0.783      0.415
edema                       0.911  0.969  0.859      0.620
enlarged_cardiomediastinum  0.823  0.895  0.750      0.556
fracture                    0.685  0.562  0.963      0.034
lung_lesion                 0.757  0.847  0.732      0.152
lung_opacity                0.862  0.877  0.807      0.852
pleural_effusion            0.922  0.965  0.885      0.781
pleural_other               0.795  0.837  0.760      0.026
pneumonia                   0.778  0.823  0.745      0.295
pneumothorax                0.889  0.921  0.880      0.404
support_devices             0.929  0.887  0.894      0.821
```

- **Final head hierarchy violations (val):** per edge {'isa_001': 0, 'isa_002': 0, 'isa_003': 0, 'isa_004': 0, 'isa_005': 0, 'isa_006': 0, 'isa_101': 0, 'isa_102': 0, 'isa_103': 0, 'isa_104': 0, 'isa_105': 0, 'isa_106': 0}; total 0; studies with ≥ 1 violation 0.
- **8. Final head − DenseNet (main, inner-holdout epoch), patient bootstrap 2,000 (`20261007-084025_c-bootstrap`):** AUROC per finding:

                              diff  ci_low  ci_high  excludes_zero
finding                                                           
any_abnormality             0.0110  0.0026   0.0186           True
lung_opacity                0.0110  0.0021   0.0206           True
atelectasis                 0.0066 -0.0080   0.0198          False
consolidation               0.0360  0.0157   0.0568           True
pneumonia                   0.0055 -0.0277   0.0364          False
edema                       0.0076 -0.0029   0.0189          False
lung_lesion                 0.0585  0.0199   0.1098           True
enlarged_cardiomediastinum  0.0047 -0.0105   0.0182          False
cardiomegaly                0.0101 -0.0050   0.0243          False
pleural_effusion            0.0075  0.0003   0.0146           True
pleural_other               0.0172 -0.0435   0.0846          False
pneumothorax                0.0268 -0.0218   0.0703          False
fracture                    0.0443 -0.1021   0.1692          False
support_devices             0.0148  0.0041   0.0270           True
MACRO                       0.0187  0.0065   0.0303           True

  ECE (negative = final head better calibrated):

                              diff  ci_low  ci_high  excludes_zero
finding                                                           
any_abnormality            -0.0045 -0.0182   0.0113          False
lung_opacity                0.0041 -0.0229   0.0204          False
atelectasis                -0.0002 -0.0157   0.0186          False
consolidation               0.0163 -0.0019   0.0255          False
pneumonia                   0.0102 -0.0048   0.0218          False
edema                       0.0109 -0.0087   0.0204          False
lung_lesion                -0.0046 -0.0077   0.0057          False
enlarged_cardiomediastinum -0.0134 -0.0243   0.0063          False
cardiomegaly               -0.0081 -0.0229   0.0131          False
pleural_effusion           -0.0009 -0.0186   0.0141          False
pleural_other               0.0064  0.0005   0.0109           True
pneumothorax               -0.0027 -0.0079   0.0082          False
fracture                    0.0004 -0.0015   0.0038          False
support_devices             0.0011 -0.0154   0.0133          False
MACRO                       0.0011 -0.0037   0.0051          False


### Lanes A2–A5 results (2026-10-07 morning; val, seen by backbone; patient bootstrap 2,000)
- **Final head** (`runs/20261007-081339_final-head`: 77 concepts + CheXmask anatomy + region set B + retrieval k=25): macro AUROC **0.8362** (+0.0056 vs letterbox Stage 3), ECE 0.0242, **0 hierarchy violations**. Final head − Stage 1: AUROC +0.0057 [−0.0024, +0.0145], ECE 0.0000 n.s.
- **Retrieval re-check with region features present** (`runs/20261007-084002_final-vs-car`): final head − C+A+R = AUROC **+0.0018 [−0.0006, +0.0043], n.s.**; ECE +0.0006 n.s. Under the accuracy-only rule (CI must exclude 0), retrieval does **not** earn its place once region features are in the head. **Decision pending (user):** drop retrieval → final head = C+A+R (`runs/20261007-065311_a1-s3-s4b`).
- **Item 4, out-of-fold check on calib+thresh pooled** (`…a4-oof-calib-thresh`): 22 region "max" features (sets A and B); train AUROC inside the calib+thresh 95% CI for 21/22, **above it for 0/22**; mean train − compare gap −0.0052. No sign of in-sample leakage.
- **Item 5, knowledge rules** (`…a5-rules`; region thresholds fitted on thresh): **R1 localisation support DROPPED**: present-tier precision 0.788 → 0.791, +0.0031 [−0.0015, +0.0078]; sensitivity 0.378 → 0.361; 63 demoted (46 true positives). **R2 side agreement KEPT**: side accuracy among stated sides 0.897 → 0.910, **+0.0134 [+0.0022, +0.0257]**; share of stated lateralisable findings that keep a side 39.5% → 29.4% (cost: fewer sides stated).
- **Item 7, association links** (`…a7-aw-head`, bootstrap `…a7-bootstrap`): final head + out-of-fold Stage 1 logits of KG partners − final head = AUROC −0.0002 [−0.0010, +0.0004], ECE −0.0008 n.s. **Negative result: dropped.**
- **Item A, MLP head ablation** (`runs/20261007-0817_lane-a3-h100`, bootstrap `last_mlp-bootstrap`), against the linear final head. Main (early stopping on the inner holdout): hidden 64 AUROC −0.0054 [−0.0120, +0.0011], ECE −0.0018 n.s.; hidden 256 −0.0040 [−0.0097, +0.0012], ECE −0.0018 n.s. Optimistic upper bound (early stopping on validate): hidden 64 −0.0044, hidden 256 −0.0055 (both n.s.). 0 hierarchy violations (factorised). **The MLP does not beat the linear head**; the linear head stays.
- **Item B, case-based evidence** (`runs/20261007-084848_case-eval`; k = 5, IDs and labels only, no probability or band changed): similar cases agree with the model's band in **88.0%** of 12,616 non-silent finding-study pairs. Present calls with ≤ 1 of 5 similar cases positive (46 calls) are wrong 23.9% of the time vs 21.0% for the other 1,734: +0.029 [−0.088, +0.152], n.s. **Not enough evidence that a disagreement rule would help**; per-finding tables in the run dir.
- **Input-group table** (`runs/20261007-0753_lane-a4-l40`, bootstrap vs C+A+R; head macro AUROC differences): C −0.0054 [−0.0131, +0.0003]; **A −0.1715 [−0.1974, −0.1430]**; **R −0.0161 [−0.0276, −0.0048]**; C+A −0.0038 [−0.0108, +0.0019]; **C+R −0.0023 [−0.0044, −0.0004]**; **A+R −0.0153 [−0.0264, −0.0044]**; ECE n.s. for all. Coverage: R (set B) has region classifiers for 10 findings: none for any_abnormality and support_devices (no region classifier at all), enlarged_cardiomediastinum and fracture (0 explicit positive CheXmask regions in train; `runs/20261006-224243_lb-stage4`), A is anatomy geometry relevant mainly to cardiomegaly/ECM/effusion; all heads still output all 14 findings (hierarchy, 0 violations).
- **DenseNet vs final head** (`last_c-bootstrap` in `runs/20261007-0817_lane-c-post`): DenseNet − final head AUROC **−0.0187 [−0.0303, −0.0065]**; DenseNet − Stage 1 **−0.0130 [−0.0250, −0.0010]**; ECE n.s. Per-finding table under "Report-draft points 7 and 8".

### Decisions (user, 2026-10-07 ~09:20)
1. **Retrieval dropped from the head**; final head = **C+A+R** (`runs/20261007-065311_a1-s3-s4b`). Retrieval kept only as case-based evidence (context, no effect on probabilities or bands). Reason: with region features present it adds +0.0018 [−0.0006, +0.0043] (n.s.).
2. The Stage 8 run on the earlier head **with retrieval** (`runs/20261007-084341_stage8-val`) finishes and is kept, labelled "head with retrieval".
3. Rerun on C+A+R, one orchestrator `runs/20261007-0925_lane-final` (rules, case evidence, per-view metrics, hierarchy violations, DenseNet bootstraps, MLP comparison, Stage 8 on all validate studies); its results are appended below automatically by `results_final.py`.
4. **R2 (side agreement) kept; R1 (localisation support) and the association links dropped.**
5. Task 3: the hand-written may_occur_in table is kept, every row marked `source: author-curated, needs review` / `review: needed` (Chest ImaGenome has no finding-to-region validity table; no empirical table built).
6. External tests: not yet.

### Final head = C+A+R (user decision 2026-10-07; written automatically by `scripts/results_final.py`, run `20261007-101645_results-final`; val, seen by backbone)
Head run `runs/20261007-065311_a1-s3-s4b`: 77 concepts + 17 CheXmask anatomy features + region scores (CheXmask set B); calibration (Platt) on calib and four-band thresholds on thresh were fitted inside that run. Retrieval is used only as case-based evidence (k = 5, context only). R2 kept, R1 and association links dropped (user decision).
- **Head metrics:** macro AUROC 0.8344, AUPRC 0.4911, ECE 0.0236; AUROC PA 0.8640 / AP 0.8129. Hierarchy violations: total {'HEAD': 0, 'S1': 392, 'DenseNet': 164}, studies with ≥ 1 {'HEAD': 0, 'S1': 362, 'DenseNet': 157}.
  Per finding AUROC (all / PA / AP):
```
view                          all     PA     AP
finding                                        
any_abnormality             0.883  0.898  0.820
atelectasis                 0.826  0.880  0.780
cardiomegaly                0.827  0.905  0.757
consolidation               0.809  0.834  0.783
edema                       0.910  0.969  0.857
enlarged_cardiomediastinum  0.823  0.893  0.750
fracture                    0.678  0.566  0.955
lung_lesion                 0.757  0.848  0.728
lung_opacity                0.856  0.871  0.799
pleural_effusion            0.921  0.963  0.884
pleural_other               0.790  0.834  0.750
pneumonia                   0.778  0.818  0.748
pneumothorax                0.896  0.934  0.879
support_devices             0.927  0.881  0.891
```
- **Rules on this head** (`runs/20261007-093037_rules`; region thresholds on thresh): R1 precision 0.788 → 0.796, CI [0.0028, 0.0131], sensitivity 0.362 → 0.346, demoted 66; R2 side accuracy 0.898 → 0.907, CI [-0.0022, 0.0212], share keeping a side 0.394 → 0.291. Applied: R2 on, R1 off (user decision).
- **MLP head vs linear C+A+R head** (main = early stopping on the inner holdout; `es-val` = optimistic upper bound):
  MLP64inner - LIN MACRO AUROC: -0.0057 [-0.0134, +0.0018]
  MLP64inner - LIN MACRO ECE: -0.0016 [-0.0043, +0.0013]
  MLP256inner - LIN MACRO AUROC: -0.0009 [-0.0074, +0.0051]
  MLP256inner - LIN MACRO ECE: +0.0012 [-0.0019, +0.0040]
  MLP64val - LIN MACRO AUROC: -0.0048 [-0.0128, +0.0031]
  MLP64val - LIN MACRO ECE: -0.0004 [-0.0034, +0.0021]
  MLP256val - LIN MACRO AUROC: -0.0036 [-0.0115, +0.0046]
  MLP256val - LIN MACRO ECE: -0.0012 [-0.0036, +0.0019]
- **DenseNet-121 vs C+A+R head and Stage 1** (main = inner-holdout epoch):
  HEAD - S1 MACRO AUROC: +0.0039 [-0.0049, +0.0124]
  HEAD - S1 MACRO ECE: -0.0006 [-0.0032, +0.0029]
  DenseNet - S1 MACRO AUROC: -0.0130 [-0.0250, -0.0010] (CI excludes 0)
  DenseNet - S1 MACRO ECE: -0.0011 [-0.0048, +0.0035]
  DenseNet - HEAD MACRO AUROC: -0.0169 [-0.0288, -0.0048] (CI excludes 0)
  DenseNet - HEAD MACRO ECE: -0.0004 [-0.0045, +0.0040]
  Per finding, DenseNet − HEAD AUROC:
  DenseNet - HEAD any_abnormality AUROC: -0.0056 [-0.0132, +0.0026]
  DenseNet - HEAD lung_opacity AUROC: -0.0052 [-0.0151, +0.0037]
  DenseNet - HEAD atelectasis AUROC: -0.0054 [-0.0187, +0.0097]
  DenseNet - HEAD consolidation AUROC: -0.0344 [-0.0548, -0.0146] (CI excludes 0)
  DenseNet - HEAD pneumonia AUROC: -0.0054 [-0.0363, +0.0274]
  DenseNet - HEAD edema AUROC: -0.0065 [-0.0180, +0.0045]
  DenseNet - HEAD lung_lesion AUROC: -0.0584 [-0.1069, -0.0235] (CI excludes 0)
  DenseNet - HEAD enlarged_cardiomediastinum AUROC: -0.0045 [-0.0185, +0.0103]
  DenseNet - HEAD cardiomegaly AUROC: -0.0093 [-0.0240, +0.0056]
  DenseNet - HEAD pleural_effusion AUROC: -0.0066 [-0.0139, +0.0010]
  DenseNet - HEAD pleural_other AUROC: -0.0121 [-0.0746, +0.0462]
  DenseNet - HEAD pneumothorax AUROC: -0.0333 [-0.0711, +0.0015]
  DenseNet - HEAD fracture AUROC: -0.0368 [-0.1608, +0.1075]
  DenseNet - HEAD support_devices AUROC: -0.0128 [-0.0258, -0.0002] (CI excludes 0)
  DenseNet - HEAD MACRO AUROC: -0.0169 [-0.0288, -0.0048] (CI excludes 0)
  DenseNet - HEAD any_abnormality ECE: +0.0021 [-0.0116, +0.0186]
  DenseNet - HEAD lung_opacity ECE: -0.0040 [-0.0178, +0.0213]
  DenseNet - HEAD atelectasis ECE: -0.0011 [-0.0209, +0.0180]
  DenseNet - HEAD consolidation ECE: -0.0141 [-0.0242, +0.0028]
  DenseNet - HEAD pneumonia ECE: -0.0059 [-0.0185, +0.0066]
  DenseNet - HEAD edema ECE: -0.0058 [-0.0164, +0.0101]
  DenseNet - HEAD lung_lesion ECE: +0.0042 [-0.0038, +0.0075]
  DenseNet - HEAD enlarged_cardiomediastinum ECE: +0.0144 [-0.0064, +0.0244]
  DenseNet - HEAD cardiomegaly ECE: +0.0096 [-0.0132, +0.0226]
  DenseNet - HEAD pleural_effusion ECE: +0.0050 [-0.0137, +0.0215]
  DenseNet - HEAD pleural_other ECE: -0.0064 [-0.0109, -0.0003] (CI excludes 0)
  DenseNet - HEAD pneumothorax ECE: +0.0037 [-0.0073, +0.0092]
  DenseNet - HEAD fracture ECE: -0.0004 [-0.0030, +0.0023]
  DenseNet - HEAD support_devices ECE: -0.0075 [-0.0174, +0.0117]
  DenseNet - HEAD MACRO ECE: -0.0004 [-0.0045, +0.0040]
  Optimistic DenseNet (epoch chosen on validate):
  HEAD - S1 MACRO AUROC: +0.0039 [-0.0049, +0.0124]
  HEAD - S1 MACRO ECE: -0.0006 [-0.0032, +0.0029]
  DenseNetOPTIMISTIC - S1 MACRO AUROC: -0.0106 [-0.0209, +0.0013]
  DenseNetOPTIMISTIC - S1 MACRO ECE: -0.0019 [-0.0047, +0.0028]
  DenseNetOPTIMISTIC - HEAD MACRO AUROC: -0.0145 [-0.0263, -0.0029] (CI excludes 0)
  DenseNetOPTIMISTIC - HEAD MACRO ECE: -0.0013 [-0.0045, +0.0028]
- **Case evidence on this head:** agreement 0.877; present calls with ≤ 1 of 5 similar positive: error 0.224 (n 49) vs 0.208 (n 1633), diff +0.017 [-0.110, +0.159].
- **Stage 8, C+A+R head (final)** (`runs/20261007-093237_stage8-car`): n 1733; first-pass match 0.875; regeneration rate 0.125; template fallback 0.125; final LLM match 0.875; template round-trip (comparator check) 0.962; LLM calls 2411, errors 0 (status counts {'429': 246}).
  Mismatch kinds, first attempt: {'omitted': 199, 'side': 42, 'polarity': 2, 'zone': 1}; all attempts: {'omitted': 599, 'side': 126, 'polarity': 6, 'zone': 3}; template: {'omitted': 65, 'side': 1}. Most common first-attempt mismatches: ['omitted:support_devices=118', 'omitted:enlarged_cardiomediastinum=77', 'side:atelectasis=16', 'side:edema=11', 'side:lung_opacity=6', 'side:pleural_effusion=5', 'side:cardiomegaly=2', 'polarity:edema=2', 'omitted:pleural_effusion=2', 'side:support_devices=1'].
  Three sample reports with belief graphs: `runs/20261007-093237_stage8-car/samples.md`.
- **Stage 8, head with retrieval (earlier, kept for the record)** (`runs/20261007-084341_stage8-val`): n 1733; first-pass match 0.863; regeneration rate 0.137; template fallback 0.137; final LLM match 0.863; template round-trip (comparator check) 0.966; LLM calls 2486, errors 0 (status counts {'429': 277}).
  Mismatch kinds, first attempt: {'omitted': 221, 'side': 38, 'polarity': 2, 'certainty': 1}; all attempts: {'omitted': 663, 'side': 112, 'polarity': 6, 'certainty': 3}; template: {'omitted': 57, 'side': 2}. Most common first-attempt mismatches: ['omitted:support_devices=140', 'omitted:enlarged_cardiomediastinum=71', 'side:atelectasis=16', 'side:edema=8', 'side:pleural_effusion=6', 'omitted:pleural_effusion=5', 'side:lung_opacity=4', 'omitted:edema=3', 'polarity:edema=2', 'side:consolidation=2'].
  Three sample reports with belief graphs: `runs/20261007-084341_stage8-val/samples.md`.
- **Notes on the C+A+R results above (2026-10-07 10:30):**
  - On the C+A+R head the CI rule gives the **opposite of the earlier decision**: R1 (localisation support) precision +0.008 [+0.0028, +0.0131] (CI excludes 0; sensitivity 0.362 → 0.346, 66 demotions), R2 (side agreement) +0.009 [−0.0022, +0.0212] (n.s.). Applied as decided by the user (R2 on, R1 off); flagged for the user.
  - C+A+R head vs Stage 1: +0.0039 [−0.0049, +0.0124], n.s.; DenseNet − C+A+R head −0.0169 [−0.0288, −0.0048] (significant); optimistic DenseNet − head −0.0145 [−0.0263, −0.0029]. MLP: no variant differs significantly from the linear head (best: 256 units inner-holdout −0.0009 [−0.0074, +0.0051]).
  - Stage 8 on C+A+R: 87.5% of reports matched the belief graph on the first attempt and 12.5% fell back to the template. The regenerations rescued none (first-pass and final match rates are identical), so the feedback loop is not effective as written. Mismatches are mostly omissions (support devices 118, enlarged cardiomediastinum 77 on the first attempt) and sides (42). The template itself round-trips 96.2% (65 template omissions = comparator/RadGraph limits), so part of the "omitted" count is the parser, not the LLM. Head with retrieval: 86.3% first-pass, 13.7% fallback. No LLM errors after retries (HTTP 429 rate limits retried 246 times).

### Knowledge-base tasks 1–6 (2026-10-07; see `KG.md`)
- **Task 1, RadLex IDs** (`kg/radlex_map.yaml`, `kg/devices.yaml`, `runs/*_build-radlex-map`; candidates `runs/*_radlex-candidates`): findings 7 exact / 5 approximate / 2 none (cardiomegaly, enlarged cardiomediastinum); report zones 3 exact (upper RID1348, mid RID1351, lower RID1354) / 1 approximate (apical → apex of lung); anatomy nodes 30 exact / 14 approximate / 4 none (newly matched: the six sided lung zones, RID1349–1356); device types 7 exact / 6 approximate / 4 none (chest port, sternotomy wires, CABG grafts, aortic graft). All IDs checked present and non-obsolete in RadLex 4.3; no match forced. Validated on load (2 new tests; 51 pass).
- **Task 2, hierarchy comparison** (`runs/*_compare-hierarchy/hierarchy_comparison.csv`): 29 edges (12 finding is_a + 17 device types) → **13 agree, 10 differ, 6 not comparable**. All 13 comparable device edges agree (tubes, catheters, valve, pacemaker under RadLex "medical device"). All 10 comparable finding edges differ: our CheXpert hierarchy is appearance-based (lung opacity as the parent of atelectasis, consolidation, edema, lung lesion; pneumonia under consolidation), RadLex is mechanism-based (atelectasis < collapse < architectural distortion; consolidation < displaced substance; edema < fluid disorder; pneumonia and pneumothorax < respiratory disorder; RadLex "opacity" < imaging observation; RadLex "abnormal" is a descriptor, so no root edge can agree). Not comparable: cardiomegaly < enlarged cardiomediastinum, enlarged cardiomediastinum < root, and 4 devices without RadLex classes. Our hierarchy unchanged.
- **Task 3**: Chest ImaGenome has no finding-to-region validity table (semantics/ has attribute categories, UMLS mapping and object lists; annotation_utils/ has image-only PDFs and a 500-row agreement sample). Hand-written may_occur_in kept, all 14 rows marked needs review (user decision).
- **Task 4, Fleischner glossary** (`findings.yaml: glossary`; headwords and pages `runs/20261007-0820_glossary-text`): 8 of 14 findings matched (lung opacity → Opacity p9; atelectasis p3; consolidation p5; pneumonia p11; lung lesion → Nodule p9 / Mass p8; pleural effusion → Pleura: Effusion p10; pleural other → Pleura: Thickening / Plaque p11, Apical Cap p3; pneumothorax p11). No term: any_abnormality, edema, enlarged cardiomediastinum, cardiomegaly, fracture, support devices. Differences in use: lung opacity is a category for us but a non-specific descriptor in the glossary; pneumonia is a diagnosis that can show as any opacity, not only consolidation (our is_a edge is narrower); lung lesion merges nodule (≤ 30 mm) and mass (> 30 mm); pleural other lumps thickening, plaque and scarring. Paraphrases are in our own words.
- **Task 5, RadReport template** (`kg/proposals/radreport_comparison.md`): the template has anatomical subsections, section-level normal sentences, severity grades, vascular congestion, aorta/spine findings, device tip positions and interval change; we add certainty tiers, side/zone, positive pneumothorax/lesion/fracture/pleural-thickening sentences. Proposals (layout by subsection, procedure line, "The lungs are clear." / "The mediastinal contours are normal." pertinent negatives; no osseous normal sentence) **await approval; not applied**.
- **Task 6, sources** (`runs/20261007-1056_add-sources/source_counts.csv`): 255 rows across 13 tables, all with a source: **RadLex ID 130; author-curated, needs review 67; Chest ImaGenome 21; CheXpert label hierarchy 19; Fleischner glossary 8; RadReport template 4; no external source (no matching glossary term) 6.** (The 13 ImaGenome-map and 28 external-map rows name their label source but the mappings themselves are author-curated drafts.)

### DenseNet-121 external predictions (2026-10-07; `runs/20261007-081007_densenet-external`)
Inner-holdout epoch with its Platt calibration; letterbox 256 → centre 224 crop; external DICOMs windowed at the 0.5–99.5 percentiles (MONOCHROME1 inverted). VinDr-CXR test 3,000/3,000 and PadChest-GR 4,555/4,555 images predicted, 0 unreadable. Manifests read with image columns only: **no labels read, no metric computed**. For the final external comparison (on hold).

### KG task 7 — glossary-derived concepts, ABLATION ONLY (2026-10-07; lane `runs/20261007-1047_lane-kg7`; val, seen by backbone)
- Phrases: `data/concepts/glossary_phrases.csv` (52 short phrases written from the Fleischner glossary terms for 9 findings; no glossary wording for cardiomegaly, enlarged cardiomediastinum, fracture, support devices). Encoded with CLEAR's text encoder; check: re-encoding 277 bank sentences reproduced the stored embeddings (cosine 1.000000), so phrases are scored exactly as the 77 (`runs/20261007-104703_glossary-concepts`, letterbox).
- Fit-split AUROC (187,754 studies), own finding, 0.60 floor: **31 of 52 kept** (`runs/20261007-104703_glossary-concepts/glossary_phrases_scored.csv`). Best per finding: pleural effusion "pleural effusion" 0.877, edema "pulmonary edema" 0.872, pneumothorax 0.846, pleural other "pleural fibrosis" 0.814, lung opacity "increased lung attenuation" 0.791, lung lesion "lung mass" 0.775, atelectasis "lung collapse" 0.755, consolidation "masslike consolidation" 0.727, pneumonia "lung infection" 0.727. Below floor include "kerley lines" 0.33, "blunting of the costophrenic angle" 0.29, "air bronchograms" 0.45, "plate-like atelectasis" 0.49, "nodule" 0.55.
- Head with the 31 glossary concepts in place of the 77 (+ the same A and R; `runs/20261007-104915_kg7-s3-gloss`): macro AUROC 0.8339, AUPRC 0.4907, ECE 0.0235; 0 hierarchy violations.
- Patient bootstrap vs C+A+R (2,000; `runs/20261007-110233_kg7-bootstrap`): **GLOSS − HEAD macro AUROC −0.0004 [−0.0037, +0.0029]; macro ECE −0.0002 [−0.0017, +0.0028]** — no difference. Per finding, CI excludes 0 only for consolidation +0.0033 [+0.0007, +0.0065] and cardiomegaly −0.0027 [−0.0052, −0.0004].
- Reading: with A and R present, the concept list barely matters; 31 glossary phrases match the 77 data-driven ones. The 77 stay the main list (user instruction).


## 2026-10-07 midday — end-to-end runner, Stage 8 fixes, evaluation plan (validate only; seen by backbone)

No test or external label was read and no test or external metric computed. No test-split head predictions were generated.

### End-to-end runner (`scripts/run_pipeline.py`, config `configs/pipeline.yaml`)
- Steps: cached features → C+A+R head → calibrated marginals → four bands → belief graph (G6 side, R2, G7 zone, D2/D3) → case evidence (context only) → template → Stage 8.
- Config switches: R1 off, R2 on; report layout `current` (`radreport` raises: proposals not approved); `show_case_evidence` off; Stage 8 comparator / retry / attempts.
- Each run writes `manifest.json` (also inside `config.json`): SHA-256 of 52 files. These are the head model and calibrators, thresholds, region scores and thresholds, concept list and text embeddings, image embeddings, CheXmask features, case table, splits, every KG YAML, the configs and the code. The git field is `de121e6…+dirty` because v2 is not committed.
- Guards: `--split test` is refused without `--allow-test`; `vindr_test` / `padchest_gr` are refused without `--allow-external`, and external feature assembly is not built. The runner never loads the label file: splits come from the frozen patient split file. Tests: `tests/test_eval_v2.py` (guards, label rules, comparator v2); 59/59 tests pass.
- **Equivalence with the existing C+A+R results** (`runs/20261007-114553_pipeline-val-nostage8`, check `runs/20261007-114734_equivalence`), all 1,733 validate studies:
  - probabilities (44 columns incl. side head): same study set, max |diff| 4.3e-6 (float32 concept scores);
  - bands, sides, zones, normal calls: 0 differences;
  - templates: 0 differences against the 216 stored fallback texts, the stored expected statements and prompts, and re-rendering from the reference probabilities.
  - **PASS.**

### Stage 8 diagnosis (`runs/20261007-114307_stage8-diag`)
Why regeneration rescued 0 of 216 in run `20261007-093237_stage8-car`:
1. **Most mismatches were comparator false alarms, not LLM errors.** In 215 of 216 fallbacks the same mismatch was reported on every attempt, and in 214 the three texts were identical. The texts were correct:
   - "Support devices are in place." is parsed by RadGraph as just "in place", unlinked;
   - RadGraph drops whole short sentences such as "Possible support device." and "No focal consolidation." depending on context;
   - "Possible enlarged cardiomediastinum." failed because "cardiomediastinum" was not a mediastinum word;
   - side words in the LLM's summary sentences ("Edema, possible cardiomegaly, and bilateral pleural effusion.") were attached to every finding in the sentence.
2. **The retry did name the mismatch** ("'support devices' (present) is missing"). The model saw a correct report and repeated it: 20 of 20 retries with feedback were identical to the first attempt, although without feedback at T = 0 it reproduced its own first attempt only 14 of 20 times.
3. **The 65 template omissions** (66 template mismatches): support_devices 48 (possible 40, present 8), consolidation "No focal consolidation." 15, cardiomegaly "Heart size is normal." 2, plus 1 side. All are parser misses of the same kind.

**Fix: comparator v2** (`nesy/rg_match.claims(version=2)`; v1 unchanged and selectable):
- sentences in which RadGraph found no finding are read lexically with the same finding words;
- device and heart/mediastinum context words are taken from the sentence;
- "cardiomediastinum" is now a mediastinum word;
- in sentences that list several findings, the sentence's side, zone and hedge words are not attached to every finding;
- "No X is present." reads as absent.

Also, the retry feedback (v2, `nesy/stage8.py`) quotes the template's own sentence for the finding and forbids summary sentences.

**Tests** (`runs/20261007-115109_stage8-retry-test`, `runs/20261007-115447_stage8-repair-test`):
- **A.** 100 random validate studies that fell back before, with comparator v2: 100/100 first-pass matches, 0 retries needed, 0 fallbacks, 0 template round-trip failures. Re-scored offline on the stored parses, v2 clears 19/19 template failures and 214/216 LLM first attempts. The 2 left are a genuine summary-sentence side error and "Possible pleural other." (the prompt uses the KG label "pleural other").
- **B. Guard sensitivity** (v2 must still catch wrong text). Corrupted template reports caught by v1 / v2:
  - dropped statement 100/100 / 100/100;
  - negated statement 100/100 / 100/100;
  - added side 79/79 / 79/79.
- **C. Does a retry repair a genuine error?** The model was shown a corrupted report as its previous answer, plus the feedback. Repaired: v1 feedback 150/150 (54 dropped, 54 negated, 42 side), v2 feedback 150/150.
- **Conclusion:** the retry loop works when the error is real; the 0 rescues came from comparator false alarms.

### External inputs: CheXmask anatomy features (`runs/20261007-114214_chexmask-ext`; images and masks only, no labels)
They did not exist; computed now with the same feature function from CheXmask OriginalResolution (Preprocessed vs Original agreed ≤ 0.004 on CTR/heart shift, `runs/20261006-0922_chexmask-smoke`):
- **VinDr test:** 3,000/3,000 images, QC ok 2,985 → `data/features/chexmask_vindr_test.parquet`.
- **PadChest-GR:** 4,310/4,555, 245 without a CheXmask row (missing → `cm_missing` indicator, never zero), QC ok 4,307 → `data/features/chexmask_padchest_gr.parquet`.
- Side convention 100% on both. **View is unknown for all external images** (no DICOM ViewPosition in 300 VinDr headers checked; no projection field in PadChest-GR metadata); open item in EVAL_PLAN.md.

### H3 comparator: linear probe on CLEAR embedding + A + R (`runs/20261007-114130_h3-probe-ear`, bootstrap `runs/20261007-120331_h3-bootstrap`)
Did not exist; trained now on the fit split (C on the inner holdout, Platt on calib):
- **val macro AUROC 0.8354, AUPRC 0.4977, ECE 0.0210**, against the C+A+R head's 0.8344 / 0.4911 / 0.0236.
- PROBE − HEAD: macro AUROC +0.0010 [−0.0031, +0.0052], macro ECE −0.0026 [−0.0053, +0.0005]; no finding's CI excludes 0.
- On val the head is within the H3 margin (head − probe lower bound −0.0052 > −0.02). The test comparison is defined in EVAL_PLAN.md.

### Evaluation script (`scripts/evaluate.py`) and EVAL_PLAN.md
- Checked on val (`runs/20261007-115716_eval-dev-val`): reproduces macro AUROC 0.8344 / AUPRC 0.4911 / ECE 0.0236, 0 violations.
- Normal call: rate 13.9%, precision 0.938, sensitivity 0.366.
- Bootstrap ECE intervals are biased upward (macro ECE 0.0236 lies below its interval [0.026, 0.035]), so ECE comparisons use paired differences.
- Radiologist labels: 596 of the 687 labelled studies are in our test split with a frontal image (9 are validate studies, 82 have no frontal image); same label rules as training, tested to reproduce `labels_v2` on val exactly.

### Stage 8 rerun on validate, comparator v2 and feedback v2, up to 3 attempts (`runs/20261007-120046_pipeline-val-v2`; written automatically)
- First-pass match **0.998** (was 0.875 with v1); final LLM match **1.000** (was 0.875); fallback **0.000** (was 0.125); rescued by a retry 4; template round trip **0.999** (was 0.962).
- Remaining first-attempt mismatches: ['side:edema=3', 'omitted:pleural_other=1']; template mismatches: {'side:atelectasis': 1}.
- LLM calls 1936, errors 0, HTTP status counts {'429': 199}. Normal-call rate 0.139.
- Retry policy: kept at 3 attempts if retries rescue most first-attempt failures (see 'rescued by a retry' vs fallback); the user's rule is to cut to 1 attempt + template if they rescue few. Decision recorded in STATE.md by Claude after reading this.

### Evaluation of the runner output on validate (`runs/20261007-123526_eval-val-v2`; CheXpert-derived labels)
- Macro AUROC 0.8344 [0.8128, 0.8554], AUPRC 0.4911, ECE 0.0236; hierarchy violations 0; normal call {'rate': 0.13906520484708598, 'n_called': 241, 'precision': 0.9377593360995851, 'sensitivity': 0.36628849270664504}.
- H3 (dev, val): head − PROBE_EAR macro AUROC -0.0010 [-0.0052, +0.0031]; margin 0.02; non-inferior: True.
- Report level vs the cached validate reference parses (positive mentions of our 13 findings, comparator v2 on both): micro F1 0.563, macro F1 0.366, RadGraph entity F1 (exact tokens + label) 0.165; n 1733. Per finding: `report_level.csv` in the eval run.

### External-run prep (user request 2026-10-07; nothing run on external images, no external labels)
- Stage 4 region classifiers retrained with saved models: `runs/20261007-121534_s4-saved` (same settings as `20261006-224243_lb-stage4`; labels of train/val only).
- Retrained vs cached set-B scores: val max |diff| 0.0078, corr 1.000000; all non-fit max |diff| 0.0078; fit (out-of-fold) max |diff| 0.0156. Scores from the saved models vs the retrained run's own val scores: max |diff| 0.0039 (float16 storage).
- External feature path on MIMIC val with the cached region scores (tests the duplicated transforms): probabilities max |diff| 6.85e-06; band differences 0, side 0, zone 0, normal call 0; template differences 0; ALL PASS True.
- External feature path on MIMIC val with region scores from the saved models (the path an external run uses): probabilities max |diff| 2.47e-03; band differences 4, side 0, zone 0, normal call 0; template differences 4; ALL PASS False.
- To use: set `inputs.s4_models_run` in `configs/pipeline.yaml` to the retrained run (or pass `--s4-models-run`). External runs still need `--allow-external`, the mapping decisions and a decision on `external.view_default`.
- Stage 8 attempts: kept at 3 (retries rescued 4/4 first-attempt failures; fallback 0). Decided 2026-10-07 14:50 under the user's rule.
