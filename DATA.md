# Data inventory

Data root: `/scratch/prj/bhi_zihe_imaging/mimic_cxr_full` (abbreviated `$D` below). Inventory taken 2026-10-06.
**V** = verified by reading or counting the files; **A** = assumed.

## Free space

- Project quota `/scratch/prj/bhi_zihe_imaging`: `ceph.quota.max_bytes` = 1,000,000,000,000 B. Used (`ceph.dir.rbytes`) = 949,916,663,613 B. **Free ≈ 50 GB** (not ~100 GB as the brief assumed). **V**
- Biggest users: `mimic_cxr_full` 852 GiB, `envs` 6.5 GiB, `project` 6.3 GiB, `NeSy-rag-rrg` 3.0 GiB, `NeSy-rag` 1.8 GiB, `data` 1.6 GiB.
- Consequence: any archive extraction that leaves less than 30 GB free needs approval, so at most ~20 GB can be extracted. Neither PadChest-GR (38.5 GB) nor ImaGenome `scene_graph.zip` (10.3 GB uncompressed; fits, but tight) should be extracted. Both can be read directly from the zip.

## Internet

- Login node: yes (git, conda work). CPU compute nodes (`slam_cpu`, `cpu`): **yes**. The dataset downloads (jobs 37795190, 37809910, 37809960, 37810334) ran there over AWS S3 and rclone. **V** (sacct)
- GPU compute nodes: not tested yet (no allocation running). **A** unknown.
- Did not consult online dataset pages during this audit; published counts below are from the files' own READMEs, or are marked as from memory.

## Summary table

| Dataset | Path | Size | Complete? | Checksums |
|---|---|---|---|---|
| MIMIC-CXR-JPG 2.1.0 | `$D/mimic-cxr-jpg/mimic-cxr-jpg-2.1.0.physionet.org` | 571 GiB | Yes: 377,110 JPGs = `IMAGE_FILENAMES` = published | Not run (571 GiB); file count exact |
| MIMIC-CXR reports (free text) | `$D/mimic-cxr-reports.zip` (added by user 2026-10-06 09:35) | 142 MB zip, 152 MB text | Yes: 227,835 reports = all MIMIC studies | No checksum file; `testzip()` CRC check passes on every member |
| CheXmask v1.0.1 | `$D/CheXmask` | 38 GiB | Yes: 243,334 MIMIC frontal images in manifest; feature coverage to be counted | **12/12 OK** |
| Chest ImaGenome 1.0.0 | `$D/Chest_ImaGenome` | 6.0 GiB | Yes | **57/57 OK** |
| MS-CXR 1.1.0 | `$D/MS_CXR` | 3.5 MiB | Yes | **OK** |
| VinDr-CXR 1.0.0 | `$D/VinDr_cxr` | 192 GiB | 15,000 train + 3,000 test DICOMs = published | **18,006/18,006 OK** |
| PadChest-GR | `$D/PadChest_GR` | 45 GiB | Yes: 4,555 PNGs in archive = 4,555 studies in JSON | No checksum file; zip CRC of last member OK; 2 members decoded |
| (extra) CXRGraph 1.0.0 | `$D/cxrgraph` | 602 MiB | not checked | not run |
| (extra) CXR-PRO | `$D/cxr-pro-mimic-cxr` | 1.5 GiB | **`mimic_test_impressions.csv` fails checksum** | 1 FAILED |

---

## MIMIC-CXR-JPG 2.1.0

Layout: `files/p1X/p<subject_id>/s<study_id>/<dicom_id>.jpg` (10 top folders p10–p19). Top-level CSVs:

| File | Rows | Key columns |
|---|---|---|
| `mimic-cxr-2.0.0-metadata.csv.gz` | 377,110 | dicom_id, subject_id, study_id, ViewPosition, Rows, Columns, StudyDate, StudyTime, PerformedProcedureStepDescription, ViewCodeSequence_CodeMeaning, PatientOrientationCodeSequence_CodeMeaning |
| `mimic-cxr-2.0.0-split.csv.gz` | 377,110 | dicom_id, study_id, subject_id, split |
| `mimic-cxr-2.0.0-chexpert.csv.gz` | 227,827 | subject_id, study_id, 14 labels (1 / 0 / −1 / blank) |
| `mimic-cxr-2.0.0-negbio.csv.gz` | 227,827 | same |
| `mimic-cxr-2.1.0-test-set-labeled.csv` | 687 | study_id + 14 radiologist labels (test split) |

Counts (V): 377,110 images, 227,835 studies, 65,379 patients. These equal the published numbers; README says 227,827 labelled studies, 8 unlabelable, 2,414 with no CheXpert label.

| Official split | Images | Studies | Patients | PA/AP images | Studies with PA/AP |
|---|---|---|---|---|---|
| train | 368,960 | 222,758 | 64,586 | 237,972 | 213,365 |
| validate | 2,991 | 1,808 | 500 | 1,959 | 1,733 |
| test | 5,159 | 3,269 | 293 | 3,403 | 3,041 |

- Patients in more than one official split: **0**. Every metadata row has a split.
- ViewPosition: AP 147,173; PA 96,161; LATERAL 82,853; LL 35,133; missing 15,769; rare others 19.
- CheXpert label counts (1 / 0 / −1 / blank), e.g. Atelectasis 45,808 / 1,531 / 10,327 / 170,161; No Finding 75,455 / 0 / 0 / 152,372. Under the new rule, blank → 0.
- Join key to everything else: `dicom_id` (images), `study_id`, `subject_id`.

## MIMIC-CXR report text

- **Resolved 2026-10-06:** the user downloaded `$D/mimic-cxr-reports.zip`. It holds 227,835 `files/pXX/pSUBJECT/sSTUDY.txt` reports (one per MIMIC study) and passes `testzip()`. It is read in place; extracting it is unnecessary at 152 MB. The old pipeline's `ReportStore` reads this zip layout directly.
- Before that, the full text was not on disk: `configs/hpc.yaml` pointed at a nonexistent `/scratch/prj/bhi_zihe_imaging/data/raw/mimic-cxr-reports.zip`.
- **Substitute A:** Chest ImaGenome `silver_dataset/cxr-mimic-v2.0.0-processed-sentences_all.txt`: TSV with `subject_id, rad_id (=study_id), sent_loc, row_id, section, sentence`; 2,340,797 sentences; covers **227,835 / 227,835** studies. Sections: `finalreport` 2,058,525, `history` 223,331, `prelimread` 58,941. FINDINGS/IMPRESSION headers appear as their own sentences ("FINDINGS:", "IMPRESSION:"), so sections can be recovered. **V**
- **Substitute B:** CXRGraph `inference.json` (622 MB): model-predicted entities/relations/attributes for all 227,835 reports, keyed by `p10_p10001884_s58196907.txt`. **V** (README only)
- Not needed for labels (CheXpert CSV), but needed for retrieval facts (D), concept selection and RadGraph round-trip checks.

## CheXmask v1.0.1

- `OriginalResolution/{ChestX-Ray8, CheXpert, MIMIC-CXR-JPG, Padchest, VinDr-CXR}.csv` and `Preprocessed/{CheXpert, MIMIC-CXR-JPG, Padchest, VinDr-CXR}.csv` (1024×1024). MIMIC original 13.0 GB, preprocessed 4.4 GB; VinDr 0.95 / 0.32 GB; Padchest 4.6 / 1.9 GB.
- Columns: `<id>, Dice RCA (Mean), Dice RCA (Max), Landmarks, Left Lung, Right Lung, Heart, Height, Width`. The id column is `dicom_id` (MIMIC), `image_id` (VinDr), `ImageID` (Padchest). **V**
- Masks are RLE strings; landmarks are a flat x,y list. README says to keep masks with Dice RCA (Mean) ≥ 0.7. Published total 657,566 masks across five sources (README).
- Rows per file: the MIMIC preprocessed file has **243,334 rows = exactly the 243,334 MIMIC PA/AP images**, with no duplicates. All three masks are present for every row; 97.6 % have Dice RCA (Mean) ≥ 0.7. (Run `v2/runs/20261006-0940_chexmask-features`.)
- Derived features: `v2/data/features/chexmask_mimic.parquet`. These are named side and CTR features, not masks. `ctr_reliable_view` is False for AP views; those rows are flagged, not dropped.
- **Side convention verified on a 2,000-image sample:** the "Left Lung" mask centroid is on the image right in 100% of them, so "Left" is the patient's left. The landmarks are ordered right lung (44 points, image left), left lung (50), heart (26).
- Landmark format differs between files: a flat "x,y,…" string in Preprocessed; a numpy-printed `[[x y] …]` array in OriginalResolution.
- Preprocessed (1024²) and OriginalResolution give the same ratio features: on 100 images, median \|ΔCTR\| was 0.0004 (max 0.0034). Step 4 therefore uses Preprocessed.

## Chest ImaGenome 1.0.0

- `silver_dataset/scene_graph.zip`: 243,310 `<dicom_id>_SceneGraph.json` files, 10.26 GB uncompressed (not extracted; readable from the zip).
- `silver_dataset/study_level_attribute_rdfgraphs.json`: 4.6 GB.
- `silver_dataset/splits/{train,valid,test}.csv`: 166,521 / 23,953 / 47,393 images (subject_id, study_id, dicom_id, path, ViewPosition). **These splits ignore the official MIMIC split.** For example, ImaGenome "test" has 12,392 patients who are in official train. Do not use them.
- `silver_dataset/splits/images_to_avoid.csv`: 3,821 images of 500 patients = exactly the gold-standard patients.
- `gold_dataset/`: `gold_bbox_coordinate_annotations_1000images.csv` (25,989 rows; 1,000 images, **500 patients**, 1,000 studies), `gold_attributes_relations_500pts_500studies1st.txt` (TSV: patient_id, study_id, image_id, bbox, relation, label_name, context, …), plus per-annotator bbox CSVs and comparison relations.
- **Gold patients by official split: train 494, validate 2, test 4.** All 500 gold patients are excluded from training splits.
- `utils/cxr-study-list_v2.csv` (227,835 studies, plus gender and age decile), `utils/cxr-record-list_view.csv` (377,110 images).
- `semantics/`: object and attribute vocabularies, UMLS mapping.
- Checksums: all 57 OK.

## MS-CXR 1.1.0

- `MS_CXR_Local_Alignment_v1.1.0.csv` (1,448 box rows; dicom_id, category_name, label_text, path, x, y, w, h, image_width, image_height, split) and the same in COCO JSON (1,120 image entries, 1,448 annotations, 8 categories).
- 1,047 unique dicom_ids, all present in MIMIC-CXR-JPG; 1,160 unique (image, phrase) pairs; 851 patients (subject_id parsed from path).
- Categories: Cardiomegaly 333, Pneumothorax 264, Pneumonia 231, Consolidation 185, Pleural Effusion 142, Lung Opacity 108, Atelectasis 98, Edema 87 (box counts).
- Views: AP 794, PA 253.
- **MS-CXR patients by official split: train 752, validate 3, test 96.** All are excluded from training splits.
- MS-CXR and gold patients overlap: 16 patients.
- Published: from memory, about 1,162 image–phrase pairs (not checked online). 1,160 found.
- Checksums OK.

## VinDr-CXR 1.0.0

- `train/` 15,000 `.dicom`, `test/` 3,000 `.dicom` (published 15,000 / 3,000). **V**
- `annotations/annotations_train.csv` (69,052 box rows: image_id, rad_id, class_name, x_min, y_min, x_max, y_max; "No finding" rows have empty boxes). There are 3 radiologists per training image.
- `annotations/annotations_test.csv` (4,748 rows, no rad_id: consensus).
- `annotations/image_labels_train.csv` (45,000 rows = 15,000 × 3 radiologists; 28 labels), `image_labels_test.csv` (3,000 rows). Column name differs: "Other diseases" (train) vs "Other disease" (test).
- No patient ID anywhere. The DICOM headers have PatientID, StudyInstanceUID and ViewPosition empty (3 files checked with pydicom). **The manifest treats each image as its own patient (assumption).** This matters only if VinDr train is ever used for training: then its test set may share patients with train, but that is unknowable from the data.
- Pixel data: PhotometricInterpretation is MONOCHROME1 in some files (needs inverting) and MONOCHROME2 in others; BitsStored is 12 or 14.
- The download job ended "FAILED" only because of a stray `EOF` line in `download.sh`; the sync had completed. Checksums: all 18,006 OK.
- pydicom 3.0.2 has now been installed in `rrg`. DICOM→PNG conversion is out of scope until the user approves it.

## PadChest-GR

- `grounded_reports_20240819.json`: list of 4,555 studies {StudyID, ImageID, PreviousStudyID, PreviousImageID, findings: [{sentence_en, sentence_es, abnormal, boxes (normalised xyxy), extra_boxes, labels, locations, progression}]}.
- `master_table.csv.zip` → `master_table.csv`: 8,787 rows (one per finding sentence), 4,555 images, 4,459 patients. Columns include StudyID, ImageID, PatientID, label, label_group, locations, prior_study, split (train 6,153 / validation 872 / test 1,762 rows), study_is_benchmark (all True). For us, the whole set is an external test set and its internal split is ignored.
- Images: `Padchest_GR_files/PadChest_GR.zip.001–.037` (37 parts, 38.5 GB). This is a split zip of 4,555 PNGs: 4,202 stored, 353 deflated; 38.53 GB uncompressed. The central directory reads correctly and the last member passes its CRC check. `v2/nesy/multizip.py` reads any member in memory; one stored and one deflated member decoded fine.
- **PNGs are 16-bit (`I;16`)**, e.g. 1824×1652. Windowing to 8-bit must be decided before CLEAR sees them.
- `PadChest_GR_progression_prior_studies/…zip.001–.009`: 1,446 prior-study PNGs (9.0 GB), not needed.
- **Extraction would leave ~11 GB free, so do not extract.** Instead, read PNGs directly from the split archive (`v2/nesy/multizip.py`).
- Join to CheXmask Padchest.csv by `ImageID`. Assumed, since the CheXmask Padchest file covers the original PadChest; the overlap has not been counted yet.

## Extras not in the brief

- **CXRGraph** (`$D/cxrgraph/files/cxrgraph/1.0.0`): RadGraph-style annotations, 550 manual and 227,835 inferred MIMIC reports.
- **CXR-PRO** (`$D/cxr-pro-mimic-cxr`): `cxr.h5` image subset, impressions with priors removed. `mimic_test_impressions.csv` **fails its checksum**. Not planned for use.

## How the datasets join

- MIMIC ↔ CheXmask MIMIC: `dicom_id`.
- MIMIC ↔ Chest ImaGenome: `dicom_id` (scene graphs, gold `image_id`), `study_id` (= `rad_id` in the sentences file), `subject_id` (= gold `patient_id`).
- MIMIC ↔ MS-CXR: `dicom_id`; `subject_id` from `path`.
- VinDr ↔ CheXmask VinDr: `image_id`.
- PadChest-GR ↔ CheXmask Padchest: `ImageID`.

## Open questions

1. ~~Report text~~: resolved, the user downloaded `mimic-cxr-reports.zip`.
2. ~~pyarrow and pydicom~~: installed into `rrg` 2026-10-06 (pyarrow 25.0.1, pydicom 3.0.2; pip dry-run showed no other package changes).

## Manifests (Stage 0 step 2) — built 2026-10-06, run `v2/runs/20261006-0918_manifests` (+ `…0919_manifests` for PadChest-GR)

All in `v2/data/manifests/`. Read them only through `v2/nesy/manifests.py: load(name)`, which joins the project split.

| Manifest | Rows | Unit | Count checks |
|---|---|---|---|
| `mimic.parquet` | 377,110 | image | images, studies, patients = published; 500/500 sampled files exist |
| `mscxr.parquet` | 1,448 | box (phrase_id groups boxes) | 1,047 images, 851 patients, all in MIMIC |
| `imagenome_silver.parquet` | 243,310 | scene graph (zip member) | all dicom_ids in MIMIC |
| `imagenome_gold.parquet` | 1,000 | image | 1,000 images / 500 patients = published |
| `vindr.parquet` | 18,000 | image | 15,000 train / 3,000 test = published; train labels = radiologist vote counts 0–3, test = consensus 0/1 |
| `padchest_gr.parquet` | 4,555 | image (= study) | 4,555 = published; all in archive |

Raw CheXpert values (1 / 0 / −1 / NaN) are kept in `cx_*`. The "blank = negative" rule is applied when labels are loaded, not in the manifest. 8 MIMIC studies have no CheXpert row.

## Derived features (`v2/data/features/`)

- `chexmask_mimic.parquet`: side and CTR features for 243,334 MIMIC frontal images (see the CheXmask section).
- `imagenome_regions.parquet`: 8,695,606 rows, one per (dicom_id, region), for 243,310 scene graphs. Boxes are given in original pixels (`original_x1..y2`) and in the 224² frame. **There are 36 regions per image** (`semantics/objects_detectable_by_bbox_pipeline_v1.txt`), not the 29 the brief mentions; choose the subset for Stage 4.
- `imagenome_region_labels.parquet`: 7,306,391 rows (dicom_id, region, category, label, n_yes, n_no) from the silver NLP over reports. Categories: anatomicalfinding, nlp, tubesandlines, disease, technicalassessment, device, and others.
- `imagenome_meta.parquet`: per-graph viewpoint, IDs, counts.
- `radgraph_reports/shard_*.jsonl.gz`: RadGraph (modern-radgraph-xl) entities for the FINDINGS+IMPRESSION text of MIMIC reports. **Stopped at 6,000 studies (3 shards)** when the user ruled out parsing the corpus (2026-10-06); not used. `radgraph_val_refs/`: the 1,733 validate reference reports, parsed once and cached (2026-10-07, `v2/runs/20261007-063428_radgraph-val-refs`).

## CheXmask coverage of the external sets (checked 2026-10-06)
- VinDr-CXR: all 18,000 images (15,000 train, 3,000 test) are in `CheXmask/OriginalResolution/VinDr-CXR.csv` (key `image_id`).
- PadChest-GR: **4,310 of 4,555 images (94.6 %)** are in `CheXmask/OriginalResolution/Padchest.csv` (key `ImageID`, same file names as PadChest-GR). 245 images have no CheXmask masks; their CheXmask-region features will be missing (flagged, never zero).

## Study set used by every model (verified 2026-10-07)
One frontal image per study: the image the CLEAR embeddings were computed from (`v2/data/embeddings/clear_frontal_lb_v1_index.parquet`; `nesy/data.py: study_table()`). Lateral images are never used. 218,139 studies:

| split | studies | AP | PA |
|---|---|---|---|
| train (fit) | 187,754 | 111,330 | 76,424 |
| heldout_loc | 13,498 | 10,528 | 2,970 |
| calib | 6,283 | 3,770 | 2,513 |
| thresh | 5,830 | 3,409 | 2,421 |
| val | 1,733 | 1,072 | 661 |
| test | 3,041 | 2,158 | 883 |

The fit split's inner holdout (5% of fit patients by hash, `scripts/probe.py: inner_holdout`) is 8,915 studies; it chooses C for the linear heads and the epoch for the DenseNet and MLP baselines.

## Derived data added 2026-10-06/07 (all under `v2/data/`)
- `image_cache_256/`: the study set above, letterboxed to 256 px (LANCZOS, aspect kept, zero pad), 54 compressed uint8 shards + `ids.parquet` in study-table order; **8.9 GB**; checked: same image as the CLEAR embedding for 218,139/218,139 studies. Used by the DenseNet-121 baseline.
- `features/regions/mimic_full_lb_shards/`: letterbox region features (global + 36 ImaGenome + 9 CheXmask region vectors, float16) for 218,187 MIMIC frontal images, 15.4 GB. Region features missing for the 242 left/right-check failures and, for CheXmask regions, the 4,485 images with CheXmask Dice RCA < 0.7. CheXmask letterbox weights in `mimic_full_lb_p{0,1}_chunks` (compressed).
- `features/regions/ext_{vindr_test,padchest_gr}_lb_shards/`: external caches (letterbox, percentile windowing; global + 9 CheXmask regions): VinDr test 3,000 (15 with region features missing), PadChest-GR 4,555 (248 missing: 245 without masks, 3 RCA < 0.7). Caching only, never evaluated.
- `features/retrieval_lb_v1.parquet`, `index_lb/`: letterbox FAISS index on the fit split and neighbour features (k 5/10/25, own patient excluded).
- `features/cases_letterbox_k5.parquet`: k = 5 most similar fit-split studies per study (study IDs, similarities, label counts; no report text; own patient excluded, 0 leaks).
- `features/stage1_oof_letterbox.parquet`: out-of-fold Stage 1 logits (2 patient folds) for the association-links stage.
- `archive/stretch_v1/` (0.615 GB): stretch-variant predictions for every split and the trained models of Stages 1-5 (see RESULTS.md).

## Deleted 2026-10-07 (user confirmed)
Stretch region features (`mimic_full_shards`, 15.4 GB), stretch CheXmask weights (`mimic_full_weights.npy`), stretch retrieval index (`data/index/`), pilot and test-mode caches; CheXmask `OriginalResolution/{CheXpert,ChestX-Ray8}.csv` and `Preprocessed/{CheXpert,Padchest,VinDr-CXR}.csv`. The CheXmask table above lists files that no longer all exist: what remains is `OriginalResolution/{MIMIC-CXR-JPG,VinDr-CXR,Padchest}.csv` and `Preprocessed/MIMIC-CXR-JPG.csv`. Free space after deletion 53.5 GB.

## Facts for the report draft (verified 2026-10-07)
**Aspect ratio of the images in use** (218,139 one-frontal-per-study images; MIMIC-CXR-JPG metadata `Rows`/`Columns`): height/width median **1.165** (range 0.229–3.272; 5th–95th percentile 0.833–1.201; median PA 1.181, AP 1.122; 60.8% taller than wide). The old stretch-to-448×448 changed the aspect ratio by the factor max(H/W, W/H): **median 1.201, 90th percentile 1.201** (max 4.38), i.e. for the typical 3056×2544 portrait image the vertical axis was compressed by about 17% relative to the horizontal one. Letterbox keeps the aspect ratio.

**The 17 CheXmask anatomy features** (feature group `chexmask`, `scripts/probe.py`; CheXmask Preprocessed MIMIC masks via `scripts/chexmask_features.py`): `cm_ctr`, `cm_ctr_maxrow`, `cm_heart_width`, `cm_thorax_width`, `cm_lung_area_frac_left`, `cm_lung_height_frac_left`, `cm_lung_base_diff_left_minus_right`, `cm_lung_apex_diff_left_minus_right`, `cm_heart_shift_to_left`, `cm_lung_l_area`, `cm_lung_r_area`, `cm_heart_area`, `cm_rca_mean`, `cm_ctr_x_ap` (CTR × AP view), `cm_ctr_maxrow_x_ap`, `cm_missing` (no mask or failed QC; values imputed with the fit-split mean), `view_ap`. All standardised on the fit split.

**Region pooling (both region sets)**: a weighted average of CLEAR patch features. Patch features are `visual.projection(x_norm_patchtokens)` on the 32×32 grid (14-px cells of the 448-px letterbox canvas). For region r with patch weights w_p ∈ [0, 1]: vector_r = Σ_p w_p · token_p / max(Σ_p w_p, 1e-6); the region is present iff Σ_p w_p > 0 (`scripts/region_features.py`). The projection is linear, so pooling then projecting equals projecting then pooling.
- Set A (36 Chest ImaGenome regions): w_p = fraction of the patch's area covered by the region's bounding box, after mapping the box from original pixels to the canvas with the same letterbox scale and offset (`nesy/regions.py: box_weights`).
- Set B (9 CheXmask regions: left lung, right lung, heart, and upper/middle/lower thirds of each lung): w_p = fraction of the patch's area covered by the region mask; the full-resolution OriginalResolution mask is resized to the letterboxed size with a BOX filter, placed on the canvas and averaged over each 14×14 cell (`mask_to_grid`), stored as uint8 (÷255). Thirds: each lung mask cut into three equal-height bands between its top and bottom rows (`chexmask_region_masks`).
- Missing: both sets for the 242 images failing the left/right check; set B for images whose CheXmask Dice RCA < 0.7 (4,485 in MIMIC).

**CLEAR preprocessing** (repo `/scratch/prj/bhi_zihe_imaging/CLEAR`, commit 57d2eae "Prepare CLEAR Nature publication release").
- Established, aspect-preserving pad: `src/clear/data_processing.py`, `preprocess(img, desired_size=320)`:
  `ratio = float(desired_size)/max(old_size)` / `new_size = tuple([int(x*ratio) for x in old_size])` / `img = img.resize(new_size, Image.LANCZOS)` / `new_img = Image.new('L', (desired_size, desired_size))` / `new_img.paste(img, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))`.
  `img_to_hdf5(...)` in the same file applies `preprocess` to every image and writes the result to the HDF5 dataset `'cxr'`; `scripts/run_preprocess.py` calls `img_to_hdf5` for MIMIC with `--resolution` default **448**.
- Established, readers of the padded HDF5: the released evaluation loaders read `'cxr'` from that HDF5 and then apply `Resize(448, BICUBIC)` + normalisation: `src/clear/zero_shot.py` (`CXRTestDataset`, `h5py.File(...)`, line 356 `Resize(448 ...)`), `examples/concepts/exp_linear.py` (`self.h5_file['cxr'][idx]`, line 271), `examples/benchmark/benchmark_base.py` (`h5py.File(img_path, 'r')['cxr']`).
- Established, the stretch: the packaged convenience transform `src/clear/hub.py: build_cxr_preprocess()` resizes to an exact square (`Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC)`); its docstring says "Images are resized to an exact square rather than aspect-ratio resized and cropped".
- Inferred, not established: that the PRE-TRAINING data loader read the padded HDF5. The released repo has no pre-training loader (`examples/train.py` only loads checkpoints). The inference rests on the preprocessing script being the repo's data pipeline at 448 px and every released evaluation loader reading its padded output. The paper's "resized to 448 × 448 with bicubic interpolation" matches the `Resize(448, BICUBIC)` applied to already-padded 448-px images (a no-op resize), and also matches `hub.py`'s stretch; the text alone does not settle which.

## Added 2026-10-07 (later)
- `v2/runs/20261007-081007_densenet-external/predictions_{vindr,padchest_gr}.parquet`: DenseNet-121 external predictions (image_id, read_ok, p_ and logit_ per finding); VinDr test 3,000, PadChest-GR 4,555; no labels.
- `v2/models/densenet121-a639ec97.pth`: torchvision ImageNet DenseNet-121 weights (downloaded from download.pytorch.org; hash prefix matches the file name).
- `v2/archive/stretch_v1/`: stretch-variant predictions and models (0.615 GB; see RESULTS.md).
- `v2/kg/sources/`: Fleischner glossary PDF, Fleischner nodule guidelines HTML (not used), RadReport "Rad Chest 2 Views" HTML (copies; the user's originals stay in the repo root).
- Free space after today's work: about 42 GB (1000 GB quota).

### External test label counts (2026-10-07, user request; `v2/runs/20261007-112537_external-label-counts`)
External test labels read once, for counts only (no predictions, no metrics, nothing selected), to support review of `v2/kg/external_maps/*.yaml`. VinDr test: 3,000 images, 2,051 "No finding", 182 with only unmapped labels; positives per finding (direct / with is_a): lung opacity 462, atelectasis 86, consolidation 96 / 250 (VinDr "Pneumonia" propagates via is_a), pneumonia 246, edema 0, lung lesion 184, cardiomegaly 309, effusion 111, pleural other 169, pneumothorax 18, fracture 13. PadChest-GR: 4,555 images, 1,456 normal, 1,307 with only unmapped labels; lung opacity 734 / 839, atelectasis 259, consolidation 184, lung lesion 237, enlarged cardiomediastinum 538, cardiomegaly 498, effusion 372, pleural other 240, pneumothorax 11, fracture 190, support devices 345. Review sheet: https://claude.ai/artifact/C1mcphtwGXA1MPJjVwavV9
