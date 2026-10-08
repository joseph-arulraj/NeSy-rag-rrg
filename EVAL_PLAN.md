# Evaluation plan for the test runs (written 2026-10-07, before any test run exists)

Nothing in this plan has been run on test data. The scripts named below have been written and checked on validate only. The MIMIC test split and the external sets are refused unless `--allow-test` / `--allow-external` is passed, and that needs the user's go-ahead.

## What is frozen before a test run

- **Pipeline:** `configs/pipeline.yaml`, run by `scripts/run_pipeline.py`. Each run's `config.json` / `manifest.json` records the git commit and the SHA-256 of every file the run depends on, which makes it the freeze manifest:
  - the head model and calibrators (`model.joblib`) and its thresholds (`thresholds_4band.json`);
  - the region scores and region thresholds;
  - the concept list and embeddings, the CheXmask features and the case table;
  - every KG YAML, the configs and the code files.

  The code is committed (2026-10-08). Runs made before the commit record `de121e6…+dirty`; commit before the test run so its manifest names a clean commit. The file hashes are complete either way.
- **Configuration switches (current values):**
  - R1 off, R2 on;
  - report layout `current` (RadReport proposals not applied);
  - `show_case_evidence` off;
  - Stage 8: comparator v2, retry and attempt count as set after the validation rerun (see RESULTS.md).
- **No refitting on test.** Nothing on test is used to refit a calibrator, refit a threshold or select any setting. One run per test set. If a bug forces a rerun, the reason is recorded and both runs are reported.

## MIMIC-CXR test (in-domain; every number labelled "seen by backbone")

- **Studies:** our frozen test split, one frontal image per study: 3,041 studies from 289 patients.
- **Primary labels: radiologist.** These are the MIMIC-CXR 2.1.0 radiologist labels (`mimic-cxr-2.1.0-test-set-labeled.csv`, 687 studies).
  - 596 of the 687 are in our test split with a frontal image. Those 596 are the primary evaluation set.
  - 9 of the 687 belong to official-validate patients and sit in our val split, so they are not test studies. The other 82 have no frontal image.
  - Mapping: the 13 CheXpert columns map to our findings, with "Airspace Opacity" → `lung_opacity`. The same label rules as training apply (`nesy/labels.py`, identical to `build_labels.py` and tested on validate):
    - blank → negative;
    - −1 → masked;
    - a positive child makes its parents positive;
    - an uncertain child with a blank parent masks the parent;
    - the root `any_abnormality` is derived from the findings.
  - The actual value coding in the file is logged when it is read.
- **Secondary labels:** CheXpert-derived report labels (`labels_v2`) on all 3,041 test studies.
- **Per finding (14, including the root) and macro mean:**
  - AUROC, AUPRC, ECE (15 equal-width bins) and Brier;
  - patient-level bootstrap, 2,000 resamples, 95% percentile intervals;
  - the number of positives shown per finding. Findings with fewer than 10 positives are reported but flagged as not interpretable, which with 596 studies probably includes fracture and pleural other.
  - Bootstrap ECE intervals are biased upward: on validate the macro-ECE point 0.0236 lies below its interval [0.026, 0.035]. ECE comparisons are therefore made as paired differences.
- **Hierarchy violations** (P(child) > P(parent)): expected 0 by construction.
- **Bands, per finding:** "present" precision and sensitivity; "present or possible" precision and sensitivity; absent-band miss rate.
- **Normal call:** rate, precision (labelled normal among called normal) and sensitivity (called normal among labelled normal), against the root label.
- **Report-to-graph agreement:**
  - Stage 8 first-pass match rate, final LLM match rate and fallback rate;
  - the template round-trip rate (a check on the comparator).
- **H3: the head against an unrestricted probe.**
  - Comparator: an unrestricted linear probe on the CLEAR embedding plus the same A (17 CheXmask anatomy features) and R (Stage 4 region scores, CheXmask regions) inputs, trained on the fit split with Platt calibration on calib (`runs/*_h3-probe-ear`).
  - Statistic: macro (mean over the 14) AUROC of the C+A+R head minus the comparator, with a paired patient bootstrap.
  - Test: the head is non-inferior if the lower 95% bound is above −0.02. This direction is my reading of "margin 0.02 mean AUROC"; **please confirm**.
  - Primary on the radiologist labels, secondary on the CheXpert-derived labels.
  - The comparator's test predictions are exported with `export_predictions.py --kind probe` at test time, not before.
- **Report level against reference reports (not decided).** This needs RadGraph parses of the MIMIC test reports, which is still the user's decision. The method is developed on validate only, using the cached validate parses:
  - per finding, the precision, recall and F1 of positive mentions in our final report against the reference (both read with comparator v2);
  - a RadGraph entity F1 (exact token and label match).

## External sets (VinDr-CXR test, PadChest-GR): not yet

These are blocked on three things:
1. **The mapping review** (sheet: https://claude.ai/artifact/C1mcphtwGXA1MPJjVwavV9). The mapping files stay `status: draft-for-review` until the user's decisions are applied.
2. **External feature assembly isn't built.** CLEAR embeddings, region features and CheXmask anatomy features are cached. Stage 4 region scores for external images need the Stage 4 classifiers, which run `20261006-224243_lb-stage4` did not save (it has no `--save-models`), so they must be retrained or saved first.
3. **View is unknown** for every VinDr and PadChest-GR image: there is no DICOM ViewPosition and no PadChest-GR projection field. The A input has a `view_ap` flag. How to treat unknown views needs a decision. The options are: PA by default (VinDr documentation describes PA images, not verified from the files), or a separate "view unknown" handling.

When unblocked, the metrics are the same as for MIMIC test, restricted to the findings marked `evaluable`, with image-level bootstrap (VinDr has no patient IDs). There is no "seen by backbone" label for these sets.

## Scripts

| Purpose | Script | Checked on validate |
|---|---|---|
| End-to-end run | `scripts/run_pipeline.py` | yes: equal to the existing C+A+R results (probabilities within 4.3e-6; bands, sides, zones, normal calls and templates identical) |
| Evaluation | `scripts/evaluate.py` | yes: reproduces head macro AUROC 0.8344 / AUPRC 0.4911 / ECE 0.0236 |
| H3 comparator | `scripts/probe.py --features emb,chexmask,s4` | trained, val metrics in RESULTS.md |
