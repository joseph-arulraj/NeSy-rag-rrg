# Evidence A — Spatial Perception: Specification and Build Plan

**Relationship to `pipeline.md`:** this document is the full specification for N02 `Spatial perception model` → N04 `Region + laterality features` → N06 `Evidence A: Spatial findings`, extracted out of [`pipeline.md`](pipeline.md) §4.2 (Revision Log rev 4) because it has grown into its own multi-dataset build project. Every cross-cutting contract that depends on A's *existence or absence* (degraded-mode policy, fusion's 3-vs-4-source behaviour, the dependency graph, build order) stays in `pipeline.md` — this file owns A's *internal* design only. `pipeline.md`'s GAP-2 is the pointer into this document.

**Status:** N02 is **deferred for the v1 base build** (`pipeline.md` Revision Log rev 1) — the base pipeline builds and validates against the `EvidenceA=absent` contract first. This document specifies what N02 *becomes* when built, not a change to that deferral decision.

---

## 0. Revision Log

| Rev | Decision | Rationale / driven by |
|---|---|---|
| 1 | Document created by extracting `pipeline.md` §4.2's diagram-derived spec, and adding a concrete three-dataset build plan (VinDr-CXR, LATTE-CXR, CheXmask-U) that did not exist when N02 was originally deferred. | User decision, this session |

---

## 1. Role in the Architecture

**Purpose (unchanged from the original diagram):** independent localised perception. Answers *what* and *where*, without reference to the CLEAR embedding space. N02 reads `N01` (the raw image) directly — it is the only Stage-1 branch not downstream of N03/CLEAR — so it gives the system explicit spatial grounding instead of relying on global image-level similarity.

**Why A is structurally different from B/C/D, not just "another evidence source":**

| Source | Mechanism | Spatial grounding | Supervision |
|---|---|---|---|
| B (concept similarity + grouping) | Cosine similarity to 368k concept phrases mined from other patients' reports | None | None — deterministic aggregation, no learned/calibrated parameters at inference |
| C (CBM) | Linear head over 68 raw concept scores, predicting CheXpert labels | None | Real, supervised on CheXpert ground truth — but flat pathology labels only, no location |
| D (retrieval, paused) | k-NN image similarity → borrow neighbours' report facts | None | None — compounding indirection (image similarity → borrowed report → extracted facts), which is precisely why it was deprioritised |
| **A** | Direct pixel-level detection/localisation on **this** image | **Yes — the only source with real coordinates** | Real, supervised, and (per the plan below) validated specifically for localisation on MIMIC's own domain |

The honest framing: A has a genuinely different *structural* case for being the strongest of the four — it is the only source that is directly grounded rather than analogy-based, and (per §5 below) the only one with a real path to validating its own spatial accuracy on MIMIC. That is not the same as a proven result. Until built and measured, this is a justified expectation, not a fact — see §8.

---

## 2. Diagram-Derived Specification (moved from `pipeline.md` §4.2, unchanged in substance)

### 2.1 N02 — the detector/localiser

| Aspect | Specification |
|---|---|
| Input | `ImageTensor` (spatial branch preprocessing) + invertible resize transform |
| Output | `list[SpatialDetection]` |
| Model class | Region-level CXR model — detection, anatomy segmentation, or grounded classification. Architecture and checkpoint were unspecified in the original diagram (`pipeline.md` GAP-2) — §4–§5 below now propose a concrete answer |

```
SpatialDetection:
  finding_label: str
  score: float                    # model-native, NOT calibrated
  bbox: [x0,y0,x1,y1]             # original-image pixel coords
  mask: RLE | null
```

### 2.2 N04 — region + laterality derivation

Pure geometric/symbolic post-processing. No learned parameters.

| Step | Specification |
|---|---|
| Coordinate restoration | Invert N01's spatial-branch resize/pad to original pixel space |
| Anatomical assignment | Map each bbox/mask to `AnatomyID` — either via an anatomy segmentation mask (preferred, deterministic overlap rule — **CheXmask-U's lung/heart contours are a direct candidate for this**, see §4.3) or a zone lookup (upper/mid/lower × left/right lung field, cardiac silhouette, mediastinum, costophrenic angles, apices, hila, diaphragm) |
| Laterality derivation | Derived from anatomy assignment **and** `patient_orientation`. **Must not** be inferred from raw image-x alone: on an AP film the patient's left is image-right, and flipped/mirrored acquisitions exist |
| Overlap resolution | Deterministic tie-break when a box straddles regions — e.g. assign to max-IoU region; emit `bilateral` only above a configured bilateral-overlap ratio |

### 2.3 Output contract → N06 `Evidence A: Spatial findings`

```
EvidenceA = list[{
  label: str, anatomy: AnatomyID, laterality: Laterality,
  score: float, bbox: [...], area_frac: float
}]
```

**Downstream consumers:** N22 (fusion, via E21a) only. *(Historical note: the original diagram also fed A into N12 as a refinement signal via E21b — that consumer was permanently removed in `pipeline.md` rev 3, not deferred. See `pipeline.md` §4.6.)*

### 2.4 Failure paths

| Condition | Behaviour |
|---|---|
| Zero detections | Emit empty `EvidenceA`; **do not** treat as failure — a normal CXR legitimately yields none. Must be distinguishable from "module errored" |
| Anatomy mapping fails for a detection | Emit with `anatomy=null`, `laterality=unspecified`; must not be silently dropped |
| `patient_orientation` missing | Fall back to `view_position`; if both missing, emit `laterality=unspecified` and flag `study_meta.laterality_unreliable=true`. Never guess |
| Model timeout/OOM | Degraded mode — see `pipeline.md` §7.2 |

### 2.5 Configuration

`spatial.checkpoint`, `spatial.score_threshold`, `spatial.nms_iou`, `spatial.max_detections`, `spatial.anatomy_atlas_version`, `spatial.bilateral_overlap_ratio` — plus the new, dataset-specific config surface implied by §4–§6 below (not yet finalised: pretraining/fine-tuning checkpoints, taxonomy mapping table paths, CheXmask gating thresholds).

---

## 3. Current Status: Deferred, With a Concrete Buildout Plan

N02 remains deferred for the v1 base build (`pipeline.md` rev 1). What changed in this session is that "eventually build N02" now has a specific, evaluable plan instead of an open architecture choice (`pipeline.md` GAP-2) — driven by identifying three complementary, already-licensed datasets that between them solve the three problems that made A hard to build well: scale, domain match to MIMIC, and validation.

---

## 4. Data Strategy: Three Complementary Datasets

Individually, each of the three datasets has a specific hole. Together, each hole is closed by one of the other two — this is the actual reason to use all three rather than picking one.

| Dataset | Gives you | Doesn't give you |
|---|---|---|
| VinDr-CXR | Scale (18,000 images), real pathology bounding boxes, broad finding coverage | Wrong domain (not MIMIC) — no way to validate transfer on its own |
| LATTE-CXR | Small (~2,700 images) but genuinely MIMIC-native radiologist-drawn boxes, explicitly paired with report text and a certainty rating | Far too small to train a detector from scratch |
| CheXmask-U | Full(ish)-coverage, MIMIC-native pixel-level anatomy (lung/heart) segmentation, with self-estimated quality | Zero pathology information — anatomy shape only |

### 4.1 VinDr-CXR

**Licensing:** confirmed obtained by the user (PhysioNet-credentialed access pattern), not yet downloaded as of this writing.

**Labelling structure — 28 findings, split into local vs. global:**

- **Local labels (1–22), bounding-box annotated:** Aortic enlargement, Atelectasis, Cardiomegaly, Calcification, Clavicle fracture, Consolidation, Edema, Emphysema, Enlarged PA, Interstitial lung disease (ILD), Infiltration, Lung cavity, Lung cyst, Lung opacity, Mediastinal shift, Nodule/Mass, Pulmonary fibrosis, Pneumothorax, Pleural thickening, Pleural effusion, Rib fracture, Other lesion.
- **Global labels (23–28), image-level diagnostic impression only — no location data:** Lung tumor, **Pneumonia**, Tuberculosis, Other diseases, COPD, No finding.

**Critical consequence:** VinDr provides **no box supervision for pneumonia** — it's global-only. A cannot be trained to localise pneumonia from VinDr at all, and LATTE-CXR (§4.2) doesn't have a distinct pneumonia class either. **Pneumonia is structurally out of scope for A**; B and C remain the only sources for it. This must not be silently discovered later.

**Annotation process and data-quality structure:**
- 18,000 images total: 15,000 train (10,606 normal / 4,394 abnormal), 3,000 test (2,052 normal / 948 abnormal). Roughly 70/30 normal/abnormal in both splits — a real class-imbalance consideration once you look at per-local-finding positive counts (some, e.g. clavicle fracture or lung cyst, will have few positive boxes in absolute terms).
- **Training set: each image annotated independently by 3 radiologists, blind — three separate, not-yet-reconciled box sets per image.** This needs an explicit aggregation decision (union, majority-vote / weighted-box-fusion, or treating each rater's annotation as a separate training example) before it's usable as standard detector training data. Not yet decided — see EA-GAP-3.
- **Test set: 2-stage consensus.** 3 radiologists annotate independently, then 2 more senior radiologists reconcile disagreements into one consensus label per image. This is meaningfully higher-trust ground truth than the raw training annotations.

### 4.2 LATTE-CXR

**Full name:** *Locally Aligned TexT and imagE, Explainable dataset for Chest X-Rays.* Derived from REFLACX (Reports and Eye-Tracking Data for Localization of Abnormalities in Chest X-rays), itself sampled directly from MIMIC-CXR.

**Licensing:** confirmed obtained by the user, already downloaded.

**Two annotation sources — not equal quality, must be treated differently:**

1. **`bbox_statement.csv` — the primary, trustworthy ground truth.** Rectangular bounding boxes converted from REFLACX's radiologist-drawn ellipses, explicitly paired with `statement` (the concatenated report sentence(s) describing what's in that box). Carries `certainty` (radiologist confidence, 1–5), `chosen` (primary-box flag when multiple boxes apply to one statement), and 16 pathology booleans. **This is real, deliberate ground truth — a radiologist intentionally drew this box around what they were describing.** Use this as the primary fine-tuning and validation data.
2. **`etbox_sentence.csv` — a weaker, secondary proxy signal.** Boxes derived automatically from accumulated eye-gaze fixation heatmaps during dictation (1.5s pre-sentence interval + sentence duration, 40% peak-intensity threshold), paired with individual report sentences, cross-linked to `bbox_statement.csv` via `Ann_box_index` where they share a sentence. **Gaze reflects visual attention during dictation, not a deliberate delineation of a finding's boundary** — do not treat this as equivalent ground truth to (1). Plausible future use as weak/auxiliary supervision; not for validating spatial accuracy.

**The `certainty` column is a genuine asset:** it lets you filter to a high-trust validation subset (e.g. certainty ≥ 4) rather than diluting the one real spatial-accuracy measurement available on MIMIC with ambiguous boxes.

**16 pathology classes (REFLACX-derived):** Abnormal mediastinal contour, Acute fracture, Atelectasis, Consolidation, Enlarged cardiac silhouette, Enlarged hilum, Groundglass opacity, Hiatal hernia, High lung volume/emphysema, Interstitial lung disease, Lung nodule or mass, Pleural abnormality, Pneumothorax, Pulmonary edema, Support devices, Other. (`"Other"` means *some uncategorized abnormality is present* — **not** equivalent to our `no_finding` bucket; do not conflate.)

**Patient overlap and split discipline — real leakage risk, correctly identified by the user:**
- LATTE-CXR's images are drawn from MIMIC-CXR patients, so its ~2,700 readings' `subject_id`s **do** overlap with the broader MIMIC-CXR-JPG population.
- LATTE-CXR ships its own `split` (train/validation/test) and `REFLACX_split` columns.
- **Required discipline:** extract every unique `subject_id` in LATTE-CXR and exclude them from any MIMIC-wide evaluation/production set — the same pattern as the project's existing train-only FAISS retrieval corpus + `assert_patient_disjoint()` mechanism. This should become **one shared held-out-patient registry**, not a one-off exclusion rule local to A — see EA-GAP-1.
- **Not yet verified:** whether LATTE's own `split`/`REFLACX_split` columns actually respect MIMIC-CXR's *official* train/validate/test boundaries, or cut across them. Must be checked before trusting LATTE's own split labels at face value — see EA-GAP-2.

### 4.3 CheXmask-U

*(Full name: CheXmask-U: Uncertainty Estimation for Landmark-Based Anatomical Segmentation Masks in Chest X-ray Images.)*

**Licensing:** confirmed obtained by the user, already downloaded.

**What it is:** per-image anatomical **shape** segmentation — Left Lung, Right Lung, Heart contours, as landmark polygons, produced by a learned (HybridGNet-style) model. **Zero pathology information** — pure anatomy, not findings.

**Schema:** `dicom_id`; `Dice RCA (Mean/Max)` — almost certainly Reverse Classification Accuracy, i.e. the model's own *self-estimated* segmentation quality (there is no ground-truth mask for MIMIC to compute a real Dice against — treat this as a confidence proxy, not verified ground truth); `Landmarks (Mean/Std)` — contour point coordinates and per-point positional uncertainty; per-structure encoded contours for Left Lung, Right Lung, Heart; Height/Width.

**Two concrete uses, neither of which requires A to work at all:**
1. **Cardiothoracic ratio (CTR) → a free, deterministic cardiomegaly signal.** CTR = max heart width ÷ max thoracic width; CTR > 0.5 is the literal textbook diagnostic threshold for cardiomegaly. This is a direct geometric measurement, arguably more interpretable and trustworthy than any learned probe, and entirely independent of whether the VinDr/LATTE-trained detector (§5) turns out to work well.
2. **Full-scale deployment-time anatomical plausibility gate.** Once A is deployed across the full ~200k MIMIC images, LATTE-CXR's ~2,700-image ground truth can't tell you anything about the other ~197k. CheXmask, if it covers most/all of MIMIC (coverage not yet verified — see EA-GAP-4), gives an always-available per-image sanity check: does A's predicted box actually fall inside the anatomically segmented lung/heart region for that image. This catches grossly wrong boxes (e.g. floating outside the thorax); it does **not** confirm fine-grained localisation accuracy — see §9 for the sharp version of this distinction.

**Trust gating:** any CTR computation or plausibility check from a low-quality-estimated segmentation (low Dice RCA, high landmark Std near the relevant border) should be down-weighted or excluded, not trusted blindly — the dataset hands you the exact signal to do this gating with.

---

## 5. Proposed Build Pipeline

```
1. Pretrain   a finding-localisation detector on VinDr-CXR (scale, broad pathology coverage, external domain)
2. Fine-tune  on LATTE-CXR's bbox_statement.csv TRAIN split (radiologist-drawn boxes only — not etbox_sentence.csv),
              closing the domain gap with real MIMIC-native data
3. Validate   on LATTE-CXR's HELD-OUT split, filtered to high-certainty boxes (certainty >= threshold) —
              the first real, measured number for A's spatial accuracy on MIMIC, not a proxy
4. Deploy     across the full ~200k MIMIC images, with CheXmask-U gating gross anatomical plausibility
              per-image, plus the independent CTR-based cardiomegaly signal
5. Check      A's own output for logical consistency via RadLex/N25 (see pipeline.md §5.4) before it
              reaches fusion — this is a backstop, not a training step (see §9)
```

This directly closes the gap that made A previously hard to trust: MIMIC-CXR-JPG has no spatial ground truth of its own, so before LATTE-CXR there was no way to measure whether a VinDr-trained detector actually transfers. Step 3 is the first point in this whole project where A's spatial accuracy becomes an actual number instead of an assumption.

---

## 6. Taxonomy Reconciliation

VinDr's 28-class taxonomy and LATTE/REFLACX's 16-class taxonomy do not match each other, and neither matches the project's existing canonical finding vocabulary ([`configs/finding_synonyms.yaml`](../configs/finding_synonyms.yaml)). This is the same category of problem already solved twice elsewhere in this project (RadGraph text → RadLex anatomy terms; RadGraph observations → `finding_synonyms.yaml`) and the same rule applies: **never let two external taxonomies map to each other directly — both map independently onto one shared canonical target vocabulary.**

### 6.1 Required work (not yet done)

1. Confirm/extend the canonical target vocabulary for A specifically (current 14 CheXpert + `pericardial_effusion`, `hyperinflation`, `scoliosis`, `hilar_abnormality`, `costophrenic_blunting`, plus candidates identified below).
2. Build two separate, hand-curated mapping tables — VinDr-taxonomy→canonical and LATTE/REFLACX-taxonomy→canonical — same disciplined pattern as `finding_synonyms.yaml`, not automated clustering.
3. For each canonical class, explicitly record whether it has VinDr pretraining coverage, LATTE fine-tune/validate coverage, or both. A class present in only one dataset is a real, specific weak spot (either no domain-adapted fine-tuning, or no broad pretraining base) — list it, don't discover it later.

### 6.2 Findings from the mapping exercise so far

**Clean 3-way matches (VinDr ↔ LATTE ↔ canonical):** atelectasis, cardiomegaly (VinDr "Cardiomegaly" / LATTE "Enlarged cardiac silhouette" — both distinct surface forms of the same synonym already listed in `finding_synonyms.yaml`), consolidation, edema, pneumothorax, lung_lesion (VinDr "Nodule/Mass" / LATTE "Lung nodule or mass").

**Asymmetric coverage, flagged not silently merged:**
- `support_devices` — LATTE has it, VinDr has nothing resembling it anywhere in its 28 labels. Zero external pretraining signal; relies entirely on ~2,700 LATTE images.
- `pneumonia` — no box supervision in **either** dataset (VinDr: global-only; LATTE: no equivalent class at all). Out of scope for A regardless of which dataset is used.

**Do-not-conflate traps (clinically distinct despite similar names):**
- Mediastinal shift (VinDr) ≠ `enlarged_cardiomediastinum`. "Shift" = displacement; "enlarged" = widening. LATTE's "Abnormal mediastinal contour" is broader still. Probably deserves its own canonical bucket(s) rather than forced merging — same mistake category as the earlier pericardial/pleural effusion conflation.
- **Pulmonary fibrosis (VinDr) vs. the existing `pleural_other` bucket** — `finding_synonyms.yaml` currently lists bare **"fibrosis"** as a `pleural_other` synonym. Pulmonary fibrosis is a *lung parenchyma* finding; pleural fibrosis/scarring is a *pleural membrane* finding — anatomically distinct structures. **This is a real, currently-live bug in the concept bank's tagging, found by cross-referencing VinDr's taxonomy against our own table** — worth fixing independent of anything to do with A. Not yet fixed. See EA-GAP-5.

**Ambiguous/coarse, needs a decision:**
- `pleural_effusion` vs. `pleural_other` vs. LATTE's "Pleural abnormality" (a superclass spanning both, arguably pneumothorax too). Recommended resolution: don't map LATTE's coarse boolean directly — instead run the box's paired `statement` text through the **existing** RadGraph + `finding_vocab` tagging machinery (the same pipeline built for the 368k concept bank) to get a precise canonical label. Reuses validated infrastructure instead of inventing a new one-off mapping.
- Lung opacity vs. Infiltration (VinDr treats these as separate local labels; the current `lung_opacity` bucket already merges "infiltrate/infiltrates" as a synonym) vs. Groundglass opacity (LATTE, a specific radiographic subtype). Decide whether to preserve this granularity or accept the existing merge knowingly.
- Fracture — VinDr splits into Clavicle fracture / Rib fracture (location-specific); LATTE and the current canonical bucket are both generic. Merge (lose specificity) or split (more curation work) — undecided.

**No canonical home yet, explicit yes/no needed:** Aortic enlargement, Calcification, Lung cavity, Lung cyst, Enlarged PA (hilar_abnormality-adjacent but not identical — PA is one specific vessel, hilum is a broader region), Hiatal hernia (LATTE-only).

**Worth adding as a new canonical class:** **Interstitial lung disease (ILD)** — supported by *both* VinDr (#10, local) and LATTE (#10), the only non-trivial new candidate that would actually get full pretrain + fine-tune + validate coverage if added, unlike the one-sided classes above.

---

## 7. Patient-Disjointness and Leakage Discipline

Two concrete requirements, not yet implemented:

1. **A shared held-out-patient registry.** LATTE-CXR's `subject_id`s must be excluded from any MIMIC-wide evaluation/production activity — not just A's own fine-tune/validate split, but coordinated with the project's *existing* exclusion mechanism (the train-only FAISS retrieval corpus + `assert_patient_disjoint()`, see [`pipeline.md`](pipeline.md) §4.8/§10.1). A fragmented, per-component patchwork of exclusion rules risks a future component (CBM training, a retrieval rebuild) not knowing about this exclusion and leaking anyway.
2. **Verify LATTE's own split boundaries against MIMIC's official splits** before trusting LATTE's `split`/`REFLACX_split` columns for train/validate/test assignment — if LATTE's "train" includes patients that are in MIMIC's official validate/test split, using it naively could contaminate whatever the project considers its official held-out evaluation later.

---

## 8. Honest Assessment: A vs. B/C/D

A has a genuinely different structural case for being the strongest evidence source: it is the only one that is directly pixel-grounded rather than analogy-based (B: text-similarity to other patients; D: image-similarity to other patients, then borrowed facts — compounding indirection at every step), and — uniquely among the four — it has a real path (§5, step 3) to *measuring* its own spatial accuracy on MIMIC's actual domain, via LATTE-CXR's radiologist-drawn, certainty-rated boxes.

**What this does not mean:** that A is guaranteed to outperform B/C/D once built. That depends on real, currently-unmeasured quantities — how much of VinDr's domain shift survives LATTE fine-tuning, how large and reliable the certainty-filtered LATTE validation set actually turns out to be, and how well coverage holds up outside the ~9–10 cleanly-matched finding classes (§6.2). D's weakness was structural and predictable in advance (three lossy indirection steps, no direct grounding at any of them); A's risks are of a different, more measurable kind (domain adaptation sufficiency, validation-set size) — not the same failure mode, and not something to assume away either.

---

## 9. Role of the KG / N25 Relative to A

Two distinct claims, and it matters to keep them separate:

1. **RadLex/N25 cannot make A's underlying detection more accurate.** That's a pixel-level task; a symbolic ontology has no access to pixels, and this doesn't change no matter how the rules are written.
2. **What N25 genuinely gives A:** (a) the same structural-consistency backstop it gives Evidence B — catching things like a `cardiomegaly` finding with an invalid `laterality=left` attribute (per `pipeline.md` §5.4's "attribute validity: drop attribute, keep finding" rule) — and (b) uniquely among the four sources, **A's real coordinates let N25 arbitrate cross-source conflicts for the whole belief graph** (e.g. resolving a laterality disagreement between A and B) in a way none of B/C/D can, since none of them have actual spatial grounding to break a tie with. That precedence (trusting A's coordinates over B's lexical inference for a spatial attribute like laterality) is a *reasoned default*, not a proven rule — it should be validated empirically once A exists, and conditioned on A's own confidence rather than applied unconditionally.

Separately, **CheXmask-U's anatomy masks** provide a different, complementary kind of check to N25's: a genuine (if partial) *geometric* plausibility filter — does A's box fall inside the actual segmented lung/heart region for this image — which RadLex, being purely symbolic, cannot provide at all. Neither mechanism can confirm fine-grained spatial accuracy; only LATTE-CXR's ground truth (§5, step 3) can.

---

## 10. Open Gaps and Decisions

*(Independent numbering from `pipeline.md`'s GAP-n/AMBIG-n — prefixed `EA-` to avoid collision.)*

| ID | Issue | Status |
|---|---|---|
| **EA-GAP-1** | LATTE-CXR's patient exclusions need to become one shared held-out-patient registry, coordinated with the existing retrieval-corpus disjointness mechanism, not a one-off local rule | Open |
| **EA-GAP-2** | Whether LATTE-CXR's own `split`/`REFLACX_split` columns respect MIMIC-CXR's official train/validate/test boundaries is unverified | Open |
| **EA-GAP-3** | VinDr-CXR's training set has 3 independent, unreconciled radiologist box sets per image — aggregation method (union / majority-vote / weighted-box-fusion / treat-as-separate-examples) not decided | Open |
| **EA-GAP-4** | CheXmask-U's coverage of the full ~200k MIMIC-CXR-JPG image set is unverified (does it cover all images, or a subset?) | Open |
| **EA-GAP-5** | `configs/finding_synonyms.yaml`'s `pleural_other` bucket includes bare "fibrosis" as a synonym, which would likely mis-bucket "pulmonary fibrosis" (a lung-parenchyma finding, anatomically distinct from pleural fibrosis/scarring) — found via the VinDr taxonomy cross-reference, not yet fixed | Open — affects the current concept bank tagging today, independent of A |
| **EA-GAP-6** | Canonical vocabulary extension decisions: add ILD (recommended — dual VinDr+LATTE support); decide fate of hiatal hernia, aortic enlargement, calcification, lung cavity/cyst, enlarged PA (no current home) | Open |
| **EA-GAP-7** | Mediastinal shift vs. `enlarged_cardiomediastinum` — likely needs its own bucket(s) rather than merging; not yet decided | Open |
| **EA-GAP-8** | Fracture granularity — merge VinDr's clavicle/rib fracture into the generic `fracture` bucket, or split | Open |
| **EA-AMBIG-1** | Lung opacity vs. Infiltration vs. Groundglass opacity — preserve VinDr/LATTE's finer granularity or keep the current `lung_opacity` merge | Open |
| **EA-AMBIG-2** | LATTE's "Pleural abnormality" — resolve via its own coarse boolean, or re-tag from the paired `statement` text using existing RadGraph/`finding_vocab` machinery (recommended) | Recommendation given, not yet implemented |
| **EA-GAP-9** | Detector architecture/checkpoint choice for N02 (the original `pipeline.md` GAP-2) — not yet chosen | Open |
| **EA-GAP-10** | Fine-tune/validation split sizing on LATTE-CXR's ~2,700 images — too few held out gives a noisy validation number, too few used for fine-tuning limits domain adaptation; not yet planned | Open |

---

## 11. Next Steps

1. Download VinDr-CXR (licensing already in hand).
2. Resolve EA-GAP-3 (VinDr multi-rater aggregation) before any training data prep.
3. Build the two taxonomy mapping tables (§6.1) and resolve the EA-GAP/EA-AMBIG items in §10 that block them (EA-GAP-6, 7, 8, EA-AMBIG-1, 2).
4. Fix EA-GAP-5 (the `pleural_other`/"fibrosis" conflation) in the live concept bank — independent of A, worth doing now.
5. Verify EA-GAP-2 (LATTE split vs. MIMIC official splits) and EA-GAP-4 (CheXmask-U coverage) before relying on either for anything load-bearing.
6. Only then: detector architecture choice (EA-GAP-9), fine-tune/validate split plan (EA-GAP-10), and actual training.
