# Glossary and stage guide (2026-10-08)

This file explains every term, rule ID and stage used in the rebuilt pipeline (`v2/`): what it means, where it lives in the code, which stage it belongs to, and why it exists. Numbers are on MIMIC validate unless stated, and MIMIC numbers are always "seen by backbone".

**Watch out for three name clashes:**
- **R1 / R2** mean two different things:
  - as **knowledge rules** in the belief graph (R1 localisation support, R2 side agreement);
  - as **label rules** in `build_labels.py` (R0–R4).

  In this repo "R1/R2" almost always means the knowledge rules.
- **A** means two different things:
  - the **CheXmask anatomy features** (input group "A", also called A1);
  - **region set A** (the 36 Chest ImaGenome boxes) in Stage 4.
- **B** means two different things:
  - the old pipeline's evidence source B (concept similarity);
  - **region set B** (the 9 CheXmask regions).

  Neither is a current input group.

---

## Part 1. The pipeline at a glance (one frontal chest X-ray study in, one report out)

```
frontal image
  │  CLEAR image encoder (frozen)        → 768-d embedding + 32×32 patch tokens
  │  CheXmask lung/heart masks (given)   → 17 anatomy features                      (A, "A1")
  │  patch tokens pooled in 9 CheXmask regions → Stage 4 region classifier → region scores  (R, "A2")
  │  embedding · 77 concept text embeddings → 77 concept scores                     (C)
  ▼
C+A+R head (factorised, linear, named features)  → 14 calibrated probabilities + side probabilities
  ▼
belief graph: bands (D1), side (G6, R2), zone (G7), parent raise (D2), normal call (D3), case evidence (C1)
  ▼                                                     every change logged in the audit trail
template report (rules P1–P8)
  ▼
Stage 8: LLM rewrites the template's statements → RadGraph round-trip check → accept / retry / template fallback
  ▼
final report (FINDINGS + IMPRESSION)
```

---

## Part 2. The stages

Stage numbers follow `PIPELINE_BRIEF.md` (0–7). "Stage 8" is our name for the LLM half of the brief's stage 7. Each build stage had to beat the previous one on a named metric or be dropped; the verdicts are in RESULTS.md.

| Stage | What it is | Input | Output | Status (val) |
|---|---|---|---|---|
| **0 Preprocessing** | Manifests for every dataset; frozen patient-level splits; cached CLEAR embeddings; labels | Raw MIMIC-CXR-JPG, CheXpert labels, MS-CXR, ImaGenome gold list | `data/manifests/*.parquet`, `data/splits/mimic_splits_v1.parquet`, `data/embeddings/clear_frontal_lb_v1.npy`, `data/labels/labels_v2.parquet` | Done. 218,139 studies, one frontal image each |
| **1 Baseline probe** | One logistic regression per finding on the raw CLEAR embedding; Platt calibration; four bands; template | 768-d embedding | 14 probabilities, bands | macro AUROC 0.8305. The reference everything is compared to |
| **2 + CheXmask anatomy** | Stage 1 plus the 17 anatomy features (A) | Embedding + A | 14 probabilities | 0.8350 (+0.0044 vs S1, n.s.) |
| **3 Factorised head** | Concept bottleneck on named features only; predicts P(child \| parent) along is_a, then multiplies down; side head | 77 concept scores + A | 14 probabilities with 0 hierarchy violations; side probabilities | 0.8306; 0 violations (S1 had 392) |
| **4 Region classifier (A2)** | Logistic regression per finding on CLEAR patch features pooled inside anatomical regions | Pooled region vectors (36 ImaGenome boxes = set A; 9 CheXmask regions = set B) | One score per region and finding | MS-CXR: the top region overlaps the box 0.838 (set B); side correct 0.867 |
| **5 Retrieval + association links** | Neighbour label fractions (k most similar fit-split studies); stage-two inputs from KG `associated_with` partners | Embedding index; Stage 1 out-of-fold logits | Extra head inputs | **Both dropped** (no significant gain). Retrieval kept only as case evidence (C1) |
| **Final head C+A+R** | Stage 3 structure with C + A + R as inputs | 77 concepts + 17 anatomy + 31 region features | 14 probabilities, side probabilities | **0.8344**, AUPRC 0.491, ECE 0.024, 0 violations |
| **6 Knowledge graph, grounding, belief graph** | YAML knowledge base; rules that turn probabilities into decisions; audit trail | Head outputs, region scores, KG | Belief graph per study (bands, side, zone, normal call, audit) | Coherent by construction; one unit test per rule (59 tests pass in total) |
| **7 Presentation + template** | Rules P1–P8 decide what is said and how | Belief graph, KG phrases | Template report | Always produces a report |
| **"Stage 8" LLM phrasing + guard** | The LLM rewrites only the template's statements; RadGraph parses the result; a comparator checks it against the graph | Template statements | Final report (LLM text or template fallback) | First pass 0.998, fallback 0 (comparator v2) |
| **Runner** | All of the above in one script, with a frozen manifest | Cached features of a split | `predictions.parquet`, `graphs.jsonl`, `stage8.jsonl`, `reports.jsonl` | Identical to the stage-by-stage results on val |
| **Evaluation** | Metrics fixed in `EVAL_PLAN.md` | Runner output + labels | `eval.json`, per-finding tables | Checked on val; test not run |

---

## Part 3. Rule IDs (they appear in the audit trail of every belief graph)

### D: decision rules (`v2/nesy/belief.py`, Stage 6)
| ID | Name | What it does | Why |
|---|---|---|---|
| **D1** | band | Puts each probability into **present** (p ≥ 0.70), **possible** (0.40 ≤ p < 0.70), **absent** (p at or below a threshold that misses ≤ 5% of positives on the thresh split; 2% for pneumothorax and the root) or **silent** (everything else, never mentioned). Where absent and possible overlap, absent wins. | Turns a probability into a statement with an explicit certainty. The absent band is how the report earns the right to say "No X". |
| **D2** | parent_raise | A parent's band is raised to its most positive *stated* child (present > possible) along an is_a edge. Silent children change nothing. | Keeps the graph coherent: if atelectasis is present, lung opacity cannot be absent. |
| **D3** | no_acute_abnormality | The normal call is true **only if the root `any_abnormality` is in its absent band**. A critical-finding list exists in the KG but is empty, so it is inactive. | One defensible rule for "No acute cardiopulmonary abnormality". The critical lists were a negative result. |
| **D4** | band_check | A check, not a graph change: fails if a finding has an absent threshold but no validate study falls in its absent band. | Stops a configuration where "No X" could never be said. |

### B: graph construction rules (`belief.py`)
| ID | What it does | Why |
|---|---|---|
| **B1** | One node per finding, never one per side; side and anatomy are attributes of the node. | Fixes old-pipeline defect 6, which produced "bilateral and unspecified atelectasis". |
| **B2** | Edges only come from the KG (is_a, associated_with). | Nothing is invented. |
| **B3** | Every change goes through `BeliefGraph.set()`, which writes an audit entry (rule, before, after, reason). | Traceability: every statement can be followed back. |

### G: grounding rules (`v2/nesy/grounding.py`, `pipeline_graph.py`, Stage 6)
| ID | What it does | Active now? |
|---|---|---|
| **G1** | Map a localised region name to an anatomy node (`kg/anatomy.yaml`). | Not used: the current path produces no explicit localisations. |
| **G2** | Walk `part_of` to find lung, side and zone; **never output a lobe** (one frontal image can't determine the lobe). | The "zone not lobe" policy is active through G7 and P8. |
| **G3** | Image-space left/right → the patient's side (the patient's left is on the image right). | Used inside the region features (left/right lung regions). |
| **G4** | `may_occur_in`: flag, never delete, a localisation in anatomy the KG doesn't allow. | **Inactive**: there are no explicit localisations to check. Region relevance is hard-coded (heart for cardiomegaly and enlarged cardiomediastinum, lungs otherwise). |
| **G5** | Attach the cardiothoracic ratio to cardiomegaly; AP views flagged. | Not used by the runner. CTR enters the head as an anatomy feature instead. |
| **G6** | Attach a side to a **stated, lateralisable** finding. In the current pipeline the side comes from the **side head**, P(side \| finding), with its confidence. | **Active.** |
| **G7** | Zone: for lung-parenchyma findings with a side, the highest-scoring lung third on that side ("left lower") is attached if its region score is at least the finding's region threshold. | **Active.** |

### R: knowledge rules on the belief graph (`pipeline_graph.py`; switches in `configs/pipeline.yaml`)
| ID | What it does | Measured on val | Status |
|---|---|---|---|
| **R1** localisation support | A **present** finding whose relevant regions all score below its region threshold is demoted to **possible**. | Precision of present calls 0.788 → 0.796 (+0.008 [+0.003, +0.013]); sensitivity 0.362 → 0.346; 66 demoted | **Off** (your decision; flagged because the CI rule would keep it) |
| **R2** side agreement | A side is stated only if the side head and the side implied by the region scores agree; otherwise the side is withheld. | Side accuracy 0.898 → 0.907 (n.s.); share of findings with a side 39% → 29% | **On** (your decision) |

### C: case evidence (`v2/nesy/case_evidence.py`)
| ID | What it does |
|---|---|
| **C1** | Attaches the 5 most similar fit-split studies (own patient excluded) with how many are positive per finding. Context only: it never changes a probability or band, is not sent to the LLM, and only appears in the report if `show_case_evidence` is on (it is off). Similar cases agree with the model's band in 88% of pairs. |

### P: presentation rules (`v2/nesy/report.py`, Stage 7)
| ID | What it does |
|---|---|
| **P1** omit_parent | Omit a parent when a more specific child is stated with at least the same certainty ("consolidation", not "lung opacity and consolidation"). |
| **P2** pertinent_negative | Say an absent finding only if it is on the KG pertinent-negatives list: pneumothorax, pleural effusion, consolidation, cardiomegaly; edema only when cardiomegaly or effusion is present. |
| **P3** silent | Never mention a silent finding, or a non-reportable node (the root). |
| **P4** hedge_possible | "Possible" findings use the KG's hedged phrase. |
| **P5** always_report | Always return a report, even for an empty or failed graph. |
| **P6** confident_side | State a side only for lateralisable findings with side confidence ≥ 0.8 (`side_min_confidence`). |
| **P7** impression | "No acute cardiopulmonary abnormality." if and only if D3 is true. If the root isn't absent and nothing is stated: "No finding meets the reporting threshold." |
| **P8** zone | Lung-parenchyma findings only: state the zone together with a stated side ("in the right lower zone"); never a lobe. |

### Label rules (`v2/scripts/build_labels.py`, `v2/nesy/labels.py`; Stage 0). Not the same as knowledge rules R1/R2.
| ID | What it does |
|---|---|
| R0 | Study without a CheXpert row → all labels masked. |
| R1 | Blank (not mentioned) → negative (project rule). |
| R2 | −1 (uncertain) → masked, excluded from training and metrics. |
| R3 | A positive child makes every is_a ancestor positive. |
| R4 | An uncertain child with a blank parent masks the parent, cascading upward. |

### Stage 8 comparator mismatch kinds (`v2/nesy/rg_match.py`)
| Kind | Meaning |
|---|---|
| omitted | A statement the graph licenses is missing from the text. |
| added | The text claims a finding the graph doesn't state. |
| added_negative | The text says "no X" where X isn't a pertinent negative. |
| polarity | Present/possible vs absent the wrong way round. |
| certainty | Present vs possible. |
| side / zone | Wrong side or zone, or one given where the graph has none. |
| lobe | Any lobe mentioned. |
| malformed / llm_error | Not in FINDINGS/IMPRESSION form, or no reply. |

---

## Part 4. Terms

### Data and splits
- **MIMIC-CXR-JPG.** The training and in-domain dataset: frontal images, reports and CheXpert labels. **One frontal image per study** (PA before AP); laterals are never used.
- **Splits** (`data/splits/mimic_splits_v1.parquet`, frozen, by patient):
  - **train / fit**: model fitting;
  - **inner holdout**: 5% of fit patients by hash, used to choose C, epochs and early stopping;
  - **calib**: Platt calibration;
  - **thresh**: band thresholds and region thresholds;
  - **val (validate)**: all metrics so far, 1,733 studies;
  - **heldout_loc**: localisation checks;
  - **test**: untouched, 3,041 studies from 289 patients.

  Calibration, thresholds and metrics always use different splits.
- **Exclusions.** MS-CXR patients and Chest ImaGenome gold-standard patients are never in a training split.
- **Seen by backbone.** CLEAR was pretrained on all of MIMIC-CXR, so every MIMIC number is optimistic for that reason. The external sets are not seen.
- **External test sets.** **VinDr-CXR test** (3,000 Vietnamese images, consensus labels, no reports) and **PadChest-GR** (4,555 Spanish studies with grounded report sentences). Never used for training or selection. Their labels map to ours through `kg/external_maps/*.yaml`, which are drafts for review.
- **MS-CXR.** MIMIC images with phrase-level boxes, used only to check Stage 4 localisation.
- **Chest ImaGenome.** Scene graphs over MIMIC with 36 anatomical regions. Its explicit region-level finding labels train Stage 4.
- **CheXmask.** Precomputed lung and heart masks for MIMIC, VinDr and PadChest. **RCA** (Dice RCA) is its per-image quality score; **QC ok** means RCA ≥ 0.7 and all three masks present.
- **Radiologist labels (the 687).** `mimic-cxr-2.1.0-test-set-labeled.csv`. 596 are in our test split with a frontal image; these are the primary test labels in EVAL_PLAN.md.
- **Labels (labels_v2).** CheXpert labels from the reports, after label rules R0–R4, plus the root.

### Image encoder and preprocessing
- **CLEAR.** The frozen vision-language model: a DINOv2 ViT-B/14 image encoder with registers, plus a text encoder. It provides the 768-d image embedding, the 32×32 patch tokens, and text embeddings for concept sentences.
- **Letterbox vs stretch.** Letterbox keeps the aspect ratio and zero-pads to a square (the CLEAR authors' preprocessing). Stretch distorts the image to a square. We use **letterbox**: it was better on every stage, for example +0.0065 AUROC at Stage 1.
- **Concept bank.** CLEAR's 368,294 report sentences with precomputed text embeddings.

### Inputs to the head
- **C, concepts.** 77 short affirmative phrases ("bibasilar opacities", "cardiomegaly"…), selected on the fit split from the concept bank:
  - temporal, negated and hedged sentences dropped;
  - the phrase must name its finding;
  - AUROC ≥ 0.60 for its own finding;
  - 6 per finding, then pruned.

  The feature is the cosine of image and text embeddings, standardised. File: `data/concepts/concepts_final_k6_pruned.csv`.
- **A, anatomy features (A1).** 17 features from the CheXmask masks:
  - cardiothoracic ratio, in two versions;
  - heart and thorax widths;
  - lung area and height fractions;
  - base and apex height differences;
  - heart shift;
  - mask areas;
  - RCA;
  - CTR × AP;
  - a missing flag;
  - the AP view flag.

  **Missing is never encoded as zero**: values are imputed with the fit-split mean and an indicator is set.
- **R, region scores (A2).** From Stage 4 on the 9 CheXmask regions (set B): per finding, the max over all regions, the max over left-lung regions and the max over right-lung regions, plus a missing flag. That makes 31 features.
- **Region sets.** **A** is the 36 Chest ImaGenome boxes; **B** is the 9 CheXmask regions: left and right lung, their upper/middle/lower thirds, and the heart. Set B is primary.
- **Region thresholds.** Per-finding thresholds on region scores, fitted on thresh. They decide when a region "supports" a finding (used by R1, R2, G7).
- **Cross-fitting / out-of-fold.** Stage 4 is trained in 2 patient folds. A fit-split image is scored by the fold model that didn't see it, so the head never trains on in-sample region scores. Checked: no leakage.
- **D, retrieval** (old letter). Neighbour label fractions from FAISS nearest neighbours. Not a head input now; used only as case evidence (C1).
- **Association links** (`associated_with`). KG pairs such as effusion–atelectasis whose Stage 1 logits were fed to each other's models. Dropped.

### The head
- **CBM (concept bottleneck model).** The prediction is a linear function of named, human-readable features only, so each finding's evidence can be read as weights times values.
- **Factorised head.** For each is_a edge it models **P(child | parent)**, trained only on studies where the parent is positive. Roots get P(f). Marginals are multiplied down the tree, so a child can never exceed its parent: **0 hierarchy violations by construction**.
- **Side head.** Multinomial logistic regression P(left / right / bilateral | finding), only for **lateralisable** findings. Trained on ImaGenome side labels where the report states the side explicitly.
- **Platt calibration.** A 1-D logistic regression on the model score, fitted on calib. Isotonic calibration is banned (old defect 2).
- **H3 comparator / probe on embedding + A + R.** An unconstrained linear model with the same information plus the raw embedding. It measures what the constraints cost: on val, head − probe = −0.0010 AUROC (n.s.).
- **DenseNet-121.** A conventional CNN baseline fine-tuned on the same images and labels. 0.8175 AUROC, 164 hierarchy violations.
- **MLP head.** A non-linear head tried as an ablation. No gain; dropped.

### Bands and decisions
- **Bands.** present / possible / absent / silent (rule D1). **Silent** means the model isn't confident either way, so the report says nothing.
- **Root `any_abnormality`.** A KG node above every pathology, not a CheXpert label. It is **non-reportable** (never a sentence); its absent band drives the normal call. **no_finding** = the root is absent.
- **Normal call.** "No acute cardiopulmonary abnormality." On val it covers 13.9% of studies with precision 0.938.
- **Pertinent negatives.** The few absent findings a radiologist routinely states (see P2).

### Knowledge graph (`v2/kg/`, read by plain Python; no triple store, Prolog or LTN)
- **findings.yaml.** The 14 nodes, each with:
  - a label;
  - the CheXpert column;
  - the `lateralisable` and `reportable` flags;
  - report phrases for present, possible and absent;
  - a glossary definition and its source.

  It also holds:
  - **is_a**: isa_001–006 from the CheXpert hierarchy (atelectasis, consolidation, edema and lung lesion under lung opacity; pneumonia under consolidation; cardiomegaly under enlarged cardiomediastinum), and isa_101–106 linking every pathology to the root (support devices are not a pathology);
  - **may_occur_in**: hand-written, needs review, not consulted at run time;
  - **associated_with**: dropped as inputs;
  - **pertinent_negatives** (pn_001–005);
  - **normal_call**;
  - **side_min_confidence** (0.8).
- **anatomy.yaml.** Anatomy nodes with **part_of** (for example "left lower lung zone" is part of "left lung"), RadLex IDs, and the ImaGenome region → node map.
- **radlex_map.yaml / devices.yaml.** RadLex 4.3 IDs for findings, zones, anatomy and device types, each marked exact / approximate / none.
- **RadLex.** RSNA's radiology lexicon, used for IDs and the hierarchy comparison only.
- **Fleischner glossary.** The Fleischner Society's thoracic imaging glossary (2024), used for definitions and the task-7 concept ablation.
- **RadReport "Rad Chest 2 Views".** RSNA template, used for the report-layout proposal (not applied).
- **Source field.** Every KG row names where it comes from. 67 of 255 rows are "author-curated, needs review".

### Stage 8 (LLM phrasing)
- **LLM.** KCL's endpoint, model `arc:nano` (Qwen 27B, no thinking mode). The key is read at run time from `~/.rrg_llm_key` and never logged. It receives **only** the template's statements: finding, certainty, side, zone, normal yes/no. No IDs, report text or case evidence.
- **RadGraph.** An information-extraction model (`modern-radgraph-xl`) that tags observations and anatomy in report text, with certainty: definitely present / uncertain / definitely absent.
- **Comparator** (`rg_match.py`). Turns a RadGraph parse into claims (finding, certainty, side, zone) and compares them with the graph.
  - **v1** was the first version.
  - **v2** (2026-10-07) also reads sentences RadGraph missed, takes device and heart words from the sentence, knows "cardiomediastinum", and doesn't spread a summary sentence's side to every finding in it.
- **Guard / round trip.** Generated text is accepted only if the comparator finds exactly the licensed statements.
- **First-pass match.** Accepted on the first attempt. **Retry**: resend with feedback naming the mismatch and quoting the template sentence (up to 3 attempts in total). **Fallback**: the template is used after the last failure.
- **Template round trip.** The comparator run on the template itself, as a check on the comparator (0.999 with v2).

### Evaluation and statistics
- **AUROC.** Ranking quality per finding; 0.5 is chance. **Macro** = the mean over the 14 outputs.
- **AUPRC.** Precision-recall area; sensitive to prevalence.
- **ECE.** Expected calibration error, with 15 bins: how far predicted probabilities are from observed rates. **Brier**: mean squared error of the probabilities.
- **Hierarchy violation.** P(child) > P(parent) in one study.
- **Patient bootstrap.** Resample patients with replacement 2,000 times; 95% percentile intervals. Paired when comparing two models. Bootstrap ECE intervals are biased upward, so ECE comparisons use paired differences.
- **CI rule.** An accuracy-only add-on is kept only if the CI of its gain excludes 0. Applied to retrieval, association links and R1/R2.
- **No-harm rule.** A and R are dropped only if removing them is significantly better, because they also feed side, zone and measurement.
- **H3.** The hypothesis that the C+A+R head is non-inferior to the unconstrained probe: the lower 95% bound of the head-minus-probe macro AUROC must be above **−0.02** (margin). Direction to be confirmed by you.
- **Report-level F1.** Per finding, positive mentions in our final report vs in the radiologist's report, both read by comparator v2. **RadGraph entity F1**: exact token-and-label entity overlap. Val: micro F1 0.563, entity F1 0.165.
- **Present precision / sensitivity, absent miss rate.** Band-level operating points (`bands.csv`).

### Infrastructure
- **Runner** (`scripts/run_pipeline.py`, `configs/pipeline.yaml`). One command from features to report. Every undecided choice is a config switch. It refuses test (`--allow-test`) and external sets (`--allow-external`).
- **Manifest** (`manifest.json` in each runner run). The SHA-256 of every model, threshold, KG and code file, used as the freeze record.
- **Equivalence check.** Proves the runner reproduces earlier results: probabilities within 1e-5, identical bands and templates.
- **External feature path** (`nesy/ext_features.py`). Builds C, A and R for non-MIMIC images from cached features and the **retrained Stage 4 models** (`runs/20261007-121534_s4-saved`).
- **Run directory** (`v2/runs/<time>_<name>/`): `config.json`, `log.txt`, `metrics.jsonl`, `STATUS` (RUNNING / DONE / FAILED).
- **Lane / orchestrator** (`scripts/lane_*.sh`). A chain of SLURM steps run unattended; it logs START/END to `orchestrator.log`.
- **Allocations.** Held SLURM jobs (H100, L40S, CPU) that steps run inside. See COMPUTE.md.
