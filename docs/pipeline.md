# Evidence-Based Radiology Report Generation — Technical Specification

**Source:** `pipeline.pdf` (single page, 1640 × 1367 pt logical canvas), title *"Evidence-based radiology report generation"*, subtitle *"Evidence generation → fusion into a belief graph → report generation with fact verification"*.

**Status:** Specification only. No implementation decisions are final; every item marked **[AMBIG-n]** or **[GAP-n]** requires a decision before coding.

**How this document was derived:** node labels and connector geometry were extracted directly from the PDF content stream (text matrices + stroked polylines + arrowhead placement matrices). Section 2 is a faithful transcription of the drawing. Sections 3–8 are the specification built on top of it, incorporating the author's accompanying prose description. Where the prose and the drawing disagree, both readings are recorded and the conflict is raised as an ambiguity. **§2 is retained as a historical record of the original diagram and is not re-derived from the PDF on subsequent edits — all revisions below are tracked in the Revision Log and applied on top of it.**

---

## 0. Revision Log

| Rev | Decision | Rationale / driven by |
|---|---|---|
| 1 | **Verification loop removed for v1.** Stage 3 is now single-pass: `Draft report → RadGraph entity extraction → Facts verification (filter, don't gate) → Final report`. `Regenerate report (LLM)` (N32) and the `pass`/`fail` branch (E31–E33) are **deferred, not deleted** — the diagram transcription in §2 is untouched, but §6 now describes the v1 (no-loop) behaviour as the authoritative spec, with the original loop design kept as "Phase 2 (deferred)". This resolves **GAP-10** and **AMBIG-9** by deferral rather than by design decision. | User decision, this session |
| 1 | **368k concept refinement (N12) redesigned around automatic, not manual, concept tagging.** Manual per-concept labelling of 368,294 items is infeasible; §4.4/§4.6 now specify a two-tier scheme — lightweight automatic tagging (regex laterality + gazetteer/parser anatomy) applied to all 368k concepts, with heavier ontological reasoning reserved for the much smaller canonical vocabulary at N25. This resolves **GAP-4** with a concrete construction method (still requires validation, see below). | User decision, this session |
| 1 | **N02 (Spatial perception) treated as phase-deferred, not failed.** §4.2/§7.2 degraded-mode guidance is split into "N02 unavailable at runtime" (hard-fail, production) vs. "N02 not yet built" (planned phase-1 mode: Evidence A absent by design, N12 runs in identity mode, 3-source fusion B+C+D). Build to the "Evidence A absent" contract first. | User decision, this session |
| 1 | **KG recommendation added.** §5.4/§8 now recommend a layered KG: RadLex as the CXR-specific anatomy/finding lexicon, SNOMED CT's finding-site/laterality qualifier model (via UMLS) for the N25 consistency rules, and UMLS as the entity-linking backbone that automates the N12 concept-tagging step above. | User decision, this session |
| 2 | **KG stack simplified to RadLex-only.** SNOMED CT and UMLS dropped from the v1 plan — not needed at this scale and both carry licensing overhead RadLex doesn't. RadLex (sourced via the BioPortal API/ontology download, no licensing gate) now covers both the N12 auto-tagging lexicon and the N25 rule vocabulary; N25's finding/anatomy/laterality/exclusivity rules are hand-curated directly over the small canonical vocabulary instead of derived from SNOMED's relationship model. SNOMED/UMLS become an optional later enhancement, only if RadLex-only concept-tagging coverage (§4.4's QA sample) proves insufficient. See §5.4. | User decision, this session |
| 3 | **N12 redesigned: Evidence A removed as a refinement input, permanently — not just deferred.** The original mechanism (down-weight 368k concepts using Evidence A's spatial context, edge `+ Evidence A` into N12) is **superseded**, not merely paused for v1. Reasons: (a) gating B on A's raw output means any error A makes propagates directly into what counts as evidence, with no cross-check on A's own reliability; (b) flat top-K over 368k concepts was independently found to be the wrong mechanism regardless of A — it surfaces whichever finding has the most templated phrasing variants in MIMIC's reporting style (measured: 4 of the top 5 hits for one sample image were near-duplicate device-tube phrasings, §10), not the most likely finding. Replaced by: RadGraph-parse every concept once offline, group by `(canonical_finding, AnatomyID, laterality)`, resolve each concept's temporal/comparison language into a present/absent polarity, and rank **groups** rather than raw phrasings. See the rewritten §4.6. Evidence A, once N02 exists, still reaches fusion independently via the E21a bus (N06→N22) — only its use as an N12 refinement signal is removed. This retires **AMBIG-2** (suppressive-vs-promotive no longer applies — there is no more A-conditioned refinement to be suppressive or promotive about) and adds **GAP-19/20/21** (laterality-in-group-key, the temporal/polarity gazetteer, and RadGraph's compound-sentence behaviour are all new, unvalidated design choices — see §8). | User decision, this session |

---

## 1. Scope and Intent

The pipeline accepts a chest X-ray (CXR) image and emits a natural-language radiology report. Its defining architectural claim is that **the language model is not the locus of clinical decision-making**. Clinical content is decided by a neuro-symbolic evidence stack, frozen into an inspectable **verified belief graph**, and only then handed to an LLM for *linguistic realisation*. A fact-verification pass then checks that the generated prose asserts only what the belief graph licenses, and strips out anything it doesn't (v1: single-pass filtering — see Revision Log rev 1; the original design closes this into a regenerate/re-verify loop, deferred to Phase 2).

Three sequential stages, as labelled in the diagram:

| Stage | Banner text in diagram | Responsibility |
|---|---|---|
| 1 | `STAGE 1 · EVIDENCE GENERATION` | Produce four heterogeneous, independent evidence streams (A, B, C, D) from one image |
| 2 | `STAGE 2 · EVIDENCE FUSION & BELIEF GRAPH` | Normalise, calibrate, fuse; build and symbolically verify a belief graph |
| 3 | `STAGE 3 · REPORT GENERATION & VERIFICATION` | Realise prose, re-extract its claims, verify against the belief graph, **filter out unsupported claims (v1: single pass, no regeneration loop — rev 1)** |

---

## 2. Diagram Fidelity — Verbatim Node and Edge Inventory

### 2.1 Node inventory

Nodes are laid out on a 5-column grid (column centres at x ≈ 200, 510, 820, 1130, 1440) and 13 rows. Column/row given for traceability against the source drawing.

| # | Node label (verbatim) | Col | Row (y) | Stage |
|---|---|---|---|---|
| N01 | `X-ray image` | 3 | 201 | 1 |
| N02 | `Spatial perception model` | 1 | 289 | 1 |
| N03 | `CLEAR encoder` | 3 | 289 | 1 |
| N04 | `Region + laterality features` | 1 | 377 | 1 |
| N05 | `Image embedding` | 3 | 377 | 1 |
| N06 | `Evidence A: Spatial findings` | 1 | 457 | 1 |
| N07 | `Similarity vs 368k concepts` | 3 | 465 | 1 |
| N08 | `368k concept vocabulary bank` | 4 | 457 | 1 |
| N09 | `FAISS retrieval` | 5 | 465 | 1 |
| N10 | `368,294-d concept similarity scores` | 3 | 545 | 1 |
| N11 | `Top-k similar reports` | 5 | 553 | 1 |
| N12 | `KG-guided refinement` | 2 | 641 | 1 |
| N13 | `Filter 68 concept scores` | 3 | 641 | 1 |
| N14 | `Predefined CLEAR 68 concepts` | 4 | 633 | 1 |
| N15 | `RadGraph parsing` | 5 | 641 | 1 |
| N16 | `Revised 368k scores` | 2 | 729 | 1 |
| N17 | `CBM` | 3 | 729 | 1 |
| N18 | `Evidence D: Retrieved facts` | 5 | 721 | 1 |
| N19 | `Top-k concepts` | 2 | 817 | 1 |
| N20 | `Evidence C: Pathology predictions` | 3 | 809 | 1 |
| N21 | `Evidence B: Refined concepts` | 2 | 897 | 1 |
| N22 | `Evidence normalisation` | 1 | 1061 | 2 |
| N23 | `Score calibration + fusion` | 2 | 1061 | 2 |
| N24 | `Initial belief graph` | 3 | 1061 | 2 |
| N25 | `KG + medical rule check` | 4 | 1061 | 2 |
| N26 | `Verified belief graph` | 5 | 1061 | 2 |
| N27 | `LLM report generation` | 1 | 1181 | 3 |
| N28 | `Draft report` | 2 | 1181 | 3 |
| N29 | `RadGraph entity extraction` | 3 | 1181 | 3 |
| N30 | `Facts verification` | 4 | 1181 | 3 |
| N31 | `Final report` | 5 | 1181 | 3 |
| N32 | `Regenerate report (LLM)` — *deferred to Phase 2, rev 1* | 3 | 1269 | 3 |

Edge annotations present in the drawing: `+ Evidence A` (at x≈241, y≈655, attached to E21b), `pass` (y≈1163, on E28), `fail` (y≈1227, on E29).

### 2.2 Edge inventory (all 33 connectors, with arrowhead direction)

| # | From → To | Geometry note |
|---|---|---|
| E01 | N01 `X-ray image` → N02 `Spatial perception model` | branches left at y=245 |
| E02 | N01 → N03 `CLEAR encoder` | straight down |
| E03 | N02 → N04 `Region + laterality features` | |
| E04 | N04 → N06 `Evidence A: Spatial findings` | |
| E05 | N03 → N05 `Image embedding` | |
| E06 | N05 → N07 `Similarity vs 368k concepts` | |
| E07 | N05 → N09 `FAISS retrieval` | branches right at y=417 |
| E08 | N08 `368k concept vocabulary bank` → N07 | **right-to-left**; bank is a resource fed *into* the similarity op |
| E09 | N07 → N10 `368,294-d concept similarity scores` | |
| E10 | N10 → N13 `Filter 68 concept scores` | |
| E11 | N10 → N12 `KG-guided refinement` | branches left at y=593 |
| E12 | N14 `Predefined CLEAR 68 concepts` → N13 | **right-to-left**; resource feed |
| E13 | N13 → N17 `CBM` | |
| E14 | N17 → N20 `Evidence C: Pathology predictions` | |
| E15 | N12 → N16 `Revised 368k scores` | |
| E16 | N16 → N19 `Top-k concepts` | |
| E17 | N19 → N21 `Evidence B: Refined concepts` | |
| E18 | N09 → N11 `Top-k similar reports` | |
| E19 | N11 → N15 `RadGraph parsing` | |
| E20 | N15 → N18 `Evidence D: Retrieved facts` | |
| E21a | N06 `Evidence A` → evidence bus (vertical trunk x=200, y 489→967) | |
| E21b | N06 trunk → N12 `KG-guided refinement` | **horizontal branch at y=637, arrowhead entering N12 from the left, labelled `+ Evidence A`** |
| E22 | N21, N20, N18 → evidence bus (y=967) → N22 `Evidence normalisation` | single compound path: drops from B (x=510), C (x=820), D (x=1440) onto a horizontal bus at y=967 spanning x 200→1440, then down at x=200 into N22 |
| E23 | N22 → N23 `Score calibration + fusion` | |
| E24 | N23 → N24 `Initial belief graph` | |
| E25 | N24 → N25 `KG + medical rule check` | |
| E26 | N25 → N26 `Verified belief graph` | |
| E27 | N26 → N27 `LLM report generation` | wraps: right edge down to y=1121, left to x=200, down into N27 |
| E28 | N27 → N28 `Draft report` | |
| E29 | N28 → N29 `RadGraph entity extraction` | |
| E30 | N29 → N30 `Facts verification` | |
| E31 | N30 → N31 `Final report` | labelled `pass` — *v1 (rev 1): this is now the only path out of N30, taken unconditionally after filtering; see §6* |
| E32 | N30 → N32 `Regenerate report (LLM)` | labelled `fail`; down from x=1130 to y=1265, then left into N32 — *deferred, rev 1* |
| E33 | N32 → N28 `Draft report` | left to x=510, then **upward** arrowhead into N28 — *deferred, rev 1* |

### 2.3 Topological facts worth stating explicitly

1. **Two independent perception paths from the raw image.** N02 (spatial) and N03 (CLEAR) both read N01 directly. Spatial evidence is *not* derived from the CLEAR embedding.
2. **The CBM branch is fed from the RAW 368k score vector, not the refined one.** E10 originates at N10 (`368,294-d concept similarity scores`), not at N16 (`Revised 368k scores`). Evidence C is therefore *not* KG-corrected. See **[AMBIG-3]**.
3. **Evidence A has two consumers in the original diagram.** It was both a first-class evidence stream into fusion (E21a) and a conditioning input to concept refinement (E21b). *(Superseded, rev 3: the E21b consumer — Evidence A gating N12 — is removed from the design entirely, not just deferred. Only E21a (fusion) remains; Evidence A is no longer structurally privileged. See Revision Log rev 3 and §4.6.)*
4. **The two concept resources (N08, N14) are data assets, not compute stages** — both are drawn feeding right-to-left into an operation.
5. **The regeneration loop re-enters at `Draft report` (N28), not at `LLM report generation` (N27).** Regeneration is a distinct node with its own (constrained) prompt contract. *(Deferred to Phase 2 — rev 1.)*
6. **`Regenerate report (LLM)` (N32) has exactly one drawn input: the `fail` edge.** It has no drawn edge from N26 `Verified belief graph`, although the prose states regeneration uses the verified evidence. See **[AMBIG-9]**. *(Moot in v1 — N32 deferred, rev 1.)*
7. **The verification loop has no drawn terminating condition.** No max-iteration counter, no fallback path. See **[GAP-10]**. *(Moot in v1 — no loop, rev 1. Re-evaluate when Phase 2 reintroduces regeneration.)*

---

## 3. Global Conventions and Shared Data Contracts

These types are referenced throughout. They are the contract surface between components; every module boundary below is expressed in terms of them.

### 3.1 Identifier spaces

| Space | Description | Cardinality | Owner |
|---|---|---|---|
| `ConceptID` | Index into the 368k concept bank | 368,294 | N08 |
| `CBMConceptID` | Index into CLEAR's predefined concept set | 68 | N14 |
| `PathologyLabel` | CBM output label space (e.g. CheXpert-style findings) | **unspecified** — see [GAP-1] | N17 |
| `AnatomyID` | Anatomical region term, KG-resolvable | unspecified | KG |
| `Laterality` | Enum: `left \| right \| bilateral \| midline \| unspecified` | 5 | fixed |
| `EvidenceSourceID` | Enum: `A \| B \| C \| D` | 4 | fixed |

**Requirement:** `ConceptID` ordering must be *byte-stable across releases*. The 368,294-d score vector is positional; any reordering or re-versioning of the bank silently invalidates every cached score, every fitted calibrator, and every KG mapping. The bank must ship with a content hash that is recorded in every artifact produced downstream.

### 3.2 Core types

```
Finding:
  finding_id: str                 # stable within a study
  label: str                      # canonical finding term
  anatomy: AnatomyID | null
  laterality: Laterality
  attributes: dict[str, str]      # e.g. severity=mild, size=2cm, character=patchy
  polarity: "present" | "absent" | "uncertain"
  confidence: float               # [0,1], calibrated post-fusion
  support: list[EvidenceRef]

EvidenceRef:
  source: EvidenceSourceID
  raw_score: float                # source-native scale, pre-normalisation
  norm_score: float               # [0,1] after N22
  provenance: dict                # source-specific: concept_id, bbox, report_id, ...

BeliefGraph:
  findings: list[Finding]
  relations: list[Relation]       # located_at, modifies, suggests, contradicts
  study_meta: StudyMeta
  graph_version: "initial" | "verified"
  audit: list[RuleApplication]    # populated by N25
```

### 3.3 Non-negotiable invariants

- **INV-1 (no orphan claims):** every `Finding` in the initial belief graph carries ≥1 `EvidenceRef`.
- **INV-2 (traceability):** every field of every `Finding` in the verified graph is traceable to a source evidence item or a named KG/medical rule.
- **INV-3 (LLM containment):** the LLM prompt in N27/N32 contains *only* content derivable from the verified belief graph plus fixed style instructions. Raw retrieved report text (N11) must never reach the LLM verbatim — retrieval contributes through N15 (`RadGraph parsing`) as structured facts only. This is what makes N30 a meaningful check rather than a self-consistency check.
- **INV-4 (determinism of the symbolic layer):** Stages 1 and 2 must be reproducible bit-for-bit given a fixed image, fixed model checkpoints, and fixed KG/rule versions. Only the LLM steps are permitted to be stochastic.
- **INV-5 (no silent partial evidence):** if any of A/B/C/D is unavailable, the pipeline must record the degradation in `study_meta` and propagate it to the final report metadata — it must not silently produce a confident report from three sources. See §7.2.

---

## 4. Stage 1 — Evidence Generation

### 4.1 N01 · `X-ray image`

**Role:** pipeline entry point.

| Aspect | Specification |
|---|---|
| Input | DICOM file/series, or pre-decoded PNG/JPEG |
| Output | `ImageTensor` + `StudyMeta` |
| Fan-out | E01 → N02, E02 → N03 (identical source, two independent preprocessings) |

**Required preprocessing (must be specified per-branch — the two branches will not share it):**
- DICOM decode; apply `RescaleSlope`/`Intercept`, VOI LUT / window-level; honour `PhotometricInterpretation` (MONOCHROME1 must be inverted).
- Resize/pad policy. **Critical:** N02's outputs are spatial, so its resize must be *recorded* (scale + pad offsets) to map region predictions back to original image coordinates. N03's resize need not be invertible.
- Normalisation constants are checkpoint-specific and must come from the respective model's config, not a global default.

**`StudyMeta` fields to capture:** `study_uid`, `series_uid`, `view_position` (PA/AP/LATERAL), `patient_orientation`, `pixel_spacing`, `acquisition_datetime`, `is_inverted`.

**Failure paths:**
| Condition | Behaviour |
|---|---|
| Corrupt/unreadable DICOM | Hard fail, `E_INPUT_DECODE`, no report |
| Non-CXR modality or body part | Hard fail, `E_INPUT_MODALITY` — must be checked, a knee radiograph must not silently produce a chest report |
| Lateral-only view | **[AMBIG-1]** — diagram assumes one image; see §8 |
| Multi-image study | **[AMBIG-1]** |

**Configuration:** `input.accepted_modalities`, `input.accepted_body_parts`, `input.target_size_spatial`, `input.target_size_clear`, `input.window_policy`.

---

### 4.2 N02 · `Spatial perception model` → N04 `Region + laterality features` → N06 `Evidence A: Spatial findings`

**Role:** independent localised perception. Answers *what* and *where*, without reference to the CLEAR embedding space. This is the branch that gives the system explicit spatial grounding instead of relying on global image-level similarity.

**[Rev 1] N02 deferred — build-phase note.** N02 is architecturally independent of every other Stage-1 branch (it is the only module not downstream of N03/CLEAR), so it can be built last without blocking anything else. While deferred:
- `EvidenceA` is `absent` (not merely empty) for every study — flag `study_meta.spatial_unavailable=true`, distinct from "spatial model ran and found nothing" (§ Failure paths below).
- **N12 is unaffected by N02's absence** *(rev 3: N12 no longer takes Evidence A as an input at all — see §4.6 and Revision Log rev 3)*. Evidence B is produced by RadGraph-based grouping and temporal/polarity resolution regardless of whether N02 exists; there is no degraded path to specify here any more because the mechanism never depended on Evidence A.
- **Fusion runs on 3 sources (B+C+D)** instead of 4 — the belief graph loses its only pixel-grounded anatomy/laterality source. Anatomy/laterality attribution for findings falls back to whatever lexical anatomy/laterality the 368k concept text happens to encode (via §4.4's auto-tags) and to RadGraph's anatomy entities from Evidence D / the draft extraction — i.e. attribution becomes *"the vocabulary asserts a location"* rather than *"the model localised it in this image"*. Record this as a known quality limitation of the N02-deferred build, not silently.
- **Recommendation:** build against the `EvidenceA=absent` contract from day one (§5.2, §5.4 already specify required behaviour for this case) so that adding N02 later is additive and doesn't require touching Stage 2/3 contracts. The **runtime** degraded-mode policy in §7.2 (hard-fail on N02 failure) is a *separate, production-time* statement and does not apply to this planned build-phase absence — see §7.2's rev-1 note.

#### N02 — the detector/localiser

| Aspect | Specification |
|---|---|
| Input | `ImageTensor` (spatial branch preprocessing) + invertible resize transform |
| Output | `list[SpatialDetection]` |
| Model class | Region-level CXR model — detection, anatomy segmentation, or grounded classification. **Architecture and checkpoint unspecified in the diagram → [GAP-2]** |

```
SpatialDetection:
  finding_label: str
  score: float                    # model-native, NOT calibrated
  bbox: [x0,y0,x1,y1]             # original-image pixel coords
  mask: RLE | null
```

#### N04 — region + laterality derivation

Pure geometric/symbolic post-processing. No learned parameters.

| Step | Specification |
|---|---|
| Coordinate restoration | Invert N01's spatial-branch resize/pad to original pixel space |
| Anatomical assignment | Map each bbox/mask to `AnatomyID` — either via an anatomy segmentation mask (preferred, deterministic overlap rule) or a zone lookup (upper/mid/lower × left/right lung field, cardiac silhouette, mediastinum, costophrenic angles, apices, hila, diaphragm) |
| Laterality derivation | Derived from anatomy assignment **and** `patient_orientation`. **Must not** be inferred from raw image-x alone: on an AP film the patient's left is image-right, and flipped/mirrored acquisitions exist |
| Overlap resolution | Deterministic tie-break when a box straddles regions — e.g. assign to max-IoU region; emit `bilateral` only above a configured bilateral-overlap ratio |

**Output → N06 `Evidence A: Spatial findings`:**
```
EvidenceA = list[{
  label: str, anatomy: AnatomyID, laterality: Laterality,
  score: float, bbox: [...], area_frac: float
}]
```

**Downstream consumers:** N22 (fusion, via E21a) **and** N12 (concept refinement, via E21b, labelled `+ Evidence A`).

**Failure paths:**
| Condition | Behaviour |
|---|---|
| Zero detections | Emit empty `EvidenceA`; **do not** treat as failure — a normal CXR legitimately yields none. Must be distinguishable from "module errored" |
| Anatomy mapping fails for a detection | Emit with `anatomy=null`, `laterality=unspecified`; must not be silently dropped |
| `patient_orientation` missing | Fall back to `view_position`; if both missing, emit `laterality=unspecified` and flag `study_meta.laterality_unreliable=true`. Never guess |
| Model timeout/OOM | Degraded mode — see §7.2 |

**Configuration:** `spatial.checkpoint`, `spatial.score_threshold`, `spatial.nms_iou`, `spatial.max_detections`, `spatial.anatomy_atlas_version`, `spatial.bilateral_overlap_ratio`.

---

### 4.3 N03 · `CLEAR encoder` → N05 `Image embedding`

**Role:** map the image into the joint image–text embedding space shared with the concept bank. Single point of dependency for both the concept branch and the retrieval branch.

| Aspect | Specification |
|---|---|
| Input | `ImageTensor` (CLEAR-branch preprocessing) |
| Output | `ImageEmbedding: float32[D]` |
| Contract | L2-normalised if downstream similarity is cosine — normalisation must happen **exactly once** and in a documented place |
| Fan-out | E06 → N07 (concept similarity), E07 → N09 (FAISS retrieval) — **the same vector feeds both**; the FAISS index and the concept bank must be built in this identical space |

**Hard requirements:**
- Embedding dimension `D`, the checkpoint hash, and the normalisation convention must be recorded in every artifact. A checkpoint swap invalidates the concept bank embeddings, the FAISS index, and all calibration.
- The 368k concept bank text embeddings, the FAISS image index, and this encoder must be produced by **one** model version. A version triple mismatch is a hard startup failure, not a runtime warning.

**Failure paths:** encoder load failure → hard fail (`E_CLEAR_LOAD`); NaN/Inf in embedding → hard fail (`E_CLEAR_NAN`), never propagate.

**Configuration:** `clear.checkpoint`, `clear.embed_dim`, `clear.normalise`, `clear.device`, `clear.batch_size`, `clear.precision`.

**[GAP-3]** The diagram does not identify the CLEAR model, its checkpoint, or its training corpus. This must be pinned before anything else is built — it is the root dependency of three of the four evidence streams.

---

### 4.4 N08 · `368k concept vocabulary bank` (resource)

**Role:** the large-vocabulary radiological observation lexicon against which the image is scored. Drawn feeding *into* N07 (E08) — it is a static asset, not a compute stage.

| Aspect | Specification |
|---|---|
| Contents | 368,294 concept entries |
| Per-entry fields (required) | `concept_id: int`, `text: str`, and — for grouping and temporal/polarity resolution to be possible at all (§4.6, rev 3) — `anatomy: AnatomyID \| null`, `laterality: Laterality`, `resolved_polarity: present\|absent`, `temporal_class: stationary\|implies_present\|implies_absent\|indeterminate`, `canonical_finding: str \| null`, `group_key: str \| null`, `attributes: dict`, `kg_uri: str \| null` |
| Materialised form | `float32[368294, D]` text-embedding matrix (≈ 1.4 GB at D=1024, fp32; ≈ 0.7 GB fp16) + a metadata sidecar table |
| Versioning | `bank_version`, `bank_sha256`, `built_with_clear_checkpoint` |

**[GAP-4] — RESOLVED (rev 1, extended rev 3): automatic tagging, not manual annotation.** Grouping and temporal/polarity resolution (§4.6, rev 3) can only work if each of the 368,294 concepts carries `(anatomy, laterality, polarity)` tags — this requirement is unchanged from rev 1, only its downstream consumer changed (grouping instead of Evidence-A-conditioned down-weighting). Manually labelling 368k items is infeasible and is **not** the intended construction method — the concepts are short, near-templated phrases (e.g. *"left lower lobe consolidation"*), which makes them tractable for a one-time, fully automatic, offline batch-tagging job:

1. **Laterality — deterministic keyword match.** Regex/gazetteer over `{left, right, bilateral, midline}` and their synonyms (`Lt`, `Rt`, `L/R`, …) against the concept text. Laterality is almost always lexicalised in these phrases, so this step is near-100%-precision and needs no model. Concepts with no lateral token get `laterality=unspecified` (correct for genuinely non-lateral findings like "cardiomegaly").
2. **Anatomy — automatic entity extraction, reusing infrastructure already in the pipeline.** Run the **same RadGraph model already used at N15/N29** over each of the 368k concept strings (once, offline) and take its `Anatomy` entities as the tag; or, more cheaply, substring/fuzzy-match against a chest-anatomy gazetteer derived from RadLex's chest subtree (~500–800 terms — see the KG recommendation below). Concepts matching neither get `anatomy=null` and are excluded from the anatomy-consistency rule (see policy below) but the laterality rule can still fire independently.
3. **Polarity** — default every concept to `asserted` (the bank is built from positive-observation phrases); flip to `explicitly-negative` only on a detected negation cue (`no`, `without`, `clear of`, …). No manual work required.
4. **Anatomy-term → `AnatomyID` mapping** — the one part of this pipeline worth a small amount of human curation, because it's small: mapping the few hundred distinct gazetteer/RadLex anatomy *terms* (not all 368k concepts) onto the fixed `AnatomyID` zone taxonomy shared with the spatial model's output (§4.2). This is a one-time table on the order of tens-to-low-hundreds of rows, radiologist-reviewable in an afternoon.
5. **QA by sampling, not exhaustive re-annotation.** Draw a stratified random sample (e.g. 300–500 concepts) and manually score the auto-tagger's precision/recall on it. This is the only manual-labelling step, and it scales with sample size, not with bank size.

**Unmappable-concept policy (resolved):** `anatomy=null` / `laterality=unspecified` concepts are **passed through unrefined** by the relevant rule (identity for that axis), not down-weighted by default — this is the safer default given imperfect automatic-parser coverage, consistent with the "no-Evidence-A ⇒ identity" rule below.

**Precomputation:** this tagging pass runs once per `bank_version` (368k short-string NER/regex pass is cheap — well under an hour on CPU), is cached in the N08 sidecar table, and is never run per-request.

**Should the KG be dropped instead ("no KG")?** Not recommended. Dropping RadLex/RadGraph removes the mechanism that makes grouping and anatomy normalisation possible at all (§4.6, rev 3) — without it, Evidence B degrades back to flat top-K over raw phrasings, which was independently shown to be the wrong mechanism regardless of Evidence A (§10's measured near-duplicate device phrasings). The insight is that this step doesn't need *formal KG reasoning* (subsumption, exclusivity, multi-hop relations) — it only needs the lightweight per-concept `(anatomy, laterality, polarity, temporal_class)` tags above, used for grouping. Genuine ontology-scale reasoning (hierarchy, mutual exclusivity, contradiction) belongs at **N25**, where the vocabulary is the much smaller canonical finding list (hundreds, not hundreds of thousands, of terms) and is realistically curatable. See §5.4 for the recommended KG.

**Memory/latency requirement:** the 368k × D matrix must be memory-resident or memory-mapped; it must not be loaded per-request.

---

### 4.5 N07 · `Similarity vs 368k concepts` → N10 `368,294-d concept similarity scores`

| Aspect | Specification |
|---|---|
| Inputs | `ImageEmbedding[D]` (E06); concept embedding matrix `[368294, D]` (E08) |
| Operation | Dense similarity — `scores = C @ e` for L2-normalised inputs (cosine). Compute cost: one 368k × D matvec per image, ≈ 0.4 GFLOP at D=1024 |
| Output | `ConceptScores: float32[368294]`, positionally aligned to `concept_id` |
| Fan-out | E10 → N13 (`Filter 68 concept scores` — **raw scores**), E11 → N12 (`KG-guided refinement`) |

**Requirements:**
- Output is a **dense** vector, not a top-k list, because N13 must index arbitrary positions into it. Do not sparsify here.
- Score scale is model-native (cosine ∈ [-1,1], or logit-scaled). **Whatever the scale, it is not a probability.** Two different downstream branches consume it with different semantics — N12/N16/N19 rank it, N13/N17 feed it into a learned head — so the scale must be documented, not assumed.
- Per-study artifact size: 368,294 × 4 B ≈ 1.47 MB dense. If retained for audit, store fp16 or top-N + threshold; retaining full fp32 for every study is ~1.5 MB/study.

**Failure paths:** dimension mismatch between embedding and bank → hard fail `E_BANK_DIM_MISMATCH`; NaN in scores → hard fail.

---

### 4.6 N12/N16/N19 · Concept grouping and temporal/polarity resolution (rev 3) → N21 `Evidence B`

**[Rev 3 — supersedes the original "KG-guided refinement (+ Evidence A)" design.]** N12 no longer takes Evidence A as an input, at all, even once N02 exists (see Revision Log rev 3). It solves a different, more fundamental problem: flat top-K over the raw 368k scores is dominated by whichever finding has the most templated phrasing variants in MIMIC's reporting style, not by which finding is most likely present — measured directly (§10: 4 of the top-5 hits for one sample image were near-duplicate paraphrasings of "right basal pleural tube"). Grouping by finding, not by raw phrase score, is required regardless of whether Evidence A exists; Evidence A was never the right fix for this specific problem.

**Role:** group the 368k concepts by the clinical fact they describe, resolve each concept's temporal/comparison language and negation into a current-state polarity, and rank *groups* — not individual phrasings — to produce Evidence B.

#### Offline, once per `bank_version` — parse and tag all 368,294 concepts

This extends the auto-tagging pipeline already specified in §4.4 (laterality regex, RadGraph-based anatomy) with two new steps (5 and the resolution in 6) needed specifically to make grouping and temporal handling correct:

| Step | Specification |
|---|---|
| 1. RadGraph parse | Parse every concept string once. **Implementation simplification (rev 3, post-GAP-21 measurement):** RadGraph correctly extracts every fact in a compound concept as separate entities (confirmed — see GAP-21), but each `concept_id` still gets exactly **one** tag/group membership, not one per fact. A "core observation" heuristic (`concepts/tagging/finding_vocab.pick_core_observation`) picks the fact most likely to be the finding rather than a modifier — the entity that is the *source* of a `located_at` relation, falling back to the longest observation text. On *"left pleural effusion and basilar opacification resolved"* this picks "opacification"; the effusion fact from that specific concept is not separately counted. This is an accepted information loss, not a bug: the 368k bank's redundancy (§10: median 7 words/concept, heavy paraphrase overlap) means the same finding is very likely also asserted by other concepts scored independently, so the loss is diluted, not silent — but it has not been measured at scale. Revisit (fan a concept into multiple group memberships) if the offline coverage report shows this matters |
| 2. Anatomy normalisation | Map each Anatomy entity to `AnatomyID` via the RadLex snapshot (is-a + part-of family, §5.4 rev 2) — unchanged from §4.4 |
| 3. Laterality tag | Regex/gazetteer over `{left, right, bilateral, midline}` and synonyms — unchanged from §4.4 |
| 4. Base polarity tag | Negation-cue regex (`no`, `without`, `clear of`, …) → `present` or `explicitly-negative` — unchanged from §4.4 |
| 5. **[New] Temporal/comparison classification** | Classify each concept as `stationary` (no comparison language) / `implies_present` (comparison language that still asserts current presence — `unchanged`, `stable`, `persistent`, `interval increase`, `new`, `worsening`) / `implies_absent` (comparison language asserting resolution — `resolved`, `cleared`, `no longer seen`) / `indeterminate` (a bare comparison with no resolvable current-state direction — `compared to`, `since`, `interval` alone). Gazetteer-based, same construction method as steps 3–4. **Proposed, not yet validated on real data — see [GAP-20]** |
| 6. **[New] Final `resolved_polarity`** | Combine step 4 and step 5: `present`+`implies_present`/`stationary` → `present`; `present`+`implies_absent` → `absent` (this is the case a naive text-strip gets backwards — stripping "resolved" from "resolved effusion" without this combination would assert the opposite of what the phrase means); `present`+`indeterminate` → concept excluded from grouping entirely, not defaulted; rare `explicitly-negative`+`implies_present` combinations are flagged for manual review rather than guessed |
| 7. Canonical finding assignment | Map each concept's Observation entity to a canonical finding label. **Built primarily from RadGraph's own observation-entity strings** (clustered for synonyms), anchored by the 14 CheXpert labels (§10) — **not primarily from RadLex**: RadLex's finding/observation coverage is measurably thin for common CXR terms (a 26-term probe of the real `RadLex.owl` found `cardiomegaly`, `cardiac silhouette`, `widened mediastinum`, `lung opacity`, and `hyperinflation` all absent). This is the B-relevant slice of the canonical vocabulary, **[GAP-6]** |
| 8. Grouping | Assign each concept to a group key `(canonical_finding, AnatomyID, laterality)`. **Laterality is folded into the key for v1** — "right pleural effusion" and "left pleural effusion" are different groups — rather than resolved within a group; this is a deliberate simplification that defers cross-source laterality arbitration to N25 (§5.4, GAP-7), where it already belongs. **v1 default, not empirically validated — see [GAP-19]** |

The result is a sidecar table, keyed by `concept_id`, of `(anatomy, laterality, resolved_polarity, temporal_class, canonical_finding, group_key)` — an extension of the §4.4 sidecar, not a separate asset.

#### Online, per image — group and rank, using N07's unmodified raw scores

**No concept is ever rescored or re-embedded.** Selection and ranking operate entirely on the scores N07 already computed against CLEAR's original, frozen concept embeddings; grouping and polarity resolution are a post-hoc relabelling of already-computed scores, not a new scoring pass.

| Step | Specification |
|---|---|
| 9. Look up tags | For every one of the 368,294 scored concepts, look up its precomputed `(group_key, resolved_polarity)` |
| 10. Aggregate within group, **split by polarity** | Per group: `present_score` = max (or top-N mean) over the group's `present`-polarity members, `present_count` = how many scored above a threshold; `absent_score`/`absent_count` computed identically over `absent`-polarity members. **These are never netted into one number here** — a high-scoring absent-polarity member (e.g. a confident "no pneumothorax") is evidence *against* the finding, not evidence *for* it with equal weight to a present-polarity phrase of the same score. Collapsing to a single `max()` across the whole group loses the sign. Resolving a present-vs-absent conflict *across sources* is N25's job (§5.4, contradiction rule); this step must not pre-empt it |
| 11. Rank groups | By `max(present_score, absent_score)` — a confidently-asserted absence is as informative as a confidently-asserted presence and belongs in the belief graph as an explicit negative finding (consistent with the requirement in §6.1 to include explicit negatives in the LLM prompt), not discarded for scoring lower on the "present" axis alone |
| 12. Select top-K groups | **K is not specified → [GAP-5]** (unchanged open item). Each selected group carries a **bounded** list of contributing concepts for audit, not its full membership — same top-N-plus-threshold principle already applied to the raw 368k vector (§4.5) |

**Output → N21 `Evidence B`:**
```
FindingGroup:
  finding: str                    # canonical finding label
  anatomy: AnatomyID
  laterality: Laterality
  present_score: float
  present_count: int
  absent_score: float
  absent_count: int
  top_contributing_concepts: list[{concept_id, text, raw_score, resolved_polarity, temporal_class}]  # bounded, e.g. top 3-5

Evidence B = list[FindingGroup]     # replaces the flat list[RefinedConcept] design (a genuine reshape, not just a rename)
```

`top_contributing_concepts` is required for INV-2 (traceability), for the same reason the CBM's `top_contributing_concepts` field is required in §4.7 — without it a group's score is an unexplainable number.

**Required design decisions carried over or newly introduced (not in the diagram):**
- **[AMBIG-2] retired.** "Is refinement purely suppressive or also promotive" no longer applies — there is no A-conditioned refinement left to be suppressive or promotive about.
- **[GAP-19, new]** Folding laterality into the group key (step 8) is a v1 default, not validated against real data. The alternative — group by `(finding, anatomy)` only and resolve laterality disagreement within the group — was considered and deferred as added complexity with no evidence yet that it's needed.
- **[GAP-20, new]** The temporal-classification gazetteer (step 5) and the polarity-combination rule (step 6) are both proposed designs, unvalidated. Needs the same QA-sampling check already used for the anatomy/laterality tags (§4.4 item 5), applied to a sample of temporally-flagged concepts, before trusting it in the belief graph.
- **[GAP-21, new]** RadGraph's actual behaviour on compound sentences (does a trailing "resolved" scope over every conjunct or only the nearest one?) has not been checked against real model output. The per-fact decomposition assumed in step 1 needs verifying before it's trusted.

**Failure paths:** RadGraph unavailable at offline build time → **hard fail at startup** — there is no "identity fallback" any more, since this mechanism no longer depends on a runtime input (Evidence A) that could legitimately be absent; the sidecar table simply cannot be built without RadGraph. Temporal/polarity gazetteer misses a concept (no trigger word matched) → default to `stationary`/`present` (the safer of the two default directions) and log for QA, rather than silently drop. A group with only `indeterminate`-classified members after resolution → excluded from ranking entirely, not defaulted to either polarity.

**Configuration:** `grouping.top_k_groups`, `grouping.contributing_concepts_per_group`, `grouping.present_absent_count_threshold`, `grouping.radgraph_version`, `grouping.canonical_vocab_version`, `grouping.temporal_gazetteer_version`, `grouping.unmapped_finding_policy`, `grouping.dedup_threshold`.

---

### 4.7 N14 · `Predefined CLEAR 68 concepts` → N13 `Filter 68 concept scores` → N17 `CBM` → N20 `Evidence C`

**Role:** a small, fully interpretable concept-bottleneck diagnostic stream. Deliberately low-dimensional so that each pathology prediction is attributable to a handful of named concepts.

| Node | Specification |
|---|---|
| N14 (resource) | Fixed list of 68 concepts, each with a `concept_id` **into the 368k bank's index space** so that filtering is a gather. Must be version-locked to `bank_version` |
| N13 | `cbm_scores = ConceptScores[idx_68]` → `float32[68]`. Pure gather. **Source is N10 (raw), not N16 (refined)** — see fact 2 in §2.3 and [AMBIG-3] |
| N17 `CBM` | Concept-bottleneck head: `float32[68] → PathologyLogits`. Typically a single linear layer (`[68 → P]`) so that weights are directly inspectable |
| N20 `Evidence C` | `list[{pathology: PathologyLabel, score: float, top_contributing_concepts: list[{concept_id, text, contribution: float}]}]` |

**Requirements:**
- The 68 input scores must be transformed onto the scale the CBM head was **trained** on. If the head was trained on normalised/standardised concept activations, the raw cosine scores must be put through the same transform. A silent scale mismatch here produces confidently wrong pathology predictions with no visible error. Store the training-time transform alongside the head weights.
- `top_contributing_concepts` should be computed as `w_p,i · x_i` per concept — this is the interpretability payoff of the CBM and should not be omitted.
- The CBM head is the only *trained* component in the pipeline besides the two encoders and the spatial model. Its training data, label space, and calibration must be documented.

**[GAP-1]** The pathology label space `P` is not given. It determines the vocabulary of the whole downstream belief graph and must be fixed first, along with its mapping into the KG's anatomy/finding terminology.

**[AMBIG-3] Raw vs refined input to the CBM.** The drawing is unambiguous (E10 originates at N10), and there is a defensible rationale: keeping C independent of B preserves evidence diversity, so fusion is combining genuinely different signals rather than two views of one KG-corrected vector. But it also means Evidence C can assert a laterality that the KG already rejected for Evidence B, guaranteeing contradictions that N25 must clean up. Both behaviours are reasonable; the choice must be made deliberately and recorded. Recommend: keep as drawn (raw), and rely on N25 to arbitrate — this preserves the independence assumption that makes fusion meaningful.

**Failure paths:** any of the 68 indices out of range for the current bank → hard fail `E_CBM_INDEX`; CBM head/bank version mismatch → hard fail at startup.

**Configuration:** `cbm.concept_index_file`, `cbm.head_weights`, `cbm.input_transform`, `cbm.label_space`, `cbm.threshold_per_label`.

---

### 4.8 N09 · `FAISS retrieval` → N11 `Top-k similar reports` → N15 `RadGraph parsing` → N18 `Evidence D`

**Role:** case-based evidence. Finds visually similar prior studies and contributes their *structured clinical facts* — not their prose.

#### N09 — FAISS retrieval

| Aspect | Specification |
|---|---|
| Input | `ImageEmbedding[D]` (E07) |
| Resource | FAISS index over the reference/training corpus image embeddings, built with the **same** CLEAR checkpoint |
| Output | `list[{report_id, distance/similarity, study_uid}]`, length k |
| Index type | Flat (exact) for correctness-critical use, or IVF/HNSW/PQ for scale. **Must be declared** — an approximate index makes retrieval non-deterministic w.r.t. index build, violating INV-4 unless the index build is itself pinned |
| Metric | Must match the embedding convention: inner product for L2-normalised embeddings, or L2. A metric/normalisation mismatch silently degrades retrieval without erroring |

**Critical requirement — leakage control:** the index must exclude the query study and, for evaluation runs, any study from the same patient. Without a patient-level exclusion filter, retrieval evidence is contaminated and every downstream metric is invalid. This must be enforced in the retrieval call, not left to index construction.

#### N11 — `Top-k similar reports`

Joins retrieved IDs to their report text. **Retrieval k is not specified → [GAP-5].** A similarity floor is also needed: if the nearest neighbour is below threshold, Evidence D should be emitted empty rather than contributing facts from a dissimilar case.

#### N15 — `RadGraph parsing`

| Aspect | Specification |
|---|---|
| Input | k report texts |
| Operation | RadGraph entity/relation extraction — entities typed `Observation` / `Anatomy` with certainty `definitely-present` / `uncertain` / `definitely-absent`; relations `located_at`, `modify`, `suggestive_of` |
| Output | k structured graphs |
| Caching | **Must be precomputed offline** for the entire reference corpus and stored alongside the index. Running RadGraph on k reports per query is a latency and cost error — the corpus is static |

#### N18 — `Evidence D: Retrieved facts`

Aggregates k graphs into evidence items:
```
EvidenceD = list[{
  label: str, anatomy: AnatomyID|null, laterality: Laterality,
  polarity: present|absent|uncertain,
  support_count: int,              # how many of the k neighbours asserted it
  weighted_support: float,         # Σ similarity of supporting neighbours
  attributes: dict,
  source_report_ids: list[str]
}]
```

**[AMBIG-4] Aggregation semantics are unspecified.** Is Evidence D a similarity-weighted vote across neighbours, or a union of all facts? A union over k=10 neighbours will assert nearly every common finding; a strict vote may lose rare true findings. Recommend similarity-weighted voting with an explicit support threshold, and record `support_count` so fusion can reason about consensus.

**Failure paths:** empty index / index load failure → hard fail at startup; zero results above the similarity floor → empty Evidence D, flagged in `study_meta`, not an error; RadGraph cache miss for a retrieved report → skip that neighbour and log, do not fail the study.

**Configuration:** `retrieval.index_path`, `retrieval.index_type`, `retrieval.metric`, `retrieval.k`, `retrieval.similarity_floor`, `retrieval.exclude_same_patient`, `retrieval.radgraph_cache_path`, `retrieval.aggregation` (`vote`|`union`), `retrieval.min_support`.

---

## 5. Stage 2 — Evidence Fusion and Belief Graph

### 5.1 N22 · `Evidence normalisation`

**Role:** the four evidence streams arrive on mutually incomparable scales — detector confidences (A), cosine/adjusted similarities (B), CBM logits or probabilities (C), and neighbour vote counts (D). This stage makes them commensurable *before* any fusion arithmetic is attempted. Averaging them directly would be meaningless.

| Aspect | Specification |
|---|---|
| Inputs | Evidence A (E21a), B, C, D (all via the E22 bus) |
| Output | `list[NormalisedEvidenceItem]` — all scores on a common `[0,1]` scale with declared semantics |

**Two distinct operations, both required:**

1. **Score normalisation** — per-source monotone mapping onto `[0,1]`. Candidate methods per source: min-max over the study, rank-percentile, z-score→sigmoid, or a fitted mapping from a held-out set. **[AMBIG-5]: the method is unspecified.** Note that within-study min-max is scale-destroying for a normal film (it will stretch the top noise concept to 1.0) — recommend a **fixed, globally fitted** mapping per source rather than a within-study one, precisely so that "nothing scored highly" remains representable.

2. **Semantic alignment / entity resolution** — the harder half. The four sources name findings differently: A emits detector labels, B emits free-text concept strings from a 368k vocabulary, C emits CBM pathology labels, D emits RadGraph observation entities. They must be mapped to a **single canonical finding vocabulary** keyed by `(label, anatomy, laterality)` so that fusion can recognise that all four are talking about the same thing.

**[GAP-6] The canonical finding vocabulary and the four source→canonical mappings are not in the diagram and are prerequisites for Stage 2 to function at all.** Without them, "fusion" degenerates into four parallel unmerged lists. Required: a canonical term list, A→canonical, B(368k concept)→canonical, C(pathology label)→canonical, D(RadGraph observation)→canonical, plus anatomy term unification across the spatial atlas, the concept bank annotations, and RadGraph's anatomy entities.

**Failure paths:** unmappable evidence item → retain with `canonical=null` and exclude from fusion, but log and count; a rising unmapped rate is a silent-degradation signal and should be monitored.

**Configuration:** `normalise.method_per_source`, `normalise.canonical_vocab_version`, `normalise.mapping_tables`, `normalise.unmapped_policy`.

---

### 5.2 N23 · `Score calibration + fusion`

**Role:** estimate, for each candidate finding, how strongly it is *actually supported across sources* — explicitly contrasted in the prose with "simply averaging incompatible scores".

| Aspect | Specification |
|---|---|
| Input | normalised, canonically-keyed evidence items |
| Output | `list[FusedFinding]` with a calibrated `confidence ∈ [0,1]` and full `support` list |

**Calibration:** per-source reliability mapping fitted on a held-out validation set — Platt scaling, isotonic regression, or temperature scaling — so that a source's normalised score becomes an estimate of `P(finding present | that source's score)`. Calibration must be fitted per `(source, finding-class)` where data allows; a single global calibrator per source is the fallback.

**Fusion:** combine per-source calibrated posteriors into one confidence. Candidate strategies, in increasing order of assumption strength:
- weighted sum of calibrated scores with learned per-source weights (simplest defensible baseline);
- noisy-OR / log-odds pooling (models independent corroboration — appropriate given the sources are designed to be independent);
- a small learned fusion head over the 4-source feature vector.

**Required behaviours regardless of strategy:**
- **Missing-source handling** must be principled, not zero-filling. A source that did not run is not a source that scored zero — zero-filling a missing Evidence D silently suppresses every finding. Use explicit presence indicators, or renormalise over available sources.
- **Agreement/disagreement must be preserved, not averaged away.** Record `n_supporting_sources` and per-source contributions on each `FusedFinding`; the belief graph and the downstream verifier both need them.
- **[AMBIG-6] Per-source weights are unspecified.** Given the architecture's intent, Evidence A should plausibly dominate anatomy/laterality attribution while C dominates pathology presence — but this must be fitted and validated, not assumed.
- **Attribute and laterality fusion** is a separate problem from confidence fusion: when A says *right* and D says *left*, the fused finding must not average to "bilateral". Conflicting categorical attributes must be resolved by an explicit precedence/voting rule, or emitted as a conflict for N25 to arbitrate. **[GAP-7]**

**Output:** the candidate finding set for the belief graph, with confidences, support, and recorded conflicts.

**Failure paths:** calibrator missing for a source → fall back to identity mapping and flag `study_meta.uncalibrated_sources`; all sources missing → hard fail `E_NO_EVIDENCE`.

**Configuration:** `fusion.strategy`, `fusion.source_weights`, `fusion.calibrators`, `fusion.missing_source_policy`, `fusion.min_confidence_to_admit`, `fusion.attribute_conflict_policy`.

---

### 5.3 N24 · `Initial belief graph`

**Role:** the first inspectable intermediate clinical state — an explicit, structured hypothesis about the study, prior to symbolic validation.

| Aspect | Specification |
|---|---|
| Input | `list[FusedFinding]` |
| Output | `BeliefGraph(graph_version="initial")` |

**Construction:**
- **Nodes:** `Finding`, `Anatomy`, `Attribute`. Each finding node carries label, polarity, confidence, and `support: list[EvidenceRef]` (INV-1).
- **Edges:** `located_at` (finding→anatomy), `has_attribute` (finding→attribute), `suggests` (finding→finding, from KG or RadGraph `suggestive_of`), `contradicts` (finding→finding, recorded but **not yet resolved** — resolution is N25's job).
- **Admission threshold:** findings below `fusion.min_confidence_to_admit` are excluded from the graph but retained in a `rejected` sidecar for audit.

**Serialisation requirement:** must be persisted per study in a stable, diffable format (JSON or RDF/JSON-LD). This artifact plus the verified graph are the pipeline's primary auditability deliverable and the substrate for every debugging session. Store both; the diff between them is the record of what the rule layer did.

---

### 5.4 N25 · `KG + medical rule check` → N26 `Verified belief graph`

**Role:** symbolic validation. Removes or down-weights implausible and conflicting claims using terminology relationships, anatomy/laterality constraints, contradiction rules, and other clinical consistency rules.

| Aspect | Specification |
|---|---|
| Inputs | `BeliefGraph(initial)`; KG; medical rule set |
| Output | `BeliefGraph(verified)` + `audit: list[RuleApplication]` |

**Rule categories (all must be represented):**

| Category | Example | Action |
|---|---|---|
| Anatomical validity | finding asserted at an anatomically impossible site | drop |
| Laterality consistency | same finding asserted both left-only and right-only from different sources | resolve by confidence/precedence, or widen to `bilateral` only if licensed |
| Mutual exclusivity | mutually exclusive findings both asserted | keep higher-confidence, drop/downweight other |
| Contradiction | `present` and `definitely-absent` for the same `(label, anatomy, laterality)` | resolve, record both in audit |
| Hierarchy / subsumption | a specific finding and its parent term both asserted | collapse to the appropriate specificity level |
| Device/context plausibility | finding requires a device or prior state not otherwise supported | downweight |
| Attribute validity | attribute value outside the permitted range for that finding | drop attribute, keep finding |

**Required properties:**
- **Deterministic:** fixed rule ordering, declared precedence, confluent outcome. Same input graph ⇒ same output graph (INV-4).
- **Auditable:** every removal/downweight emits a `RuleApplication {rule_id, rule_version, targets, action, before, after, rationale}`.
- **Non-destructive:** removed claims are moved to a `suppressed` list on the graph, never deleted. N30 may need them to explain why a drafted claim is unsupported.
- **Bounded:** if rules can cascade, cap iterations and detect cycles.

**[AMBIG-7] "Remove or down-weight" — the split is unspecified.** Which violations are fatal to a claim vs. merely confidence-reducing? This needs an explicit per-rule-category policy table, since it directly sets the system's precision/recall operating point. And down-weighting raises a second question: after down-weighting, is the admission threshold re-applied? Recommend yes, once, with the re-application recorded in the audit.

**[GAP-8] — RESOLVED (rev 2): RadLex only, no SNOMED CT / UMLS in v1.** The diagram doesn't specify a KG despite consuming one at two structurally different points (N12: lightweight per-concept tags over 368k items; N25: rules over a small canonical vocabulary). Rev 1 originally recommended a three-layer stack (RadLex + SNOMED CT + UMLS); **rev 2 drops SNOMED CT and UMLS** — at this scale their marginal value doesn't justify their licensing/integration overhead, and everything both stages need is achievable with RadLex alone plus hand-curated rules.

| Layer | v1 choice | Used at | Why |
|---|---|---|---|
| **Anatomy/finding lexicon** | **RadLex**, via the **BioPortal API** (Annotator for term matching, or a one-time full-ontology download for offline batch matching — prefer the download for the 368k-item batch job, so the annotation pass doesn't depend on a live external service at build time, per INV-4) | N12 anatomy gazetteer (§4.4); base vocabulary for the N25 canonical term list | Purpose-built for radiology, strong CXR anatomy coverage (lungs/lobes, pleura, mediastinum, heart, bones), the same terminology family RadGraph's own schema draws from, **no licensing gate** — BioPortal serves it openly, unlike SNOMED CT/UMLS which need a national licence or a UMLS affiliate agreement |
| **Formal laterality/finding-site relations, exclusivity/contradiction rules** | **Hand-curated table**, not an ontology | N25 rule set | N25's canonical vocabulary is only on the order of a few hundred terms — small enough that a radiologist-reviewable `finding × valid anatomy × valid laterality × exclusivity-pairs` table is *less* integration work than correctly parsing SNOMED's relationship-group/attribute model, and it avoids SNOMED entirely. RadLex's own `is_a`/`part_of` hierarchy covers most subsumption needs (item 6 in §5.4's rule table) without SNOMED |
| **Entity-linking backbone** | *(dropped — not needed)* | — | Rev 1 reached for UMLS to boost 368k-concept tagging coverage, but §4.4's pipeline (RadGraph parse + RadLex gazetteer match) is expected to get solid coverage on its own since the concepts are short, templated phrases. The §4.4 QA sample (300–500 concepts) is the check: only add UMLS-based linking later if that sample shows RadLex-only coverage is materially insufficient |

**Practical split (unchanged in spirit from rev 1):** N12's per-concept tags (§4.4) need no formal reasoning — regex laterality + RadLex-anatomy match is enough. N25's rules need real curation, but over hundreds of terms, not hundreds of thousands, which makes hand-authoring the pragmatic choice rather than a shortcut. **RadGraph itself is still not a substitute KG** — it's an extraction schema, not a reference ontology with hierarchy/exclusivity axioms.

Open items: pin the exact RadLex release/version (via BioPortal's versioned ontology submissions); decide whether N12 and N25 draw from the *same* pinned RadLex release (recommended); and the "medical rules" beyond pure anatomy/laterality/exclusivity (e.g. device/context plausibility) still need either hand-authoring by a radiologist or mining from a report corpus — RadLex alone doesn't supply those either. **Revisit SNOMED/UMLS if:** (a) the §4.4 QA sample shows RadLex-only tagging coverage is too low, or (b) N25's rule table grows past what's comfortably hand-maintained and formal relational lookup becomes worth the licensing cost.

**N26 `Verified belief graph`** is the trust boundary. Everything downstream treats it as ground truth; nothing downstream may add clinical content.

---

## 6. Stage 3 — Report Generation and Verification

**[Rev 1] v1 architecture: single pass, no regeneration loop.** `Draft report → RadGraph entity extraction → Facts verification → Final report`, run exactly once per study. `Facts verification` (N30) is redefined from a **pass/fail gate** to a **filter**: it classifies each drafted claim against the verified belief graph and produces per-claim keep/remove/correct decisions; `Final report` (N31) deterministically applies those decisions to the draft text. There is no `Regenerate report (LLM)` step and no `pass`/`fail` branch in v1 — see §6.5 for the original (deferred) design. This resolves GAP-10/AMBIG-9 by deferral, not by design decision — the original loop remains the intended Phase 2 upgrade once v1 is validated.

### 6.1 N27 · `LLM report generation` → N28 `Draft report`

**Role:** linguistic realisation only. Per the prose, the LLM is "primarily responsible for linguistic realisation rather than independently deciding what abnormalities are present."

| Aspect | Specification |
|---|---|
| Input | `BeliefGraph(verified)` (E27) — **and nothing else clinical** (INV-3) |
| Output | `DraftReport: {text, sections: {findings, impression}, model, params, prompt_hash}` — *(`attempt` field dropped in v1: single pass, no retries-by-regeneration; API-level retries on transient errors still apply, see Failure paths)* |

**Prompt contract:**
- Serialise the verified graph into a compact, unambiguous textual rendering — one line per finding with label, anatomy, laterality, attributes, polarity, confidence.
- Include explicit **negative** findings (`polarity=absent`) so the model can write the customary negations rather than inventing them.
- Include an explicit closed-world instruction: *state every listed finding; state nothing not listed; do not add hedges, causes, or recommendations not present in the input.*
- Include section structure and style requirements (Findings / Impression, tense, phrasing conventions).
- **Must not** include: raw retrieved report text, raw concept score lists, or any Stage-1 output that bypassed verification.

**Requirements:**
- `temperature` low (0–0.3) for the first attempt; determinism is desirable but not guaranteed by any vendor, so `prompt_hash` and sampling params must be recorded per attempt.
- Confidence must be verbalised through a **fixed, declared mapping** from confidence bands to hedging language (e.g. ≥0.8 → assertive, 0.5–0.8 → "probable", <0.5 → "possible"). Free-form hedging by the model is unverifiable downstream. **[GAP-9]**
- Token budget must accommodate the largest plausible graph; graph serialisation must be truncation-safe (truncating the graph silently drops findings — if truncation is ever necessary it must be a recorded error, not a silent shortening).

**Failure paths:** LLM API error/timeout → bounded retry with backoff (this is an infrastructure retry on the same prompt, not the content-driven regeneration loop — that loop is deferred, see §6.5); exhausted → `E_LLM_UNAVAILABLE`, no report emitted; empty/malformed output → `E_LLM_MALFORMED`, no report emitted (v1 has no regeneration path to recover into); refusal/safety block → hard fail with the reason recorded.

**Configuration:** `llm.provider`, `llm.model`, `llm.temperature`, `llm.max_tokens`, `llm.system_prompt_version`, `llm.confidence_verbalisation_map`, `llm.timeout_s`, `llm.retries`.

---

### 6.2 N29 · `RadGraph entity extraction`

**Role:** reconstruct what the LLM *actually wrote*, as structured claims. This is deliberately the same class of parser used on retrieved reports (N15), so drafted claims and evidence facts live in one representation.

| Aspect | Specification |
|---|---|
| Input | `DraftReport.text` |
| Output | `DraftClaims: {entities, relations}` with certainty labels and **character offsets into the draft text** |

**Requirements:**
- Character offsets are mandatory — N30's filtering decisions must point N31 at specific spans to delete/correct, not just at claim labels. *(In v1 this drives text editing at N31 instead of driving a regeneration prompt; the requirement is unchanged either way.)*
- Extracted entities must be mapped into the **same canonical vocabulary** used in §5.1, otherwise comparison against the belief graph is string-matching and will produce false "unsupported finding" flags on paraphrases. This mapping is the same asset as [GAP-6].
- The same RadGraph model version must be used in N15 and N29; a version skew makes evidence facts and drafted claims subtly incomparable.

**Failure paths:** parse failure → cannot verify ⇒ fail closed, `E_EXTRACT_FAILED`, no report emitted (v1 has no regeneration path to recover into — see §6.5); empty extraction on a non-empty draft → same, fail closed.

---

### 6.3 N30 · `Facts verification` (v1: filter, not gate — rev 1)

**Role:** compares drafted claims against the verified belief graph and classifies each one — it no longer decides `pass`/`fail` for the whole report; it decides, per claim, whether that claim survives into the final text. N31 then applies the decisions.

| Aspect | Specification |
|---|---|
| Inputs | `DraftClaims` (E30); `BeliefGraph(verified)` — **implicit resource dependency, not drawn** |
| Output | `ClaimDecisions` (replaces the v0 `VerificationReport` verdict) |

**Error classes detected (same taxonomy as the original design; disposition differs per class in v1 — see table):**

| Class | Definition | v1 disposition |
|---|---|---|
| `UNSUPPORTED_FINDING` | drafted claim has no corresponding verified finding (hallucination) | **remove** the claim/sentence from the draft |
| `WRONG_ANATOMY` | claim matches a verified finding but asserts a different anatomical site | **correct** if a simple substitution (swap the anatomy term for the verified one); **remove** if the sentence can't be cleanly corrected by substitution |
| `WRONG_LATERALITY` | ditto for laterality | **correct** by substitution (e.g. "left" → "right") where the span is a simple lexical swap; **remove** otherwise |
| `WRONG_ATTRIBUTE` | severity/size/character contradicts the verified attributes | **correct** by substitution where possible; **remove** otherwise |
| `NEGATION_ERROR` | polarity inverted relative to the verified graph | **correct** by substitution (flip the negation) where possible; **remove** otherwise |
| `INTERNAL_CONTRADICTION` | the draft contradicts itself | **remove** the losing claim (prefer the one matching the verified graph) |
| `MISSING_FINDING` | verified finding above the reporting threshold is absent from the draft (omission) | **v1: logged only, not fixed.** See limitation below |

```
ClaimDecisions:
  decisions: list[{
    drafted_claim, char_span, class,
    action: "keep" | "remove" | "correct",
    corrected_text: str | null,        # only for action="correct"
    expected, observed, rationale
  }]
  coverage: {verified_findings_total, findings_reported, findings_missing: list[Finding]}
```

**[Rev 1 limitation — carried forward explicitly, not silently dropped] `MISSING_FINDING` is not remediated in v1.** Without a regeneration loop there is no mechanism to make the LLM add an omitted finding, and a v1 goal is "never assert more than the evidence supports" (precision) rather than "always assert everything the evidence supports" (recall) — a single-pass filter can only ever improve precision by removing claims, not improve recall by inventing prose for missing ones. `coverage.findings_missing` is still populated and surfaced in the final report's provenance bundle (§6.4) so omissions are visible and auditable, not silently lost. Treat this as the primary argument for reintroducing the loop in Phase 2 (§6.5): regeneration is specifically what lets the system *add* a missing finding safely (LLM writes it, then it's re-verified), which filtering alone cannot do.

**Matching requirements (unchanged from v0):** claim↔finding matching must be canonical-vocabulary-based with a defined synonym/subsumption tolerance. A parent-term claim ("opacity") against a child-term verified finding ("consolidation") is a *specificity* question, not automatically an error — the policy must state whether generalisation is permitted, whether specialisation is (it should not be: asserting more than the evidence supports is exactly the failure mode the pipeline exists to prevent), and how partial attribute matches score.

**`keep`/`remove`/`correct` split policy [AMBIG-8, revised for v1]:** whether a wrong-anatomy/laterality/attribute/negation error is corrected-in-place vs. removed wholesale depends on whether the erroneous span is isolable as a simple lexical substitution without changing surrounding sentence grammar/meaning (e.g. one adjective or noun swap) — if not cleanly isolable, prefer `remove` over a risky in-place edit that could produce ungrammatical or misleading text. This bias toward `remove` is deliberate: v1 has no re-verification pass to catch a bad correction, so corrections should only be attempted where they're mechanically safe.

**Failure paths:** verifier internal error → fail-closed (`E_VERIFY_ERROR`, no report released). The system must never emit an unfiltered/unverified report.

**Configuration:** `verify.match_tolerance`, `verify.allow_generalisation`, `verify.correction_policy` (which classes attempt `correct` vs. always `remove`), `verify.fail_closed` (must be `true`).

---

### 6.4 N31 · `Final report` (v1: deterministic assembly from `ClaimDecisions` — rev 1)

**Role:** apply N30's `ClaimDecisions` to `DraftReport.text` to produce the final text. This is a deterministic text-editing step (span deletion/substitution driven by N29's character offsets), not a second LLM call, unless the optional smoothing pass below is enabled.

**Assembly algorithm:**
1. Sort decisions by `char_span` start, descending (edit from the end of the text backward so earlier offsets stay valid).
2. For each `remove` decision: delete the corresponding sentence/clause span. Prefer deleting whole sentences over sub-sentence fragments to avoid leaving grammatically broken remnants — if a decision's span is sub-sentential, expand the deletion to the enclosing sentence boundary.
3. For each `correct` decision: substitute `char_span` with `corrected_text`.
4. **[GAP: new, rev 1]** After edits, section structure (Findings/Impression) may end up with an empty or truncated section if every claim in it was removed — the assembler must handle this explicitly (e.g. render "No other findings." or omit an empty section per a declared style rule), not leave a dangling heading.
5. **Optional smoothing pass (not a loop):** a single, non-conditional LLM call may be used to lightly re-fluent-ify the edited text (fix any resulting grammar seams) — this is explicitly *not* the regeneration loop: it has no pass/fail gate, does not re-run verification, and must not be allowed to add or reintroduce clinical content (same closed-world constraint as N27's prompt, INV-3). Recommend deferring this too for a first v1 build (pure deterministic span editing) and adding it only if raw span-deletion output reads too roughly; if added, its own output should still be diffed against `ClaimDecisions` in tests to confirm it introduced no new claims.

```
FinalReport:
  text, sections
  verified_belief_graph_ref
  claim_decisions                       # replaces v0's verification_report
  omitted_findings: list[Finding]       # from coverage.findings_missing — surfaced, not silently dropped
  evidence_summary: per-finding source attribution
  degradations: list[str]               # any skipped/failed evidence source (INV-5), incl. spatial_unavailable
  model_versions: {clear, spatial, cbm, radgraph, llm, kg, rules, bank}
```

The `degradations`, `omitted_findings`, and `model_versions` fields are required: a report generated with, say, retrieval unavailable, or with known omissions, must be identifiable as such after the fact.

---

### 6.5 Deferred (Phase 2) · N32 `Regenerate report (LLM)` — the original loop design

**Not built in v1** (Revision Log rev 1). Retained here so the original design isn't lost when Phase 2 reintroduces it — e.g. once `MISSING_FINDING` remediation (§6.3's limitation) becomes a priority.

**Original role:** controlled regeneration using the verified evidence, closing `N28 → N29 → N30 → N32 → N28` into a loop until N30 passes or attempts are exhausted.

| Aspect | Specification |
|---|---|
| Drawn input | `fail` edge from N30 (E32) only |
| Implicit inputs | `VerificationReport`; `BeliefGraph(verified)`; previous draft |
| Output | a new draft, routed to N28 `Draft report` (E33) — re-entering the loop at the draft node, so the new draft is re-extracted (N29) and re-verified (N30) |

**Prompt contract (distinct from N27's):** the regeneration prompt must include the previous draft, the specific errors with their character spans, and the corrective instruction per error class (remove this unsupported sentence; add this missing finding; correct laterality here). Blind resampling of N27's prompt is not "controlled regeneration" and will loop.

**[AMBIG-9] N32 has no drawn edge from N26.** The prose says regeneration uses the verified evidence, so the verified graph must be available to N32. Either the diagram omits the edge, or N32 is intended to receive the graph transitively through the verification report. Specify explicitly when Phase 2 is built — recommend an explicit N26 → N32 dependency so that the regeneration prompt is built from the same authoritative source as N27's.

**[GAP-10] Loop termination is entirely unspecified in the diagram.** The drawing shows an unbounded cycle. Required when Phase 2 is built:
- `max_attempts` (recommend 3).
- **Behaviour on exhaustion**, which is a product decision, not a technical one: (a) emit the last draft flagged as unverified — *contradicts the pipeline's central guarantee and should be rejected*; (b) emit a conservative template report rendered deterministically from the verified graph, bypassing the LLM; (c) emit nothing and escalate for human review; (d) **v1-compatible fallback:** fall back to the v1 filter (§6.3/§6.4) on the last attempt instead of emitting unverified text — since the filter is already fail-closed-safe by construction, this reuses v1's machinery as Phase 2's safety net. Recommend (d), or (b)/(c) — never (a).
- **Non-convergence detection:** if attempt *n* produces the same error set as *n−1*, the loop is stuck; break early rather than burning the remaining attempts.
- **Temperature schedule** across attempts (a stuck loop at temperature 0 will repeat verbatim; a small increase per attempt is the usual remedy).
- Per-attempt cost/latency accounting.

**Configuration (Phase 2):** `regen.max_attempts`, `regen.on_exhaustion`, `regen.temperature_schedule`, `regen.prompt_version`, `regen.stuck_detection`.

---

## 7. Cross-Cutting Requirements

### 7.1 Dependency graph and startup validation

| Component | Hard dependencies |
|---|---|
| N03 CLEAR encoder | checkpoint |
| N07 similarity | CLEAR checkpoint **+** concept bank built with that same checkpoint |
| N09 FAISS | CLEAR checkpoint **+** index built with that same checkpoint |
| N12/N19 grouping *(rev 3)* | RadGraph model **+** RadLex snapshot **+** canonical finding vocabulary **+** temporal gazetteer, all keyed to `bank_version` — **no longer depends on Evidence A / N02** |
| N13/N17 CBM | 68-concept index list keyed to `bank_version` **+** CBM head trained on that index order |
| N15/N29 RadGraph | one model version shared by both |
| N22/N29 | canonical vocabulary + all source mapping tables |
| N25 | KG + rule set |

**Requirement:** a startup compatibility check must verify the full version matrix (`clear_checkpoint`, `bank_version`, `faiss_index_version`, `cbm_index_version`, `kg_version`, `rule_set_version`, `radgraph_version`, `canonical_vocab_version`) and refuse to start on mismatch. Every one of these mismatches fails *silently and plausibly* at runtime — wrong-but-confident output rather than an exception — which makes a startup gate the only reliable defence.

### 7.2 Degraded-mode policy

| Failed component | Policy |
|---|---|
| N02 spatial (runtime failure, N02 exists and errors) | Evidence A empty; flag `degraded: no_spatial`; fusion runs on 3 sources. *(Rev 3: N12/Evidence B is unaffected either way — it no longer depends on A. The "more than half the architecture's premise" concern from rev 1 is reduced accordingly, though A's own independent evidence stream to fusion is still lost.)* |
| N02 spatial (**planned absence — N02 not yet built, rev 1**) | Not a failure — this is the v1 build-phase default. `EvidenceA=absent`, `study_meta.spatial_unavailable=true`, 3-source fusion. Do **not** hard-fail on this; hard-fail is a *runtime* policy for a module that exists and errors, not a build-time gate on a module not yet implemented |
| N03 CLEAR | Hard fail — B, C, D all depend on it |
| N09 FAISS | Evidence D empty; flag `degraded: no_retrieval`; continue |
| N12/N19 grouping *(rev 3)* | RadGraph/RadLex/canonical-vocab unavailable → **hard fail at startup** (no identity fallback — see §4.6's rev-3 failure paths) |
| N17 CBM | Evidence C empty; flag; continue |
| N25 rules | **Hard fail** — an unverified belief graph must never reach the LLM |
| N30 verifier | **Hard fail** — fail closed |

Every degradation must appear in `FinalReport.degradations` (INV-5).

### 7.3 Persistence and auditability

Per study, persist: `StudyMeta`; Evidence A/B/C/D (B compressed or top-N); initial belief graph; verified belief graph + rule audit; every draft attempt with its prompt hash; every verification report; final report + provenance bundle. Retention must respect PHI policy (§7.6). The initial↔verified graph diff and the draft↔verified comparison are the two artifacts that make failures diagnosable.

### 7.4 Observability

Metrics: per-stage latency (p50/p95); evidence-source availability rate (incl. `spatial_unavailable` rate while N02 is deferred, rev 1); unmapped-evidence rate at N22 (silent-degradation canary); auto-tagger coverage rate at N08/N12 (fraction of the 368k bank with resolved anatomy/laterality tags — the key health metric for the resolved GAP-4 pipeline, §4.4); distribution of N25 rule firings by `rule_id`; **v1 (rev 1):** claims removed/corrected per report, `MISSING_FINDING` (omission) rate from `coverage.findings_missing` — the headline metric motivating Phase 2's loop; error-class histogram from N30; retrieval similarity distribution; LLM token spend per report. *(Phase 2, once §6.5 is built: verification pass rate at attempt 1/2/3, loop-exhaustion rate.)*

### 7.5 Performance envelope

Dominant costs: the 368k × D matvec (N07), FAISS search (N09), CLEAR + spatial forward passes, and — in v1 — exactly 1 LLM call (2 if the optional §6.4 smoothing pass is enabled), almost certainly the latency floor. *(Phase 2, once §6.5's loop is built: 1–`max_attempts` LLM calls.)* Precompute obligations: concept bank embeddings, FAISS index, the RadLex/UMLS-based concept auto-tagging pass (§4.4, one-time per `bank_version`), RadGraph over the reference corpus — none of these may be computed per request. A target end-to-end latency is not stated in the diagram and should be set, since it governs the choice between an exact and an approximate FAISS index.

### 7.6 Safety, privacy, and clinical governance

- Reports are clinical text; PHI handling governs logs, caches, persisted artifacts, and — critically — anything sent to an external LLM provider. If the LLM is hosted externally, de-identification before the prompt is built is a hard requirement, and the verified-graph serialisation must be checked to ensure it carries no identifiers.
- The retrieval corpus contains other patients' reports. INV-3 (facts only, never verbatim text) is a privacy control as well as an architectural one.
- Intended use, limitations, and the research/non-diagnostic status of outputs must be stated on every emitted report.
- The rule set and KG are clinical content and require radiologist review and versioned sign-off, independent of the software release process.

---

## 8. Ambiguities, Gaps, and Required Decisions

Ordered by blocking impact.

### Resolved (rev 1)

| ID | Issue | Resolution |
|---|---|---|
| **GAP-4** | The 368k concepts must carry (anatomy, laterality, polarity, KG link) for N12 to function; the diagram shows a plain vocabulary | **Resolved by construction method, not by data:** automatic tagging (regex laterality + RadLex/UMLS-linked anatomy), not manual labelling. See §4.4. Still needs the coverage/QA sampling pass run and its results recorded before N12 is trusted |
| **GAP-8** | The KG is consumed at N12 and N25 but never specified | **Resolved (rev 2):** RadLex only, via BioPortal — no SNOMED CT / UMLS in v1. N12 tags come from RadLex-gazetteer + RadGraph matching; N25's rules are hand-curated over the small canonical vocabulary instead of derived from a formal ontology. See §5.4. Still needs RadLex version pinning; revisit SNOMED/UMLS only if the §4.4 coverage QA or N25's rule count outgrows this |
| **GAP-10** | Verification loop has no termination condition or exhaustion behaviour | **Deferred, not resolved:** the loop itself is not built in v1 (§6, §6.5), so termination doesn't arise yet. Re-open when Phase 2 reintroduces the loop |
| **AMBIG-9** | N32 has no drawn input from N26 despite the prose | **Moot in v1:** N32 doesn't exist yet. Re-open when Phase 2 builds it (§6.5) |
| **AMBIG-2** | Is N12's refinement purely suppressive, or also promotive? | **Retired (rev 3):** the question no longer applies — N12 no longer refines B using Evidence A at all, in any direction. See §4.6, Revision Log rev 3 |

### Blocking — cannot build without resolution

| ID | Issue | Decision needed |
|---|---|---|
| **GAP-6** | No canonical finding vocabulary and no source→canonical mapping tables; N22, N24, N29, N30 all require them | Define the canonical vocabulary and the four mappings |
| **GAP-1** | CBM pathology label space undefined | Fix the label set and its mapping into the canonical vocabulary |
| **GAP-3** | CLEAR model/checkpoint unidentified — root dependency of B, C, D | Pin model, checkpoint, embedding dim, normalisation convention |
| **GAP-2** | Spatial perception model architecture/checkpoint unspecified | **Not blocking v1** (N02 is deliberately deferred — Revision Log rev 1, §4.2, §7.2) — but still needs an eventual choice of detector vs. segmentation vs. grounded classifier, and a fixed anatomy atlas, before Phase-1.5 adds spatial perception back in |

### Significant — affect correctness and the operating point

| ID | Issue | Recommendation |
|---|---|---|
| **AMBIG-8** | N30 keep/remove/correct split undefined *(v1: no longer a "pass criterion" — N30 filters rather than gates, rev 1)* | Per-class disposition policy (see §6.3): default to `remove` unless a correction is a mechanically safe, isolable substitution |
| **AMBIG-7** | N25 "remove vs. down-weight" split undefined | Per-rule-category policy table; re-apply admission threshold once after down-weighting |
| **AMBIG-3** | CBM fed from raw (N10) not refined (N16) scores | Keep as drawn to preserve evidence independence; let N25 arbitrate resulting conflicts — but record the decision |
| **AMBIG-5** | Normalisation method per source undefined | Globally fitted per-source mapping, **not** within-study min-max (which destroys the normal-study case) |
| **AMBIG-6** | Fusion weights / strategy undefined | Fit per-source weights on held-out data; start from log-odds pooling |
| **GAP-7** | Categorical attribute conflict across sources (left vs. right) has no resolution rule | Explicit precedence rule; never average categorical attributes |
| **AMBIG-4** | Evidence D aggregation (vote vs. union) undefined | Similarity-weighted vote with minimum support; retain `support_count` |
| **GAP-5** | Three distinct `k`/`K` values (retrieval k, top-K **groups**, RadGraph neighbours) are all written as "top-k" | Name and configure them separately (rev 3: `grouping.top_k_groups` for B, `retrieval.k` for D); add a similarity floor for retrieval — the top-K **dedup** half of this is now resolved by §4.6's grouping mechanism (rev 3), which ranks findings rather than raw phrasings |
| **GAP-9** | Confidence→hedging language mapping undefined | Fixed declared band mapping; free-form hedging is unverifiable at N30 |
| **GAP-18** *(new, rev 1)* | v1 has no mechanism to remediate `MISSING_FINDING` (omissions) — filtering only ever removes, never adds | Accepted limitation for v1 (precision over recall); track `coverage.findings_missing` rate as the metric that justifies building Phase 2's loop (§6.3, §6.5) |
| **GAP-19** *(new, rev 3)* | Folding laterality into Evidence B's group key `(finding, anatomy, laterality)` is a v1 default, not validated | Revisit once real grouping output exists — check whether left/right splits of the same finding behave sensibly, or whether within-group laterality resolution would serve better |
| **GAP-20** *(new, rev 3)* | The temporal-classification gazetteer and the present/absent polarity-combination rule (§4.6 steps 5–6) are proposed, unvalidated designs | Same QA-sampling check as §4.4's anatomy/laterality tags, applied to a sample of temporally-flagged concepts, before trusting the resolved polarity |
| **GAP-21** | RadGraph's actual behaviour on compound sentences | **Resolved by measurement (rev 3):** ran the real `radgraph` package (`modern-radgraph-xl`, pip-installable, CPU, weights download openly from Hugging Face — no PhysioNet credential needed for inference). On *"left pleural effusion and basilar opacification resolved"*, RadGraph extracts both findings correctly decomposed, **but "resolved" produces no entity at all and both findings stay tagged `definitely present`** — RadGraph's own certainty label does not capture temporal/resolution semantics, confirming step 5-6's separate temporal classifier is necessary, not redundant. Negation, by contrast, RadGraph gets right natively (*"no definite pneumothorax is seen"* → `definitely absent`) — base polarity tagging can reuse RadGraph's own certainty label instead of a separate regex. **New, more serious finding:** on a 13-phrase probe of concept-bank-style short noun phrases, RadGraph returned **zero entities for 5 of 13 (38%)**, including unambiguous findings like bare `"cardiomegaly"`, `"right pleural effusion"`, and `"left upper lobe opacity"` — likely because RadGraph is trained on full report sentences, not CLEAR's templated fragments. Laterality words (`"left"`, `"right"`) are never extracted as entities in any example — confirms laterality must stay a wholly separate regex layer, never derived from RadGraph. **Consequence: the RadLex/CheXpert gazetteer fallback in §4.4/§4.6 step 7 is now load-bearing, not a "cheaper alternative"** — a meaningful fraction of concepts will need it because RadGraph alone misses them. This was checked on ~18 hand-picked phrases only; the real 368k-concept batch run (needed to get an actual coverage rate) has not been done — that run is estimated at roughly 2+ hours on CPU from a small-batch throughput measurement (~44 texts/s unbatched; real throughput on a large batch is unmeasured) and belongs on the HPC, not this exploration |

### Missing from the diagram entirely

| ID | Issue |
|---|---|
| **AMBIG-1** | **Multi-image studies.** Real CXR studies are frequently PA + lateral. The diagram shows one image. Does the pipeline take one image, or fuse views? View fusion would change N01, N02, N03, and the belief graph's identity model |
| **GAP-11** | **Prior studies / comparison.** Radiology reports routinely reference change over time ("improved since…"). No prior-study input exists anywhere in the pipeline; the LLM must therefore be explicitly forbidden from writing comparative language, since no evidence stream can support it |
| **GAP-12** | **Clinical indication / reason for exam.** Not an input. This is standard context in real reporting and its absence should be a stated limitation |
| **GAP-13** | **Normal-study path.** What does the pipeline emit when all evidence is negative? The belief graph should carry explicit negative findings, and N27 must render a normal report. This case must be tested first, not last — several components (within-study normalisation, top-K selection) behave badly on it |
| **GAP-14** | **Report structure.** Findings vs. Impression is not modelled. The Impression is a *summary judgement*, which is arguably a clinical decision the LLM should not be making unaided — consider deriving it deterministically from the verified graph's highest-confidence findings |
| **GAP-15** | **Training/fitting story.** Which components are trained, on what data, with what splits? At minimum: the spatial model, the CBM head, the per-source calibrators, and the fusion weights. Leakage control between the retrieval corpus and any evaluation set is essential and is not addressed |
| **GAP-16** | **Evaluation.** No metrics are specified. Needed: clinical-efficacy metrics (RadGraph F1, CheXbert-style label F1), NLG metrics as secondary, and — specific to this architecture — verification pass rate at attempt 1 and per-error-class rates, which directly measure whether the containment design is working |
| **GAP-17** | **Human-in-the-loop.** No review or escalation surface. Given GAP-10's exhaustion case, a review queue is implied |

---

## 9. Recommended Build Order

Derived from the dependency graph in §7.1; each step is independently testable.

1. **Contracts first** — canonical vocabulary (GAP-6), pathology label space (GAP-1), core types (§3.2), version matrix + startup gate (§7.1).
2. **Assets** — pin CLEAR (GAP-3); build concept bank embeddings + run the auto-tagging pipeline for the sidecar table (§4.4, resolved GAP-4) + its QA sample; build the FAISS index with patient-level exclusion; precompute RadGraph over the reference corpus.
3. **Stage 1 branches except spatial, independently** — CLEAR → similarity → 368k scores; the 68-filter → CBM; FAISS → RadGraph → Evidence D. Each with fixture-based tests on known cases. **N02/Evidence A deliberately deferred** (Revision Log rev 1) — build and test the `EvidenceA=absent` identity path (§4.2, §4.6) as the first-class case, not as an afterthought.
4. **KG + RadGraph** (§5.4, resolved GAP-8: RadLex only, via BioPortal, for both N12/N19 tags and the N25 vocabulary base; RadGraph for entity parsing) **+ hand-curated N25 rules**, then N12/N19 grouping (§4.6, rev 3), validated by checking that near-duplicate phrasings of a known finding collapse into one group and that a hand-picked normal-film image ranks no findings above threshold.
5. **Stage 2** — normalisation, calibration, fusion, initial graph, rule check, running on 3 sources (B+C+D) while N02 is deferred. Validate on the normal-study case (GAP-13) before anything else.
6. **Stage 3 (v1: single pass, rev 1)** — generation, extraction, filtering, assembly. Build **N30 before N27**: the filter can be tested against hand-written good and deliberately corrupted drafts, and it defines the target the generator must hit.
7. **Evaluation harness** (GAP-16), with claim-removal rate and `MISSING_FINDING` (omission) rate as the headline v1 metrics (§7.4) — the omission rate is what justifies building Phase 2.
8. **Phase 2 (later)** — build N02 (spatial perception, GAP-2) and flip Evidence A back on; build the regeneration loop (§6.5) to remediate `MISSING_FINDING` (GAP-18).


---

## 10. Post-rev-2 decisions and measurements (implementation phase)

| Item | Decision / finding |
|---|---|
| **GAP-1 (CBM label space)** | **Resolved:** the 14 MIMIC-CXR CheXpert labels, used for the CBM branch only (Evidence C). They are not used for retrieval or any other stage |
| **Data** | MIMIC-CXR-JPG + reports, **official splits only**. Local sample: 100 studies (50 train / 50 test), one frontal PA/AP image each. Full data lives on the HPC |
| **N05 embedding** | One 768-d L2-normalised vector X from CLEAR's `encode_image`. It is dotted with the 368,294 x 768 CLEAR text embeddings (N07) **and** used for FAISS (N09). No LLM-projected embedding |
| **Concept bank measured** (`scripts/explore_concepts.py`) | 368,294 unique phrases, no duplicates/empties; median 7 words, max 52. 52.7% contain left/right/bilateral wording; 21.6% mention devices; 9.0% are hedged. **29.8% use temporal/comparison wording** ("unchanged", "interval", "new", "increased"...), i.e. change against a prior exam. This quantifies **GAP-11**: with no prior-study input, those concepts cannot be supported and need to be filtered or down-weighted before Top-K (N19). Some phrases keep MIMIC's `___` de-identification token |
| **Top-K on raw similarity** (3 sample images) | Top hits are dominated by device/position phrases and near-duplicates (e.g. four "right basal pleural tube" variants), and include temporal phrases ("in unchanged position"). Confirms the need for the Top-K dedup rule (**GAP-5**) and the temporal filter above |

### 10.1 Retrieval protocol, splits and view policy (decided)

| Item | Decision |
|---|---|
| **Corpus** | Train split only (official MIMIC-CXR split), one frontal image per study: PA preferred over AP, ties -> smallest `dicom_id`, lateral-only / unknown-view studies excluded. Same policy for corpus and queries. One image per study for now; multi-image studies are a later extension |
| **Splits** | Train = retrieval corpus (and any fitted components). **Validate = calibration, rule parameters, thresholds, Top-K, fusion settings.** **Test = untouched until the final evaluation** |
| **Index** | Exact `IndexFlatIP` on fp32, L2-normalised 768-d X (inner product = cosine). Query = the same X, fp32. No bf16/mixed precision anywhere on this path |
| **Leakage control (fixes §4.8's requirement)** | Removing only the query's own `dicom_id` is insufficient: it lets the same patient's other studies return. The retriever removes every index entry of the query's patient, exactly (over-fetch by that patient's row count). MIMIC's official splits are patient-disjoint, so this only ever changes results for train-split queries; the loader verifies disjointness and fails otherwise |
| **Known leakage risk outside our code (unverified)** | CLEAR's 368,294 concepts were mined from the report text of 227,835 studies, which is the entire MIMIC-CXR dataset, so phrases from validation/test reports are in the vocabulary. CLEAR was pretrained on 873,342 image-report pairs from MIMIC-CXR, CheXpert-Plus and ReXGradient; whether MIMIC-CXR test patients were excluded from pretraining was **not confirmed** (paper Methods not read). Our train-only index and untouched-test rule cannot undo exposure that happened inside the released encoder / concept bank. Check the paper's Methods; if MIMIC test was included, report it as a limitation and prefer an external evaluation set for headline numbers |
| **Neighbour reports contain prior-study language** | Retrieved reports include "COMPARISON:", "WET READ", "___" de-identification tokens and change-over-time statements. Evidence D (N15/N18) must not turn comparison statements into facts about the current image (same problem as the 29.8% temporal concepts, GAP-11) — **paused for now, not designed** (see below) |

### 10.2 Evidence B redesign: grouping replaces Evidence-A-conditioned refinement (rev 3)

Evidence D is paused while this is worked out; the design below covers Evidence B only.

| Question | Decision |
|---|---|
| Should Evidence A gate/refine the 368k concept scores? | **No, removed permanently (not just deferred).** Gating on an unvalidated source propagates its errors with no cross-check, and the dedup problem (measured, §10) exists independent of A anyway. Full rationale and mechanism: §4.6 |
| How is Evidence B actually built, then? | RadGraph-parse all 368k concepts once, offline; normalise anatomy via RadLex, laterality via regex, base polarity via negation cues (extends §4.4); classify temporal/comparison language and combine it with base polarity into a final `resolved_polarity`; group concepts by `(canonical_finding, AnatomyID, laterality)`; at query time, aggregate each group's `present_score`/`absent_score` separately (never netted) from the *already-computed, unmodified* N07 scores; rank groups by `max(present_score, absent_score)`; select top-K groups |
| Does resolving temporal language mean re-scoring or re-embedding a concept? | **No.** N07's scores are computed once, against CLEAR's fixed, frozen concept embeddings, and never touched again. Grouping and polarity resolution are a relabelling of already-computed scores — selection happens on the original score, and only concepts that already cleared consideration get relabelled with a finding + polarity |
| Where does the canonical finding vocabulary (for B) come from, if not RadLex? | RadGraph's own observation-entity strings, clustered for synonyms, anchored by the 14 CheXpert labels. RadLex's finding/observation coverage was measured to be too thin for this (missing `cardiomegaly`, `cardiac silhouette`, `widened mediastinum`, `lung opacity`, `hyperinflation` in a 26-term probe) — RadLex remains the anatomy backbone, not the finding backbone |
| What survives into Evidence B, per selected group — just the group label and score? | No — a bounded `top_contributing_concepts` list (concept_id, text, raw_score, resolved_polarity, temporal_class) travels with each group, for the same traceability (INV-2) reason the CBM's `top_contributing_concepts` field exists. This detail stays behind the Evidence B / `Finding` boundary — it is never serialised into the LLM prompt (INV-3 unaffected) |
| Why not just use max/noisy-OR/entropy-weighting instead of a trained calibrator downstream? | Assessed and rejected as a *replacement* for trained fusion (though useful as baselines/features): max-pooling picks whichever source is most inflated and ignores agreement; noisy-OR assumes source independence that doesn't hold here (C is computed from a subset of B's own scores, both come from the same image embedding X); entropy-weighting conflates confidence with correctness and has no natural definition for B's raw cosine scores. Calibration/fusion design itself is still open — see the discussion this rev is drawn from |
| Calibration data source (not yet built, recorded for continuity) | Proposed: label each candidate finding by whether it's asserted in that study's own ground-truth report (via RadGraph on the report), not by the 14 CheXpert labels alone — CheXpert is too narrow (misses everything outside its 14 labels) and calibrating the CBM on its own labels would just confirm what it was trained to say. Fit only on the **validation** split (train-split outputs are optimistic); "not mentioned" vs. "explicitly negated" in the report need different label treatment, still undecided |

**What's still open after this rev:** the calibration design above (labels, missing-source handling, how much to discount Evidence B for possible MIMIC-exposure inflation); Evidence D's temporal handling (paused); the KG/rule-based revision-loop idea discussed alongside this (deliberately not adopted for v1 — flagged as a possible v1.5 extension, not specified here).
