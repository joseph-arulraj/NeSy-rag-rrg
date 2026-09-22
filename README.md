# NeSy-RAG-RRG — evidence-based radiology report generation (v1 in progress)

Specification: `docs/pipeline.md` (source of truth) · module design: `docs/architecture.md`.

**Built so far:** N01 image loader · N03/N05 CLEAR encoder · N08 concept bank · N07 concept similarity ·
N09 retrieval (FAISS index over the train split) · N12/N16/N19 concept grouping + temporal/polarity
resolution (rev 3 — no Evidence A) · RadLex snapshot + client · concept-bank tagging pipeline (RadGraph +
RadLex + finding vocabulary) · a concept explorer. Next: Evidence C (CBM) and Evidence D (N11/N15/N18).

## Setup (Mac and HPC)

```bash
conda activate <your-env>
pip install -e ./CLEAR            # the cloned CLEAR repo (not on PyPI)
pip install -e '.[dev]'           # this project (torch, numpy, pillow, pyyaml, pytest)
pip install -e '.[faiss]'         # HPC only: faiss-cpu for retrieval.backend: faiss
```

**All paths and tunables live in `configs/default.yaml`** (Mac profile). `configs/hpc.yaml` overlays it for the
cluster: replace every `CHANGE_ME`, then pass `--config configs/hpc.yaml`. Single values can be overridden with
`--set section.key=value`. Device is `auto` = CUDA > MPS > CPU; a forced device that is unavailable raises.

### One-time, on a machine with internet (HPC login node)
CLEAR builds its image tower from the DINOv2 *source code* (weights come from `best_model.pt`) via `torch.hub`,
which needs GitHub. Compute nodes usually can't reach it, so cache it once:

```bash
python scripts/cache_dinov2.py --config configs/hpc.yaml     # then keep clear.local_files_only: true
```

## Use

```bash
python scripts/explore_concepts.py                      # profile of the 368,294 concepts -> outputs/concept_exploration.md
python scripts/explore_concepts.py --grep "left lower lobe" --max 25
python scripts/run_demo.py                              # image -> X (768-d) -> concept scores -> top-K, on the MIMIC sample
python scripts/run_demo.py --config configs/hpc.yaml
pytest -q                                               # real-asset tests skip themselves if files are missing

python scripts/build_faiss_index.py                     # offline: encode every TRAIN study (fp32) -> <index_dir>/train.*
python scripts/run_retrieval_demo.py --query-split test # top-k similar train studies + their reports for a few query images
python scripts/run_retrieval_demo.py --query-split train --no-exclude   # shows the self/same-patient leak the exclusion removes

python scripts/build_radlex_snapshot.py                  # offline, once: parse the downloaded RadLex.owl -> model_weights/radlex_snapshot.json.gz
python scripts/build_concept_tags.py                     # offline: RadGraph-parse all 368,294 concepts, tag anatomy/laterality/polarity/temporal/finding
python scripts/build_concept_tags.py --limit 500         # smoke test (~10s); the full run is ~2h on CPU at the measured throughput, resumable if interrupted
python scripts/run_evidence_b_demo.py                    # image -> X -> concept scores -> grouped, temporal/polarity-resolved findings (Evidence B)
```

## Retrieval protocol (decisions, see docs/pipeline.md section 10)
- **Corpus = train split only**, one frontal image per study (PA preferred over AP, ties -> smallest `dicom_id`; lateral-only
  studies excluded). Validate = tuning (calibration, thresholds, top-K, fusion, rule parameters). **Test is untouched until the
  final evaluation.** Official MIMIC splits are patient-disjoint; the loader verifies it and raises if not.
- **fp32 everywhere** on the embedding path (index and queries). `clear.precision` other than `fp32` is a config error.
- **Same-patient exclusion** is exact (over-fetch by that patient's index rows), not just the same `dicom_id`.
- **`retrieval.backend`**: `faiss` (IndexFlatIP, the design; HPC) or `numpy` (identical exact search as a matmul; default in the Mac
  profile). On macOS, pip's `torch` and `faiss-cpu` abort when loaded in one process (clashing OpenMP runtimes).
  A full `pytest -q` therefore skips the FAISS cases on a Mac; run them alone: `pytest tests/test_faiss_retrieval.py`. On the HPC
  everything runs in one `pytest`.

## Data handling
`data/`, `model_weights/`, `outputs/`, `CLEAR/` and the RadLex downloads are git-ignored on purpose. MIMIC-CXR is
under the PhysioNet credentialed-data agreement and must not be pushed to any remote. On the HPC, point
`paths.*` at the real locations in `configs/hpc.yaml`.
