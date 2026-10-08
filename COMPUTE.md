# Compute

## Python environment

- Conda env **`rrg`**: `/scratch/users/k23031260/.conda/envs/rrg`
- Interpreter (use by full path everywhere, including inside `srun`): `/scratch/users/k23031260/.conda/envs/rrg/bin/python` (Python 3.11)
- Verified imports, 2026-10-06: torch 2.14.0+cu130, scikit-learn 1.9.1, faiss 1.15.1, radgraph 0.1.18, clear (`clear-med` 0.1.0, editable install from `/cephfs/volumes/hpc_data_prj/bhi_zihe_imaging/91a35342-d50a-4aa5-8a4c-a1ac0086fb4c/CLEAR/src/clear`), pandas 3.0.6, numpy 2.4.6, PIL 12.3.0, scipy 1.17.1, yaml 6.0.3.
- Installed 2026-10-06: pyarrow 25.0.1, pydicom 3.0.2 (no other packages changed). Still missing: pycocotools, cv2.
- Imports are slow from cephfs (torch + radgraph ≈ 2 min the first time).

## Allocations

| Job ID | Partition | Resources | Submitted | State | Node | Expires | Lane / runs |
|---|---|---|---|---|---|---|---|
| **37814480** | slam_gpu | **H100 NVL 96 GB**, 16 CPU, 256G, **1 day** | 2026-10-06 11:29 (by user) | **RUNNING** (started 11:31) | erc-hpc-comp231 | **2026-10-07 11:31** | GPU lane: checks, patch-token pilot, region pass |
| **37811782** | slam_cpu | 16 CPU, 400G, 2 days, no GPU | 2026-10-06 09:39 | **RUNNING** (started 09:39) | erc-hpc-comp214 | 2026-10-08 09:39 | CPU lane: `…0940_chexmask-features` (DONE 09:52), `…1004_imagenome-parse` (DONE), `…1010_radgraph-reports` (RUNNING, ETA ~15:30) |
| 37811829 | slam_gpu | H100, 16 CPU, 256G, 1 day | 2026-10-06 09:44 | cancelled 09:50: est. start 2026-10-08 11:21 (Resources) | — | — | — |
| 37811052 | biomed_a100_gpu | A100 512G 2 days | 08:58 | cancelled 11:4x at user's request (one GPU is enough) | | | |
| 37812110 | biomed_a100_gpu | A100, 16 CPU, 256G, 1 day | 2026-10-06 09:54 | cancelled 10:00: est. start 2026-10-10 11:00 (Priority), later than the 512G job | — | — | — |
| 37811050, 37811051 | slam_gpu | H100 512G | 08:58 | cancelled while pending | | | |
| 37811507 | slam_gpu | H100 512G 2 days | 09:2x | cancelled while pending (replaced by 37811829) | | | |
| 37811738 | interruptible_gpu | H100 512G | 09:3x | cancelled while pending (user) | | | |
| 37811770 | slam_cpu | 512G | 09:38 | cancelled: comp214 had only ~467G free | | | |

Notes:
- slam_gpu has one H100 node (erc-hpc-comp231: 4 GPUs, 60 CPUs). At 09:44 all 4 GPUs were allocated and 58 of 60 CPUs were in use, so a 16-CPU request waits for others' jobs to end.
- slam_cpu nodes comp214/comp215 have 128 CPUs and 2 TB each. A 512G request did not fit; 400G started immediately.
- Smaller GPU requests did not help: H100 256G/1 day was estimated to start 2026-10-08, and A100 256G/1 day 2026-10-10.
- interruptible_gpu has H100 nodes (comp228, comp229) with PreemptMode=CANCEL. It did not start within ~5 min.

CPU allocation command that works:

```bash
sbatch --partition=slam_cpu --cpus-per-task=16 --mem=400G --signal=USR2 --time=2-00:00:00 --wrap="sleep infinity"
```

## Step command (verified 2026-10-06 on 37811782)

```bash
PY=/scratch/users/k23031260/.conda/envs/rrg/bin/python
cd /scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/v2
RUN=runs/$(date +%Y%m%d-%H%M)_<name>; mkdir -p $RUN
nohup srun --jobid=<id> --overlap bash -c "cd $PWD && $PY -u <script> --run-dir $RUN > $RUN/stdout.txt 2>&1" > $RUN/srun.out 2>&1 &
squeue -s -j <id>
```

The ten-second test (`v2/runs/20261006-0940_step-test`) printed live into the run directory and wrote STATUS DONE. Inside the step, `os.cpu_count()` reports 128 (the whole node), so always pass `--workers 16` explicitly. Scripts write `log.txt` themselves through `nesy.runlog`; `stdout.txt` duplicates it.

## GPU step test (verified 2026-10-06 on 37814480)

The same detached pattern works on the GPU allocation: `setsid nohup srun --jobid=37814480 --overlap bash -c "…" > $RUN/srun.out 2>&1 < /dev/null &`. Step start took about 50 s. The node is erc-hpc-comp231 with an NVIDIA H100 NVL (95,830 MiB); torch.cuda works in `rrg`. Always launch with `setsid … < /dev/null` and confirm the step appears in `squeue -s`.

## Internet

- Login node: yes.
- CPU compute nodes: yes. comp214 inside the step reached https://pypi.org (HTTP 200); the S3 downloads also ran on CPU nodes.
- GPU compute node comp231 (job 37814480): **yes**, https://pypi.org returned HTTP 200 from inside the step (`v2/runs/*_gpu-step-test`).

## Other partitions seen

`cpu` (default, 2-day limit, 54 nodes, 16+ CPUs), `paid_cpu`, `interruptible_cpu`, `slam_cpu` (2 nodes × 128 CPUs, 2 TB, 7-day limit), `interruptible_gpu` (A100/A30/L40S/H100, preemptible, no time limit).

## Allocations as of 2026-10-07 ~10:55
| job | partition | resources | expires | note |
|---|---|---|---|---|
| 37811782 | slam_cpu | 16 CPU, 400G | 2026-10-08 ~09:40 | CPU lane |
| 37814480 | slam_gpu | H100 NVL, 16 CPU, 256G | **2026-10-07 11:31** | current H100 |
| 37831235 | slam_gpu | H100 (constraint h100), 16 CPU, **512G**, 2 days | pending; expected start 11:31 today | time limit raised to 2 days while pending (`scontrol update`) |
| 37831634 | slam_gpu | **L40S** (comp232), 16 CPU, 256G, 2 days | 2026-10-09 ~06:06 | DenseNet, KG work; has internet |
The user allowed four allocations today (normally at most three).

### 2026-10-07 11:36 update
- 37814480 (H100, comp231) reached its 24 h limit at 11:33 and ended.
- **37831235** (slam_gpu, H100, 512G, 16 CPU) started 11:36 on erc-hpc-comp231; ends ~2026-10-09 11:36.
- Held now (3): CPU 37811782 (comp214, to ~2026-10-08 09:40), H100 37831235 (comp231, to 2026-10-09 11:36), L40S 37831634 (comp232, to 2026-10-09 ~06:06).

### 2026-10-07 11:40–12:00: runner build and validation (37814480 not used; it ended 11:33)
| Step | Allocation |
|---|---|
| H3 comparator probe + bootstrap (`runs/20261007-1141_lane-h3`) | L40S 37831634 (CPU work) |
| External CheXmask features (`runs/20261007-114214_chexmask-ext`) | CPU 37811782 |
| Runner on val without Stage 8, equivalence check, evaluation test | CPU 37811782 |
| Stage 8 diagnosis, retry and repair tests, full val runner with Stage 8 (`runs/20261007-120046_pipeline-val-v2`) | H100 37831235 (RadGraph on GPU; LLM calls 2 concurrent) |
