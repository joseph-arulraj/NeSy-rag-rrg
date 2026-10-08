# Project instructions

**At the start of every session and after any compaction, read `STATE.md` and `QUEUE.md` (repo root) before doing anything else.**

This repo is being rebuilt into a neurosymbolic chest X-ray report pipeline. The full plan, context and stage list are in `PIPELINE_BRIEF.md` at the repo root. Read it in full at the start of every session, and re-read the relevant stage before starting it.

Progress lives in these files. Read them at the start of a session and keep them current:

- `AUDIT.md`: findings about the old pipeline.
- `DATA.md`: verified layout and counts for each dataset.
- `COMPUTE.md`: running SLURM allocations, the step command that works, expiry times.
- `RESULTS.md`: every experiment result, including negative ones.
- `QUEUE.md`: what is running, queued, finished, failed and waiting on the user.
- `STATE.md`: decisions with reasons, conventions, pitfalls, open items.
- `v2/KG.md`: the knowledge base, its sources and what still needs review.

## Rules that always hold

- Splits are patient-level and frozen once written. Never change them without asking.
- VinDr-CXR test and PadChest-GR are external test sets. Never use them for training, calibration, thresholds or model selection.
- MS-CXR patients and Chest ImaGenome gold-standard patients are excluded from every MIMIC training split.
- A finding not mentioned in a report is a negative label.
- Calibrators, thresholds and test metrics each use a different split. Use Platt or beta scaling, not isotonic.
- CLEAR was pretrained on all of MIMIC-CXR. Label every MIMIC result as "seen by backbone".
- Scratch quota is 1000 GB and nearly full. Never copy a dataset. Check free space before writing anything large.
- Build in a new directory. Never delete, move or overwrite the old pipeline, its outputs or any downloaded data.
- GPU work runs as steps inside held SLURM allocations (see `COMPUTE.md`). Hold at most three (the user allowed four on 2026-10-07; ask before going above three again). Every long run checkpoints and can resume.
- Do not add a triple store, Prolog or Logic Tensor Networks.

## Python environment

- Use the conda env `rrg` for the whole project: `/scratch/users/k23031260/.conda/envs/rrg`.
- Call its interpreter by full path, `/scratch/users/k23031260/.conda/envs/rrg/bin/python`, everywhere, including inside `srun` steps. Do not rely on `conda activate`.
- The path and the verified package list are recorded in `COMPUTE.md`.

## Logging

Every script and every training run logs to a file, so progress can be followed with `tail -f` and a failure can be diagnosed without rerunning.

- Each run gets its own directory: `runs/<YYYYMMDD-HHMM>_<short-name>/`.
- That directory holds `config.json` (all arguments and hyperparameters, seed, git commit, split-file version, SLURM job ID and node), `log.txt`, `metrics.jsonl` and any checkpoints.
- `log.txt` has timestamped lines and captures stdout and stderr, including full tracebacks.
- Long loops log progress at a regular interval: items done out of total, elapsed time, estimated time remaining. Use plain log lines, not progress bars, which are unreadable in a file.
- Training logs loss and validation metrics every epoch or fixed number of steps, and says when a checkpoint was written and where.
- Preprocessing logs counts in and out of every step, and how many items were skipped and why.
- Each run ends with a clear final line: finished or failed, total time, and where the outputs are.
- After a run, add its headline numbers and run directory to `RESULTS.md`.

## Waiting on runs

Runs take anywhere from a few minutes to a few hours, so check on them by their own estimate, not on a fixed timer.

- Every run writes a one-word `STATUS` file in its run directory: `RUNNING`, `DONE` or `FAILED`. It is written at start and updated at exit, including on an exception.
- To check a run, read `STATUS` and the last 20 or so lines of `log.txt`. Do not re-read whole logs.
- Schedule the next check from the estimated time remaining in the log: soon when it is nearly done, at most every 30 minutes for a long run. Use the session's scheduling or monitoring tools for this, not a blocking `sleep`.
- A run with `STATUS` of `RUNNING` whose log has not changed for 15 minutes, or whose step is gone from `squeue -s`, is treated as failed. Read the end of `log.txt` and `srun.out` to find out why.
- While a run is in progress, do other useful work: preprocessing on CPU, writing and testing the next stage on a small sample, or a second run on another allocation.
- When a run finishes or fails, report it in one or two lines with the headline numbers or the error.

## Working style

- Verify before claiming: run the code, check the counts, read the file. Say plainly what was verified and what is assumed.
- Each build stage must beat the previous one on a named metric or be dropped. Record the numbers either way.
- Stop and ask before anything listed under "Ask before" in the brief.