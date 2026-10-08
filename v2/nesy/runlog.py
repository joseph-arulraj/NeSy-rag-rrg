"""Run directories and file logging, as required by CLAUDE.md.

Every script calls `start_run(name, args)` first. It creates `runs/<YYYYMMDD-HHMM>_<name>/`, writes
`config.json` and `STATUS` (RUNNING), and tees stdout/stderr into `log.txt` with timestamps.
Use it as a context manager: on exit it writes DONE or FAILED (with the traceback in log.txt) and a
final line with the total time and the output location.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = V2_ROOT / "runs"


class _Tee:
    """Writes every line to the real stream and, timestamped, to the log file."""

    def __init__(self, stream, fh):
        self.stream, self.fh, self._at_line_start = stream, fh, True

    def write(self, text: str) -> int:
        self.stream.write(text)
        for chunk in text.splitlines(keepends=True):
            if self._at_line_start:
                self.fh.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S "))
            self.fh.write(chunk)
            self._at_line_start = chunk.endswith("\n")
        self.fh.flush()
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.fh.flush()


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(V2_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(V2_ROOT), "status", "--porcelain", "--", str(V2_ROOT)],
                               capture_output=True, text=True, timeout=30).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001 -- provenance is best effort, never fatal
        return "unknown"


class Run:
    def __init__(self, name: str, config: dict, run_dir: Path | None = None):
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        self.dir = Path(run_dir) if run_dir else RUNS_DIR / f"{stamp}_{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.config = {
            **config,
            "run_name": name,
            "git_commit": _git_commit(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "host": socket.gethostname(),
            "python": sys.executable,
            "argv": sys.argv,
            "env_nesy": {k: v for k, v in os.environ.items() if k.startswith("NESY_")},
            "started": datetime.now().isoformat(timespec="seconds"),
        }
        self._metrics = None

    def __enter__(self) -> "Run":
        (self.dir / "config.json").write_text(json.dumps(self.config, indent=2, default=str))
        self._status("RUNNING")
        self._fh = open(self.dir / "log.txt", "a", encoding="utf-8")
        self._orig = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(sys.stdout, self._fh), _Tee(sys.stderr, self._fh)
        self.t0 = time.time()
        self.log(f"run {self.name} started in {self.dir} on {self.config['host']} (job {self.config['slurm_job_id']})")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = time.time() - self.t0
        if exc_type is not None:
            traceback.print_exception(exc_type, exc, tb)
            self.log(f"FAILED after {elapsed / 60:.1f} min; outputs in {self.dir}")
            self._status("FAILED")
        else:
            self.log(f"FINISHED in {elapsed / 60:.1f} min; outputs in {self.dir}")
            self._status("DONE")
        sys.stdout, sys.stderr = self._orig
        self._fh.close()
        if self._metrics:
            self._metrics.close()
        return False

    def _status(self, word: str) -> None:
        (self.dir / "STATUS").write_text(word + "\n")

    def log(self, msg: str) -> None:
        print(msg, flush=True)

    def metric(self, **kv) -> None:
        if self._metrics is None:
            self._metrics = open(self.dir / "metrics.jsonl", "a", encoding="utf-8")
        self._metrics.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), **kv}, default=str) + "\n")
        self._metrics.flush()


class Progress:
    """Plain-line progress logging: done/total, elapsed, ETA, at most every `every_s` seconds."""

    def __init__(self, run: Run, total: int, label: str, every_s: float = 60.0):
        self.run, self.total, self.label, self.every_s = run, total, label, every_s
        self.t0 = self.last = time.time()

    def update(self, done: int, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last < self.every_s and done < self.total:
            return
        self.last = now
        el = now - self.t0
        rate = done / el if el > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else float("nan")
        self.run.log(f"{self.label}: {done:,}/{self.total:,} done, elapsed {el / 60:.1f} min, ETA {eta / 60:.1f} min")
