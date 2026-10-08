#!/bin/bash
# Launch a python script as a detached SLURM step. Usage: launch.sh <jobid> <run-name> <script.py> [args...]
# Writes <run-dir>/run.sh (the exact command) and launches it with srun --overlap; no nested quoting.
set -euo pipefail
cd "$(dirname "$0")/.."
JOB=$1; NAME=$2; SCRIPT=$3; shift 3
RUN="runs/$(date +%Y%m%d-%H%M%S)_$NAME"; mkdir -p "$RUN"
{
  echo '#!/bin/bash'
  echo "cd $(pwd)"
  echo 'export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}'
  printf 'exec /scratch/users/k23031260/.conda/envs/rrg/bin/python -u %q' "$SCRIPT"
  for a in "$@"; do printf ' %q' "$a"; done
  printf ' --run-dir %q > %q 2>&1\n' "$RUN" "$RUN/stdout.txt"
} > "$RUN/run.sh"
chmod +x "$RUN/run.sh"
setsid nohup srun --jobid="$JOB" --overlap "$RUN/run.sh" > "$RUN/srun.out" 2>&1 < /dev/null &
echo "$RUN"
