# Shared helpers for lane orchestrators (sourced). Each step is a foreground SLURM step launched from a
# run.sh file in its own run directory (no nested bash -c), logged to $ORCH/orchestrator.log.
V2=/scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/v2
PY=/scratch/users/k23031260/.conda/envs/rrg/bin/python
log() { echo "$(date '+%F %T') $*" | tee -a "$ORCH/orchestrator.log"; }
free_gb() { getfattr --only-values -n ceph.dir.rbytes /scratch/prj/bhi_zihe_imaging 2>/dev/null | awk '{printf "%.1f", (1e12-$1)/1e9}'; }
# step NAME JOB [ENV=val ...] -- script.py args...
step() {
  local name=$1 job=$2; shift 2
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local rd="runs/$(date +%Y%m%d-%H%M%S)_$name"; mkdir -p "$V2/$rd"
  {
    echo '#!/bin/bash'; echo "cd $V2"; echo 'export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}'
    for e in "${envs[@]}"; do echo "export $e"; done
    printf 'exec %q -u' "$PY"; for a in "$@"; do printf ' %q' "$a"; done
    printf ' --run-dir %q > %q 2>&1\n' "$rd" "$rd/stdout.txt"
  } > "$V2/$rd/run.sh"; chmod +x "$V2/$rd/run.sh"
  log "START $name on $job -> $rd"
  (cd $V2 && srun --jobid="$job" --overlap "$rd/run.sh" > "$rd/srun.out" 2>&1)
  local rc=$?
  local st; st=$(cat "$V2/$rd/STATUS" 2>/dev/null || echo MISSING)
  log "END   $name rc=$rc STATUS=$st free=$(free_gb) GB"
  echo "$rd" > "$ORCH/last_$name"
  [ "$st" = DONE ] && [ $rc -eq 0 ]
}
lastdir() { cat "$ORCH/last_$1"; }
