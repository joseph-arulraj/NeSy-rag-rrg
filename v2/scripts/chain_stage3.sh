#!/bin/bash
# Final concept lists -> Stage 3 (k=6, k=12) -> normal-call check (root 5 % / 2 %) for each. Usage: chain_stage3.sh <chain-dir> <stage1-run>
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/scratch/users/k23031260/.conda/envs/rrg/bin/python
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
CHAIN=$1; S1=$2; mkdir -p "$CHAIN"; echo RUNNING > "$CHAIN/STATUS"; trap 'echo FAILED > "$CHAIN/STATUS"' ERR
step() { local name=$1; shift; local rd="runs/$(date +%Y%m%d-%H%M)_$name"
  echo "$(date '+%F %T') START $name -> $rd" >> "$CHAIN/chain.log"
  if "$PY" -u "$@" --run-dir "$rd" > "$rd.stdout" 2>&1; then echo "$(date '+%F %T') DONE  $name -> $rd" >> "$CHAIN/chain.log"; echo "$rd" > "$CHAIN/last_$name"
  else echo "$(date '+%F %T') FAIL  $name -> $rd" >> "$CHAIN/chain.log"; exit 1; fi; }
step concepts-final scripts/select_concepts.py --ks 6,12
for K in 6 12; do
  step "stage3-k$K" scripts/stage3_factorised.py --name "k$K" --concepts "data/concepts/concepts_final_k$K.csv" --baseline "$S1"
  step "normal-call-stage3-k$K" scripts/abnormal_absent_check.py --pred-run "$(cat $CHAIN/last_stage3-k$K)" --label "stage3-k$K" --rules "root 5%;root 2%"
done
echo DONE > "$CHAIN/STATUS"; echo "$(date '+%F %T') CHAIN FINISHED" >> "$CHAIN/chain.log"
