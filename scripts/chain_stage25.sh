#!/bin/bash
# Stage 2 and Stage 5 ablations against a given Stage 1 run. Usage: bash chain_stage25.sh <chain-dir> <stage1-run-dir>
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/scratch/users/k23031260/.conda/envs/rrg/bin/python
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
CHAIN=$1; S1=$2
mkdir -p "$CHAIN"; echo RUNNING > "$CHAIN/STATUS"
trap 'echo FAILED > "$CHAIN/STATUS"' ERR
step() { local name=$1; shift; local rd="runs/$(date +%Y%m%d-%H%M)_$name"
  echo "$(date '+%F %T') START $name -> $rd" >> "$CHAIN/chain.log"
  if "$PY" -u "$@" --run-dir "$rd" > "$rd.stdout" 2>&1; then echo "$(date '+%F %T') DONE  $name -> $rd" >> "$CHAIN/chain.log"
  else echo "$(date '+%F %T') FAIL  $name -> $rd" >> "$CHAIN/chain.log"; exit 1; fi; }
step stage2 scripts/probe.py --name stage2 --features emb,chexmask --baseline "$S1"
for k in 5 10 25; do step "stage5-nb$k" scripts/probe.py --name "stage5-emb-nb$k" --features "emb,nb$k" --baseline "$S1"; done
echo DONE > "$CHAIN/STATUS"; echo "$(date '+%F %T') CHAIN FINISHED" >> "$CHAIN/chain.log"
