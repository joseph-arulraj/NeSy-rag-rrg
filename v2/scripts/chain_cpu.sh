#!/bin/bash
# Runs the CPU work list items that depend on the embedding store, in order, each in its own run dir.
# Usage (inside an allocation step): bash v2/scripts/chain_cpu.sh <chain-run-dir>
# A failing item stops the chain; its run dir holds the traceback.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/scratch/users/k23031260/.conda/envs/rrg/bin/python
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
CHAIN=$1
mkdir -p "$CHAIN"
stamp() { date +%Y%m%d-%H%M; }
step() {   # name, then command args
  local name=$1; shift
  local rd="runs/$(stamp)_$name"
  echo "$(date '+%F %T') START $name -> $rd" >> "$CHAIN/chain.log"
  if "$PY" -u "$@" --run-dir "$rd" > "$rd.stdout" 2>&1; then
    echo "$(date '+%F %T') DONE  $name -> $rd" >> "$CHAIN/chain.log"
  else
    echo "$(date '+%F %T') FAIL  $name -> $rd" >> "$CHAIN/chain.log"; exit 1
  fi
  echo "$rd" > "$CHAIN/last_$name"
}
echo RUNNING > "$CHAIN/STATUS"
trap 'echo FAILED > "$CHAIN/STATUS"' ERR
# retrieval done in runs/20261006-1148_retrieval
step stage1 scripts/probe.py --name stage1 --features emb
S1=$(cat "$CHAIN/last_stage1")
step sample-reports scripts/sample_reports.py --pred-run "$S1" --n 20
step concepts scripts/select_concepts.py
step stage2 scripts/probe.py --name stage2 --features emb,chexmask --baseline "$S1"
S2=$(cat "$CHAIN/last_stage2")
for k in 5 10 25; do
  step "stage5-nb$k" scripts/probe.py --name "stage5-emb-nb$k" --features "emb,nb$k" --baseline "$S1"
done
echo DONE > "$CHAIN/STATUS"
echo "$(date '+%F %T') CHAIN FINISHED" >> "$CHAIN/chain.log"
