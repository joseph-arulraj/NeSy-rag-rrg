#!/bin/bash
# Overnight letterbox plan (user, 2026-10-06). Runs on the LOGIN node, detached; each step is a foreground
# SLURM step launched from a script file in its run directory (no nested bash -c). Every step can resume.
# Stops if the pilot gate fails or free space would fall below the floor. Status in $ORCH/orchestrator.log.
#   CPU=37811782 (slam_cpu)  GPU=37814480 (slam_gpu H100, expires 2026-10-07 11:31)
set -uo pipefail
cd "$(dirname "$0")/.."
V2=$(pwd)
PY=/scratch/users/k23031260/.conda/envs/rrg/bin/python
CPU=37811782; GPU=37814480
ORCH=$1; mkdir -p "$ORCH"; LOG="$ORCH/orchestrator.log"
FREE_FLOOR_GB=15
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
free_gb() { getfattr --only-values -n ceph.dir.rbytes /scratch/prj/bhi_zihe_imaging 2>/dev/null | awk '{printf "%.1f", (1e12-$1)/1e9}'; }
SKIPPED=()

# step NAME JOB [ENV=val ...] -- script.py args...   -> runs foreground; echoes run dir; returns script status
step() {
  local name=$1 job=$2; shift 2
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local rd="runs/$(date +%Y%m%d-%H%M%S)_$name"; mkdir -p "$rd"
  {
    echo '#!/bin/bash'; echo "cd $V2"; echo 'export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}'
    for e in "${envs[@]}"; do echo "export $e"; done
    printf 'exec %q -u' "$PY"; for a in "$@"; do printf ' %q' "$a"; done
    printf ' --run-dir %q > %q 2>&1\n' "$rd" "$rd/stdout.txt"
  } > "$rd/run.sh"; chmod +x "$rd/run.sh"
  log "START $name on $job -> $rd"
  srun --jobid="$job" --overlap "$rd/run.sh" > "$rd/srun.out" 2>&1
  local rc=$?
  local st; st=$(cat "$rd/STATUS" 2>/dev/null || echo MISSING)
  log "END   $name rc=$rc STATUS=$st free=$(free_gb) GB"
  echo "$rd" > "$ORCH/last_$name"
  [ "$st" = DONE ] && [ $rc -eq 0 ]
}
lastdir() { cat "$ORCH/last_$1"; }

log "orchestrator started; free $(free_gb) GB"

# ---------------------------------------------------------------- 2. letterbox pilot + gate
step weights-pilot-lb $CPU -- scripts/chexmask_region_weights.py --dataset mimic --ids data/features/regions/mimic_pilot_ids.parquet \
     --out-name mimic_pilot_lb --geometry letterbox --workers 16 --compressed || { log "pilot weights FAILED; stopping"; exit 1; }
step region-pilot-lb $GPU -- scripts/region_features.py --ids data/features/regions/mimic_pilot_ids.parquet --weights-name mimic_pilot_lb \
     --out-name mimic_pilot_lb --geometry letterbox --overlays 10 --shard-size 512 --batch 128 --workers 15 || { log "pilot pass FAILED; stopping"; exit 1; }
if ! step pilot-gate $CPU -- scripts/pilot_gate.py --pilot "$(lastdir region-pilot-lb)" --stretch-pilot runs/20261006-1147_region-pilot; then
  log "PILOT GATE FAILED -> stopping as instructed (see $(lastdir pilot-gate)/log.txt)"; exit 1
fi
log "pilot gate passed"

# ---------------------------------------------------------------- 3. full letterbox weights (2 halves in parallel) + GPU pass
[ -f data/features/regions/mimic_full_lb_p0_ids.parquet ] || $PY - <<'EOF'
import pandas as pd, hashlib
ids = pd.read_parquet("data/features/regions/mimic_full_ids.parquet")
h = ids.image_id.map(lambda d: int(hashlib.sha256(d.encode()).hexdigest()[:8], 16) % 2)
ids[h == 0].to_parquet("data/features/regions/mimic_full_lb_p0_ids.parquet", index=False)
ids[h == 1].to_parquet("data/features/regions/mimic_full_lb_p1_ids.parquet", index=False)
print(len(ids[h == 0]), len(ids[h == 1]))
EOF
( step weights-full-lb-p0 $CPU -- scripts/chexmask_region_weights.py --dataset mimic --ids data/features/regions/mimic_full_lb_p0_ids.parquet \
      --out-name mimic_full_lb_p0 --geometry letterbox --workers 16 --compressed ) &
P0=$!
( step weights-full-lb-p1 $GPU -- scripts/chexmask_region_weights.py --dataset mimic --ids data/features/regions/mimic_full_lb_p1_ids.parquet \
      --out-name mimic_full_lb_p1 --geometry letterbox --workers 15 --compressed ) &
P1=$!
wait $P0; R0=$?; wait $P1; R1=$?
if [ $R0 -ne 0 ] || [ $R1 -ne 0 ]; then log "full weights FAILED (p0 rc=$R0, p1 rc=$R1); stopping"; exit 1; fi
F=$(free_gb); log "free before full pass: $F GB (pass needs ~15.4 GB)"
if awk -v f="$F" 'BEGIN{exit !(f - 15.4 < 13.5)}'; then log "not enough space for the full pass; stopping"; exit 1; fi
until step region-full-lb $GPU -- scripts/region_features.py --ids data/features/regions/mimic_full_ids.parquet \
      --weights-name mimic_full_lb_p0,mimic_full_lb_p1 --out-name mimic_full_lb --geometry letterbox \
      --exclude-ids runs/20261006-1304_region-full/flip_check_failures.csv --shard-size 8192 --batch 128 --workers 15; do
  n=$(( ${n:-0} + 1 )); log "full pass attempt $n failed; resuming from finished shards"; [ $n -ge 3 ] && { log "giving up on full pass"; exit 1; }
done
step build-lb-store $CPU -- scripts/build_lb_store.py --shards mimic_full_lb || { log "store build FAILED; stopping"; exit 1; }

# ---------------------------------------------------------------- 4. retrain on letterbox (CPU) || Stage 4 + MS-CXR (GPU)
cpu_chain() {
  local E=NESY_EMB=letterbox
  step lb-retrieval $CPU $E -- scripts/retrieval_features.py --threads 16 || return 1
  step lb-stage1 $CPU $E -- scripts/probe.py --name lb-stage1 --features emb || return 1
  local S1; S1=$(lastdir lb-stage1)
  step lb-stage2 $CPU $E -- scripts/probe.py --name lb-stage2 --features emb,chexmask --baseline "$S1" || return 1
  local S2; S2=$(lastdir lb-stage2)
  step lb-stage3 $CPU $E -- scripts/stage3_factorised.py --name lb-k6-pruned --concepts data/concepts/concepts_final_k6_pruned.csv --baseline "$S2" || return 1
  step lb-stage5-emb-nb10 $CPU $E -- scripts/probe.py --name lb-stage5-emb-nb10 --features emb,nb10 --baseline "$S1" || return 1
  step lb-stage5-s3-nb10 $CPU $E -- scripts/stage3_factorised.py --name lb-k6-pruned-nb10 --concepts data/concepts/concepts_final_k6_pruned.csv \
       --extra chexmask,nb10 --baseline "$(lastdir lb-stage3)" || return 1
  step lb-normal-call $CPU $E -- scripts/abnormal_absent_check.py --pred-run "$(lastdir lb-stage3)" --label lb-stage3 --rules "root 5%;root 2%" || return 1
  step lb-band-report $CPU $E -- scripts/band_report.py --pred-run "$(lastdir lb-stage3)" || return 1
  step lb-bootstrap $CPU $E -- scripts/bootstrap_compare.py --runs "S1=$S1" "S2=$S2" "S3=$(lastdir lb-stage3)" \
       "S5emb=$(lastdir lb-stage5-emb-nb10)" "S3nb=$(lastdir lb-stage5-s3-nb10)" --n-boot 2000 || return 1
}
gpu_chain() {
  step lb-stage4 $GPU NESY_EMB=letterbox -- scripts/stage4_regions.py --shards mimic_full_lb --crossfit 2 --skip-mscxr || return 1
  step lb-mscxr-loc $CPU -- scripts/mscxr_loc_eval.py --stage4-run "$(lastdir lb-stage4)" --geometry letterbox \
       --weights mimic_full_lb_p0,mimic_full_lb_p1 --label letterbox || return 1
}
cpu_chain & PC=$!
gpu_chain & PG=$!
wait $PC; RC=$?; wait $PG; RG=$?
[ $RC -ne 0 ] && { log "CPU retrain chain FAILED (continuing with what exists)"; SKIPPED+=("part of the CPU retrain chain"); }
[ $RG -ne 0 ] && { log "Stage 4 / MS-CXR chain FAILED"; SKIPPED+=("Stage 4 or MS-CXR (letterbox)"); }
if [ $RC -eq 0 ] && [ $RG -eq 0 ]; then
  S3=$(lastdir lb-stage3)
  step lb-stage3-s4 $CPU NESY_EMB=letterbox "NESY_S4_RUN=$(lastdir lb-stage4)" -- scripts/stage3_factorised.py --name lb-k6-pruned-s4 \
       --concepts data/concepts/concepts_final_k6_pruned.csv --extra chexmask,s4 --baseline "$S3" || SKIPPED+=("Stage 3 + Stage 4 ablation")
  step lb-bootstrap-s4 $CPU NESY_EMB=letterbox -- scripts/bootstrap_compare.py --runs "S2=$(lastdir lb-stage2)" "S3=$S3" \
       "S3s4=$(lastdir lb-stage3-s4)" --n-boot 2000 || SKIPPED+=("ablation bootstrap")
fi

# ---------------------------------------------------------------- 5. stretch vs letterbox comparison
step compare-variants $CPU -- scripts/compare_variants.py --orch "$ORCH" || SKIPPED+=("stretch vs letterbox comparison table")

# ---------------------------------------------------------------- 7. external caching (letterbox, percentile windowing), if time
HOURS_LEFT=$(squeue -h -j $GPU -o %L | awk -F'[-:]' '{ if (NF==4) print $1*24+$2; else if (NF==3) print $1; else print 0 }')
log "GPU allocation hours left: $HOURS_LEFT"
if [ "${HOURS_LEFT:-0}" -ge 2 ] && [ -f scripts/external_features.py ]; then
  step ext-weights-vindr $CPU -- scripts/chexmask_region_weights.py --dataset vindr --ids data/features/regions/ext_vindr_test_ids.parquet \
       --out-name ext_vindr_test_lb --geometry letterbox --workers 16 --compressed || SKIPPED+=("VinDr CheXmask weights")
  step ext-weights-padchest $CPU -- scripts/chexmask_region_weights.py --dataset padchest --ids data/features/regions/ext_padchest_gr_ids.parquet \
       --out-name ext_padchest_gr_lb --geometry letterbox --workers 16 --compressed || SKIPPED+=("PadChest CheXmask weights")
  step ext-features $GPU -- scripts/external_features.py || SKIPPED+=("external feature cache")
else
  SKIPPED+=("external feature cache (not enough GPU time or script missing)")
fi
log "ORCHESTRATOR FINISHED. Skipped: ${SKIPPED[*]:-none}. Free space $(free_gb) GB"
echo DONE > "$ORCH/STATUS"
