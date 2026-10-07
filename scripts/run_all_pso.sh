#!/bin/bash
# Run PSO across all combos x 4 conditions x both cities. Output follows the
# same structure as the MARL evals: results/<city>/grid<G>/pso/<A>agents_<cond>/
#
# Bounded parallelism (POOL concurrent) to avoid the OOM we hit with heavy
# concurrent runs. PSO is one-shot (no training) so this is CPU-bound but light.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${ENV_NAME:-ars}"
# PSO is CPU-only but imports TensorFlow (for tensorboard); many concurrent TF
# processes exhaust GPU memory (CUDA_ERROR_OUT_OF_MEMORY) and crash at startup.
# Hide the GPU so TF stays on CPU -- PSO needs no GPU compute.
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=3

GRIDS="${GRIDS:-15 30 45}"
AGENTS="${AGENTS:-100 150 200}"
CONDS="${CONDS:-uniform_fixed uniform_bd gaussian_fixed gaussian_bd}"
SIM_DAYS="${SIM_DAYS:-100}"
PARTS="${PARTS:-50}"; ITERS="${ITERS:-80}"
POOL="${POOL:-5}"                    # max concurrent PSO runs
LOG_ROOT="pso_traffic_logs/$(date +%Y%m%d-%H%M%S)"; mkdir -p "$LOG_ROOT"
SUMMARY="$LOG_ROOT/summary.csv"; echo "city,grid,agents,cond,status,dist,traffic" > "$SUMMARY"

pflags(){ case "$1" in
  uniform_fixed)  echo "--altruism-dist uniform  --agent-dynamics fixed" ;;
  uniform_bd)     echo "--altruism-dist uniform  --agent-dynamics birth_death" ;;
  gaussian_fixed) echo "--altruism-dist gaussian --agent-dynamics fixed" ;;
  gaussian_bd)    echo "--altruism-dist gaussian --agent-dynamics birth_death" ;;
esac; }

run_one(){
  local city=$1 g=$2 a=$3 cond=$4
  local ds; if [ "$city" = beijing ]; then ds="beijing_${a}_agents_grid${g}"; else ds="${a}_agents_grid${g}"; fi
  [ -d "dataset/$ds" ] || { echo "skip $city $a/g$g (no dataset)"; return; }
  local sdir="results/${city}/grid${g}/pso/${a}agents_${cond}"
  # skip if a valid traffic-bearing summary already exists
  if grep -q 'gini_combined_benefits' "$sdir/run/results_summary.txt" 2>/dev/null; then
    echo "skip $city $a/g$g $cond (already has traffic)"; return
  fi
  mkdir -p "$sdir"
  local plog="$LOG_ROOT/pso_${city}_${a}_g${g}_${cond}.log"
  # shellcheck disable=SC2046
  nice -n 15 python -m ars.baselines.run_pso \
    --num-agents "$a" --initial-active-agents "$a" --days "$SIM_DAYS" $(pflags "$cond") \
    --dataset "$ds" --grid-size "$g" --experiment-name run --save-dir "$sdir" \
    --pso-particles "$PARTS" --pso-iterations "$ITERS" --alpha 0.4 > "$plog" 2>&1
  local rc=$? f="$sdir/run/results_summary.txt"
  local dr tr
  dr=$(grep -oE 'distance_reduction_percent: [-0-9.]+' "$f" 2>/dev/null | grep -oE '[-0-9.]+$')
  tr=$(grep -oE 'traffic_reduction_percent: [-0-9.]+' "$f" 2>/dev/null | grep -oE '[-0-9.]+$')
  echo "$city,$g,$a,$cond,$([ $rc -eq 0 ] && echo ok || echo FAIL),${dr:-},${tr:-}" >> "$SUMMARY"
  echo "  PSO $city $a/g$g $cond: dist=${dr:-?} traffic=${tr:-?}"
}

# job pool
for city in nyc beijing; do for g in $GRIDS; do for a in $AGENTS; do for cond in $CONDS; do
  run_one "$city" "$g" "$a" "$cond" &
  while [ "$(jobs -rp | wc -l)" -ge "$POOL" ]; do wait -n; done
done; done; done; done
wait
echo "=== PSO traffic rerun DONE: $SUMMARY ==="
column -t -s, "$SUMMARY" 2>/dev/null | tail -20
