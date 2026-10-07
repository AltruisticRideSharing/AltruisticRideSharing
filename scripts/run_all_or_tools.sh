#!/bin/bash
# Run OR-Tools across all combos x 4 conditions x both cities. Output to
# results/<city>/grid<G>/or_tools/<A>agents_<cond>/.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${ENV_NAME:-ars}"
GRIDS="${GRIDS:-15 30 45}"; AGENTS="${AGENTS:-100 150 200}"
CONDS="${CONDS:-uniform_fixed uniform_bd gaussian_fixed gaussian_bd}"
SIM_DAYS="${SIM_DAYS:-100}"; POOL="${POOL:-3}"
LOG_ROOT="ortools_traffic_logs/$(date +%Y%m%d-%H%M%S)"; mkdir -p "$LOG_ROOT"
SUMMARY="$LOG_ROOT/summary.csv"; echo "city,grid,agents,cond,status,dist,traffic" > "$SUMMARY"
dflags(){ case "$1" in
  uniform_fixed)  echo "--altruism-dist uniform  --agent-dynamics fixed" ;;
  uniform_bd)     echo "--altruism-dist uniform  --agent-dynamics birth_death" ;;
  gaussian_fixed) echo "--altruism-dist gaussian --agent-dynamics fixed" ;;
  gaussian_bd)    echo "--altruism-dist gaussian --agent-dynamics birth_death" ;;
esac; }
run_one(){
  local city=$1 g=$2 a=$3 cond=$4 ds
  if [ "$city" = beijing ]; then ds="beijing_${a}_agents_grid${g}"; else ds="${a}_agents_grid${g}"; fi
  [ -d "dataset/$ds" ] || return
  local sdir="results/${city}/grid${g}/or_tools/${a}agents_${cond}"
  grep -q 'gini_combined_benefits' "$sdir/run/results_summary.txt" 2>/dev/null && { echo "skip $city $a/g$g $cond"; return; }
  mkdir -p "$sdir"
  nice -n 15 python -m ars.baselines.run_or_tools --num-agents "$a" --initial-active-agents "$a" \
    --days "$SIM_DAYS" $(dflags "$cond") --dataset "$ds" --grid-size "$g" \
    --experiment-name run --save-dir "$sdir" > "$LOG_ROOT/or_${city}_${a}_g${g}_${cond}.log" 2>&1
  local rc=$? f="$sdir/run/results_summary.txt"
  local dr tr
  dr=$(grep -oE 'distance_reduction_percent: [-0-9.]+' "$f" 2>/dev/null|grep -oE '[-0-9.]+$')
  tr=$(grep -oE 'traffic_reduction_percent: [-0-9.]+' "$f" 2>/dev/null|grep -oE '[-0-9.]+$')
  echo "$city,$g,$a,$cond,$([ $rc -eq 0 ]&&echo ok||echo FAIL),${dr:-},${tr:-}" >> "$SUMMARY"
  echo "  OR $city $a/g$g $cond: dist=${dr:-?} traffic=${tr:-?}"
}
for city in nyc beijing; do for g in $GRIDS; do for a in $AGENTS; do for cond in $CONDS; do
  run_one "$city" "$g" "$a" "$cond" &
  while [ "$(jobs -rp|wc -l)" -ge "$POOL" ]; do wait -n; done
done; done; done; done
wait
echo "=== OR-Tools traffic rerun DONE: $SUMMARY ==="
