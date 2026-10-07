#!/bin/bash
#
# Train the DQN baseline for all agent populations (100, 150, 200) across all
# grid sizes (15, 30, 45).
#
# Mirrors scripts/run_training.sh (ORACLE), with two differences dictated by
# train_dqn.py's interface:
#   - No --exploration / --model / --num-envs / --device / --gpu-id / --alpha-r
#     (train_dqn.py doesn't expose these; DQN trains single-env, CPU, PyTorch,
#     with default env reward/altruism params, matching train_oracle.py's own
#     behaviour of not varying altruism-distribution or birth-death at train time).
#   - Loops over GRID as well as AGENTS, since (unlike run_training.sh) we need
#     all three grids, not just one per invocation.
#
# Checkpoints are written by ars.train_dqn to:
#   <save_root>/dqn/grid<GRID>/<A>agents/dqn_final.pkl
#
# Config is overridable via environment variables, e.g.:
#   AGENT_COUNTS="100 200" GRIDS="15 30" DAYS=20 bash scripts/run_training_dqn.sh
set -uo pipefail

# ---- configuration (overridable via env) ----------------------------------
AGENT_COUNTS="${AGENT_COUNTS:-100 150 200}"
GRIDS="${GRIDS:-15 30 45}"
# DATASET_PREFIX mirrors run_training.sh: empty => canonical NYC datasets and
# savedagents/. Set e.g. DATASET_PREFIX=beijing_ to isolate a different city.
DATASET_PREFIX="${DATASET_PREFIX:-}"
DAYS="${DAYS:-30}"
NUM_EPISODES="${NUM_EPISODES:-200}"
WANDB_MODE="${WANDB_MODE:-online}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- timing / summary ------------------------------------------------------
LOG_ROOT="${REPO_ROOT}/ablation_logs"
mkdir -p "${LOG_ROOT}"
SUMMARY="${LOG_ROOT}/train_dqn_summary_$(date +%Y%m%d-%H%M%S).txt"
echo "model,agents,grid,status,seconds,hms" > "${SUMMARY}"
hms() { printf '%02d:%02d:%02d' $(( $1/3600 )) $(( ($1%3600)/60 )) $(( $1%60 )); }

# Checkpoint root: isolate per city so e.g. beijing never overwrites NYC.
SAVE_ROOT="savedagents"
[ -n "${DATASET_PREFIX}" ] && SAVE_ROOT="savedagents_${DATASET_PREFIX%_}"
echo "Dataset prefix: '${DATASET_PREFIX}' | checkpoint root: ${SAVE_ROOT}"

# Dataset folder naming follows generate_grid_dataset.py:
#   grid 15 (canonical)      -> <A>_agents
#   grid 30 / 45 (ablation)  -> <A>_agents_grid<G>
dataset_name() {
  local agents=$1
  local grid=$2
  if [ "${grid}" -eq 15 ]; then
    echo "${DATASET_PREFIX}${agents}_agents"
  else
    echo "${DATASET_PREFIX}${agents}_agents_grid${grid}"
  fi
}

for GRID in ${GRIDS}; do
  for AGENTS in ${AGENT_COUNTS}; do
    DATASET="$(dataset_name "${AGENTS}" "${GRID}")"
    if [ ! -d "dataset/${DATASET}" ]; then
      echo "!! dataset/${DATASET} missing; skipping ${AGENTS} agents @ grid${GRID}."
      echo "dqn,${AGENTS},${GRID},NO_DATASET,0,00:00:00" >> "${SUMMARY}"
      continue
    fi

    echo "=============================================="
    echo "=== Training DQN | ${AGENTS} agents | grid ${GRID} ==="
    echo "=============================================="

    start=$(date +%s)
    python ars/train_dqn.py \
      --dataset "${DATASET}" \
      --grid-size "${GRID}" \
      --num-agents "${AGENTS}" \
      --days "${DAYS}" \
      --num-episodes "${NUM_EPISODES}" \
      --save-dir "${SAVE_ROOT}" \
      --wandb-mode "${WANDB_MODE}"
    rc=$?
    elapsed=$(( $(date +%s) - start ))

    if [ "${rc}" -eq 0 ]; then
      echo "Completed DQN training for ${AGENTS} agents @ grid${GRID} in $(hms ${elapsed}) (${elapsed}s)."
      echo "dqn,${AGENTS},${GRID},ok,${elapsed},$(hms ${elapsed})" >> "${SUMMARY}"
    else
      echo "FAILED DQN training for ${AGENTS} agents @ grid${GRID} after $(hms ${elapsed}) (${elapsed}s)."
      echo "dqn,${AGENTS},${GRID},FAILED,${elapsed},$(hms ${elapsed})" >> "${SUMMARY}"
    fi
    echo
  done
done

echo "All DQN training runs complete."
echo "Checkpoints: ${SAVE_ROOT}/dqn/grid<G>/<A>agents/dqn_final.pkl"
echo "------------------------------------------------------------"
echo "Timing summary: ${SUMMARY}"
column -t -s, "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
TOTAL_S=$(awk -F, 'NR>1 {s+=$5} END {print s+0}' "${SUMMARY}")
echo "Total wall-clock: $(hms ${TOTAL_S}) (${TOTAL_S}s)"
