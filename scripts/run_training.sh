#!/bin/bash
#
# Train ORACLE for all three agent populations (100, 150, 200) using the
# TAGS (Temperature-Annealed Gumbel-Softmax) exploration strategy.
#
# Checkpoints are written by ars.train_oracle to:
#   savedagents/oracle/grid<GRID>/<A>agents/final_weights_attention_<A>_tags.pkl
#
# Config is overridable via environment variables, e.g.:
#   AGENT_COUNTS="100 200" DAYS=20 bash scripts/run_training.sh
#   DEVICE=gpu GPU_ID=0 bash scripts/run_training.sh
set -uo pipefail

# ---- configuration (overridable via env) ----------------------------------
AGENT_COUNTS="${AGENT_COUNTS:-100 150 200}"
GRID="${GRID:-15}"
# DATASET_PREFIX selects a city's datasets and isolates its outputs. Empty =>
# the canonical NYC datasets (<A>_agents) and savedagents/ (unchanged). Set
# DATASET_PREFIX=beijing_ to train on dataset/beijing_<A>_agents and write
# checkpoints to a separate tree so NYC and Beijing never overwrite each other.
DATASET_PREFIX="${DATASET_PREFIX:-}"
DAYS="${DAYS:-30}"
NUM_EPISODES="${NUM_EPISODES:-200}"
NUM_ENVS="${NUM_ENVS:-10}"
EXPLORATION="${EXPLORATION:-tags}"   # ORACLE uses TAGS only
# Reward trade-off weight (rider benefit vs driver detour). NYC default 0.4;
# Beijing's denser short-trip core needs ~0.6 for positive reward (a PSO sweep
# on Beijing showed mean reward goes -5.4 @0.4 -> +37 @0.6 while every day
# already saves ~6% distance). Set ALPHA_R=0.6 when training on beijing_*.
ALPHA_R="${ALPHA_R:-0.4}"
DEVICE="${DEVICE:-cpu}"
GPU_ID="${GPU_ID:-0}"
WANDB_MODE="${WANDB_MODE:-online}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- timing / summary ------------------------------------------------------
LOG_ROOT="${REPO_ROOT}/ablation_logs"
mkdir -p "${LOG_ROOT}"
SUMMARY="${LOG_ROOT}/train_summary_$(date +%Y%m%d-%H%M%S).txt"
echo "model,agents,grid,status,seconds,hms" > "${SUMMARY}"
hms() { printf '%02d:%02d:%02d' $(( $1/3600 )) $(( ($1%3600)/60 )) $(( $1%60 )); }

# Checkpoint root: isolate per city so e.g. beijing never overwrites NYC.
#   no prefix     -> savedagents            (canonical NYC)
#   beijing_      -> savedagents_beijing
SAVE_ROOT="savedagents"
[ -n "${DATASET_PREFIX}" ] && SAVE_ROOT="savedagents_${DATASET_PREFIX%_}"
echo "Dataset prefix: '${DATASET_PREFIX}' | checkpoint root: ${SAVE_ROOT}"

for AGENTS in ${AGENT_COUNTS}; do
  DATASET="${DATASET_PREFIX}${AGENTS}_agents"
  if [ ! -d "dataset/${DATASET}" ]; then
    echo "!! dataset/${DATASET} missing; skipping ${AGENTS} agents."
    echo "oracle,${AGENTS},${GRID},NO_DATASET,0,00:00:00" >> "${SUMMARY}"
    continue
  fi

  echo "=============================================="
  echo "=== Training ORACLE (TAGS) | ${AGENTS} agents | grid ${GRID} ==="
  echo "=============================================="

  start=$(date +%s)
  python -m ars.train_oracle \
    --model attention \
    --dataset "${DATASET}" \
    --grid-size "${GRID}" \
    --num-agents "${AGENTS}" \
    --days "${DAYS}" \
    --num-episodes "${NUM_EPISODES}" \
    --num-envs "${NUM_ENVS}" \
    --exploration "${EXPLORATION}" \
    --alpha-r "${ALPHA_R}" \
    --save-dir "${SAVE_ROOT}" \
    --device "${DEVICE}" --gpu-id "${GPU_ID}" \
    --wandb-mode "${WANDB_MODE}"
  rc=$?
  elapsed=$(( $(date +%s) - start ))

  if [ "${rc}" -eq 0 ]; then
    echo "Completed ORACLE training for ${AGENTS} agents in $(hms ${elapsed}) (${elapsed}s)."
    echo "oracle,${AGENTS},${GRID},ok,${elapsed},$(hms ${elapsed})" >> "${SUMMARY}"
  else
    echo "FAILED ORACLE training for ${AGENTS} agents after $(hms ${elapsed}) (${elapsed}s)."
    echo "oracle,${AGENTS},${GRID},FAILED,${elapsed},$(hms ${elapsed})" >> "${SUMMARY}"
  fi
  echo
done

echo "All ORACLE training runs complete."
echo "Checkpoints: ${SAVE_ROOT}/oracle/grid${GRID}/<A>agents/final_weights_attention_<A>_${EXPLORATION}.pkl"
echo "------------------------------------------------------------"
echo "Timing summary: ${SUMMARY}"
column -t -s, "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
TOTAL_S=$(awk -F, 'NR>1 {s+=$5} END {print s+0}' "${SUMMARY}")
echo "Total wall-clock: $(hms ${TOTAL_S}) (${TOTAL_S}s)"
