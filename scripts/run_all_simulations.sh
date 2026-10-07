#!/bin/bash
#
# Run evaluation simulations across all three agent populations (100, 150, 200),
# both altruism distributions (uniform, gaussian), and both population modes
# (fixed, birth-death) for one or more models.
#
# Choose the model(s) via the MODELS env var (space-separated). Any of:
#   oracle  -> ars.evaluate_oracle  (parameter-sharing ORACLE)
#   maac    -> ars.evaluate         (attention, unshared)
#   maddpg  -> ars.evaluate         (maddpg, unshared)
# Default runs all three.
#
# Each model restores its trained checkpoint from:
#   savedagents/<model>/grid<GRID>/<A>agents/<final_weights_*.pkl>
# Train first with scripts/run_training.sh (ORACLE) or scripts/run_grid_ablation.sh.
#
# Examples:
#   bash scripts/run_all_simulations.sh                 # all models, all sizes
#   MODELS="oracle" bash scripts/run_all_simulations.sh # ORACLE only
#   MODELS="oracle maddpg" AGENT_COUNTS="100 200" bash scripts/run_all_simulations.sh
#   DEVICE=gpu GPU_ID=0 bash scripts/run_all_simulations.sh
set -uo pipefail

# ---- configuration (overridable via env) ----------------------------------
MODELS="${MODELS:-oracle maac maddpg}"
AGENT_COUNTS="${AGENT_COUNTS:-100 150 200}"
# DATASET_PREFIX selects a city's datasets and isolates its checkpoints/results.
# Empty => canonical NYC (<A>_agents, savedagents/, results/). beijing_ =>
# dataset/beijing_<A>_agents, savedagents_beijing/, results_beijing/grid.../...
DATASET_PREFIX="${DATASET_PREFIX:-}"
DISTRIBUTIONS="${DISTRIBUTIONS:-uniform gaussian}"
MODES="${MODES:-fixed birthdeath}"
GRID="${GRID:-15}"
DAYS="${DAYS:-100}"
# Reward weight; MUST match what the policy trained with.
ALPHA_R="${ALPHA_R:-0.4}"
# Attention heads; MUST match the checkpoint's architecture or the restore loads
# a differently-shaped network. NYC checkpoints: 4. Beijing (HPO-best): 1.
NUM_HEADS="${NUM_HEADS:-4}"
BD_INIT_AGENTS="${BD_INIT_AGENTS:-40}"   # initial active agents in birth-death mode
DEVICE="${DEVICE:-cpu}"
GPU_ID="${GPU_ID:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- timing / summary ------------------------------------------------------
LOG_ROOT="${REPO_ROOT}/ablation_logs"
mkdir -p "${LOG_ROOT}"
SUMMARY="${LOG_ROOT}/sim_summary_$(date +%Y%m%d-%H%M%S).txt"
echo "model,agents,dist,mode,status,seconds,hms" > "${SUMMARY}"
hms() { printf '%02d:%02d:%02d' $(( $1/3600 )) $(( ($1%3600)/60 )) $(( $1%60 )); }

# map a model label to its evaluation entry point
eval_module_for() {
  case "$1" in
    oracle)        echo "ars.evaluate_oracle" ;;
    maac|maddpg)   echo "ars.evaluate" ;;
    *)             echo "" ;;
  esac
}
model_type_for() {
  case "$1" in
    oracle|maac)   echo "attention" ;;
    maddpg)        echo "maddpg" ;;
    *)             echo "" ;;
  esac
}
# Checkpoint filename, matching each trainer's save convention:
#   ars.train_oracle -> final_weights_attention_<A>_tags.pkl
#   ars.train (maac) -> final_weights_attention.pkl
#   ars.train (mddpg)-> final_weights_maddpg.pkl
ckpt_file_for() {
  local model="$1" agents="$2"
  case "${model}" in
    oracle) echo "final_weights_attention_${agents}_tags.pkl" ;;
    maac)   echo "final_weights_attention.pkl" ;;
    maddpg) echo "final_weights_maddpg.pkl" ;;
  esac
}

run_eval() {  # MODEL AGENTS DIST MODE
  local MODEL="$1" AGENTS="$2" DIST="$3" MODE="$4"
  local EVAL_MOD MTYPE CKPT_FILE CKPT_DIR DATASET SAVE_ROOT
  EVAL_MOD="$(eval_module_for "${MODEL}")"
  MTYPE="$(model_type_for "${MODEL}")"
  CKPT_FILE="$(ckpt_file_for "${MODEL}" "${AGENTS}")"
  # Mirror run_training.sh: empty prefix => savedagents; else savedagents_<city>.
  SAVE_ROOT="savedagents"
  [ -n "${DATASET_PREFIX}" ] && SAVE_ROOT="savedagents_${DATASET_PREFIX%_}"
  CKPT_DIR="${SAVE_ROOT}/${MODEL}/grid${GRID}/${AGENTS}agents"
  DATASET="${DATASET_PREFIX}${AGENTS}_agents"

  # population-mode parameters
  local ENABLE_BD INIT_AGENTS
  if [ "${MODE}" == "fixed" ]; then
    ENABLE_BD=""; INIT_AGENTS="${AGENTS}"
  else
    ENABLE_BD="--enable-birth-death"; INIT_AGENTS="${BD_INIT_AGENTS}"
  fi

  # distribution parameters
  local DIST_ARGS
  if [ "${DIST}" == "uniform" ]; then
    DIST_ARGS="--altruism-distribution uniform --altruism-mean 0.5"
  else
    DIST_ARGS="--altruism-distribution gaussian --altruism-mean 0.5 --altruism-std 0.2"
  fi

  if [ ! -d "dataset/${DATASET}" ]; then
    echo "!! dataset/${DATASET} missing; skipping ${MODEL} ${AGENTS} agents."
    echo "${MODEL},${AGENTS},${DIST},${MODE},NO_DATASET,0,00:00:00" >> "${SUMMARY}"
    return
  fi
  if [ ! -f "${CKPT_DIR}/${CKPT_FILE}" ]; then
    echo "!! checkpoint ${CKPT_DIR}/${CKPT_FILE} missing; skipping ${MODEL} ${AGENTS}/${DIST}/${MODE}."
    echo "   Train it first (run_training.sh for oracle, run_grid_ablation.sh for maac/maddpg)."
    echo "${MODEL},${AGENTS},${DIST},${MODE},NO_CKPT,0,00:00:00" >> "${SUMMARY}"
    return
  fi

  echo "=============================================="
  echo "Running: model=${MODEL} agents=${AGENTS} dist=${DIST} mode=${MODE}"
  echo "=============================================="

  # Results dir: only override when a city prefix is set, so NYC keeps the
  # canonical results/grid.../... layout untouched. Mirrors the structure of
  # ars.metrics.get_results_dir but rooted at results_<city> to avoid clobber.
  local RESULTS_ARG=""
  if [ -n "${DATASET_PREFIX}" ]; then
    local MODE_LBL CFG
    MODE_LBL=$([ "${MODE}" == "fixed" ] && echo fixed || echo bd)
    CFG="${AGENTS}agents_${DIST}_${MODE_LBL}"
    RESULTS_ARG="--results-dir results_${DATASET_PREFIX%_}/grid${GRID}/${MODEL}/${CFG}"
  fi

  # --num-heads exists only on ars.evaluate_oracle (the ORACLE evaluator); the
  # maac/maddpg path (ars.evaluate) does not take it.
  local HEADS_ARG=""
  [ "${MODEL}" = "oracle" ] && HEADS_ARG="--num-heads ${NUM_HEADS}"

  local start elapsed rc
  start=$(date +%s)
  # shellcheck disable=SC2086
  python -m "${EVAL_MOD}" \
    --model-type "${MTYPE}" \
    --dataset "${DATASET}" --grid-size "${GRID}" \
    --num-agents "${AGENTS}" --initial-active-agents "${INIT_AGENTS}" \
    --days "${DAYS}" \
    ${ENABLE_BD} ${DIST_ARGS} ${RESULTS_ARG} ${HEADS_ARG} \
    --alpha-r "${ALPHA_R}" \
    --save-dir "${CKPT_DIR}" --restore --weights-file "${CKPT_FILE}" \
    --device "${DEVICE}" --gpu-id "${GPU_ID}"
  rc=$?
  elapsed=$(( $(date +%s) - start ))

  if [ "${rc}" -eq 0 ]; then
    echo "Completed: model=${MODEL} agents=${AGENTS} dist=${DIST} mode=${MODE} in $(hms ${elapsed}) (${elapsed}s)"
    echo "${MODEL},${AGENTS},${DIST},${MODE},ok,${elapsed},$(hms ${elapsed})" >> "${SUMMARY}"
  else
    echo "FAILED: model=${MODEL} agents=${AGENTS} dist=${DIST} mode=${MODE} after $(hms ${elapsed}) (${elapsed}s)"
    echo "${MODEL},${AGENTS},${DIST},${MODE},FAILED,${elapsed},$(hms ${elapsed})" >> "${SUMMARY}"
  fi
  echo
}

echo "Starting simulation experiments | models: ${MODELS} | agents: ${AGENT_COUNTS}"
for MODEL in ${MODELS}; do
  if [ -z "$(eval_module_for "${MODEL}")" ]; then
    echo "!! unknown model '${MODEL}'; skipping"; continue
  fi
  for AGENTS in ${AGENT_COUNTS}; do
    for DIST in ${DISTRIBUTIONS}; do
      for MODE in ${MODES}; do
        run_eval "${MODEL}" "${AGENTS}" "${DIST}" "${MODE}"
      done
    done
  done
done

echo "All simulation experiments completed!"
echo "------------------------------------------------------------"
echo "Timing summary: ${SUMMARY}"
column -t -s, "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
TOTAL_S=$(awk -F, 'NR>1 {s+=$6} END {print s+0}' "${SUMMARY}")
echo "Total wall-clock: $(hms ${TOTAL_S}) (${TOTAL_S}s)"
