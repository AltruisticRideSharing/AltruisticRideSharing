#!/bin/bash
#
# Evaluate the DQN baseline on the NYC datasets across every combination of:
#   grid size      : 15, 30, 45
#   agent count    : 100, 150, 200
#   altruism dist. : uniform, gaussian
#   population mode: fixed, birth-death
# => 36 runs.
#
# Checkpoints are read from (one per grid/agent-count, reused across all four
# distribution/mode combinations, mirroring how ORACLE is evaluated):
#   savedagents/dqn/grid<G>/<A>agents/dqn_final.pkl
#
# Results are written to (the dqn/ level is created if absent):
#   results/nyc/grid<G>/dqn/<A>agents_<dist>_<mode>/
#
# This layout is passed explicitly via --results-dir because it does not match
# what ars.metrics_dqn.get_results_dir() would derive on its own
# (results/grid<G>/<model>/...) — the extra nyc/ level is intentional so other
# cities can sit alongside it.
#
# initial-active-agents follows the convention used in previous runs:
#   fixed       -> equal to num-agents (everyone active from day 1)
#   birth-death -> BD_INITIAL_ACTIVE (default 100), leaving the remainder in the
#                  never_joined pool so births can actually occur.
#   NOTE: at 100 agents this makes bd equal to fixed (empty never_joined pool),
#   so birth-death is a no-op there. Kept for consistency with prior runs.
#
# Config is overridable via environment variables, e.g.:
#   AGENT_COUNTS="150 200" GRIDS="30 45" DAYS=10 bash scripts/run_evaluation_dqn_nyc.sh
#   SKIP_EXISTING=0 bash scripts/run_evaluation_dqn_nyc.sh    # force re-run everything
set -uo pipefail

# ---- configuration (overridable via env) ----------------------------------
AGENT_COUNTS="${AGENT_COUNTS:-100 150 200}"
GRIDS="${GRIDS:-15 30 45}"
DAYS="${DAYS:-30}"
ALTRUISM_MEAN="${ALTRUISM_MEAN:-0.5}"
ALTRUISM_STD="${ALTRUISM_STD:-0.15}"
SAVE_ROOT="${SAVE_ROOT:-savedagents}"
RESULTS_ROOT="${RESULTS_ROOT:-results/nyc}"
BD_INITIAL_ACTIVE="${BD_INITIAL_ACTIVE:-100}"
# Skip a combination if its results_summary.txt already exists, so an
# interrupted sweep can be resumed without redoing completed work.
SKIP_EXISTING="${SKIP_EXISTING:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "Checkpoint root: ${SAVE_ROOT}/dqn"
echo "Results root:    ${RESULTS_ROOT}"
echo "Days per run:    ${DAYS}"
echo

# ---- timing / summary ------------------------------------------------------
LOG_ROOT="${REPO_ROOT}/ablation_logs"
mkdir -p "${LOG_ROOT}"
SUMMARY="${LOG_ROOT}/eval_dqn_nyc_summary_$(date +%Y%m%d-%H%M%S).txt"
echo "model,agents,grid,distribution,mode,status,seconds,hms" > "${SUMMARY}"
hms() { printf '%02d:%02d:%02d' $(( $1/3600 )) $(( ($1%3600)/60 )) $(( $1%60 )); }

# Dataset folder naming follows generate_grid_dataset.py:
#   grid 15 (canonical)      -> <A>_agents
#   grid 30 / 45 (ablation)  -> <A>_agents_grid<G>
dataset_name() {
  local agents=$1
  local grid=$2
  if [ "${grid}" -eq 15 ]; then
    echo "${agents}_agents"
  else
    echo "${agents}_agents_grid${grid}"
  fi
}

# Four (distribution, mode) combinations per grid/agent-count pair.
# The mode label matches the <dist>_<mode> results-folder suffix convention.
COMBOS=(
  "uniform:fixed"
  "uniform:bd"
  "gaussian:fixed"
  "gaussian:bd"
)

log_skip() {
  # $1=agents $2=grid $3=dist $4=mode $5=status
  echo "dqn,$1,$2,$3,$4,$5,0,00:00:00" >> "${SUMMARY}"
}

for GRID in ${GRIDS}; do
  for AGENTS in ${AGENT_COUNTS}; do
    DATASET="$(dataset_name "${AGENTS}" "${GRID}")"
    CKPT="${SAVE_ROOT}/dqn/grid${GRID}/${AGENTS}agents/dqn_final.pkl"

    if [ ! -d "dataset/${DATASET}" ]; then
      echo "!! dataset/${DATASET} missing; skipping all combos for ${AGENTS} agents @ grid${GRID}."
      for COMBO in "${COMBOS[@]}"; do
        log_skip "${AGENTS}" "${GRID}" "${COMBO%%:*}" "${COMBO##*:}" "NO_DATASET"
      done
      echo
      continue
    fi

    if [ ! -f "${CKPT}" ]; then
      echo "!! checkpoint ${CKPT} missing; skipping all combos for ${AGENTS} agents @ grid${GRID}."
      for COMBO in "${COMBOS[@]}"; do
        log_skip "${AGENTS}" "${GRID}" "${COMBO%%:*}" "${COMBO##*:}" "NO_CHECKPOINT"
      done
      echo
      continue
    fi

    for COMBO in "${COMBOS[@]}"; do
      DIST="${COMBO%%:*}"
      MODE="${COMBO##*:}"

      # Birth-death needs a non-empty never_joined pool; fixed activates everyone.
      if [ "${MODE}" == "bd" ]; then
        BD_FLAG="--enable-birth-death"
        INITIAL_ACTIVE="${BD_INITIAL_ACTIVE}"
        # Never exceed the total population.
        [ "${INITIAL_ACTIVE}" -gt "${AGENTS}" ] && INITIAL_ACTIVE="${AGENTS}"
      else
        BD_FLAG=""
        INITIAL_ACTIVE="${AGENTS}"
      fi

      RESULTS_DIR="${RESULTS_ROOT}/grid${GRID}/dqn/${AGENTS}agents_${DIST}_${MODE}"

      if [ "${SKIP_EXISTING}" -eq 1 ] && [ -f "${RESULTS_DIR}/results_summary.txt" ]; then
        echo "-- results already exist at ${RESULTS_DIR}; skipping."
        log_skip "${AGENTS}" "${GRID}" "${DIST}" "${MODE}" "SKIPPED_EXISTING"
        continue
      fi

      mkdir -p "${RESULTS_DIR}"

      echo "=================================================================="
      echo "=== DQN eval | grid ${GRID} | ${AGENTS} agents | ${DIST} | ${MODE} "
      echo "===   dataset=${DATASET}  initial_active=${INITIAL_ACTIVE}"
      echo "=================================================================="

      start=$(date +%s)
      python ars/evaluate_dqn.py \
        --dataset "${DATASET}" \
        --grid-size "${GRID}" \
        --num-agents "${AGENTS}" \
        --initial-active-agents "${INITIAL_ACTIVE}" \
        --days "${DAYS}" \
        --altruism-distribution "${DIST}" \
        --altruism-mean "${ALTRUISM_MEAN}" \
        --altruism-std "${ALTRUISM_STD}" \
        --save-dir "${SAVE_ROOT}" \
        --weights-file "${CKPT}" \
        --results-dir "${RESULTS_DIR}" \
        ${BD_FLAG}
      rc=$?
      elapsed=$(( $(date +%s) - start ))

      if [ "${rc}" -eq 0 ]; then
        echo "Completed in $(hms ${elapsed}) (${elapsed}s) -> ${RESULTS_DIR}"
        echo "dqn,${AGENTS},${GRID},${DIST},${MODE},ok,${elapsed},$(hms ${elapsed})" >> "${SUMMARY}"
      else
        echo "FAILED after $(hms ${elapsed}) (${elapsed}s)."
        echo "dqn,${AGENTS},${GRID},${DIST},${MODE},FAILED,${elapsed},$(hms ${elapsed})" >> "${SUMMARY}"
      fi
      echo
    done
  done
done

echo "All NYC DQN evaluation runs complete."
echo "Results: ${RESULTS_ROOT}/grid<G>/dqn/<A>agents_<dist>_<mode>/"
echo "------------------------------------------------------------"
echo "Timing summary: ${SUMMARY}"
column -t -s, "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
TOTAL_S=$(awk -F, 'NR>1 {s+=$7} END {print s+0}' "${SUMMARY}")
echo "Total wall-clock: $(hms ${TOTAL_S}) (${TOTAL_S}s)"
