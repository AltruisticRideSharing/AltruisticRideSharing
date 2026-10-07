#!/usr/bin/env bash
#
# Sensitivity / ablation sweep over the ARS altruism-economy design choices.
#
# One-factor-at-a-time sensitivity analysis of the key design choices that
# are NOT learned by the policy:
#   * forced-driver threshold  (paper default 0.2)
#   * reward trade-off alpha_r (paper default 0.4)
#   * driver/rider altruism scaling alpha_s / beta_s (defaults 0.5 / 0.7)
#
# We hold a single trained ORACLE policy fixed and re-run the evaluation
# simulation while varying ONE knob at a time (one-factor-at-a-time). This is
# cheap (no retraining) and isolates the effect of each altruism-economy choice
# on efficiency (distance reduction, detour, utilization) and fairness (Gini,
# acceptance, mean altruism, driver/rider balance).
#
# NOTE: alpha_r also shapes the training reward. Holding the policy fixed while
# sweeping alpha_r at evaluation measures the *altruism-economy* sensitivity, not
# a full retrain-per-value response. State this caveat in the paper; a retrain
# sweep can be layered on later if time permits.
#
# Results land in:   results/ablation/<knob>/<value>/oracle/...
# Collected table:   results/ablation/sensitivity_summary.csv  (via scripts/collect_ablation.py)
#
# Usage:
#   ABLATION_CKPT_DIR=savedagents/oracle/grid15/100agents \
#   ABLATION_CKPT_FILE=final_weights_attention_100_tags.pkl \
#   bash scripts/run_ablation_sensitivity.sh
#
# Override the sweep grids / budget via env vars (see below).
set -uo pipefail

ENV_NAME="${ENV_NAME:-ars}"
# Defaults target the healthy grid45 ORACLE checkpoint. A sensitivity sweep does
# not need the full 100-day horizon: SIM_DAYS=20 is plenty to measure the
# relative response to each knob and keeps the 20-config sweep affordable on CPU.
GRID="${GRID:-45}"
DATASET="${DATASET:-100_agents_grid${GRID}}"
NUM_AGENTS="${NUM_AGENTS:-100}"
SIM_DAYS="${SIM_DAYS:-20}"
ALTRUISM_DIST="${ALTRUISM_DIST:-uniform}"
DEVICE="${DEVICE:-cpu}"
GPU_ID="${GPU_ID:-0}"

# Trained ORACLE checkpoint to hold fixed across the sweep.
CKPT_DIR="${ABLATION_CKPT_DIR:-savedagents/oracle/grid${GRID}/${NUM_AGENTS}agents}"
CKPT_FILE="${ABLATION_CKPT_FILE:-final_weights_attention_${NUM_AGENTS}_tags.pkl}"

# Defaults (the paper configuration). Held when another knob is swept.
DEF_ALPHA_R="${DEF_ALPHA_R:-0.4}"
DEF_THRESH="${DEF_THRESH:-0.2}"
DEF_ALPHA_S="${DEF_ALPHA_S:-0.5}"
DEF_BETA_S="${DEF_BETA_S:-0.7}"

# Sweep grids (space-separated). Centred on the defaults.
SWEEP_ALPHA_R="${SWEEP_ALPHA_R:-0.2 0.3 0.4 0.5 0.6}"
SWEEP_THRESH="${SWEEP_THRESH:-0.1 0.15 0.2 0.25 0.3}"
SWEEP_ALPHA_S="${SWEEP_ALPHA_S:-0.3 0.4 0.5 0.6 0.7}"
SWEEP_BETA_S="${SWEEP_BETA_S:-0.5 0.6 0.7 0.8 0.9}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LOG_ROOT="${REPO_ROOT}/ablation_logs"
mkdir -p "${LOG_ROOT}"

CONDA_BASE="$(conda info --base 2>/dev/null)"
# shellcheck disable=SC1090
[ -n "${CONDA_BASE}" ] && source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}" || { echo "Could not activate env ${ENV_NAME}"; exit 1; }
echo "Using python: $(which python)"

if [ ! -f "${CKPT_DIR}/${CKPT_FILE}" ]; then
  echo "!! checkpoint ${CKPT_DIR}/${CKPT_FILE} not found."
  echo "   Train ORACLE first (ars.train_oracle) or set ABLATION_CKPT_DIR/FILE."
  exit 1
fi

SUMMARY="${LOG_ROOT}/ablation_summary_$(date +%Y%m%d-%H%M%S).txt"
echo "knob,value,status,seconds,hms,results_dir,log" > "${SUMMARY}"
hms() { printf '%02d:%02d:%02d' $(( $1/3600 )) $(( ($1%3600)/60 )) $(( $1%60 )); }

# run_one <knob> <value> <alpha_r> <thresh> <alpha_s> <beta_s>
run_one() {
  local knob="$1" value="$2" ar="$3" th="$4" as="$5" bs="$6"
  local out="results/ablation/${knob}/${value}"
  local log="${LOG_ROOT}/ablation_${knob}_${value}.log"
  echo ""
  echo ">>> ablation ${knob}=${value}  (alpha_r=${ar} thresh=${th} alpha_s=${as} beta_s=${bs})"
  echo "    results: ${out}"
  local start elapsed
  start=$(date +%s)
  if python -m ars.evaluate_oracle \
       --model-type attention \
       --dataset "${DATASET}" --grid-size "${GRID}" \
       --num-agents "${NUM_AGENTS}" --days "${SIM_DAYS}" \
       --altruism-distribution "${ALTRUISM_DIST}" \
       --alpha-r "${ar}" --forced-driver-threshold "${th}" \
       --alpha-s "${as}" --beta-s "${bs}" \
       --save-dir "${CKPT_DIR}" --restore --weights-file "${CKPT_FILE}" \
       --results-dir "${out}" \
       --device "${DEVICE}" --gpu-id "${GPU_ID}" > "${log}" 2>&1; then
    elapsed=$(( $(date +%s) - start ))
    echo "    OK ($(hms ${elapsed}))"
    echo "${knob},${value},ok,${elapsed},$(hms ${elapsed}),${out},${log}" >> "${SUMMARY}"
  else
    elapsed=$(( $(date +%s) - start ))
    echo "    FAILED in $(hms ${elapsed}) (see ${log})"
    echo "${knob},${value},FAILED,${elapsed},$(hms ${elapsed}),${out},${log}" >> "${SUMMARY}"
  fi
}

echo "============================================================"
echo "Sensitivity sweep | ckpt ${CKPT_DIR}/${CKPT_FILE}"
echo "Defaults: alpha_r=${DEF_ALPHA_R} thresh=${DEF_THRESH} alpha_s=${DEF_ALPHA_S} beta_s=${DEF_BETA_S}"
echo "============================================================"

for v in ${SWEEP_ALPHA_R};  do run_one alpha_r "$v" "$v" "${DEF_THRESH}" "${DEF_ALPHA_S}" "${DEF_BETA_S}"; done
for v in ${SWEEP_THRESH};   do run_one threshold "$v" "${DEF_ALPHA_R}" "$v" "${DEF_ALPHA_S}" "${DEF_BETA_S}"; done
for v in ${SWEEP_ALPHA_S};  do run_one alpha_s "$v" "${DEF_ALPHA_R}" "${DEF_THRESH}" "$v" "${DEF_BETA_S}"; done
for v in ${SWEEP_BETA_S};   do run_one beta_s  "$v" "${DEF_ALPHA_R}" "${DEF_THRESH}" "${DEF_ALPHA_S}" "$v"; done

echo ""
echo "============================================================"
echo "Sweep complete. Summary: ${SUMMARY}"
column -t -s, "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
TOTAL_S=$(awk -F, 'NR>1 {s+=$4} END {print s+0}' "${SUMMARY}")
echo "Total wall-clock: $(hms ${TOTAL_S}) (${TOTAL_S}s)"
echo "------------------------------------------------------------"
echo "Collect into a table with:"
echo "  python scripts/collect_ablation.py --root results/ablation \\"
echo "      --out results/ablation/sensitivity_summary.csv"
echo "============================================================"
