#!/bin/bash
# Re-run NYC ORACLE (and PSO) at grid 30x30 and 45x45 with the NYC HPO-BEST
# hyperparameters. The prior grid-ablation trained ORACLE with train_oracle
# DEFAULTS (run_grid_ablation.sh passes no lr/gamma/tau flags), which left
# grid30 undertrained (9.49%, below grid45's ~20%). This script applies the
# NYC HPO-best config (runs/hpo/oracle_grid15/best.json, trial 6) so ORACLE
# reaches its true capability.
#
# Outputs go under the results/nyc/ structure:
#   checkpoints -> savedagents/nyc/oracle/grid<G>/100agents/
#   ORACLE eval -> results/nyc/grid<G>/oracle/100agents_uniform_fixed/
#   PSO         -> results/nyc/grid<G>/pso/pso_100agents_uniform_fixed/
#
# Usage:
#   bash scripts/nyc_rerun_oracle_pso_grid3045.sh
#   GRIDS="45" MODELS="oracle" bash scripts/nyc_rerun_oracle_pso_grid3045.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GRIDS="${GRIDS:-30 45}"
MODELS="${MODELS:-oracle pso}"
NUM_AGENTS="${NUM_AGENTS:-100}"
TRAIN_DAYS="${TRAIN_DAYS:-30}"
SIM_DAYS="${SIM_DAYS:-100}"
NUM_ENVS="${NUM_ENVS:-10}"
NUM_EPISODES="${NUM_EPISODES:-200}"      # 200 total ep/day (ECML config)
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"     # cosine decay reduces over-train collapse
USE_LAYER_NORM="${USE_LAYER_NORM:-0}"    # LayerNorm in attention critic
DEVICE="${DEVICE:-cpu}"
GPU_ID="${GPU_ID:-0}"
DIST="${DIST:-uniform}"
PSO_PARTICLES="${PSO_PARTICLES:-50}"
PSO_ITERS="${PSO_ITERS:-80}"

# NYC HPO-best (runs/hpo/oracle_grid15/best.json, trial 6, objective mean_eval_reward)
NYC_LR="${NYC_LR:-0.0005452335737684413}"
NYC_ACTOR_LR="${NYC_ACTOR_LR:-0.010280029617905608}"
NYC_GAMMA="${NYC_GAMMA:-0.9006792587384664}"
NYC_TAU="${NYC_TAU:-0.08164088498895099}"
NYC_REG_COEF="${NYC_REG_COEF:-0.0007676092714795293}"
NYC_GUMBEL_END="${NYC_GUMBEL_END:-0.21914469472126635}"
NYC_ATTN_TEMP="${NYC_ATTN_TEMP:-1.7213090816545344}"
NYC_HEADS="${NYC_HEADS:-4}"

LOG_ROOT="nyc_rerun_logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "${LOG_ROOT}"
SUMMARY="${LOG_ROOT}/summary.csv"
echo "model,grid,stage,status,seconds,log" > "${SUMMARY}"
hms() { printf '%02d:%02d:%02d' $(($1/3600)) $(($1%3600/60)) $(($1%60)); }
run_step() { local label="$1" logf="$2"; shift 2; local s r e; echo ">>> ${label}"; s=$(date +%s); "$@" >"${logf}" 2>&1; r=$?; e=$(( $(date +%s)-s )); LAST_SECONDS=$e
  if [ $r -eq 0 ]; then echo "    ok ($(hms $e))"; LAST_STATUS=ok; else echo "    FAILED rc=$r -> ${logf}"; LAST_STATUS=FAILED; fi; return $r; }

for GRID in ${GRIDS}; do
  DATASET="${NUM_AGENTS}_agents_grid${GRID}"
  if [ ! -d "dataset/${DATASET}" ]; then
    echo "!! dataset/${DATASET} missing; skip grid${GRID}."; continue
  fi

  for MODEL in ${MODELS}; do
    if [ "${MODEL}" = "pso" ]; then
      exp="pso_${NUM_AGENTS}agents_${DIST}_fixed"
      # write into results/nyc/grid<G>/pso/ via save-dir
      sdir="results/nyc/grid${GRID}/pso"
      mkdir -p "${sdir}"
      plog="${LOG_ROOT}/pso_grid${GRID}.log"
      run_step "PSO nyc grid${GRID}" "${plog}" \
        python -m ars.baselines.run_pso \
          --num-agents "${NUM_AGENTS}" --initial-active-agents "${NUM_AGENTS}" \
          --days "${SIM_DAYS}" --agent-dynamics fixed --altruism-dist "${DIST}" \
          --dataset "${DATASET}" --grid-size "${GRID}" \
          --experiment-name "${exp}" --save-dir "${sdir}" \
          --pso-particles "${PSO_PARTICLES}" --pso-iterations "${PSO_ITERS}" \
          --alpha 0.4 --enable-benefit-analysis --verbose
      echo "pso,${GRID},sim,${LAST_STATUS},${LAST_SECONDS},${plog}" >> "${SUMMARY}"
      continue
    fi

    # ORACLE: train with NYC HPO-best config, then eval
    ckpt_dir="savedagents/nyc/oracle/grid${GRID}/${NUM_AGENTS}agents"
    ckpt_file="best_weights_attention_${NUM_AGENTS}_tags.pkl"  # peak-val, not final
    tlog="${LOG_ROOT}/train_oracle_grid${GRID}.log"
    elog="${LOG_ROOT}/eval_oracle_grid${GRID}.log"

    LN_FLAG=$([ "${USE_LAYER_NORM:-0}" = "1" ] && echo --use-layer-norm)
    run_step "TRAIN oracle nyc grid${GRID} (HPO-best)" "${tlog}" \
      python -m ars.train_oracle --model attention \
        --dataset "${DATASET}" --grid-size "${GRID}" \
        --num-agents "${NUM_AGENTS}" --initial-active-agents "${NUM_AGENTS}" \
        --days "${TRAIN_DAYS}" --num-envs "${NUM_ENVS}" --num-episodes "${NUM_EPISODES}" \
        --exploration tags \
        --lr "${NYC_LR}" --actor-lr "${NYC_ACTOR_LR}" --gamma "${NYC_GAMMA}" \
        --tau "${NYC_TAU}" --reg-coef "${NYC_REG_COEF}" \
        --gumbel-temp-end "${NYC_GUMBEL_END}" --attn_temp "${NYC_ATTN_TEMP}" \
        --num-heads "${NYC_HEADS}" --lr-schedule "${LR_SCHEDULE}" ${LN_FLAG} \
        --save-dir savedagents/nyc --device "${DEVICE}" --gpu-id "${GPU_ID}"
    echo "oracle,${GRID},train,${LAST_STATUS},${LAST_SECONDS},${tlog}" >> "${SUMMARY}"
    [ "${LAST_STATUS}" != "ok" ] && continue

    run_step "EVAL oracle nyc grid${GRID}" "${elog}" \
      python -m ars.evaluate_oracle --model-type attention \
        --dataset "${DATASET}" --grid-size "${GRID}" \
        --num-agents "${NUM_AGENTS}" --initial-active-agents "${NUM_AGENTS}" \
        --days "${SIM_DAYS}" --altruism-distribution "${DIST}" --altruism-mean 0.5 \
        --num-heads "${NYC_HEADS}" --attn_temp "${NYC_ATTN_TEMP}" ${LN_FLAG} \
        --results-dir "results/nyc/grid${GRID}/oracle/${NUM_AGENTS}agents_${DIST}_fixed" \
        --save-dir "${ckpt_dir}" --restore --weights-file "${ckpt_file}" \
        --device "${DEVICE}"
    echo "oracle,${GRID},eval,${LAST_STATUS},${LAST_SECONDS},${elog}" >> "${SUMMARY}"
  done
done

echo; echo "=== NYC re-run done. summary: ${SUMMARY} ==="; column -t -s, "${SUMMARY}"
