#!/bin/bash
# Train + evaluate ALL Beijing baselines across the multi-scale grid:
#   models  : oracle (attention-MADDPG, shared) | maac (attention, unshared)
#             | maddpg (unshared) | pso (global optimizer, no training)
#   agents  : 100 150 200
#   grids   : 15 30 45
#
# For each (model, agents, grid):
#   * learning models (oracle/maac/maddpg): TRAIN then EVAL
#   * pso: single simulation (no training)
# Everything is Beijing-isolated:
#   datasets      dataset/beijing_<A>_agents_grid<G>   (generate first with
#                 scripts/beijing_generate_all_datasets.sh)
#   checkpoints   savedagents/beijing/<model>/grid<G>/<A>agents/
#   metrics       results/beijing/grid<G>/<model>/<A>agents_<dist>_fixed/
#
# Beijing uses ALPHA_R=0.6 and (for oracle) 4 attention heads -- the config that
# trains well on the compact Beijing core (alpha_r=0.4 is too detour-averse).
#
# Usage:
#   bash scripts/beijing_run_all_baselines.sh                  # everything
#   MODELS="oracle pso" bash scripts/beijing_run_all_baselines.sh
#   AGENT_COUNTS="100" GRID_SIZES="15" bash scripts/beijing_run_all_baselines.sh
#   STAGES="eval" MODELS="oracle" bash ...   # eval only (checkpoints must exist)
#
# Detached (survives logout):
#   setsid nohup bash scripts/beijing_run_all_baselines.sh > beijing_all.log 2>&1 &
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# -------- knobs --------
MODELS="${MODELS:-oracle maac maddpg pso}"
AGENT_COUNTS="${AGENT_COUNTS:-100 150 200}"
GRID_SIZES="${GRID_SIZES:-15 30 45}"
STAGES="${STAGES:-train eval}"          # which stages to run for learning models
DIST="${DIST:-uniform}"                 # altruism distribution
ALPHA_R="${ALPHA_R:-0.6}"               # Beijing reward trade-off
NUM_HEADS="${NUM_HEADS:-4}"             # oracle attention heads
# ORACLE HPO-best hyperparameters (distance-reduction HPO, trial 19). The
# NYC-inherited train_oracle DEFAULTS give NEGATIVE Beijing rewards (~-4.4);
# this config is what reaches ~8% distance reduction on Beijing. Applied to
# oracle only (maac/maddpg use their own defaults).
ORACLE_LR="${ORACLE_LR:-0.0008706757351599968}"
ORACLE_ACTOR_LR="${ORACLE_ACTOR_LR:-0.2878901730914787}"
ORACLE_GAMMA="${ORACLE_GAMMA:-0.9246993293529511}"
ORACLE_TAU="${ORACLE_TAU:-0.007355365959885428}"
ORACLE_REG_COEF="${ORACLE_REG_COEF:-0.00038676102055288703}"
ORACLE_GUMBEL_END="${ORACLE_GUMBEL_END:-0.17308581899910433}"
ORACLE_ATTN_TEMP="${ORACLE_ATTN_TEMP:-1.831412133536444}"
TRAIN_DAYS="${TRAIN_DAYS:-30}"
SIM_DAYS="${SIM_DAYS:-100}"
NUM_ENVS="${NUM_ENVS:-10}"                # ORACLE: 10 envs / 30 days (HPO-tuned)
NUM_EPISODES="${NUM_EPISODES:-200}"      # ORACLE: 200 total ep/day (200//10 envs = 20/env/day), ECML config
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"     # cosine decay reduces over-train collapse
USE_LAYER_NORM="${USE_LAYER_NORM:-0}"    # 1=on: LayerNorm in attention critic (stability)
# maac/maddpg train an independent net per agent and are much slower; run them
# with more parallel envs and fewer days for throughput.
MADDPG_ENVS="${MADDPG_ENVS:-25}"
MADDPG_DAYS="${MADDPG_DAYS:-10}"
MADDPG_EPISODES="${MADDPG_EPISODES:-100}"   # 100 total ep/day (100//25 envs = 4/env/day)
DEVICE="${DEVICE:-cpu}"
GPU_ID="${GPU_ID:-0}"
PSO_PARTICLES="${PSO_PARTICLES:-50}"
PSO_ITERS="${PSO_ITERS:-80}"

SAVE_ROOT="savedagents/beijing"
RESULTS_ROOT="results/beijing"
LOG_ROOT="beijing_logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "${LOG_ROOT}"
SUMMARY="${LOG_ROOT}/summary.csv"
echo "model,agents,grid,stage,status,seconds,log" > "${SUMMARY}"

hms() { printf '%02d:%02d:%02d' $(($1/3600)) $(($1%3600/60)) $(($1%60)); }

# distribution args (shared by eval + pso)
if [ "${DIST}" = "uniform" ]; then
  DIST_EVAL="--altruism-distribution uniform --altruism-mean 0.5"
  DIST_PSO="--altruism-dist uniform"
else
  DIST_EVAL="--altruism-distribution gaussian --altruism-mean 0.5 --altruism-std 0.2"
  DIST_PSO="--altruism-dist gaussian"
fi

model_type_for() { case "$1" in oracle|maac) echo attention ;; maddpg) echo maddpg ;; esac; }
train_mod_for()  { case "$1" in oracle) echo ars.train_oracle ;; maac|maddpg) echo ars.train ;; esac; }
eval_mod_for()   { case "$1" in oracle) echo ars.evaluate_oracle ;; maac|maddpg) echo ars.evaluate ;; esac; }
ckpt_file_for()  { case "$1" in
    oracle) echo "best_weights_attention_${2}_tags.pkl" ;;   # peak-val, not final (avoids over-train collapse)
    maac)   echo "final_weights_attention.pkl" ;;
    maddpg) echo "final_weights_maddpg.pkl" ;;
  esac; }

run_step() {  # LABEL LOGFILE  -- remaining args are the command
  local label="$1" logf="$2"; shift 2
  local start rc elapsed
  echo ">>> ${label}"
  start=$(date +%s)
  "$@" > "${logf}" 2>&1
  rc=$?
  elapsed=$(( $(date +%s) - start ))
  LAST_SECONDS=${elapsed}
  if [ ${rc} -eq 0 ]; then echo "    ok ($(hms ${elapsed}))"; LAST_STATUS=ok
  else echo "    FAILED rc=${rc} ($(hms ${elapsed})) -> ${logf}"; LAST_STATUS=FAILED; fi
  return ${rc}
}

record() { echo "$1,$2,$3,$4,$5,$6,${7:-}" >> "${SUMMARY}"; }

for GRID in ${GRID_SIZES}; do
  for A in ${AGENT_COUNTS}; do
    DATASET="beijing_${A}_agents_grid${GRID}"
    if [ ! -d "dataset/${DATASET}" ]; then
      echo "!! dataset/${DATASET} missing -> skip agents=${A} grid=${GRID}."
      echo "   Generate first: GRID_SIZES=\"${GRID}\" AGENTS=${A} bash scripts/beijing_generate_all_datasets.sh"
      for M in ${MODELS}; do record "${M}" "${A}" "${GRID}" all NO_DATASET 0; done
      continue
    fi

    for MODEL in ${MODELS}; do
      # ---------------- PSO: single simulation, no training ----------------
      if [ "${MODEL}" = "pso" ]; then
        exp="pso_${A}agents_${DIST}_fixed_grid${GRID}"
        plog="${LOG_ROOT}/pso_${A}_grid${GRID}.log"
        run_step "PSO agents=${A} grid=${GRID}" "${plog}" \
          python -m ars.baselines.run_pso \
            --num-agents "${A}" --initial-active-agents "${A}" \
            --days "${SIM_DAYS}" --agent-dynamics fixed ${DIST_PSO} \
            --dataset "${DATASET}" --grid-size "${GRID}" \
            --experiment-name "${exp}" \
            --pso-particles "${PSO_PARTICLES}" --pso-iterations "${PSO_ITERS}" \
            --alpha "${ALPHA_R}" --enable-benefit-analysis --verbose
        record pso "${A}" "${GRID}" sim "${LAST_STATUS}" "${LAST_SECONDS}" "${plog}"
        continue
      fi

      # ---------------- learning models: train then eval -------------------
      MTYPE="$(model_type_for "${MODEL}")"
      CKPT_DIR="${SAVE_ROOT}/${MODEL}/grid${GRID}/${A}agents"
      CKPT_FILE="$(ckpt_file_for "${MODEL}" "${A}")"
      HEADS_ARG=""; [ "${MODEL}" = "oracle" ] && HEADS_ARG="--num-heads ${NUM_HEADS}"

      # TRAIN
      if [[ " ${STAGES} " == *" train "* ]]; then
        tlog="${LOG_ROOT}/train_${MODEL}_${A}_grid${GRID}.log"
        if [ "${MODEL}" = "oracle" ]; then
          run_step "TRAIN oracle agents=${A} grid=${GRID}" "${tlog}" \
            python -m ars.train_oracle --model attention \
              --dataset "${DATASET}" --grid-size "${GRID}" \
              --num-agents "${A}" --initial-active-agents "${A}" \
              --days "${TRAIN_DAYS}" --num-envs "${NUM_ENVS}" --num-episodes "${NUM_EPISODES}" \
              --alpha-r "${ALPHA_R}" ${HEADS_ARG} \
              --lr "${ORACLE_LR}" --actor-lr "${ORACLE_ACTOR_LR}" \
              --gamma "${ORACLE_GAMMA}" --tau "${ORACLE_TAU}" \
              --reg-coef "${ORACLE_REG_COEF}" --gumbel-temp-end "${ORACLE_GUMBEL_END}" \
              --attn_temp "${ORACLE_ATTN_TEMP}" --lr-schedule "${LR_SCHEDULE}" \
              $([ "${USE_LAYER_NORM}" = "1" ] && echo --use-layer-norm) \
              --save-dir "${SAVE_ROOT}" --device "${DEVICE}" --gpu-id "${GPU_ID}"
        else
          # maac/maddpg: 25 envs / 10 days / 100 ep/day (4/env)
          run_step "TRAIN ${MODEL} agents=${A} grid=${GRID}" "${tlog}" \
            python -m ars.train --model "${MTYPE}" \
              --dataset "${DATASET}" --grid-size "${GRID}" \
              --num-agents "${A}" --initial-active-agents "${A}" \
              --days "${MADDPG_DAYS}" --num-envs "${MADDPG_ENVS}" \
              --num-episodes "${MADDPG_EPISODES}" \
              --save-dir "${SAVE_ROOT}" --device "${DEVICE}" --gpu-id "${GPU_ID}"
        fi
        record "${MODEL}" "${A}" "${GRID}" train "${LAST_STATUS}" "${LAST_SECONDS}" "${tlog}"
        if [ "${LAST_STATUS}" != "ok" ]; then
          echo "   training failed -> skipping eval for ${MODEL} ${A}/${GRID}"
          continue
        fi
      fi

      # EVAL
      if [[ " ${STAGES} " == *" eval "* ]]; then
        if [ ! -f "${CKPT_DIR}/${CKPT_FILE}" ]; then
          echo "!! checkpoint ${CKPT_DIR}/${CKPT_FILE} missing -> skip eval ${MODEL} ${A}/${GRID}"
          record "${MODEL}" "${A}" "${GRID}" eval NO_CKPT 0
          continue
        fi
        elog="${LOG_ROOT}/eval_${MODEL}_${A}_grid${GRID}.log"
        RESULTS_ARG="--results-dir ${RESULTS_ROOT}/grid${GRID}/${MODEL}/${A}agents_${DIST}_fixed"
        # oracle eval must match training's attention_temp + layernorm; the
        # maac/maddpg evaluator (ars.evaluate) does not take these flags.
        ORACLE_EVAL_ARGS=""
        if [ "${MODEL}" = "oracle" ]; then
          ORACLE_EVAL_ARGS="--attn_temp ${ORACLE_ATTN_TEMP}"
          [ "${USE_LAYER_NORM}" = "1" ] && ORACLE_EVAL_ARGS="${ORACLE_EVAL_ARGS} --use-layer-norm"
        fi
        run_step "EVAL ${MODEL} agents=${A} grid=${GRID}" "${elog}" \
          python -m "$(eval_mod_for "${MODEL}")" \
            --model-type "${MTYPE}" \
            --dataset "${DATASET}" --grid-size "${GRID}" \
            --num-agents "${A}" --initial-active-agents "${A}" \
            --days "${SIM_DAYS}" ${DIST_EVAL} ${RESULTS_ARG} ${HEADS_ARG} ${ORACLE_EVAL_ARGS} \
            --alpha-r "${ALPHA_R}" \
            --save-dir "${CKPT_DIR}" --restore --weights-file "${CKPT_FILE}" \
            --device "${DEVICE}" --gpu-id "${GPU_ID}"
        record "${MODEL}" "${A}" "${GRID}" eval "${LAST_STATUS}" "${LAST_SECONDS}" "${elog}"
      fi
    done
  done
done

echo
echo "=== ALL DONE. summary: ${SUMMARY} ==="
column -t -s, "${SUMMARY}"
