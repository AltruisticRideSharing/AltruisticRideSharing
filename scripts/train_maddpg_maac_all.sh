#!/bin/bash
# Train + eval MADDPG and MAAC across both cities.
# Defaults: 10 parallel envs, 100 episodes/day, 5 days (override via env vars).
# DeepPool is intentionally NOT trained here.
#
# Coverage:
#   Beijing : grid 15/30/45 x agents 100/150/200   (all 9 combos each model)
#   NYC     : grid 30/45     x agents 100/150/200   (grid 15 skipped by default)
#
# Paths mirror the Beijing driver:
#   checkpoints -> savedagents/<city>/<model>/grid<G>/<A>agents/
#   metrics     -> results/<city>/grid<G>/<model>/<A>agents_<dist>_fixed/
#
# Runs combos SEQUENTIALLY (one at a time) to avoid oversubscribing cores while
# the ORACLE run is live. Launch detached; safe to log off.
#
# Usage:
#   CITIES="beijing nyc" bash scripts/train_maddpg_maac_all.sh
#   CITIES=beijing MODELS=maac GRIDS=45 AGENTS=200 bash scripts/train_maddpg_maac_all.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO_ROOT}"

CITIES="${CITIES:-beijing nyc}"
MODELS="${MODELS:-maddpg maac}"
AGENTS="${AGENTS:-100 150 200}"
STAGES="${STAGES:-train eval}"
DIST="${DIST:-uniform}"

# user-directed config
NUM_ENVS="${NUM_ENVS:-10}"
NUM_EPISODES="${NUM_EPISODES:-100}"
TRAIN_DAYS="${TRAIN_DAYS:-5}"
SIM_DAYS="${SIM_DAYS:-100}"
DEVICE="${DEVICE:-cpu}"
GPU_ID="${GPU_ID:-0}"

# per-city reward trade-off (matches the city drivers)
alpha_for() { case "$1" in beijing) echo 0.6 ;; nyc) echo 0.4 ;; *) echo 0.5 ;; esac; }
# grids per city (nyc grid15 is ECML-frozen -> excluded)
grids_for() { case "$1" in beijing) echo "${GRIDS:-15 30 45}" ;; nyc) echo "${GRIDS:-30 45}" ;; esac; }
dataset_for() { case "$1" in beijing) echo "beijing_${2}_agents_grid${3}" ;; nyc) echo "${2}_agents_grid${3}" ;; esac; }
# ars.train --model attention => saved as MAAC; --model maddpg => MADDPG
mtype_for() { case "$1" in maac) echo attention ;; maddpg) echo maddpg ;; esac; }
ckpt_for()  { case "$1" in maac) echo "final_weights_attention.pkl" ;; maddpg) echo "final_weights_maddpg.pkl" ;; esac; }

LOG_ROOT="mm_logs/$(date +%Y%m%d-%H%M%S)"; mkdir -p "${LOG_ROOT}"
SUMMARY="${LOG_ROOT}/summary.csv"; echo "city,model,grid,agents,stage,status,seconds,log" > "${SUMMARY}"
hms(){ printf '%02d:%02d:%02d' $(($1/3600)) $(($1%3600/60)) $(($1%60)); }
run_step(){ local label="$1" logf="$2"; shift 2; local s r e; echo ">>> ${label}"; s=$(date +%s)
  "$@" >"${logf}" 2>&1; r=$?; e=$(( $(date +%s)-s )); LAST_SECONDS=$e
  if [ $r -eq 0 ]; then echo "    ok ($(hms $e))"; LAST_STATUS=ok; else echo "    FAILED rc=$r -> ${logf}"; LAST_STATUS=FAILED; fi; return $r; }

for CITY in ${CITIES}; do
  ALPHA="$(alpha_for "${CITY}")"
  for GRID in $(grids_for "${CITY}"); do
    for A in ${AGENTS}; do
      DATASET="$(dataset_for "${CITY}" "${A}" "${GRID}")"
      if [ ! -d "dataset/${DATASET}" ]; then
        echo "!! dataset/${DATASET} missing -> skip ${CITY} ${A}/g${GRID}"; continue
      fi
      for MODEL in ${MODELS}; do
        MTYPE="$(mtype_for "${MODEL}")"; CKPT="$(ckpt_for "${MODEL}")"
        CKPT_DIR="savedagents/${CITY}/${MODEL}/grid${GRID}/${A}agents"
        RES_DIR="results/${CITY}/grid${GRID}/${MODEL}/${A}agents_${DIST}_fixed"
        mkdir -p "${CKPT_DIR}"

        # disk guard: abort before a run if free space is dangerously low
        # (a checkpoint written on a full disk corrupts silently).
        FREE_GB=$(df --output=avail -BG . | tail -1 | tr -dc 0-9)
        if [ "${FREE_GB:-0}" -lt "${MIN_FREE_GB:-30}" ]; then
          echo "!! only ${FREE_GB}GB free (< ${MIN_FREE_GB:-30}GB) -> ABORT before ${CITY} ${MODEL} ${A}/g${GRID}"
          echo "${CITY},${MODEL},${GRID},${A},train,ABORT_LOWDISK,0,-" >> "${SUMMARY}"
          continue
        fi

        if [[ " ${STAGES} " == *" train "* ]]; then
          # resume: skip combos that already have a VALID (loadable) checkpoint
          if [ -f "${CKPT_DIR}/${CKPT}" ] && \
             python -c "import pickle,sys; pickle.load(open(sys.argv[1],'rb'))" "${CKPT_DIR}/${CKPT}" 2>/dev/null; then
            echo ">>> SKIP ${CITY} ${MODEL} ${A}/g${GRID} (valid checkpoint exists)"
            echo "${CITY},${MODEL},${GRID},${A},train,skip_exists,0,-" >> "${SUMMARY}"
          else
          tlog="${LOG_ROOT}/train_${CITY}_${MODEL}_${A}_g${GRID}.log"
          run_step "TRAIN ${CITY} ${MODEL} ${A}/g${GRID}" "${tlog}" \
            python -m ars.train --model "${MTYPE}" \
              --dataset "${DATASET}" --grid-size "${GRID}" \
              --num-agents "${A}" --initial-active-agents "${A}" \
              --days "${TRAIN_DAYS}" --num-envs "${NUM_ENVS}" --num-episodes "${NUM_EPISODES}" \
              --save-dir "savedagents/${CITY}" --device "${DEVICE}" --gpu-id "${GPU_ID}"
          echo "${CITY},${MODEL},${GRID},${A},train,${LAST_STATUS},${LAST_SECONDS},${tlog}" >> "${SUMMARY}"
          [ "${LAST_STATUS}" != "ok" ] && { echo "   train failed -> skip eval"; continue; }
          fi
        fi

        if [[ " ${STAGES} " == *" eval "* ]]; then
          if [ ! -f "${CKPT_DIR}/${CKPT}" ]; then
            echo "!! checkpoint ${CKPT_DIR}/${CKPT} missing -> skip eval"; continue
          fi
          elog="${LOG_ROOT}/eval_${CITY}_${MODEL}_${A}_g${GRID}.log"
          run_step "EVAL ${CITY} ${MODEL} ${A}/g${GRID}" "${elog}" \
            python -m ars.evaluate --model-type "${MTYPE}" \
              --dataset "${DATASET}" --grid-size "${GRID}" \
              --num-agents "${A}" --initial-active-agents "${A}" \
              --days "${SIM_DAYS}" --altruism-distribution "${DIST}" \
              --results-dir "${RES_DIR}" --alpha-r "${ALPHA}" \
              --save-dir "${CKPT_DIR}" --restore --weights-file "${CKPT}" \
              --device "${DEVICE}" --gpu-id "${GPU_ID}"
          echo "${CITY},${MODEL},${GRID},${A},eval,${LAST_STATUS},${LAST_SECONDS},${elog}" >> "${SUMMARY}"
        fi
      done
    done
  done
done
echo; echo "=== MADDPG/MAAC sweep done: ${SUMMARY} ==="; column -t -s, "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
