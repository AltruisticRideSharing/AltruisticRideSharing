#!/usr/bin/env bash
#
# End-to-end Beijing (T-Drive) dataset setup for the ARS pipeline.
#
# Stages (each timed; summary written to ablation_logs/):
#   1. DOWNLOAD  T-Drive raw GPS (per-taxi CSVs) into a work dir.
#   2. INGEST    segment GPS -> pickup/dropoff trips  (ars.data.ingest_tdrive)
#   3. GENERATE  trips -> grid dataset folders        (ars.data.generate_grid_dataset)
#
# The result is dataset/beijing_<A>_agents[_grid<N>]/ folders that are drop-in
# for train/eval exactly like the NYC ones:
#   python -m ars.train_oracle    --dataset beijing_100_agents --grid-size 15 ...
#   bash scripts/run_all_simulations.sh   (set DATASET_PREFIX/AGENT dirs accordingly)
#
# T-Drive download is auth-gated upstream (MSR page / Kaggle / HF all need a
# credential). This script auto-uses whatever is available, in order:
#   * a local zip/dir you point it at via TDRIVE_RAW or TDRIVE_ZIP
#   * the Kaggle CLI (needs ~/.kaggle/kaggle.json)   dataset: arashnic/tdriver
# If none is usable it prints the 2-line manual fetch and exits cleanly so you
# can drop the file in and re-run (ingest/generate are idempotent + cached).
#
# Usage:
#   bash scripts/setup_beijing_dataset.sh                     # 15x15, 100 agents
#   GRID_SIZES="15 30 45" AGENTS=100 bash scripts/setup_beijing_dataset.sh
#   TDRIVE_ZIP=~/Downloads/archive.zip bash scripts/setup_beijing_dataset.sh
#   TDRIVE_RAW=~/tdrive/taxi_log_2008_by_id bash scripts/setup_beijing_dataset.sh
set -uo pipefail

ENV_NAME="${ENV_NAME:-ars}"
GRID_SIZES="${GRID_SIZES:-15}"
AGENTS="${AGENTS:-100}"
# NB: a literal {num_agents} cannot live inside ${VAR:-default} — the first '}'
# closes the expansion early. Set the default in a separate statement.
OUT_TEMPLATE_DEFAULT='beijing_{num_agents}_agents'   # +_grid{grid} if multi-grid
OUT_TEMPLATE="${OUT_TEMPLATE:-$OUT_TEMPLATE_DEFAULT}"
MAX_TAXIS="${MAX_TAXIS:-0}"        # cap taxis ingested (0=all); set small for a smoke test
BBOX="${BBOX:-}"                   # optional "min_lon min_lat max_lon max_lat" override

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

WORK="${TDRIVE_WORK:-${REPO_ROOT}/dataset/dataset_creation/tdrive}"
RIDES_PARQUET="${REPO_ROOT}/ars/data/cache/beijing_filtered_rides.parquet"
LOG_ROOT="${REPO_ROOT}/ablation_logs"
mkdir -p "${WORK}" "${LOG_ROOT}" "${REPO_ROOT}/ars/data/cache"

SUMMARY="${LOG_ROOT}/beijing_setup_$(date +%Y%m%d-%H%M%S).txt"
echo "stage,status,seconds,hms" > "${SUMMARY}"
hms() { printf '%02d:%02d:%02d' $(( $1/3600 )) $(( ($1%3600)/60 )) $(( $1%60 )); }
record() { echo "$1,$2,$3,$(hms $3)" >> "${SUMMARY}"; }

CONDA_BASE="$(conda info --base 2>/dev/null)"
# shellcheck disable=SC1090
[ -n "${CONDA_BASE}" ] && source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}" || { echo "Could not activate env ${ENV_NAME}"; exit 1; }
echo "Using python: $(which python)"

# Word count of --grid-sizes decides whether the folder name needs _grid{grid}.
NGRID=$(echo "${GRID_SIZES}" | wc -w)
if [ "${NGRID}" -gt 1 ] && [ "${OUT_TEMPLATE}" = "beijing_{num_agents}_agents" ]; then
  OUT_TEMPLATE="beijing_{num_agents}_agents_grid{grid}"
  echo "[info] multi-grid run -> out-template ${OUT_TEMPLATE}"
fi

# ----------------------------------------------------------------------------
# Stage 1: obtain raw T-Drive per-taxi files -> $RAW_DIR
# ----------------------------------------------------------------------------
find_raw_dir() {  # echo a dir that contains taxi_log_*.txt/.csv, or empty
  local base="$1"
  # the canonical layout nests files under taxi_log_2008_by_id/
  for cand in "${base}/taxi_log_2008_by_id" "${base}/release/taxi_log_2008_by_id" "${base}"; do
    if compgen -G "${cand}/*.txt" >/dev/null 2>&1 || compgen -G "${cand}/*.csv" >/dev/null 2>&1; then
      echo "${cand}"; return
    fi
  done
  # last resort: first dir under base holding many .txt files
  local d
  d=$(find "${base}" -type f -name '*.txt' 2>/dev/null | head -1)
  [ -n "${d}" ] && dirname "${d}"
}

RAW_DIR=""
start=$(date +%s)
if [ -n "${TDRIVE_RAW:-}" ]; then
  echo "[1/3] using TDRIVE_RAW=${TDRIVE_RAW}"
  RAW_DIR="$(find_raw_dir "${TDRIVE_RAW}")"
elif [ -n "${TDRIVE_ZIP:-}" ]; then
  echo "[1/3] unzipping TDRIVE_ZIP=${TDRIVE_ZIP} -> ${WORK}"
  unzip -oq "${TDRIVE_ZIP}" -d "${WORK}" && RAW_DIR="$(find_raw_dir "${WORK}")"
elif [ -n "$(find_raw_dir "${WORK}")" ]; then
  echo "[1/3] reusing previously extracted T-Drive in ${WORK}"
  RAW_DIR="$(find_raw_dir "${WORK}")"
elif command -v kaggle >/dev/null 2>&1 && [ -f "${HOME}/.kaggle/kaggle.json" ]; then
  echo "[1/3] downloading T-Drive via Kaggle (arashnic/tdriver) ..."
  if kaggle datasets download -d arashnic/tdriver -p "${WORK}" --unzip; then
    RAW_DIR="$(find_raw_dir "${WORK}")"
  fi
fi
elapsed=$(( $(date +%s) - start ))

if [ -z "${RAW_DIR}" ]; then
  record download MISSING "${elapsed}"
  cat <<EOF

------------------------------------------------------------
[1/3] No T-Drive raw data found and no usable downloader.

T-Drive needs one credential to fetch. Pick ONE, then re-run this script:

  (a) Kaggle CLI (recommended, fully scriptable):
        pip install kaggle
        # put your kaggle.json token in ~/.kaggle/kaggle.json  (chmod 600)
        bash scripts/setup_beijing_dataset.sh

  (b) Manual download, then point this script at the file/dir:
        # grab the T-Drive sample zip from
        #   https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/
        #   or https://www.kaggle.com/datasets/arashnic/tdriver
        TDRIVE_ZIP=/path/to/archive.zip bash scripts/setup_beijing_dataset.sh
        # or, if already unzipped:
        TDRIVE_RAW=/path/to/taxi_log_2008_by_id bash scripts/setup_beijing_dataset.sh

Ingest + generate are cached, so re-running only does the remaining work.
------------------------------------------------------------
EOF
  echo "Partial summary: ${SUMMARY}"
  exit 2
fi
record download ok "${elapsed}"
echo "[1/3] raw T-Drive dir: ${RAW_DIR}"

# ----------------------------------------------------------------------------
# Stage 2: segment GPS -> trips parquet  (skipped if cache present)
# ----------------------------------------------------------------------------
start=$(date +%s)
if [ -f "${RIDES_PARQUET}" ]; then
  echo "[2/3] reusing cached rides parquet ${RIDES_PARQUET}"
  rc=0
else
  echo "[2/3] segmenting T-Drive trips -> ${RIDES_PARQUET}"
  BBOX_ARG=""; [ -n "${BBOX}" ] && BBOX_ARG="--bbox ${BBOX}"
  # shellcheck disable=SC2086
  python -m ars.data.ingest_tdrive \
    --raw-dir "${RAW_DIR}" \
    --out "${RIDES_PARQUET}" \
    --max-taxis "${MAX_TAXIS}" ${BBOX_ARG}
  rc=$?
fi
elapsed=$(( $(date +%s) - start ))
[ "${rc}" -eq 0 ] && record ingest ok "${elapsed}" || { record ingest FAILED "${elapsed}"; echo "ingest failed"; exit 1; }

# ----------------------------------------------------------------------------
# Stage 3: trips -> grid dataset folder(s)
# ----------------------------------------------------------------------------
start=$(date +%s)
echo "[3/3] generating grid dataset (grids: ${GRID_SIZES}, agents: ${AGENTS})"
# 'auto' polygon so a custom --bbox in stage 2 is honoured automatically.
POLY_MODE="city"; [ -n "${BBOX}" ] && POLY_MODE="auto"
# shellcheck disable=SC2086
python -m ars.data.generate_grid_dataset \
  --city beijing \
  --rides-source "${RIDES_PARQUET}" \
  --polygon "${POLY_MODE}" \
  --grid-sizes ${GRID_SIZES} \
  --num-agents "${AGENTS}" \
  --out-template "${OUT_TEMPLATE}"
rc=$?
elapsed=$(( $(date +%s) - start ))
[ "${rc}" -eq 0 ] && record generate ok "${elapsed}" || { record generate FAILED "${elapsed}"; echo "generate failed"; exit 1; }

echo ""
echo "============================================================"
echo "Beijing dataset ready."
echo "Timing summary: ${SUMMARY}"
column -t -s, "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
TOTAL_S=$(awk -F, 'NR>1 {s+=$3} END {print s+0}' "${SUMMARY}")
echo "Total wall-clock: $(hms ${TOTAL_S}) (${TOTAL_S}s)"
echo "------------------------------------------------------------"
echo "Folders written under dataset/ :"
for g in ${GRID_SIZES}; do
  folder=$(python - "$OUT_TEMPLATE" "$AGENTS" "$g" <<'PY'
import sys
print(sys.argv[1].format(num_agents=int(sys.argv[2]), grid=int(sys.argv[3])))
PY
)
  echo "  dataset/${folder}   (--dataset ${folder} --grid-size ${g})"
done
echo ""
echo "Train ORACLE on Beijing, e.g.:"
echo "  AGENT_COUNTS=${AGENTS} GRID=$(echo ${GRID_SIZES} | awk '{print $1}') \\"
echo "    bash scripts/run_training.sh   # after pointing DATASET to beijing_* (see note)"
echo "============================================================"
