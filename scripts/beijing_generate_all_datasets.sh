#!/bin/bash
# Generate all Beijing grid datasets for the multi-scale experiments:
#   agents in {100, 150, 200}  x  grids in {15, 30, 45}
#
# Produces grid-suffixed folders so every (agents, grid) pair is a distinct,
# correctly-sized dataset:
#   dataset/beijing_<A>_agents_grid<G>/   (weight_matrix is <G>x<G>x4)
#
# IMPORTANT -- Beijing trip-distance band: trips are sampled in the 2.5-5.5 km
# band (NOT the generator's 1.5-5.0 km default, and NOT 3.0-6.0 km). This is a
# tuned sweet spot:
#   * 1.5-5.0 km (default): trips too short -> ~6% PSO savings, weak RL signal.
#   * 3.0-6.0 km: trips ~15.1 cells on 15x15 -> TOO LONG; agents spend the whole
#     episode in transit, ORACLE never learns (val stuck at -2.0 for 30 days).
#   * 2.5-5.5 km: trips ~13.0 cells -> NYC-like learning AND 16.1% PSO savings.
# Lesson (memory beijing-tdrive-pipeline "ROOT CAUSE FOUND"): trip_cells must
# stay <=13 on grid15. Longer trips give more PSO headroom but break RL.
#
# Reuses the cached T-Drive rides parquet and OSM graph
# (ars/data/cache/beijing_filtered_rides.parquet, beijing_osm_graph.pkl), so no
# re-download/re-segmentation happens -- only per-grid snapping runs. The 45x45
# snap is the slow stage; grid caches are shared across agent counts.
#
# Usage:
#   bash scripts/beijing_generate_all_datasets.sh
#   AGENT_COUNTS="100 200" GRID_SIZES="15 30" bash scripts/beijing_generate_all_datasets.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

AGENT_COUNTS="${AGENT_COUNTS:-100 150 200}"
GRID_SIZES="${GRID_SIZES:-15 30 45}"
DIST_MIN="${DIST_MIN:-2.5}"
DIST_MAX="${DIST_MAX:-5.5}"
RIDES_PARQUET="${REPO_ROOT}/ars/data/cache/beijing_filtered_rides.parquet"

if [ ! -f "${RIDES_PARQUET}" ]; then
  echo "!! ${RIDES_PARQUET} missing. Run scripts/setup_beijing_dataset.sh first"
  echo "   (it ingests T-Drive -> the rides parquet this script snaps to grids)."
  exit 1
fi

echo "Beijing dataset generation"
echo "  agents        : ${AGENT_COUNTS}"
echo "  grids         : ${GRID_SIZES}"
echo "  distance band : ${DIST_MIN}-${DIST_MAX} km  (Beijing-tuned, not the 1.5-5.0 default)"
echo "  folders       : dataset/beijing_<A>_agents_grid<G>"
echo

for A in ${AGENT_COUNTS}; do
  echo "==================================================================="
  echo "  agents=${A}  grids=${GRID_SIZES}  band=${DIST_MIN}-${DIST_MAX}km"
  echo "==================================================================="
  # Call the generator directly (rides + OSM graph are cached) so we can pass
  # the Beijing distance band, which setup_beijing_dataset.sh does not expose.
  python -m ars.data.generate_grid_dataset \
    --city beijing \
    --rides-source "${RIDES_PARQUET}" \
    --polygon city \
    --grid-sizes ${GRID_SIZES} \
    --num-agents "${A}" \
    --distance-min "${DIST_MIN}" \
    --distance-max "${DIST_MAX}" \
    --out-template 'beijing_{num_agents}_agents_grid{grid}'
done

echo
echo "=== done. generated datasets: ==="
ls -d dataset/beijing_*_agents_grid* 2>/dev/null
