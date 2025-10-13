#!/bin/bash

echo "Starting all simulation experiments..."

AGENT_COUNTS=(100 150 200)
DISTRIBUTIONS=("uniform" "gaussian")
BIRTH_DEATH_MODES=("fixed" "birthdeath")

for AGENTS in "${AGENT_COUNTS[@]}"; do
  for DIST in "${DISTRIBUTIONS[@]}"; do
    for MODE in "${BIRTH_DEATH_MODES[@]}"; do

      # Set parameters for each mode
      if [ "$MODE" == "fixed" ]; then
        ENABLE_BD=""
        INIT_AGENTS=$AGENTS
        MODE_DESC="fixed"
      else
        ENABLE_BD="--enable-birth-death"
        INIT_AGENTS=40
        MODE_DESC="birthdeath"
      fi

      # Set distribution-specific parameters
      if [ "$DIST" == "uniform" ]; then
        DIST_ARGS="--altruism-distribution uniform --altruism-mean 0.5"
        DIST_DESC="uniform"
      else
        DIST_ARGS="--altruism-distribution gaussian --altruism-mean 0.5 --altruism-std 0.2"
        DIST_DESC="gaussian"
      fi

      # Set dataset according to agent count
      DATASET_ARG="--dataset ${AGENTS}_agents"

      echo "=============================================="
      echo "Running: Agents=$AGENTS, Mode=$MODE_DESC, Distribution=$DIST_DESC, Dataset=${AGENTS}_agents"
      echo "=============================================="

      python simulation.py \
        --num-agents $AGENTS \
        --initial-active-agents $INIT_AGENTS \
        --days 100 \
        $ENABLE_BD \
        $DIST_ARGS \
        $DATASET_ARG

      echo "Completed: Agents=$AGENTS, Mode=$MODE_DESC, Distribution=$DIST_DESC"
      echo
    done
  done
done

echo "All simulation experiments completed!"