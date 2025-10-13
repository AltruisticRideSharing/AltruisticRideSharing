#!/bin/bash
# filepath: /home/divyanshu/Documents/AltruisticRideSharing/run_all_pso_simulations.sh

echo "Starting all PSO baseline experiments..."

AGENT_COUNTS=(100 150 200)
DISTRIBUTIONS=("uniform" "gaussian")
AGENT_DYNAMICS=("fixed" "birth_death")

for AGENTS in "${AGENT_COUNTS[@]}"; do
  for DIST in "${DISTRIBUTIONS[@]}"; do
    for MODE in "${AGENT_DYNAMICS[@]}"; do

      # Set parameters for each mode
      if [ "$MODE" == "fixed" ]; then
        INIT_AGENTS=$AGENTS
        MODE_DESC="fixed"
      else
        INIT_AGENTS=40
        MODE_DESC="birth_death"
      fi

      # Set distribution-specific parameters
      if [ "$DIST" == "uniform" ]; then
        DIST_DESC="uniform"
      else
        DIST_DESC="gaussian"
      fi

      # Set dataset according to agent count
      DATASET_ARG="${AGENTS}_agents"

      # Create experiment name
      EXPERIMENT_NAME="pso_${AGENTS}agents_${DIST_DESC}_${MODE_DESC}_$(date +%Y%m%d_%H%M%S)"

      echo "=============================================="
      echo "Running PSO: Agents=$AGENTS, Mode=$MODE_DESC, Distribution=$DIST_DESC, Dataset=$DATASET_ARG"
      echo "Experiment: $EXPERIMENT_NAME"
      echo "=============================================="

      python main_pso.py \
        --num-agents $AGENTS \
        --initial-active-agents $INIT_AGENTS \
        --days 100 \
        --agent-dynamics $MODE \
        --altruism-dist $DIST \
        --dataset $DATASET_ARG \
        --experiment-name $EXPERIMENT_NAME \
        --verbose \
        --pso-particles 50 \
        --pso-iterations 80 \
        --alpha 0.4

      echo "Completed PSO: Agents=$AGENTS, Mode=$MODE_DESC, Distribution=$DIST_DESC"
      echo "Results saved as: $EXPERIMENT_NAME"
      echo
    done
  done
done

echo "All PSO baseline experiments completed!"
echo "Results saved in: ./results/pso/"
echo "TensorBoard logs in: ./runs/pso/"
echo
echo "To view results with TensorBoard:"
echo "tensorboard --logdir ./runs/pso/"