# Altruistic Ride-Sharing

Research codebase for simulating and evaluating altruistic ride-sharing using multi-agent RL (MADDPG) and a PSO-based baseline.

This repository contains training, evaluation, and simulation code used to study cooperative / altruistic behaviour in ride-sharing. It includes a MADDPG implementation, a PSO baseline trainer, dataset examples, and utilities for running experiments.

## Highlights
- Multi-Agent Deep Deterministic Policy Gradient (MADDPG) training pipeline
- Particle Swarm Optimization (PSO) baseline implementation and evaluator
- Simulation runner for comparing sharing vs no-sharing outcomes

## Table of contents
- Quick start
- Datasets
- Typical workflows (train / baseline / simulate)
- Key files and directories

## Quick start

1. Clone this repository and change into the project directory:

```bash
git clone https://github.com/AltruisticRideSharing/AltruisticRideSharing.git
cd AltruisticRideSharing
```

2. Create an anaconda environment and install dependencies:

```bash
conda create -n ars python=3.10
conda activate ars
pip install -r requirements.txt
```

3. Verify you have example datasets under `dataset/` (e.g. `dataset/100_agents/`), else you can create them using the python notebook provided.

## Dataset layout
Example datasets are located in the `dataset/` folder. Each dataset folder (for example `100_agents`) contains:

- `fixed_positions.npy` — initial agent positions
- `fixed_destinations.npy` — agent destinations
- `weight_matrix.npy` / `time_weight_matrix.npy` — pairwise weights / travel time matrices

You can add other datasets by following the same structure. The code expects the dataset folder name (e.g. `100_agents`) to be passed to the CLI when required.

## Typical workflows

1) Train MADDPG (multi-agent RL)

```bash
python main.py --dataset 100_agents --days 10 --num-episodes 100
```

Key training flags are defined in `main.py`'s `parse_args`. Models and checkpoints are saved to the run/log directory you specify.

2) Run PSO baseline experiments

```bash
python main_pso.py --num-agents 100 --days 10 --pso-particles 50 --pso-iterations 80 --alpha 0.4 --verbose
```

PSO-specific logic and result-export is implemented in `models/pso/baseline_trainer.py` and related PSO modules.

3) Run full simulations

```bash
python simulation.py --num-agents 100 --days 100 --dataset 100_agents
```

There are helper scripts to run batch experiments:

- `run_all_simulations.sh` — runs configured MADDPG experiments
- `run_all_pso_simulations.sh` — runs PSO baselines in batch

Make them executable and run them when needed:

```bash
chmod +x run_all_pso_simulations.sh
./run_all_pso_simulations.sh
```

## Key files and directories

- `main.py` — MADDPG training entry point
- `main_pso.py` — PSO baseline entry point
- `simulation.py` — simulation runner and data export utilities
- `models/maddpg.py` & `models/actor.py` / `models/critic.py` — MADDPG implementation
- `models/pso/` — PSO trainer, particle and evaluator implementation
- `utils/` — helper utilities (replay buffers, TF helpers, device helpers)
- `dataset/` — example datasets and dataset creation helpers
