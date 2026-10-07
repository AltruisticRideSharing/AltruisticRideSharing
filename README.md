# Altruistic Ride-Sharing (ARS)

Research codebase for **Altruistic Ride Sharing (ARS)**, a decentralized peer-to-peer
mobility framework in which commuters alternate between driver and rider roles using
non-monetary *altruism points*, with rider selection learned by multi-agent
reinforcement learning.

This repository contains the official implementation for the paper:
**"Altruistic Ride Sharing: A Framework for Fair and Sustainable Urban Mobility via Peer-to-Peer Incentives"**.

The code for the earlier version of the paper is on the [`v1`](https://github.com/AltruisticRideSharing/AltruisticRideSharing/tree/v1) branch.

## Methods

The paper evaluates ORACLE against three learning baselines and two optimization
baselines. They map to the code as follows:

| Method | What it is | Entry point |
|--------|-----------|-------------|
| **ORACLE** (proposed) | One shared actor and one ego-conditioned attention critic for all agents | `python -m ars.train_oracle --model attention` |
| **DeepPool** | Shared DQN over the same observation, action mask, and reward | `python -m ars.train_dqn` |
| **MADDPG** | Independent actor and concatenating centralized critic per agent | `python -m ars.train --model maddpg` |
| **MAAC** | Independent actor and attention critic per agent | `python -m ars.train --model attention` |
| **PSO** | Particle-swarm rider-to-driver assignment, solved once per day | `python -m ars.baselines.run_pso` |
| **OR-Tools** | Capacity-constrained integer assignment (CBC), solved once per day | `python -m ars.baselines.run_or_tools` |

MAAC and ORACLE share the attention architecture (`ars.models.attention.AttentionTrainer`).
`ars.train_oracle` shares one network across all agents (ORACLE), whereas `ars.train`
gives each agent its own networks (MAAC and MADDPG).

## Installation

Installation uses `scripts/setup.sh`, which creates a conda environment for the
Python 3.10 interpreter and installs the project and its pinned dependencies into it
with [`uv`](https://docs.astral.sh/uv/) (`pyproject.toml` / `uv.lock`). Conda must be
on your `PATH`.

```bash
git clone https://github.com/AltruisticRideSharing/AltruisticRideSharing.git
cd AltruisticRideSharing
bash scripts/setup.sh            # or: ENV_NAME=myenv bash scripts/setup.sh
conda activate ars
```

## Datasets

The datasets used in the paper are included under `dataset/`. Each folder holds the
four files the environment loads: `weight_matrix.npy` and `time_weight_matrix.npy`
(directional road distances in km and travel times in minutes between neighboring
cells), `fixed_positions.npy`, and `fixed_destinations.npy`.

| City | 15×15 grid | 30×30 grid | 45×45 grid |
|------|-----------|-----------|-----------|
| New York City | `<N>_agents` | `<N>_agents_grid30` | `<N>_agents_grid45` |
| Beijing | `beijing_<N>_agents` | `beijing_<N>_agents_grid30` | `beijing_<N>_agents_grid45` |

with `<N>` ∈ {100, 150, 200}.

- **New York City**: NYC TLC Yellow Taxi trips (January 2016) in a Manhattan corridor.
- **Beijing**: Microsoft T-Drive taxi trajectories (February 2008) in a compact core
  of about 5×5 km.

Inter-cell distances come from the OpenStreetMap road network, with travel times at
25 km/h. To regenerate datasets:

```bash
# New York City, one agent count per call: writes 100_agents_grid30/ and 100_agents_grid45/
python -m ars.data.generate_grid_dataset --city nyc --grid-sizes 30 45 --num-agents 100
# the 15x15 folder (100_agents/)
python -m ars.data.generate_grid_dataset --city nyc --grid-sizes 15 --num-agents 100 --out-template 100_agents

# Beijing: download T-Drive, segment trips, and grid them
bash scripts/setup_beijing_dataset.sh
bash scripts/beijing_generate_all_datasets.sh   # all agent counts x grid sizes
```

The generator caches the filtered rides and the OSM road graph under `ars/data/cache/`.
The perception field scales with the grid (`n ≈ grid/3`, snapped to odd: 15→5,
30→11, 45→15).

## Reproducing the paper

Each script is configured through environment variables (agent counts, grids,
cities, training length) documented at the top of the file.

| Experiment | Script |
|------------|--------|
| ORACLE, New York City, 15×15 | `scripts/run_training.sh` |
| ORACLE, New York City, 30×30 and 45×45 (tuned hyperparameters) | `scripts/nyc_rerun_oracle_pso_grid3045.sh` |
| All Beijing methods (ORACLE, MADDPG, MAAC, PSO), train and evaluate | `scripts/beijing_run_all_baselines.sh` |
| MADDPG and MAAC, both cities, train and evaluate | `scripts/train_maddpg_maac_all.sh` |
| DeepPool, train | `scripts/run_training_dqn.sh` |
| DeepPool, evaluate | `scripts/run_evaluation_dqn_nyc.sh`, `scripts/run_evaluation_dqn_beijing.sh` |
| PSO, all conditions | `scripts/run_all_pso.sh` |
| OR-Tools, all conditions | `scripts/run_all_or_tools.sh` |
| Evaluate trained ORACLE / MADDPG / MAAC under all four conditions | `scripts/run_all_simulations.sh` |
| Bayesian hyperparameter search (Optuna TPE) | `scripts/bayesian_hpo_oracle.py`, `scripts/bayesian_hpo_dqn.py` |
| Sensitivity analysis of the design constants | `scripts/run_ablation_sensitivity.sh`, then `scripts/collect_ablation.py` |

The four evaluation conditions are uniform or Gaussian altruism, each with a fixed or a
birth-death population. Each evaluation run simulates 100 days.

## Training

ORACLE, as used in the paper (200 episodes per simulated day over 30 days, 10 parallel
environments, TAGS exploration):

```bash
python -m ars.train_oracle --model attention \
  --dataset 100_agents_grid45 --grid-size 45 --num-agents 100 \
  --days 30 --num-episodes 200 --num-envs 10 --exploration tags
```

Training keeps the checkpoint with the best held-out reward as
`best_weights_attention_<N>_tags.pkl` next to `final_weights_attention_<N>_tags.pkl`.

Learning baselines:

```bash
python -m ars.train_dqn --dataset 100_agents --grid-size 15 --num-agents 100 --days 30 --num-episodes 200
python -m ars.train --model maddpg    --dataset 100_agents --num-agents 100 --days 10 --num-episodes 100 --num-envs 25
python -m ars.train --model attention --dataset 100_agents --num-agents 100 --days 10 --num-episodes 100 --num-envs 25
```

Optimization baselines (no training):

```bash
python -m ars.baselines.run_pso --num-agents 100 --dataset 100_agents --grid-size 15 \
  --days 100 --pso-particles 50 --pso-iterations 80 --alpha 0.4
python -m ars.baselines.run_or_tools --num-agents 100 --dataset 100_agents --grid-size 15 --days 100
```

## Evaluation

```bash
python -m ars.evaluate_oracle --model-type attention \
  --dataset 100_agents_grid45 --grid-size 45 --num-agents 100 --days 100 \
  --altruism-distribution uniform \
  --save-dir savedagents/oracle/grid45/100agents --restore \
  --weights-file best_weights_attention_100_tags.pkl
# add --enable-birth-death for the birth-death population;
# MADDPG/MAAC: python -m ars.evaluate, DeepPool: python -m ars.evaluate_dqn
```

Each run writes a `results_summary.txt` with the paper's metrics:

| Key | Metric |
|-----|--------|
| `distance_reduction_percent` | Distance reduction relative to no sharing (%) |
| `mean_vehicle_utilization` | Commuters per driver's vehicle |
| `traffic_reduction_percent` | Traffic density reduction (%) |
| `avg_detour_factor` | Detour factor |
| `avg_trip_time` | Average trip time (min) |
| `mean_acceptance_rate` | Rider acceptance rate (%) |
| `gini_combined_benefits` | Benefit Gini |
| `reintegration_score` | Reintegration score, 0 to 100 (birth-death only) |

## Key parameters (`ars.train_oracle`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--grid-size` | 15 | Square grid side length |
| `--perception-field` | auto | Egocentric field n (odd); defaults to about grid/3 |
| `--days` | 1 | Simulated training days |
| `--num-episodes` | 100 | Episodes per simulated day, split across environments |
| `--num-envs` | 4 | Parallel environments |
| `--alpha-r` | 0.4 | Reward trade-off between rider benefit and driver detour (0.6 for Beijing) |
| `--lr` / `--actor-lr` | 1e-4 / 0.1 | Critic learning rate / actor learning-rate multiplier |
| `--gamma` | 0.99 | Discount factor |
| `--tau` | 0.01 | Soft target-update rate |
| `--num-heads` / `--attn_temp` | 4 / 1.0 | Attention heads / temperature |
| `--exploration` | `tags` | `tags`, `gst`, or `epsilon_greedy` |
| `--lr-schedule` | `cosine` | `constant`, `cosine`, or `linear` |
| `--wandb-mode` | `online` | `online`, `offline`, or `disabled` |

## Monitoring (Weights & Biases)

Training logs to [Weights & Biases](https://wandb.ai) under the `altruistic-ridesharing`
project. Run `wandb login` once, or pass `--wandb-mode offline` (sync later with
`wandb sync`) or `--wandb-mode disabled`.

## Project structure

```
AltruisticRideSharing/
├── pyproject.toml / uv.lock          # dependencies (installed by scripts/setup.sh)
├── ars/                              # installable package
│   ├── train_oracle.py               # ORACLE (parameter sharing)
│   ├── train.py                      # MADDPG / MAAC (independent agents)
│   ├── train_dqn.py                  # DeepPool (shared DQN)
│   ├── evaluate_oracle.py / evaluate.py / evaluate_dqn.py  # evaluation runners
│   ├── metrics.py / metrics_dqn.py   # evaluation metrics
│   ├── env/                          # environment and agents
│   ├── models/                       # networks and trainers
│   │   ├── attention.py              # AttentionTrainer (ORACLE / MAAC)
│   │   ├── maddpg.py                 # MADDPGTrainer (unshared)
│   │   ├── actor.py / critic.py / attention_critic.py
│   │   ├── deeppool/                 # DQN baseline
│   │   └── pso/ and or_tools/        # optimization baselines
│   ├── baselines/                    # PSO / OR-Tools entry points
│   ├── utils/                        # parallel envs, replay buffers, estimators
│   └── data/                         # TLC / T-Drive ingestion and grid dataset generator
├── scripts/                          # setup, training, evaluation, HPO, sensitivity
└── dataset/                          # New York City and Beijing datasets
```

## Citation

```bibtex
@misc{singh2025altruisticridesharingcommunitydriven,
      title={Altruistic Ride Sharing: A Community-Driven Approach to Short-Distance Mobility},
      author={Divyanshu Singh and Ashman Mehra and Snehanshu Saha and Santonu Sarkar},
      year={2025},
      eprint={2510.13227},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2510.13227},
}
```

## License

MIT License
