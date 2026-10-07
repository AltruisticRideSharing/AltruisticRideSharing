"""
Parallel to train_oracle.py. Uses the identical environment interface,
observation space, action masking, and reward signal as ORACLE. Only the
policy and learning mechanism differ.

Usage (mirrors train_oracle.py CLI):
    python train_dqn.py --num-agents 100 --days 10 --num-episodes 100
    python train_dqn.py --num-agents 100 --days 10 --restore --save-dir savedagents
"""

import argparse
import os
import math
import random
import pickle
import numpy as np
import wandb
from datetime import datetime
from ars.env.env import Env
from ars.models.deeppool.dqn_baseline import DQNBaseline


# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser("DQN Baseline for ARS")

    # Environment — identical to train_oracle.py
    parser.add_argument("--max-episode-len",     type=int,   default=1000)
    parser.add_argument("--num-episodes",        type=int,   default=100)
    parser.add_argument("--batch-size",          type=int,   default=256)
    parser.add_argument("--dataset",             type=str,   default="100_agents")
    parser.add_argument("--num-agents",          type=int,   default=100)
    parser.add_argument("--initial-active-agents", type=int, default=100)
    parser.add_argument("--grid-size",           type=int,   default=15)
    parser.add_argument("--perception-field",    type=int,   default=None)
    parser.add_argument("--days",                type=int,   default=1)

    # DQN-specific
    parser.add_argument("--lr",                  type=float, default=1e-4)
    parser.add_argument("--gamma",               type=float, default=0.99)
    parser.add_argument("--tau",                 type=float, default=0.01,
                        help="Polyak averaging rate for target net (matches ORACLE default)")
    parser.add_argument("--buffer-size",         type=int,   default=50_000)
    parser.add_argument("--hidden1",             type=int,   default=128,
                        help="First hidden layer (intentionally smaller than ORACLE's 256)")
    parser.add_argument("--hidden2",             type=int,   default=256,
                        help="Second hidden layer (intentionally smaller than ORACLE's 512)")

    # Exploration
    parser.add_argument("--epsilon-start",       type=float, default=1.0)
    parser.add_argument("--epsilon-min",         type=float, default=0.01)
    parser.add_argument("--epsilon-decay-episodes", type=float, default=0.8,
                        help="Fraction of total episodes over which to decay epsilon")

    # Save / restore
    parser.add_argument("--save-dir",            type=str,   default="savedagents")
    parser.add_argument("--restore",             action="store_true", default=False)

    # Wandb — identical to train_oracle.py
    parser.add_argument("--wandb-project",       type=str,   default="altruistic-ridesharing")
    parser.add_argument("--wandb-entity",        type=str,   default=None)
    parser.add_argument("--wandb-mode",          type=str,   default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-run-name",      type=str,   default=None)

    # Tune mode
    parser.add_argument("--tune-mode",           type=str,   default=None,
                        help="If set, write final metrics as JSON to this path "
                             "(consumed by the Optuna HPO driver)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_save_dir(arglist):
    """Mirrors build_save_dir() in train_oracle.py."""
    path = os.path.join(
        arglist.save_dir, "dqn",
        f"grid{arglist.grid_size}",
        f"{arglist.num_agents}agents"
    )
    os.makedirs(path, exist_ok=True)
    return path


def init_wandb(arglist):
    run_name = arglist.wandb_run_name or (
        f"dqn_grid{arglist.grid_size}_{arglist.num_agents}agents"
    )
    return wandb.init(
        project=arglist.wandb_project,
        entity=arglist.wandb_entity,
        mode=arglist.wandb_mode,
        name=run_name,
        config=vars(arglist),
    )


def has_real_choice(state_i, num_agents):
    """
    True if this driver has at least one rider visible (not just decline).
    Mirrors the 'store' condition in train_oracle.py.
    """
    grid = state_i[3:].astype(np.int32)
    return (grid >= 0).any()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(arglist):
    # --- Environment ---
    env = Env(
        dataset_folder=arglist.dataset,
        numAgents=arglist.num_agents,
        initial_active_agents=arglist.initial_active_agents,
        height=arglist.grid_size,
        width=arglist.grid_size,
        perception_field=arglist.perception_field,
    )

    obs_dim = env.observation_space().nvec.shape[0]   # 28
    num_agents = env.numAgents                         # 100

    print(f"obs_dim={obs_dim}, num_agents={num_agents}, "
          f"action_dim={num_agents + 1}")

    # --- Total episodes for epsilon decay ---
    total_episodes = arglist.days * arglist.num_episodes
    epsilon_decay_episodes = int(total_episodes * arglist.epsilon_decay_episodes)

    # --- DQN agent (one shared network for all drivers) ---
    agent = DQNBaseline(
        obs_dim=obs_dim,
        num_agents=num_agents,
        num_agents_grid=arglist.grid_size,
        hidden1=arglist.hidden1,
        hidden2=arglist.hidden2,
        lr=arglist.lr,
        gamma=arglist.gamma,
        tau=arglist.tau,
        epsilon_start=arglist.epsilon_start,
        epsilon_end=arglist.epsilon_min,
        epsilon_decay_episodes=epsilon_decay_episodes,
        buffer_size=arglist.buffer_size,
        batch_size=arglist.batch_size,
    )

    save_dir = build_save_dir(arglist)
    print(f"Checkpoints -> {save_dir}")

    if arglist.restore:
        agent.load(save_dir, tag="final")

    wandb_run = init_wandb(arglist)
    global_episode = 0
    all_rewards = []
    eval_rewards = []

    # --- Outer loop: days ---
    for day in range(arglist.days):
        env.reset_day(day + 1, enable_birth_death=False)
        env.current_day = day + 1
        print(f"\n=== Day {day + 1}/{arglist.days} | "
              f"epsilon={agent.epsilon:.4f} ===")

        episodes_per_day = arglist.num_episodes
        is_last_episode_eval = True   # final episode of each day = evaluation

        # --- Inner loop: episodes ---
        for episode in range(episodes_per_day):
            is_eval = (episode == episodes_per_day - 1) and is_last_episode_eval

            states, infos = env.reset()
            # states: (numAgents, 28)  infos: (numAgents,) — 1=driver, 0=other

            done = False
            episode_rewards = np.zeros(num_agents, dtype=np.float32)
            steps = 0
            metrics = {"loss": [], "q_mean": [], "q_max": []}

            # --- Step loop ---
            while not done:
                actions = agent.select_actions(states, infos, evaluation=is_eval)

                next_states, rewards, done, next_infos = env.step(actions)
                # rewards: (numAgents,)  done: scalar int  next_infos: (numAgents,)

                episode_rewards += rewards

                # Store transitions (only when a driver has a real choice)
                if not is_eval:
                    agent.store_transitions(
                        states=states,
                        actions=actions,
                        rewards=rewards,
                        next_states=next_states,
                        done=bool(done),
                        infos=infos,
                    )

                    # Training update
                    update_metrics = agent.update()
                    for k, v in update_metrics.items():
                        metrics[k].append(v)

                states = next_states
                infos = next_infos
                steps += 1

            # --- End of episode ---
            total_ep_reward = float(episode_rewards.sum())
            mean_ep_reward  = float(episode_rewards.mean())

            if not is_eval:
                agent.decay_epsilon()

            # Logging
            log = {
                "progress/day":          day + 1,
                "progress/episode":      episode,
                "progress/global_ep":    global_episode,
                "progress/steps":        steps,
                "parameters/epsilon":    agent.epsilon,
                "parameters/buffer_size": len(agent.buffer),
            }
            if is_eval:
                log["val_rewards/total"]  = total_ep_reward
                log["val_rewards/mean"]   = mean_ep_reward
                eval_rewards.append(total_ep_reward)
            else:
                log["rewards/total"] = total_ep_reward
                log["rewards/mean"]  = mean_ep_reward
                all_rewards.append(total_ep_reward)

            for k, v in metrics.items():
                if v:
                    log[f"training/{k}"] = float(np.mean(v))

            wandb.log(log, step=global_episode)

            mode = "EVAL" if is_eval else "train"
            print(f"  [{mode}] ep={episode:3d} | "
                  f"reward={total_ep_reward:7.3f} | "
                  f"steps={steps:4d} | "
                  f"ε={agent.epsilon:.4f} | "
                  f"buf={len(agent.buffer):6d} | "
                  f"loss={np.mean(metrics['loss']) if metrics['loss'] else 0:.4f}")

            global_episode += 1

        # Checkpoint every 10 days
        if (day + 1) % 10 == 0:
            agent.save(save_dir, tag=f"{day + 1}_days")

    # Final save
    agent.save(save_dir, tag=f"{arglist.days}_days")
    agent.save(save_dir, tag="final")
    print(f"\nTraining complete. Final checkpoint saved to {save_dir}")

    if arglist.tune_mode:
        import json
        tune_metrics = {
            "mean_eval_reward":  float(np.mean(eval_rewards)) if eval_rewards else 0.0,
            "last_eval_reward":  float(eval_rewards[-1]) if eval_rewards else 0.0,
            "mean_train_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
            "final_epsilon":     float(agent.epsilon),
            "final_buffer_size": int(len(agent.buffer)),
        }
        with open(arglist.tune_mode, "w") as f:
            json.dump(tune_metrics, f)

    if wandb_run:
        wandb_run.finish()


def main():
    arglist = parse_args()
    train(arglist)


if __name__ == "__main__":
    main()
