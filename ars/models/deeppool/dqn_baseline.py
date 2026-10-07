"""
DQN Baseline for Altruistic Ride Sharing (ARS).

Drop-in replacement for the ORACLE (AttentionTrainer) policy in train_oracle.py.
Uses the identical environment interface, observation space (28-dim), action space
(numAgents + 1), action masking logic, and reward signal as ORACLE — only the
learning mechanism changes.

Architecture decisions that mirror ORACLE for a fair comparison:
  - Shared parameters: one network used by all drivers (matches ORACLE's
    set_shared_networks pattern)
  - Same 28-dim observation input
  - Same action masking (riders visible in 5x5 local field + decline)
  - Same ARS reward signal (no changes to env dynamics or altruism updates)
  - Standard uniform replay buffer (not PER — PER is ORACLE's advantage,
    keeping it would confound the comparison)

Usage: see train_dqn.py for the full training loop.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import os
import pickle


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class DQNNetwork(nn.Module):
    """
    MLP mapping the 28-dim ARS observation to Q-values over the action space.

    Observation layout (from env.py get_observation):
        [0]    delta_x + width      (relative destination x, normalized positive)
        [1]    delta_y + height     (relative destination y, normalized positive)
        [2]    role_flag            (1=driver, 0=rider)
        [3:28] nearby_grid flat     (25 cells; each = agent index 0..N-1 or -1)

    The grid cells hold raw agent indices which are high-cardinality integers.
    We normalize them to [0, 1] by dividing by numAgents before feeding into
    the linear layer. -1 (empty cell) is mapped to 0 after normalization.

    Hidden layer sizes are kept intentionally smaller than ORACLE's actor
    (256->512) to avoid giving DQN a capacity advantage.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden1: int = 128, hidden2: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay Buffer (uniform, not PER — intentional baseline choice)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Standard uniform experience replay buffer.

    Stores transitions as (state, action, reward, next_state, done, info).
    'info' is the driver flag (1=driver) used to mask non-driver transitions
    during sampling, matching the masked_state approach in train_oracle.py.
    """

    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool, is_driver: int):
        self.buffer.append((state, action, reward, next_state, done, is_driver))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, is_drivers = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
            np.array(is_drivers,  dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN Baseline Trainer
# ---------------------------------------------------------------------------

class DQNBaseline:
    """
    Single shared-parameter DQN for all drivers in the ARS environment.

    Mirrors the ORACLE shared-parameter pattern:
        primary = DQNBaseline(...)
        all agents call primary.select_action(obs_i, valid_mask)
        training calls primary.update()

    Key design choices matching train_oracle.py:
        - Only stores transitions where at least one driver has a real choice
          (valid_actions > 1), matching the 'store' condition in train_oracle.py
        - Non-driver agents always output action = numAgents (decline)
        - Action masking applied before argmax / epsilon-greedy sampling
        - Soft target network updates (Polyak averaging) matching ORACLE's tau
    """

    def __init__(
        self,
        obs_dim: int,
        num_agents: int,
        num_agents_grid: int,     # grid_size (height or width) for obs normalization
        hidden1: int = 128,
        hidden2: int = 256,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.01,        # Polyak: same default as ORACLE's --tau
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_episodes: int = 80,   # episodes over which to decay
        buffer_size: int = 50_000,
        batch_size: int = 256,
        target_update_freq: int = 200,      # steps between hard target updates
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.num_agents = num_agents          # action_dim = num_agents + 1
        self.action_dim = num_agents + 1
        self.num_agents_grid = num_agents_grid
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / max(1, epsilon_decay_episodes)

        # Networks
        self.q_net = DQNNetwork(obs_dim, self.action_dim, hidden1, hidden2).to(self.device)
        self.target_net = DQNNetwork(obs_dim, self.action_dim, hidden1, hidden2).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=buffer_size)

        self.steps = 0
        self.episodes = 0

    # ------------------------------------------------------------------
    # Observation preprocessing
    # ------------------------------------------------------------------

    def _preprocess_obs(self, obs: np.ndarray) -> torch.Tensor:
        """
        Normalize the 28-dim observation for network input.

        Spatial coords [0:2]: already positive integers (0..2*grid_size).
            Normalize by dividing by 2*grid_size.
        Role flag [2]: already 0 or 1, no change needed.
        Grid cells [3:28]: agent indices 0..N-1, or -1 (empty).
            Map -1 -> 0.0, then divide by numAgents to get [0, 1].
        """
        obs = obs.astype(np.float32)
        # Spatial: obs[0] = delta_x + width (range 0..2*width), similarly obs[1]
        # width = height = num_agents_grid (the grid side length)
        obs[0] /= (2 * self.num_agents_grid)
        obs[1] /= (2 * self.num_agents_grid)
        # obs[2] is role flag — already {0, 1}
        # Grid cells
        grid = obs[3:]
        grid = np.where(grid == -1, 0.0, grid / self.num_agents)
        obs[3:] = grid
        return torch.tensor(obs, dtype=torch.float32, device=self.device)

    def _preprocess_obs_batch(self, obs_batch: np.ndarray) -> torch.Tensor:
        """Batch version of _preprocess_obs. obs_batch: (B, obs_dim)."""
        obs = obs_batch.astype(np.float32)
        obs[:, 0] /= (2 * self.num_agents_grid)
        obs[:, 1] /= (2 * self.num_agents_grid)
        grid = obs[:, 3:]
        grid = np.where(grid == -1, 0.0, grid / self.num_agents)
        obs[:, 3:] = grid
        return torch.tensor(obs, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Action mask construction (mirrors generate_action_masks in train_oracle.py)
    # ------------------------------------------------------------------

    @staticmethod
    def build_valid_mask(obs_i: np.ndarray, num_agents: int) -> np.ndarray:
        """
        Build a boolean valid-action mask from a single agent's observation.

        obs_i[3:] contains the 25-cell local perception grid.
        Valid actions = rider indices visible in the grid + numAgents (decline).
        Returns a bool array of shape (num_agents + 1,).
        """
        grid = obs_i[3:].astype(np.int32)                # shape (25,)
        rider_ids = grid[grid >= 0]                        # filter out -1 (empty cells)
        mask = np.zeros(num_agents + 1, dtype=bool)
        mask[rider_ids] = True
        mask[num_agents] = True                            # decline always valid
        return mask

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs_i: np.ndarray, is_driver: bool,
                      evaluation: bool = False) -> int:
        """
        Select an action for a single agent.

        Non-drivers always return numAgents (decline/no-op).
        Drivers use epsilon-greedy over masked Q-values.

        Args:
            obs_i:      28-dim observation for this agent
            is_driver:  True if this agent is a driver this episode
            evaluation: If True, act greedily (no exploration)

        Returns:
            Integer action in [0, numAgents]
        """
        if not is_driver:
            return self.num_agents   # riders / dropouts always decline

        valid_mask = self.build_valid_mask(obs_i, self.num_agents)
        valid_indices = np.where(valid_mask)[0]

        # Epsilon-greedy exploration
        if not evaluation and random.random() < self.epsilon:
            return int(np.random.choice(valid_indices))

        # Greedy: masked argmax of Q-values
        obs_t = self._preprocess_obs(obs_i).unsqueeze(0)   # (1, obs_dim)
        with torch.no_grad():
            q_values = self.q_net(obs_t).squeeze(0)         # (action_dim,)

        # Mask invalid actions to -inf before argmax
        mask_t = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)
        q_values[~mask_t] = float('-inf')
        return int(q_values.argmax().item())

    def select_actions(self, states: np.ndarray, infos: np.ndarray,
                       evaluation: bool = False) -> np.ndarray:
        """
        Select actions for all agents in one call.

        Mirrors the per-agent loop in train_oracle.py's while-not-done block.

        Args:
            states: (numAgents, obs_dim) observation array from env.reset/step
            infos:  (numAgents,) float array; 1.0 = driver, 0.0 = non-driver
            evaluation: greedy if True

        Returns:
            actions: (numAgents,) int array
        """
        actions = np.full(self.num_agents, self.num_agents, dtype=np.int32)

        driver_indices = np.where(infos == 1.0)[0]
        if len(driver_indices) == 0:
            return actions

        # Batch Q-value computation for all drivers at once (efficient)
        driver_obs = states[driver_indices]                  # (D, obs_dim)
        obs_t = self._preprocess_obs_batch(driver_obs)       # (D, obs_dim)

        with torch.no_grad():
            q_batch = self.q_net(obs_t)                      # (D, action_dim)

        for local_i, global_i in enumerate(driver_indices):
            obs_i = states[global_i]
            valid_mask = self.build_valid_mask(obs_i, self.num_agents)
            valid_indices = np.where(valid_mask)[0]

            if not evaluation and random.random() < self.epsilon:
                actions[global_i] = int(np.random.choice(valid_indices))
            else:
                q_vals = q_batch[local_i].clone()
                mask_t = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)
                q_vals[~mask_t] = float('-inf')
                actions[global_i] = int(q_vals.argmax().item())

        return actions

    # ------------------------------------------------------------------
    # Experience storage
    # ------------------------------------------------------------------

    def store_transitions(
        self,
        states: np.ndarray,       # (numAgents, obs_dim)
        actions: np.ndarray,      # (numAgents,)
        rewards: np.ndarray,      # (numAgents,)
        next_states: np.ndarray,  # (numAgents, obs_dim)
        done: bool,
        infos: np.ndarray,        # (numAgents,) current step driver flags
    ):
        """
        Store one transition per driver agent into the replay buffer.

        Mirrors the 'store' condition in train_oracle.py:
          - Only store if at least one driver has a real choice (>1 valid action,
            i.e., at least one rider visible in addition to decline).
          - Non-driver agents are skipped entirely.

        Args match the outputs of env.step() plus the actions array.
        """
        # Check if any driver has a real choice (mirrors train_oracle.py's 'store' flag)
        has_real_choice = False
        for i in range(self.num_agents):
            if infos[i] == 1.0:
                valid_mask = self.build_valid_mask(states[i], self.num_agents)
                # >1 means at least one real rider option besides decline
                if valid_mask.sum() > 1:
                    has_real_choice = True
                    break

        if not has_real_choice:
            return

        for i in range(self.num_agents):
            if infos[i] != 1.0:
                continue   # only store driver transitions
            self.buffer.add(
                state=states[i],
                action=int(actions[i]),
                reward=float(rewards[i]),
                next_state=next_states[i],
                done=done,
                is_driver=int(infos[i]),
            )

    # ------------------------------------------------------------------
    # Training update
    # ------------------------------------------------------------------

    def update(self) -> dict:
        """
        One gradient step on a sampled mini-batch.

        Returns a dict of training metrics (empty dict if buffer too small).
        Standard DQN TD loss: L = E[(r + γ * max_a' Q_target(s', a') - Q(s, a))^2]

        Target network is updated via Polyak averaging every step
        (tau=0.01 matches ORACLE's default --tau).
        """
        if len(self.buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones, is_drivers = \
            self.buffer.sample(self.batch_size)

        states_t      = self._preprocess_obs_batch(states)
        next_states_t = self._preprocess_obs_batch(next_states)
        actions_t     = torch.tensor(actions, dtype=torch.long,   device=self.device)
        rewards_t     = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t       = torch.tensor(dones,   dtype=torch.float32, device=self.device)

        # Current Q-values for taken actions
        q_current = self.q_net(states_t)                          # (B, action_dim)
        q_taken   = q_current.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (B,)

        # Target Q-values (no grad)
        with torch.no_grad():
            q_next      = self.target_net(next_states_t)          # (B, action_dim)
            q_next_max  = q_next.max(dim=1).values                # (B,)
            q_target    = rewards_t + self.gamma * q_next_max * (1.0 - dones_t)

        loss = F.mse_loss(q_taken, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (standard DQN practice, prevents instability)
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Polyak target update (same tau as ORACLE's soft target update)
        self._soft_update_target()

        self.steps += 1

        return {
            "loss": loss.item(),
            "q_mean": q_taken.mean().item(),
            "q_max": q_taken.max().item(),
        }

    def _soft_update_target(self):
        """Polyak averaging: θ_target = τ*θ_online + (1-τ)*θ_target."""
        for param, target_param in zip(self.q_net.parameters(),
                                       self.target_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------

    def decay_epsilon(self):
        """Call once per episode (after the episode ends, before the next)."""
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        self.episodes += 1

    def set_epsilon(self, value: float):
        """Manual epsilon override (e.g., set to 0 for evaluation)."""
        self.epsilon = max(self.epsilon_end, float(value))

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save(self, save_dir: str, tag: str = "final"):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"dqn_{tag}.pkl")
        payload = {
            "q_net":      self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "epsilon":    self.epsilon,
            "steps":      self.steps,
            "episodes":   self.episodes,
        }
        torch.save(payload, path)
        print(f"[DQN] saved checkpoint -> {path}")
        return path

    def load(self, save_dir: str, tag: str = "final"):
        path = os.path.join(save_dir, f"dqn_{tag}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No DQN checkpoint at {path}")
        # Use torch.load instead of pickle.load to avoid TF/PyTorch segfault
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.q_net.load_state_dict(payload["q_net"])
        self.target_net.load_state_dict(payload["target_net"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.epsilon  = payload["epsilon"]
        self.steps    = payload["steps"]
        self.episodes = payload["episodes"]
        print(f"[DQN] loaded checkpoint <- {path}")
