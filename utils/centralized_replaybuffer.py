import numpy as np
import random

class CentralizedReplayBuffer:
    def __init__(self, size, num_agents):
        """Create a centralized replay buffer for MADDPG.

        Parameters:
        size: int
            Maximum number of transitions to store in the buffer.
        num_agents: int
            Number of agents in the multi-agent system.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0
        self.num_agents = num_agents

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_n, act_n, reward_n, next_obs_n, done_n):
        """Store a joint experience tuple."""
        data = (obs_n, act_n, reward_n, next_obs_n, done_n)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    # def make_index(self, batch_size):
    #     return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
    
    # def sample_index(self, idxes):
    #     return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        idxes = random.sample(range(len(self._storage)), batch_size)
        return self._encode_sample(idxes)

    def _encode_sample(self, idxes):
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = [], [], [], [], []

        for i in idxes:
            obs_n, act_n, reward_n, next_obs_n, done_n = self._storage[i]
            obs_batch.append(np.array(obs_n))
            act_batch.append(np.array(act_n))
            rew_batch.append(np.array(reward_n))
            next_obs_batch.append(np.array(next_obs_n))
            done_batch.append(np.array(done_n))

        return (
            np.array(obs_batch),    # Shape: (batch_size, num_agents, obs_dim)
            np.array(act_batch),    # Shape: (batch_size, num_agents, act_dim)
            np.array(rew_batch),    # Shape: (batch_size, num_agents)
            np.array(next_obs_batch),  # Shape: (batch_size, num_agents, obs_dim)
            np.array(done_batch)    # Shape: (batch_size, num_agents)
        )
