import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity  # Maximum number of elements to store
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)  # Binary tree structure
        self.data_pointer = 0

    def add(self, priority):
        # Add priority to the tree
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx, priority):
        # Update a specific node and propagate the change
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        parent = (tree_idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child = 2 * parent_idx + 1
            right_child = left_child + 1
            if left_child >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child]:
                    parent_idx = left_child
                else:
                    v -= self.tree[left_child]
                    parent_idx = right_child
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    def total_priority(self):
        return self.tree[0]

    
class CentralizedReplayBuffer:
    def __init__(self, num_agents, state_dim=28, action_dim=11, max_size=600000, alpha_initial=0.3, beta_initial=0.3, alpha_final=1, beta_final=1, alpha_decay=1e-5, beta_decay=1e-5, min_priority=1e-5):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_size = max_size
        self.alpha = alpha_initial
        self.beta = beta_initial
        self.min_priority = min_priority
        self.sum_tree = SumTree(max_size)
        self.alpha_initial = alpha_initial
        self.beta_initial = beta_initial
        self.alpha_final = alpha_final
        self.beta_final = beta_final
        self.alpha_decay = alpha_decay
        self.beta_decay = beta_decay
        self.theta = 0.1

        # print(max_size, num_agents, state_dim, action_dim)

        self.states = np.zeros((max_size, num_agents, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, num_agents, action_dim), dtype=np.float32)
        # print(self.actions.shape)
        self.rewards = np.zeros((max_size, num_agents), dtype=np.float32)
        self.next_states = np.zeros((max_size, num_agents, state_dim), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)
        # Add mask storage
        self.masks = np.zeros((max_size, num_agents), dtype=np.float32)
        
        # Add separate action mask storage for current and next states
        self.state_action_masks = np.zeros((max_size, num_agents, action_dim), dtype=np.float32)
        self.next_state_action_masks = np.zeros((max_size, num_agents, action_dim), dtype=np.float32)

        self.size = 0
        self.ptr = 0

    def __len__(self):
        return self.size

    def add(self, states, actions, rewards, next_states, dones, masks, state_action_masks, next_state_action_masks, td_errors=None):
        """Store transitions for all agents with action masks for both states."""
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        # print(actions.shape)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        masks = np.array(masks, dtype=np.float32)

        max_priority = np.max(self.sum_tree.tree[-self.max_size:]) if self.size > 0 else 1.0
        priority = max_priority if td_errors is None else (np.abs(td_errors).max() + self.min_priority) ** self.alpha

        self.states[self.ptr] = states
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.next_states[self.ptr] = next_states
        self.dones[self.ptr] = dones
        self.masks[self.ptr] = masks
        
        # Store both current and next state action masks
        self.state_action_masks[self.ptr] = state_action_masks
        self.next_state_action_masks[self.ptr] = next_state_action_masks

        self.sum_tree.add(priority)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def clear_buffer(self):
        """Reset the replay buffer to its initial empty state."""
        # Reset counters
        self.size = 0
        self.ptr = 0
        
        # Reset numpy arrays
        self.states = np.zeros((self.max_size, self.num_agents, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.max_size, self.num_agents, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.max_size, self.num_agents), dtype=np.float32)
        self.next_states = np.zeros((self.max_size, self.num_agents, self.state_dim), dtype=np.float32)
        self.dones = np.zeros(self.max_size, dtype=np.float32)
        self.masks = np.zeros((self.max_size, self.num_agents), dtype=np.float32)
        self.state_action_masks = np.zeros((self.max_size, self.num_agents, self.action_dim), dtype=np.float32)
        self.next_state_action_masks = np.zeros((self.max_size, self.num_agents, self.action_dim), dtype=np.float32)
        
        # Reset sum tree
        self.sum_tree = SumTree(self.max_size)

    def sample(self, batch_size):
        if self.size < batch_size:
            return None, None, None, None, None, None, None, None

        segment = self.sum_tree.total_priority() / batch_size
        samples_idx = []
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            leaf_idx, priority, data_idx = self.sum_tree.get_leaf(s)
            samples_idx.append(leaf_idx)
            priorities.append(priority)

        probabilities = np.array(priorities) / self.sum_tree.total_priority()
        # print("Probabilities: ", probabilities)
        weights = (self.size * (probabilities + 1e-6)) ** (-self.beta)
        weights /= weights.max()

        samples_idx = np.array(samples_idx, dtype=np.int32)
        state_batch = self.states[samples_idx - (self.sum_tree.capacity - 1)]
        action_batch = self.actions[samples_idx - (self.sum_tree.capacity - 1)]
        reward_batch = self.rewards[samples_idx - (self.sum_tree.capacity - 1)]
        next_state_batch = self.next_states[samples_idx - (self.sum_tree.capacity - 1)]
        done_batch = self.dones[samples_idx - (self.sum_tree.capacity - 1)]
        mask_batch = self.masks[samples_idx - (self.sum_tree.capacity - 1)]
        state_action_mask_batch = self.state_action_masks[samples_idx - (self.sum_tree.capacity - 1)]
        next_state_action_mask_batch = self.next_state_action_masks[samples_idx - (self.sum_tree.capacity - 1)]

        # After sampling experiences
        # self.print_highest_priority_experiences(num_top=5)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, mask_batch, state_action_mask_batch, next_state_action_mask_batch, samples_idx, weights

    def update_priorities(self, indices, td_errors):
        """Update priorities for sampled experiences."""
        new_priorities = (np.abs(td_errors) + self.min_priority) ** self.alpha
        for idx, prio in zip(indices, new_priorities):
            self.sum_tree.update(idx, prio)

    def update_alpha_beta(self, episode_num, max_episodes):
        # Update alpha and beta based on episode number or training step
        self.alpha = min(self.alpha_final, self.alpha_initial + self.alpha_decay * episode_num)
        self.beta = min(self.beta_final, self.beta_initial + self.beta_decay * episode_num)

    def print_highest_priority_experiences(self, num_top=5):
        """
        Prints the highest priority experiences stored in the replay buffer.
        
        Args:
            num_top (int): Number of top priority experiences to print.
        """
        # Get indices of highest priority experiences from SumTree
        sorted_indices = np.argsort(-self.sum_tree.tree[-self.sum_tree.capacity:])  # Sort in descending order

        print("\nTop prioritized experiences (highest TD error):")
        for i in range(min(num_top, len(sorted_indices))):
            idx = sorted_indices[i]
            priority = self.sum_tree.tree[idx + self.sum_tree.capacity - 1]
            
            # Fetch corresponding reward from the centralized buffer
            if idx < len(self.rewards):  # Ensure index is within stored experiences
                total_reward = np.sum(self.rewards[idx])  # Sum of rewards for this experience
            else:
                total_reward = None  # If index is out of range

            print(f"Index: {idx}, Priority: {priority}, Sum of Rewards: {total_reward}")

