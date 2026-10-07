import tensorflow as tf
import numpy as np


def replace_gradient(value, surrogate):
    """Returns `value` but backpropagates gradients through `surrogate`.
    
    Forward pass: returns value
    Backward pass: gradients flow through surrogate
    """
    return surrogate + tf.stop_gradient(value - surrogate)


class GradientEstimator:
    """Base class for gradient estimators."""
    
    def update_state(self):
        """Called after each episode/step to update internal state (e.g., temperature)."""
        pass
    
    def reset(self):
        """Reset state (e.g., at start of each day)."""
        pass
    
    def __call__(self, logits, action_mask=None, training=True):
        raise NotImplementedError


class EpsilonGreedy(GradientEstimator):
    """Standard epsilon-greedy exploration (no gradient through exploration).
    
    This is the baseline - provides NO gradient signal through the exploration path.
    Actions are selected greedily from the actor's softmax output, with epsilon 
    probability of random action.
    """
    
    def __init__(self, epsilon_start=1.0, epsilon_min=0.01, decay_fraction=0.8, num_episodes=100):
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.decay_fraction = decay_fraction
        self.num_episodes = num_episodes
        self.epsilon = epsilon_start
        self._episode_count = 0
        self._decay_end = int(num_episodes * decay_fraction)
    
    def reset(self):
        """Reset epsilon to start value (called at beginning of each day)."""
        self.epsilon = self.epsilon_start
        self._episode_count = 0
    
    def update_state(self):
        """Decay epsilon after each episode."""
        self._episode_count += 1
        if self._episode_count <= self._decay_end:
            self.epsilon = 1.0 - (self._episode_count / self._decay_end) * (1.0 - self.epsilon_min)
            self.epsilon = max(self.epsilon_min, self.epsilon)
        else:
            self.epsilon = self.epsilon_min
    
    def __call__(self, logits, action_mask=None, training=True):
        """Select action using epsilon-greedy.
        
        Args:
            logits: Raw logits from actor network, shape (act_dim,) or (batch, act_dim)
            action_mask: Binary mask of valid actions, same shape as logits
            training: Whether in training mode (unused for epsilon-greedy)
            
        Returns:
            action_index: Selected action (int for single, array for batch)
            action_probs: Softmax probabilities (for logging, no gradient utility)
        """
        if action_mask is not None:
            invalid_mask = 1.0 - tf.cast(action_mask, tf.float32)
            logits = logits - 1e9 * invalid_mask
        
        action_probs = tf.nn.softmax(logits, axis=-1)
        
        if training and np.random.random() < self.epsilon:
            # Random action from valid actions
            if action_mask is not None:
                valid_indices = tf.where(tf.cast(action_mask, tf.bool))
                if len(valid_indices) > 0:
                    idx = np.random.randint(len(valid_indices))
                    action = int(valid_indices[idx][-1])
                else:
                    action = int(tf.argmax(action_probs, axis=-1))
            else:
                action = np.random.randint(logits.shape[-1])
        else:
            action = int(tf.argmax(action_probs, axis=-1))
        
        return action, action_probs
    
    def get_temperature(self):
        """Return epsilon as the 'temperature' equivalent for logging."""
        return self.epsilon


class STGS(GradientEstimator):
    """Straight-Through Gumbel Softmax estimator.
    
    Provides gradient signal through the exploration path by using the 
    Gumbel-Softmax reparameterization trick with straight-through gradient.
    
    Forward: hard one-hot action (argmax of perturbed logits)
    Backward: gradients flow through soft Gumbel-Softmax distribution
    """
    
    def __init__(self, temperature=1.0):
        self.temperature = temperature
        self._initial_temperature = temperature
    
    def reset(self):
        """Reset temperature to initial value."""
        self.temperature = self._initial_temperature
    
    def get_temperature(self):
        return self.temperature
    
    @tf.function(reduce_retracing=True)
    def _sample_gumbel_softmax(self, logits, temperature):
        """Sample from Gumbel-Softmax distribution with straight-through gradient.
        
        Args:
            logits: (act_dim,) or (batch, act_dim) raw logits
            temperature: scalar temperature
            
        Returns:
            y_hard: one-hot action with gradients flowing through y_soft
        """
        # Sample Gumbel(0, 1) noise
        gumbel_noise = -tf.math.log(-tf.math.log(
            tf.random.uniform(tf.shape(logits), minval=1e-8, maxval=1.0 - 1e-8)
        ))
        
        # Perturb logits and apply temperature
        perturbed_logits = (logits + gumbel_noise) / temperature
        y_soft = tf.nn.softmax(perturbed_logits, axis=-1)
        
        # Hard one-hot (argmax)
        y_hard = tf.one_hot(tf.argmax(y_soft, axis=-1), depth=tf.shape(logits)[-1])
        y_hard = tf.cast(y_hard, tf.float32)
        
        # Straight-through: forward uses y_hard, backward uses y_soft
        return replace_gradient(y_hard, y_soft)
    
    def __call__(self, logits, action_mask=None, training=True):
        """Select action using Gumbel-Softmax with straight-through gradient.
        
        Args:
            logits: Raw logits from actor network
            action_mask: Binary mask of valid actions
            training: Whether in training mode
            
        Returns:
            action_index: Selected action
            action_one_hot: Differentiable one-hot action (use for gradient computation)
        """
        if action_mask is not None:
            invalid_mask = 1.0 - tf.cast(action_mask, tf.float32)
            logits = logits - 1e9 * invalid_mask
        
        if training:
            action_one_hot = self._sample_gumbel_softmax(logits, self.temperature)
        else:
            # At eval time, just take argmax
            action_probs = tf.nn.softmax(logits, axis=-1)
            action_one_hot = tf.one_hot(tf.argmax(action_probs, axis=-1), 
                                         depth=tf.shape(logits)[-1])
            action_one_hot = tf.cast(action_one_hot, tf.float32)
        
        action = int(tf.argmax(action_one_hot, axis=-1))
        return action, action_one_hot


class TAGS(STGS):
    """Temperature-Annealed Straight-Through Gumbel Softmax estimator.
    
    Supports both exponential and linear annealing schedules.
    """
    
    def __init__(self, start_temp=1.0, end_temp=0.01, period=80, annealing='exponential'):
        super().__init__(start_temp)
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.period = period
        self.annealing = annealing
        if annealing == 'exponential':
            self.multiplier = (end_temp / start_temp) ** (1.0 / period)
        else:  # linear
            self.step_size = (start_temp - end_temp) / period
        self.updates = 0
    
    def reset(self):
        self.temperature = self.start_temp
        self.updates = 0
    
    def update_state(self):
        if self.updates < self.period:
            if self.annealing == 'exponential':
                self.temperature *= self.multiplier
            else:  # linear
                self.temperature -= self.step_size
                self.temperature = max(self.temperature, self.end_temp)
        self.updates += 1

class GST(GradientEstimator):
    """Gapped Straight-Through Estimator.
    
    Provides better gradient signal than STGS by adjusting logits before 
    applying softmax, ensuring the surrogate distribution is closer to the 
    true categorical distribution.
    
    Key idea: Move logits so that:
    - Selected class logit is moved UP to match the max logit
    - Non-selected class logits that are within `gap` of the max are moved DOWN
    
    This creates a cleaner gradient signal with less bias than STGS.
    """
    
    def __init__(self, temperature=1.0, gap=1.0):
        """
        Args:
            temperature: Softmax temperature for the surrogate
            gap: Gap parameter controlling how aggressively non-selected logits are pushed down
        """
        self.temperature = temperature
        self._initial_temperature = temperature
        self.gap = gap
    
    def reset(self):
        self.temperature = self._initial_temperature
    
    def get_temperature(self):
        return self.temperature
    
    def __call__(self, logits, action_mask=None, training=True):
        """Select action using Gapped Straight-Through estimator.
        
        Args:
            logits: Raw logits from actor network, shape (act_dim,)
            action_mask: Binary mask of valid actions
            training: Whether in training mode
            
        Returns:
            action_index: Selected action
            action_one_hot: Differentiable one-hot action
        """
        if action_mask is not None:
            invalid_mask = 1.0 - tf.cast(action_mask, tf.float32)
            logits = logits - 1e9 * invalid_mask
        
        if training:
            action_one_hot = self._sample_gst(logits)
        else:
            action_probs = tf.nn.softmax(logits, axis=-1)
            action_one_hot = tf.one_hot(tf.argmax(action_probs, axis=-1),
                                         depth=tf.shape(logits)[-1])
            action_one_hot = tf.cast(action_one_hot, tf.float32)
        
        action = int(tf.argmax(action_one_hot, axis=-1))
        return action, action_one_hot
    
    @tf.function(reduce_retracing=True)
    def _sample_gst(self, logits):
        """Sample using GST estimator.
        
        Args:
            logits: shape (act_dim,) or (batch, act_dim)
            
        Returns:
            one_hot action with GST gradients
        """
        # Sample categorical action
        gumbel_noise = -tf.math.log(-tf.math.log(
            tf.random.uniform(tf.shape(logits), minval=1e-8, maxval=1.0 - 1e-8)
        ))
        noisy_logits = logits + gumbel_noise
        
        # Hard sample
        action_idx = tf.argmax(noisy_logits, axis=-1)
        num_classes = tf.shape(logits)[-1]
        DD = tf.one_hot(action_idx, depth=num_classes)
        DD = tf.cast(DD, tf.float32)
        
        # Calculate movements
        max_logit = tf.reduce_max(logits, axis=-1, keepdims=True)
        selected_logit = tf.reduce_sum(logits * DD, axis=-1, keepdims=True)
        
        # m1: move selected logit up to max
        m1 = (max_logit - selected_logit) * DD
        
        # m2: push down non-selected logits that are within gap of max
        m2 = tf.maximum(logits + self.gap - max_logit, 0.0) * (1.0 - DD)
        
        # Compute surrogate
        adjusted_logits = (logits + m1 - m2) / self.temperature
        surrogate = tf.nn.softmax(adjusted_logits, axis=-1)
        
        return replace_gradient(DD, surrogate)

class TAGSTAnnealed(GST):
    """Temperature-Annealed Gapped Straight-Through Estimator.
    
    Supports both exponential and linear annealing for temperature and gap.
    """
    
    def __init__(self, start_temp=1.0, end_temp=0.01, period=80, 
                 gap_start=1.0, gap_end=0.1, annealing='exponential'):
        super().__init__(start_temp, gap_start)
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.period = period
        self.gap_start = gap_start
        self.gap_end = gap_end
        self.annealing = annealing
        if annealing == 'exponential':
            self.temp_multiplier = (end_temp / start_temp) ** (1.0 / period)
            self.gap_multiplier = (gap_end / gap_start) ** (1.0 / period)
        else:  # linear
            self.temp_step = (start_temp - end_temp) / period
            self.gap_step = (gap_start - gap_end) / period
        self.updates = 0
    
    def reset(self):
        self.temperature = self.start_temp
        self.gap = self.gap_start
        self.updates = 0
    
    def update_state(self):
        if self.updates < self.period:
            if self.annealing == 'exponential':
                self.temperature *= self.temp_multiplier
                self.gap *= self.gap_multiplier
            else:  # linear
                self.temperature -= self.temp_step
                self.gap -= self.gap_step
                self.temperature = max(self.temperature, self.end_temp)
                self.gap = max(self.gap, self.gap_end)
        self.updates += 1


def create_exploration_strategy(strategy_name, args, episodes_per_day):
    """Factory function to create exploration strategy.
    
    Args:
        strategy_name: 'epsilon_greedy', 'tags', or 'gst'
        args: Parsed command line arguments
        episodes_per_day: Number of episodes per day (for annealing period)
        
    Returns:
        GradientEstimator instance
    """
    decay_period = int(episodes_per_day * args.epsilon_decay_episodes)
    annealing = getattr(args, 'annealing', 'exponential')
    
    if strategy_name == "epsilon_greedy":
        return EpsilonGreedy(
            epsilon_start=args.epsilon_start,
            epsilon_min=args.epsilon_min,
            decay_fraction=args.epsilon_decay_episodes,
            num_episodes=episodes_per_day
        )
    elif strategy_name == "tags":
        return TAGS(
            start_temp=getattr(args, 'gumbel_temp_start', 1.0),
            end_temp=getattr(args, 'gumbel_temp_end', 0.01),
            period=decay_period,
            annealing=annealing
        )
    elif strategy_name == "gst":
        return TAGSTAnnealed(
            start_temp=getattr(args, 'gumbel_temp_start', 1.0),
            end_temp=getattr(args, 'gumbel_temp_end', 0.01),
            period=decay_period,
            gap_start=getattr(args, 'gst_gap', 1.0),
            gap_end=getattr(args, 'gst_gap_end', 0.1),
            annealing=annealing
        )
    else:
        raise ValueError(f"Unknown exploration strategy: {strategy_name}")