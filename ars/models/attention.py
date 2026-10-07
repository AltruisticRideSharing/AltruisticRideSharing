import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
from ars.utils.distributions import make_pdtype
from ars.models.base import AgentTrainer
from tensorflow.keras import Input
from tensorflow.keras.layers import InputLayer 
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.models import Model
from ars.models.actor import Actor
from ars.models.attention_critic import AttentionCritic
from ars.utils.gradient_estimators import replace_gradient
import gc


def update_target(model, target_model, tau=0.01):
    polyak = 1.0 - tau
    model_weights = model.get_weights()
    target_weights = target_model.get_weights()
    updated_weights = [polyak * tw + (1 - polyak) * mw 
                       for mw, tw in zip(model_weights, target_weights)]
    target_model.set_weights(updated_weights)


class AttentionTrainer(AgentTrainer):
    def __init__(self, name, learning_rate, obs_shape_n, act_space_n, centralized_buffer, agent_index, args, local_q_func=False):
        self.name = name
        self.learning_rate = learning_rate
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.obs_size = obs_shape_n[agent_index]
        self.joint_obs_size = np.sum(obs_shape_n)
        self.act_size = act_space_n[agent_index].n
        self.act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        
        self.device = None
        if args.device == "gpu":
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.device = tf.device('/GPU:0')
            else:
                self.device = tf.device('/CPU:0')
                print(f"Agent {self.name}: No GPU available, falling back to CPU")
        else:
            self.device = tf.device('/CPU:0')
        
        self.joint_act_size = sum(a.n for a in act_space_n)
        self.args = args
        self.shared_mode = False  # Default: each agent owns its own networks
        self._train_step_counter = 0
        
        # Store exploration strategy name for use during training
        self.exploration_strategy_name = getattr(args, 'exploration', 'epsilon_greedy')
        
        # Every agent builds its own networks by default
        with self.device:
            self.actor, self.critic = self.build_model()
            self.actor_target, self.critic_target = self.build_model()

            self.actor_target.set_weights(self.actor.get_weights())
            self.critic_target.set_weights(self.critic.get_weights())

            self.critic_optimizer = AdamW(learning_rate=self.learning_rate, clipnorm=0.5)
            self.actor_optimizer = AdamW(
                learning_rate=self.learning_rate * self.args.actor_lr, clipnorm=0.5
            )

        self.replay_buffer = centralized_buffer
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def set_shared_networks(self, shared_trainer):
        """Enable shared network mode: all agents share the primary trainer's networks."""
        if self.agent_index != shared_trainer.agent_index:
            self.shared_mode = True
            self.actor = shared_trainer.actor
            self.critic = shared_trainer.critic
            self.actor_target = shared_trainer.actor_target
            self.critic_target = shared_trainer.critic_target
            self.critic_optimizer = shared_trainer.critic_optimizer
            self.actor_optimizer = shared_trainer.actor_optimizer
            shared_trainer.shared_mode = True
    
    def build_model(self):
        actor_model = Actor(
            self.obs_size, 
            self.act_size, 
            hidden1=self.args.actor_hidden1,
            hidden2=self.args.actor_hidden2
        )
        
        critic_model = AttentionCritic(
            obs_dim=self.obs_size,
            act_dim=self.act_size,
            num_agents=self.n,
            embed_dim=getattr(self.args, 'embed_dim', 128),
            num_heads=getattr(self.args, 'num_heads', 4),
            hidden1=self.args.critic_hidden1,
            hidden2=self.args.critic_hidden2,
            attention_temp=getattr(self.args, 'attention_temp', 1.0),
            use_layer_norm=getattr(self.args, 'use_layer_norm', False),
            dropout_rate=getattr(self.args, 'dropout_rate', 0.0)
        )
        
        return actor_model, critic_model
    
    @tf.function(reduce_retracing=True)
    def _get_action_body(self, obs_tensor, valid_actions_mask):
        logits = self.actor(obs_tensor[None])
        invalid_mask = 1.0 - tf.cast(valid_actions_mask, tf.float32)
        masked_logits = logits - 1e9 * tf.reshape(invalid_mask, [1, -1])
        a = tf.nn.softmax(masked_logits, axis=-1)
        return a[0]

    @tf.function(reduce_retracing=True)
    def _get_raw_logits(self, obs_tensor, valid_actions_mask):
        """Get masked logits from actor (for use with Gumbel-Softmax estimators).
        
        Args:
            obs_tensor: Single observation
            valid_actions_mask: Binary mask of valid actions
            
        Returns:
            Masked logits (invalid actions set to -1e9)
        """
        logits = self.actor(obs_tensor[None])
        invalid_mask = 1.0 - tf.cast(valid_actions_mask, tf.float32)
        masked_logits = logits - 1e9 * tf.reshape(invalid_mask, [1, -1])
        return masked_logits[0]

    @tf.function(reduce_retracing=True)
    def _compute_all_target_actions_shared(self, obs_next_n, next_state_action_masks_n):
        """Compute target actions for all agents using shared actor_target"""
        target_actions = []
        for j in range(self.n):
            next_logits = self.actor_target(obs_next_n[j], training=False)
            invalid_mask = 1.0 - next_state_action_masks_n[j]
            masked_logits = next_logits - 1e9 * invalid_mask
            target_actions.append(tf.nn.softmax(masked_logits, axis=-1))
        return tf.stack(target_actions, axis=1)

    @tf.function(reduce_retracing=True)
    def _compute_all_current_actions_shared(self, obs_n, state_action_masks_n):
        """Compute current policy actions for all agents using shared actor"""
        current_actions = []
        for j in range(self.n):
            logits = self.actor(obs_n[j], training=True)
            invalid_mask = 1.0 - state_action_masks_n[j]
            masked_logits = logits - 1e9 * invalid_mask
            probs = tf.nn.softmax(masked_logits, axis=-1)
            current_actions.append(probs)
        return tf.stack(current_actions, axis=1)

    @tf.function(reduce_retracing=True)
    def _compute_all_current_actions_shared_gumbel(self, obs_n, state_action_masks_n, temperature):
        """Compute current policy actions using Gumbel-Softmax for differentiable exploration.
        
        Used during training when exploration strategy is TAGS or GST.
        Provides gradient signal through the action selection process.
        
        Args:
            obs_n: List of observation tensors per agent
            state_action_masks_n: List of action mask tensors per agent  
            temperature: Current Gumbel-Softmax temperature
            
        Returns:
            Stacked action tensors with straight-through gradients
        """
        current_actions = []
        for j in range(self.n):
            logits = self.actor(obs_n[j], training=True)
            invalid_mask = 1.0 - state_action_masks_n[j]
            masked_logits = logits - 1e9 * invalid_mask
            
            # Gumbel-Softmax with straight-through
            gumbel_noise = -tf.math.log(-tf.math.log(
                tf.random.uniform(tf.shape(masked_logits), minval=1e-8, maxval=1.0 - 1e-8)
            ))
            perturbed = (masked_logits + gumbel_noise) / temperature
            y_soft = tf.nn.softmax(perturbed, axis=-1)
            y_hard = tf.one_hot(tf.argmax(y_soft, axis=-1), depth=tf.shape(masked_logits)[-1])
            y_hard = tf.cast(y_hard, tf.float32)
            
            # Straight-through gradient
            action = y_soft + tf.stop_gradient(y_hard - y_soft)
            current_actions.append(action)
        return tf.stack(current_actions, axis=1)

    @tf.function(reduce_retracing=True)
    def _compute_all_current_actions_shared_gst(self, obs_n, state_action_masks_n, temperature, gap):
        """Compute current policy actions using GST estimator for differentiable exploration.
        
        Args:
            obs_n: List of observation tensors per agent
            state_action_masks_n: List of action mask tensors per agent
            temperature: Current GST temperature
            gap: GST gap parameter
            
        Returns:
            Stacked action tensors with GST gradients
        """
        current_actions = []
        for j in range(self.n):
            logits = self.actor(obs_n[j], training=True)
            invalid_mask = 1.0 - state_action_masks_n[j]
            masked_logits = logits - 1e9 * invalid_mask
            
            # Sample categorical action
            gumbel_noise = -tf.math.log(-tf.math.log(
                tf.random.uniform(tf.shape(masked_logits), minval=1e-8, maxval=1.0 - 1e-8)
            ))
            noisy_logits = masked_logits + gumbel_noise
            action_idx = tf.argmax(noisy_logits, axis=-1)
            num_classes = tf.shape(masked_logits)[-1]
            DD = tf.cast(tf.one_hot(action_idx, depth=num_classes), tf.float32)
            
            # Calculate GST movements
            max_logit = tf.reduce_max(masked_logits, axis=-1, keepdims=True)
            selected_logit = tf.reduce_sum(masked_logits * DD, axis=-1, keepdims=True)
            m1 = (max_logit - selected_logit) * DD
            m2 = tf.maximum(masked_logits + gap - max_logit, 0.0) * (1.0 - DD)
            
            adjusted = (masked_logits + m1 - m2) / temperature
            surrogate = tf.nn.softmax(adjusted, axis=-1)
            
            # Straight-through: forward DD, backward surrogate
            action = surrogate + tf.stop_gradient(DD - surrogate)
            current_actions.append(action)
        return tf.stack(current_actions, axis=1)

    def compute_gradients(self, agents, t, exploration_strategy=None):
        """Compute gradients - dispatches to shared or independent mode
        
        Args:
            agents: List of all agent trainers
            t: Current timestep
            exploration_strategy: Optional GradientEstimator for differentiable exploration
        """
        if self.shared_mode and self.agent_index != 0:
            # In shared mode, only the primary agent trains
            return None
        
        batch_size = min(len(self.replay_buffer), self.args.batch_size)
        sample_result = self.replay_buffer.sample(batch_size)
        obs, act, rewards, obs_next, done, masks, state_action_masks, next_state_action_masks, sample_idx, weights = sample_result
        
        # Prepare tensors
        obs_n = [tf.squeeze(tf.cast(o, tf.float32), axis=1) for o in tf.split(obs, self.n, axis=1)]
        obs_next_n = [tf.squeeze(tf.cast(o, tf.float32), axis=1) for o in tf.split(obs_next, self.n, axis=1)]
        act_n = [tf.squeeze(tf.cast(a, tf.float32), axis=1) for a in tf.split(act, self.n, axis=1)]
        state_action_masks_n = [tf.squeeze(tf.cast(m, tf.float32), axis=1) for m in tf.split(state_action_masks, self.n, axis=1)]
        next_state_action_masks_n = [tf.squeeze(tf.cast(m, tf.float32), axis=1) for m in tf.split(next_state_action_masks, self.n, axis=1)]
         
        obs_stacked = tf.stack(obs_n, axis=1)
        obs_next_stacked = tf.stack(obs_next_n, axis=1)
        act_stacked = tf.stack(act_n, axis=1)
        
        done = tf.cast(tf.reshape(done, [-1, 1]), tf.float32)
        weights = tf.cast(tf.reshape(weights, [-1, 1]), tf.float32)
        rewards = tf.cast(rewards, tf.float32)  # (batch, n_agents)
        
        # Determine which training method to use based on exploration strategy
        use_gumbel = (exploration_strategy is not None and 
                      self.exploration_strategy_name in ['tags', 'gst'])
        
        if self.shared_mode:
            if use_gumbel:
                temperature = tf.constant(exploration_strategy.get_temperature(), dtype=tf.float32)
                if self.exploration_strategy_name == 'gst':
                    gap = tf.constant(exploration_strategy.gap, dtype=tf.float32)
                    critic_loss, actor_loss, td_errors = self._train_step_shared_gst(
                        obs_stacked, act_stacked, rewards, obs_next_stacked, done, weights,
                        obs_n, obs_next_n, state_action_masks_n, next_state_action_masks_n, 
                        batch_size, temperature, gap
                    )
                else:  # TAGS
                    critic_loss, actor_loss, td_errors = self._train_step_shared_gumbel(
                        obs_stacked, act_stacked, rewards, obs_next_stacked, done, weights,
                        obs_n, obs_next_n, state_action_masks_n, next_state_action_masks_n, 
                        batch_size, temperature
                    )
            else:
                critic_loss, actor_loss, td_errors = self._train_step_shared(
                    obs_stacked, act_stacked, rewards, obs_next_stacked, done, weights,
                    obs_n, obs_next_n, state_action_masks_n, next_state_action_masks_n, batch_size
                )
        else:
            # Independent mode: each agent has its own networks, train only for this agent
            critic_loss, actor_loss, td_errors = self._train_step_independent(
                obs_stacked, act_stacked, rewards, obs_next_stacked, done, weights,
                obs_n, obs_next_n, state_action_masks_n, next_state_action_masks_n,
                agents, batch_size
            )
        
        # Update priorities
        self.replay_buffer.update_priorities(sample_idx, td_errors)
        
        # Periodic cleanup
        self._train_step_counter += 1
        if self._train_step_counter % 100 == 0:
            gc.collect()
        
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'td_errors': td_errors,
            'sample_idx': sample_idx
        }

    @tf.function(reduce_retracing=True)
    def _train_step_shared(self, obs_stacked, act_stacked, rewards, obs_next_stacked, done, weights,
                           obs_n, obs_next_n, state_action_masks_n, next_state_action_masks_n, batch_size):
        """Training step for shared network mode - standard softmax actor update"""
        
        # Pre-compute target actions for all agents
        target_act_next_stacked = self._compute_all_target_actions_shared(obs_next_n, next_state_action_masks_n)
        
        # --- CRITIC UPDATE ---
        with tf.GradientTape() as critic_tape:
            total_critic_loss = tf.constant(0.0)
            all_td_errors = []
            
            for agent_idx in range(self.n):
                rew = rewards[:, agent_idx:agent_idx+1]
                
                target_q_next = self.critic_target(obs_next_stacked, target_act_next_stacked, 
                                                    agent_index=agent_idx, training=False)
                target_q = rew + self.args.gamma * (1.0 - done) * target_q_next
                target_q = tf.stop_gradient(target_q)
                
                current_q = self.critic(obs_stacked, act_stacked, 
                                        agent_index=agent_idx, training=True)
                
                td_error = current_q - target_q
                all_td_errors.append(tf.abs(td_error))
                total_critic_loss = total_critic_loss + tf.reduce_mean(weights * tf.square(td_error))
            
            total_critic_loss = total_critic_loss / tf.cast(self.n, tf.float32)
        
        critic_grads = critic_tape.gradient(total_critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        # --- ACTOR UPDATE (standard softmax) ---
        with tf.GradientTape() as actor_tape:
            total_actor_loss = tf.constant(0.0)
            new_act_stacked = self._compute_all_current_actions_shared(obs_n, state_action_masks_n)
            
            for agent_idx in range(self.n):
                q_new = self.critic(obs_stacked, new_act_stacked, 
                                    agent_index=agent_idx, training=False)
                total_actor_loss = total_actor_loss - tf.reduce_mean(weights * q_new)
            
            total_actor_loss = total_actor_loss / tf.cast(self.n, tf.float32)
            reg_loss = self.args.reg_coef * tf.add_n([tf.reduce_sum(tf.square(w)) 
                                                       for w in self.actor.trainable_variables])
            total_actor_loss = total_actor_loss + reg_loss
        
        actor_grads = actor_tape.gradient(total_actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        avg_td_errors = tf.reduce_mean(tf.concat(all_td_errors, axis=1), axis=1)
        
        return total_critic_loss, total_actor_loss, avg_td_errors

    @tf.function(reduce_retracing=True)
    def _train_step_shared_gumbel(self, obs_stacked, act_stacked, rewards, obs_next_stacked, done, weights,
                                   obs_n, obs_next_n, state_action_masks_n, next_state_action_masks_n, 
                                   batch_size, temperature):
        """Training step with Gumbel-Softmax actor update (TAGS).
        
        The critic update is identical to standard. The actor update uses
        Gumbel-Softmax with straight-through gradient to provide exploration
        signal that flows back through the actor network.
        """
        # Pre-compute target actions
        target_act_next_stacked = self._compute_all_target_actions_shared(obs_next_n, next_state_action_masks_n)
        
        # --- CRITIC UPDATE (same as standard) ---
        with tf.GradientTape() as critic_tape:
            total_critic_loss = tf.constant(0.0)
            all_td_errors = []
            
            for agent_idx in range(self.n):
                rew = rewards[:, agent_idx:agent_idx+1]
                target_q_next = self.critic_target(obs_next_stacked, target_act_next_stacked,
                                                    agent_index=agent_idx, training=False)
                target_q = rew + self.args.gamma * (1.0 - done) * target_q_next
                target_q = tf.stop_gradient(target_q)
                
                current_q = self.critic(obs_stacked, act_stacked,
                                        agent_index=agent_idx, training=True)
                td_error = current_q - target_q
                all_td_errors.append(tf.abs(td_error))
                total_critic_loss = total_critic_loss + tf.reduce_mean(weights * tf.square(td_error))
            
            total_critic_loss = total_critic_loss / tf.cast(self.n, tf.float32)
        
        critic_grads = critic_tape.gradient(total_critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        # --- ACTOR UPDATE (Gumbel-Softmax straight-through) ---
        with tf.GradientTape() as actor_tape:
            total_actor_loss = tf.constant(0.0)
            new_act_stacked = self._compute_all_current_actions_shared_gumbel(
                obs_n, state_action_masks_n, temperature
            )
            
            for agent_idx in range(self.n):
                q_new = self.critic(obs_stacked, new_act_stacked,
                                    agent_index=agent_idx, training=False)
                total_actor_loss = total_actor_loss - tf.reduce_mean(weights * q_new)
            
            total_actor_loss = total_actor_loss / tf.cast(self.n, tf.float32)
            reg_loss = self.args.reg_coef * tf.add_n([tf.reduce_sum(tf.square(w))
                                                       for w in self.actor.trainable_variables])
            total_actor_loss = total_actor_loss + reg_loss
        
        actor_grads = actor_tape.gradient(total_actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        avg_td_errors = tf.reduce_mean(tf.concat(all_td_errors, axis=1), axis=1)
        return total_critic_loss, total_actor_loss, avg_td_errors

    @tf.function(reduce_retracing=True)
    def _train_step_shared_gst(self, obs_stacked, act_stacked, rewards, obs_next_stacked, done, weights,
                                obs_n, obs_next_n, state_action_masks_n, next_state_action_masks_n,
                                batch_size, temperature, gap):
        """Training step with GST actor update.
        
        Uses the Gapped Straight-Through estimator which provides better 
        gradient signal than standard STGS by adjusting logit movements.
        """
        # Pre-compute target actions
        target_act_next_stacked = self._compute_all_target_actions_shared(obs_next_n, next_state_action_masks_n)
        
        # --- CRITIC UPDATE (same as standard) ---
        with tf.GradientTape() as critic_tape:
            total_critic_loss = tf.constant(0.0)
            all_td_errors = []
            
            for agent_idx in range(self.n):
                rew = rewards[:, agent_idx:agent_idx+1]
                target_q_next = self.critic_target(obs_next_stacked, target_act_next_stacked,
                                                    agent_index=agent_idx, training=False)
                target_q = rew + self.args.gamma * (1.0 - done) * target_q_next
                target_q = tf.stop_gradient(target_q)
                
                current_q = self.critic(obs_stacked, act_stacked,
                                        agent_index=agent_idx, training=True)
                td_error = current_q - target_q
                all_td_errors.append(tf.abs(td_error))
                total_critic_loss = total_critic_loss + tf.reduce_mean(weights * tf.square(td_error))
            
            total_critic_loss = total_critic_loss / tf.cast(self.n, tf.float32)
        
        critic_grads = critic_tape.gradient(total_critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        # --- ACTOR UPDATE (GST) ---
        with tf.GradientTape() as actor_tape:
            total_actor_loss = tf.constant(0.0)
            new_act_stacked = self._compute_all_current_actions_shared_gst(
                obs_n, state_action_masks_n, temperature, gap
            )
            
            for agent_idx in range(self.n):
                q_new = self.critic(obs_stacked, new_act_stacked,
                                    agent_index=agent_idx, training=False)
                total_actor_loss = total_actor_loss - tf.reduce_mean(weights * q_new)
            
            total_actor_loss = total_actor_loss / tf.cast(self.n, tf.float32)
            reg_loss = self.args.reg_coef * tf.add_n([tf.reduce_sum(tf.square(w))
                                                       for w in self.actor.trainable_variables])
            total_actor_loss = total_actor_loss + reg_loss
        
        actor_grads = actor_tape.gradient(total_actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        avg_td_errors = tf.reduce_mean(tf.concat(all_td_errors, axis=1), axis=1)
        return total_critic_loss, total_actor_loss, avg_td_errors

    def _train_step_independent(self, obs_stacked, act_stacked, rewards, obs_next_stacked, done, weights,
                                obs_n, obs_next_n, state_action_masks_n, next_state_action_masks_n,
                                agents, batch_size):
        """Training step for independent network mode - each agent has its own actor/critic.
        
        Similar to standard MADDPG but uses AttentionCritic instead of concatenation-based critic.
        Each agent only updates its own networks using its own reward.
        """
        agent_idx = self.agent_index
        rew = tf.expand_dims(rewards[:, agent_idx], axis=-1)  # (batch, 1)
        
        # Compute target actions from each agent's own target network
        target_act_next_list = []
        for j in range(self.n):
            next_logits = agents[j].actor_target(obs_next_n[j], training=False)
            invalid_mask = 1.0 - next_state_action_masks_n[j]
            masked_logits = next_logits - 1e9 * invalid_mask
            target_act_next_list.append(tf.nn.softmax(masked_logits, axis=-1))
        target_act_next_stacked = tf.stack(target_act_next_list, axis=1)
        
        # --- CRITIC UPDATE ---
        with tf.GradientTape() as critic_tape:
            target_q_next = self.critic_target(obs_next_stacked, target_act_next_stacked,
                                                agent_index=agent_idx, training=False)
            target_q = rew + self.args.gamma * (1.0 - done) * target_q_next
            target_q = tf.stop_gradient(target_q)
            
            current_q = self.critic(obs_stacked, act_stacked,
                                    agent_index=agent_idx, training=True)
            
            td_error = current_q - target_q
            critic_loss = tf.reduce_mean(weights * tf.square(td_error))
        
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        # --- ACTOR UPDATE ---
        with tf.GradientTape() as actor_tape:
            # Get current actions from each agent's own actor
            current_act_list = []
            for j in range(self.n):
                if j == agent_idx:
                    # Use this agent's actor with gradients
                    logits = self.actor(obs_n[j], training=True)
                    invalid_mask = 1.0 - state_action_masks_n[j]
                    masked_logits = logits - 1e9 * invalid_mask
                    probs = tf.nn.softmax(masked_logits, axis=-1)
                    current_act_list.append(probs)
                else:
                    # Use other agents' actors without gradients
                    logits = agents[j].actor(obs_n[j], training=False)
                    invalid_mask = 1.0 - state_action_masks_n[j]
                    masked_logits = logits - 1e9 * invalid_mask
                    probs = tf.nn.softmax(masked_logits, axis=-1)
                    current_act_list.append(tf.stop_gradient(probs))
            
            new_act_stacked = tf.stack(current_act_list, axis=1)
            
            q_new = self.critic(obs_stacked, new_act_stacked,
                                agent_index=agent_idx, training=False)
            actor_loss = -tf.reduce_mean(weights * q_new)
            
            reg_loss = self.args.reg_coef * tf.add_n([tf.reduce_sum(tf.square(w)) 
                                                       for w in self.actor.trainable_variables])
            actor_loss = actor_loss + reg_loss
        
        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        td_errors = tf.squeeze(tf.abs(td_error), axis=1)  # (batch,)
        
        return critic_loss, actor_loss, td_errors

    def apply_gradients(self, gradients_dict):
        """Apply target network updates only (gradients already applied in _train_step)"""
        update_target(self.actor, self.actor_target, tau=self.args.tau)
        update_target(self.critic, self.critic_target, tau=self.args.tau)
        return gradients_dict

    def load_models(self, path, version_name):
        file_name = 'a' + str(self.agent_index) + 'A' + version_name
        self.actor.load_weights(path + file_name)
        file_name = 'a' + str(self.agent_index) + 'C' + version_name
        self.critic.load_weights(path + file_name)
        file_name = 'a' + str(self.agent_index) + 'AT' + version_name
        self.actor_target.load_weights(path + file_name)
        file_name = 'a' + str(self.agent_index) + 'CT' + version_name
        self.critic_target.load_weights(path + file_name)

    def save_models(self, path, version_name):
        file_name = 'a' + str(self.agent_index) + 'A' + version_name
        self.actor.save_weights(path + file_name)
        file_name = 'a' + str(self.agent_index) + 'C' + version_name
        self.critic.save_weights(path + file_name)
        file_name = 'a' + str(self.agent_index) + 'AT' + version_name
        self.actor_target.save_weights(path + file_name)
        file_name = 'a' + str(self.agent_index) + 'CT' + version_name
        self.critic_target.save_weights(path + file_name)