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
from ars.models.critic import Critic


# @tf.function
def update_target(model, target_model, tau=0.01):
    polyak = 1.0 - tau
    model_weights = [tf.convert_to_tensor(w, dtype=tf.float32) for w in model.get_weights()]
    target_weights = [tf.convert_to_tensor(w, dtype=tf.float32) for w in target_model.get_weights()]

    # Create a new list to store the updated weights
    updated_weights = []

    # Iterate over each layer's weights and apply Polyak averaging
    for model_w, target_w in zip(model_weights, target_weights):
        # Perform Polyak averaging for each layer
        updated_weight = tf.convert_to_tensor(polyak * target_w + (1 - polyak) * model_w, dtype=tf.float32)
        updated_weights.append(updated_weight)

    # Set the updated weights in the target model
    target_model.set_weights(updated_weights)


class MADDPGSharedTrainer(AgentTrainer):
    def __init__(self, name, learning_rate, obs_shape_n, act_space_n, centralized_buffer, agent_index, args, local_q_func=False):
        self.name = name
        self.learning_rate = learning_rate
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.obs_size = obs_shape_n[agent_index]
        self.joint_obs_size = np.sum(obs_shape_n)
        self.act_size = act_space_n[agent_index].n
        self.act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        # Update device selection to be consistent with main.py
        self.device = None
        if args.device == "gpu":
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # Use the first visible GPU (based on CUDA_VISIBLE_DEVICES)
                self.device = tf.device('/GPU:0')
                # print(f"Agent {self.name} using GPU")
            else:
                self.device = tf.device('/CPU:0')
                print(f"Agent {self.name}: No GPU available, falling back to CPU")
        else:
            self.device = tf.device('/CPU:0')
            print(f"Agent {self.name} using CPU as requested")
        
        tf.config.optimizer.set_jit(True)  # Enable JIT compilation
        
        self.joint_act_size = 0
        for i_act in act_space_n:
            self.joint_act_size += i_act.n
        self.args = args

        self.is_primary = (agent_index == 0)
        
        if self.is_primary:
            with self.device:
                self.actor, self.critic = self.build_model()
                self.actor_target, self.critic_target = self.build_model()
                self.critic_optimizer = AdamW(learning_rate=self.learning_rate, clipnorm=0.5)
                self.actor_optimizer = AdamW(learning_rate=self.learning_rate * self.args.actor_lr, clipnorm=0.5)
            
            self.actor.compile(jit_compile=True)
            self.critic.compile(jit_compile=True)
            self.actor_target.compile(jit_compile=True)
            self.critic_target.compile(jit_compile=True)
        else:
            self.actor = None
            self.critic = None
            self.actor_target = None
            self.critic_target = None
            self.critic_optimizer = None
            self.actor_optimizer = None

        # Create experience buffer
        self.replay_buffer = centralized_buffer
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def set_shared_networks(self, shared_trainer):
        if not self.is_primary:
            self.actor = shared_trainer.actor
            self.critic = shared_trainer.critic
            self.actor_target = shared_trainer.actor_target
            self.critic_target = shared_trainer.critic_target
            self.critic_optimizer = shared_trainer.critic_optimizer
            self.actor_optimizer = shared_trainer.actor_optimizer
    
    def build_model(self):
        """Build actor and critic models"""
        # Create actor model with configurable hidden layers
        actor_model = Actor(
            self.obs_size, 
            self.act_size, 
            hidden1=self.args.actor_hidden1,
            hidden2=self.args.actor_hidden2
        )
        
        # Create critic model with configurable hidden layers
        critic_model = Critic(
            self.joint_obs_size, 
            self.joint_act_size,
            hidden1=self.args.critic_hidden1,
            hidden2=self.args.critic_hidden2
        )
        
        # Compile models for better performance
        actor_model.compile(jit_compile=True)
        critic_model.compile(jit_compile=True)
        
        return actor_model, critic_model
    
    @tf.function(reduce_retracing=True)
    def _get_action_body(self, obs_tensor, valid_actions_mask):
        logits = self.actor(obs_tensor[None])
        
        # Apply mask: set logits of invalid actions to a large negative number
        invalid_mask = 1.0 - tf.cast(valid_actions_mask, tf.float32)
        masked_logits = logits - 1e9 * tf.reshape(invalid_mask, [1, -1])
        
        # Apply softmax on masked logits
        a = tf.nn.softmax(masked_logits, axis=-1)
        return a[0]

    # @tf.function(experimental_relax_shapes=True)
    def compute_gradients(self, agents, t):
        # Sample from replay buffer
        batch_size = min(len(self.replay_buffer), self.args.batch_size)
        # Update unpacking to include action masks
        obs, act, rewards, obs_next, done, masks, state_action_masks, next_state_action_masks, sample_idx, weights = self.replay_buffer.sample(batch_size)
        
        # Split observations and reshape properly
        obs_n = tf.split(obs, self.n, axis=1)
        obs_next_n = tf.split(obs_next, self.n, axis=1)
        act_n = tf.split(act, self.n, axis=1)
        
        # Split action masks and reshape
        state_action_masks_n = tf.split(state_action_masks, self.n, axis=1)
        next_state_action_masks_n = tf.split(next_state_action_masks, self.n, axis=1)
        
        # Reshape tensors to remove extra dimension
        obs_n = [tf.squeeze(o, axis=1) for o in obs_n]  # Shape: (batch_size, obs_size)
        obs_next_n = [tf.squeeze(o, axis=1) for o in obs_next_n]  # Shape: (batch_size, obs_size)
        act_n = [tf.squeeze(a, axis=1) for a in act_n]  # Shape: (batch_size, act_size)
        state_action_masks_n = [tf.squeeze(m, axis=1) for m in state_action_masks_n]  # Shape: (batch_size, act_size)
        next_state_action_masks_n = [tf.squeeze(m, axis=1) for m in next_state_action_masks_n]  # Shape: (batch_size, act_size)
        
        # Cast tensors
        obs = tf.cast(obs_n[self.agent_index], tf.float32)
        act = tf.cast(act_n[self.agent_index], tf.float32)
        rew = tf.cast(rewards[:, self.agent_index], tf.float32)
        obs_next = tf.cast(obs_next_n[self.agent_index], tf.float32)
        done = tf.cast(done, tf.float32)
        weights = tf.cast(weights, tf.float32)
        
        # Cast action masks
        state_action_mask = tf.cast(state_action_masks_n[self.agent_index], tf.float32)
        next_state_action_mask = tf.cast(next_state_action_masks_n[self.agent_index], tf.float32)
        
        # Reshape rewards and done flags
        rew = tf.expand_dims(rew, axis=-1)
        done = tf.expand_dims(done, axis=-1)

        # Get target actions with action masking
        target_act_next_n = []
        for j in range(self.n):
            next_logits = agents[j].actor_target(obs_next_n[j], training=False)
            # Apply action masking to target actions
            invalid_mask = 1.0 - next_state_action_masks_n[j]
            masked_logits = next_logits - 1e9 * invalid_mask
            target_act_next_n.append(tf.nn.softmax(masked_logits, axis=-1))
        
        # Concatenate states and actions efficiently
        next_state_action_n = [
            tf.concat([obs_next_n[j], target_act_next_n[j]], axis=-1)
            for j in range(self.n)
        ]
        next_state_action_attached = tf.concat(next_state_action_n, axis=-1)
        
        # Compute target Q-value in single pass
        target_q_next = self.critic_target(next_state_action_attached, training=False)
        target_q = rew + self.args.gamma * (1.0 - done) * target_q_next
        
        # Current state-action processing
        state_action_n = [
            tf.concat([tf.cast(obs_n[i], tf.float32), 
                      tf.cast(act_n[i], tf.float32)], axis=-1)
            for i in range(self.n)
        ]
        state_action_attached = tf.concat(state_action_n, axis=-1)
        
        # Critic update
        with tf.GradientTape(persistent=True) as tape:
            current_q = self.critic(state_action_attached, training=True)
            td_error = current_q - target_q
            critic_loss = tf.reduce_mean(weights * tf.square(td_error))
        
        # Actor update
            logits = self.actor(obs, training=True)
            # Apply action masking to policy update
            invalid_mask = 1.0 - state_action_mask
            masked_logits = logits - 1e9 * invalid_mask
            
            act_pd = self.act_pdtype_n[self.agent_index].pdfromflat(masked_logits)
            new_act = tf.nn.softmax(masked_logits, axis=-1)
            act_n[self.agent_index] = new_act
            
            # Regularization
            # p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))
            p_reg = self.args.reg_coef * sum(tf.reduce_sum(tf.square(w)) for w in self.actor.trainable_variables)            
            # Recompute state-action pairs with new actions
            state_action_n = [
                tf.concat([tf.cast(obs_n[i], tf.float32), 
                          tf.cast(act_n[i], tf.float32)], axis=-1)
                for i in range(self.n)
            ]
            state_action_attached = tf.concat(state_action_n, axis=-1)
            
            p_loss = -tf.reduce_mean(weights * self.critic(state_action_attached)) + p_reg + 1e-3
        
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        actor_gradients = tape.gradient(p_loss, self.actor.trainable_variables)

        # Update priorities in replay buffer
        td_error_abs = tf.abs(td_error)
        self.replay_buffer.update_priorities(sample_idx, td_error_abs)

        # Return metrics for logging
        return {
            'critic_grads': critic_gradients,
            'actor_grads': actor_gradients,
            'critic_vars': self.critic.trainable_variables,
            'actor_vars': self.actor.trainable_variables,
            'td_errors': td_error_abs,
            'sample_idx': sample_idx,
            'critic_loss': critic_loss,
            'actor_loss': p_loss
        }

    def apply_gradients(self, gradients_dict):
        """Apply pre-computed gradients"""
        self.critic_optimizer.apply_gradients(
            zip(gradients_dict['critic_grads'], gradients_dict['critic_vars'])
        )
        self.actor_optimizer.apply_gradients(
            zip(gradients_dict['actor_grads'], gradients_dict['actor_vars'])
        )
        
        # Update target networks with configurable tau
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
