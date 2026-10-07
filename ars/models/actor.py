import tensorflow as tf
from tensorflow.keras import layers, Model, Input

class Actor(Model):
    def __init__(self, obs_dim, act_dim, hidden1=64, hidden2=128):
        """
        Actor network for MADDPG
        
        Args:
            obs_dim: Observation space dimension
            act_dim: Action space dimension
            hidden1: Size of first hidden layer
            hidden2: Size of second hidden layer
        """
        super(Actor, self).__init__()
        
        # Define input layer explicitly
        self.input_shape_val = (obs_dim,)
        
        # Define the actor network layers
        self.dense1 = layers.Dense(hidden1, activation='relu', kernel_initializer='glorot_uniform')
        self.dense2 = layers.Dense(hidden2, activation='relu', kernel_initializer='glorot_uniform')
        self.output_layer = layers.Dense(act_dim, kernel_initializer='glorot_uniform')
        
        # Build model properly by passing a dummy input
        dummy_input = tf.keras.Input(shape=(obs_dim,))
        self(dummy_input)

    def call(self, obs, training=False):
        """Forward pass through the network"""
        x = self.dense1(obs)
        x = self.dense2(x)
        logits = self.output_layer(x)
        return logits
        
    def get_action(self, obs, valid_actions_mask=None):
        """
        Get action probabilities from observation
        
        Args:
            obs: Agent observation
            valid_actions_mask: Mask indicating valid actions (optional)
            
        Returns:
            Action probabilities
        """
        logits = self(obs)
        
        # Apply action masking if provided
        if valid_actions_mask is not None:
            invalid_mask = 1.0 - tf.cast(valid_actions_mask, tf.float32)
            logits = logits - 1e9 * invalid_mask
            
        # Apply softmax to get probabilities
        action_probs = tf.nn.softmax(logits, axis=-1)
        return action_probs