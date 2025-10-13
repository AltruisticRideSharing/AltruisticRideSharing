import tensorflow as tf
from tensorflow.keras import layers, Model

class Critic(Model):
    def __init__(self, joint_obs_dim, joint_act_dim, hidden1=128, hidden2=256):
        """
        Critic network for MADDPG
        
        Args:
            joint_obs_dim: Dimension of joint observations from all agents
            joint_act_dim: Dimension of joint actions from all agents
            hidden1: Size of first hidden layer
            hidden2: Size of second hidden layer
        """
        super(Critic, self).__init__()
        
        # Store input dimensions
        self.input_shape_val = (joint_obs_dim + joint_act_dim,)
        
        # Define the critic network layers
        self.dense1 = layers.Dense(hidden1, activation='relu', kernel_initializer='glorot_uniform')
        self.dense2 = layers.Dense(hidden2, activation='relu', kernel_initializer='glorot_uniform')
        self.output_layer = layers.Dense(1, kernel_initializer='glorot_uniform')
        
        # Build model properly by passing a dummy input
        dummy_input = tf.keras.Input(shape=(joint_obs_dim + joint_act_dim,))
        self(dummy_input)
        
    def call(self, inputs, training=False):
        """Forward pass through the network"""
        x = self.dense1(inputs)
        x = self.dense2(x)
        q_value = self.output_layer(x)
        return q_value