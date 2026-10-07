import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class AttentionCritic(Model):
    def __init__(self, obs_dim, act_dim, num_agents, embed_dim=128, num_heads=8, 
                 hidden1=256, hidden2=512, attention_temp=1.0, use_layer_norm=False, 
                 dropout_rate=0.0):
        super(AttentionCritic, self).__init__()
        
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention_temp = attention_temp
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate
        
        # Store attention weights for analysis (optional)
        self.store_attention = False
        self.last_attention_weights = None
        
        # State-only encoder: used for Query (determines "who is relevant to me")
        state_encoder_layers = [
            layers.Dense(embed_dim, activation='relu', kernel_initializer='he_normal')
        ]
        if use_layer_norm:
            state_encoder_layers.append(layers.LayerNormalization())
        if dropout_rate > 0:
            state_encoder_layers.append(layers.Dropout(dropout_rate))
        self.state_encoder = tf.keras.Sequential(state_encoder_layers, name='state_encoder')
        
        # State+Action encoder: used for Keys and Values (captures what others are doing)
        sa_encoder_layers = [
            layers.Dense(embed_dim, activation='relu', kernel_initializer='he_normal')
        ]
        if use_layer_norm:
            sa_encoder_layers.append(layers.LayerNormalization())
        if dropout_rate > 0:
            sa_encoder_layers.append(layers.Dropout(dropout_rate))
        self.sa_encoder = tf.keras.Sequential(sa_encoder_layers, name='sa_encoder')
        
        # Attention projections
        self.query_proj = layers.Dense(embed_dim, use_bias=False, 
                                        kernel_initializer='glorot_uniform',
                                        name='query_proj')
        self.key_proj = layers.Dense(embed_dim, use_bias=False,
                                      kernel_initializer='glorot_uniform',
                                      name='key_proj')
        self.value_proj = layers.Dense(embed_dim, use_bias=False,
                                        kernel_initializer='glorot_uniform',
                                        name='value_proj')
        self.output_proj = layers.Dense(embed_dim, use_bias=False,
                                         kernel_initializer='glorot_uniform',
                                         name='output_proj')
        
        # Optional layer norm for attention output
        if use_layer_norm:
            self.attention_ln = layers.LayerNormalization()
        
        # Ego action embedding (separate path, not in attention query)
        self.ego_action_encoder = layers.Dense(embed_dim, activation='relu', 
                                                kernel_initializer='he_normal',
                                                name='ego_action_encoder')
        
        # Final MLP: input = [state_embedding, attended_features, ego_action_embedding] = 3 * embed_dim
        final_layers = [
            layers.Dense(hidden1, activation='relu', kernel_initializer='he_normal')
        ]
        if dropout_rate > 0:
            final_layers.append(layers.Dropout(dropout_rate))
        final_layers.append(layers.Dense(hidden2, activation='relu', kernel_initializer='he_normal'))
        if dropout_rate > 0:
            final_layers.append(layers.Dropout(dropout_rate))
        # Scalar output: Q(s, a) is a single value
        final_layers.append(layers.Dense(1, kernel_initializer='glorot_uniform'))
        
        self.final_mlp = tf.keras.Sequential(final_layers, name='final_mlp')
        
        self._build_model()
        
    def _build_model(self):
        """Build the model with dummy inputs"""
        dummy_obs = tf.zeros((1, self.num_agents, self.obs_dim))
        dummy_act = tf.zeros((1, self.num_agents, self.act_dim))
        self(dummy_obs, dummy_act, agent_index=0)
    
    @tf.function
    def _efficient_multihead_attention(self, query, keys_values, training=False):
        """
        Efficient batched multi-head attention.
        
        Args:
            query: (batch_size, embed_dim) - Query from state-only encoder
            keys_values: (batch_size, num_others, embed_dim) - From state+action encoder
            
        Returns:
            attended_features: (batch_size, embed_dim)
            attention_weights: (batch_size, num_heads, num_others)
        """
        batch_size = tf.shape(query)[0]
        num_others = tf.shape(keys_values)[1]
        
        q = self.query_proj(query)
        k = self.key_proj(keys_values)
        v = self.value_proj(keys_values)
        
        q = tf.reshape(q, [batch_size, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch_size, num_others, self.num_heads, self.head_dim])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.reshape(v, [batch_size, num_others, self.num_heads, self.head_dim])
        v = tf.transpose(v, [0, 2, 1, 3])
        
        scale = tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        q_expanded = tf.expand_dims(q, axis=2)
        scores = tf.matmul(q_expanded, k, transpose_b=True) / scale
        scores = tf.squeeze(scores, axis=2)
        
        scores = scores / self.attention_temp
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        attn_expanded = tf.expand_dims(attention_weights, axis=2)
        attended = tf.matmul(attn_expanded, v)
        attended = tf.squeeze(attended, axis=2)
        
        attended = tf.reshape(attended, [batch_size, self.embed_dim])
        attended = self.output_proj(attended)
        
        return attended, attention_weights
    
    def call(self, obs_n, act_n, agent_index=0, training=False):
        """
        Forward pass with MAAC-style architecture:
        - Query derived from state only
        - Keys/Values derived from state+action of other agents
        - Ego action fed separately into final MLP
        - Scalar Q-value output
        
        Args:
            obs_n: (batch_size, num_agents, obs_dim)
            act_n: (batch_size, num_agents, act_dim)
            agent_index: Index of the agent this critic evaluates
            training: Whether in training mode
            
        Returns:
            q_value: (batch_size, 1)
        """
        batch_size = tf.shape(obs_n)[0]
        
        if isinstance(agent_index, int):
            agent_index = tf.constant(agent_index)
        
        # State-only encoding for query agent
        ego_obs = obs_n[:, agent_index, :]
        ego_state_embedding = self.state_encoder(ego_obs, training=training)
        
        # Ego action encoding (separate path)
        ego_act = act_n[:, agent_index, :]
        ego_action_embedding = self.ego_action_encoder(ego_act)
        
        if self.num_agents > 1:
            # State+Action encoding for other agents (Keys/Values)
            other_sa_inputs = tf.concat([obs_n, act_n], axis=-1)
            other_sa_embeddings = self.sa_encoder(other_sa_inputs, training=training)
            
            indices = tf.range(tf.shape(other_sa_embeddings)[1])
            mask = tf.not_equal(indices, agent_index)
            other_embeddings = tf.boolean_mask(other_sa_embeddings, mask, axis=1)
            
            attended_features, attention_weights = self._efficient_multihead_attention(
                ego_state_embedding, other_embeddings, training=training
            )
            
            if self.store_attention:
                self.last_attention_weights = attention_weights
            
            if self.use_layer_norm:
                attended_features = self.attention_ln(attended_features)
            
            combined_features = tf.concat([ego_state_embedding, attended_features, ego_action_embedding], axis=-1)
        else:
            combined_features = tf.concat([
                ego_state_embedding, 
                tf.zeros_like(ego_state_embedding), 
                ego_action_embedding
            ], axis=-1)
        
        q_value = self.final_mlp(combined_features, training=training)
        return q_value
    
    def get_attention_weights(self):
        return self.last_attention_weights