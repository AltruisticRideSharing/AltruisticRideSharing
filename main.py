import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import pickle
import copy
import math
import random
import os
from env.env import Env
# Import both model implementations
from models.maddpg import MADDPGAgentTrainer
from utils.prioritized_replaybuffer import CentralizedReplayBuffer
from datetime import datetime
import json

# Add TensorBoard argument to parse_args
def parse_args():
    parser = argparse.ArgumentParser("Multiagent reinforcement learning with MADDPG")
    
    # Environment and scenario parameters
    parser.add_argument("--max-episode-len", type=int, default=1000, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=100, help="number of episodes")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size for optimization")
    parser.add_argument("--dataset", type=str, default="150_agents", help="dataset folder")
    
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--actor-lr", type=float, default=10, help="actor learning rate multiplier")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--reg-coef", type=float, default=1e-4, help="Regularization coefficient for actor loss")
    
    # Network architecture
    parser.add_argument("--actor-hidden1", type=int, default=64, help="actor first hidden layer size")
    parser.add_argument("--actor-hidden2", type=int, default=128, help="actor second hidden layer size")
    parser.add_argument("--critic-hidden1", type=int, default=128, help="critic first hidden layer size")
    parser.add_argument("--critic-hidden2", type=int, default=256, help="critic second hidden layer size")
    
    # Polyak averaging coefficient
    parser.add_argument("--tau", type=float, default=0.01, help="target network update rate (1-polyak)")
    
    # Exploration parameters
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="starting epsilon for exploration")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="minimum epsilon value")
    parser.add_argument("--epsilon-decay-episodes", type=float, default=0.8, 
                        help="fraction of episodes over which to decay epsilon")
    
    # Buffer parameters
    parser.add_argument("--buffer-size", type=int, default=600000, help="size of the replay buffer")
    parser.add_argument("--alpha-initial", type=float, default=0.3, help="initial alpha for prioritized replay")
    parser.add_argument("--alpha-final", type=float, default=1.0, help="final alpha for prioritized replay")
    parser.add_argument("--beta-initial", type=float, default=0.3, help="initial beta for importance sampling")
    parser.add_argument("--beta-final", type=float, default=1.0, help="final beta for importance sampling")
    parser.add_argument("--alpha-decay", type=float, default=1e-5, help="decay rate for alpha")
    parser.add_argument("--beta-decay", type=float, default=1e-5, help="decay rate for beta")
    parser.add_argument("--min-priority", type=float, default=1e-5, help="minimum priority for samples")
    
    # Save and load
    parser.add_argument("--save-dir", type=str, default="savedagents", help="directory where models will be saved")
    parser.add_argument("--restore", action="store_true", default=False, help="whether to restore from a previously saved model")
    
    # Add days argument
    parser.add_argument("--days", type=int, default=1, help="maximum number of days to train")
    
    # Add TensorBoard log directory
    parser.add_argument("--log-dir", type=str, default="logs", help="TensorBoard log directory")
    
    # GPU configuration parameters
    parser.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"], 
                       help="Device to use: 'gpu' (default) or 'cpu'")
    parser.add_argument("--gpu-id", type=str, default="0", 
                       help="GPU ID(s) to use (comma-separated for multiple GPUs, or 'all' for all available GPUs)")
    parser.add_argument("--memory-growth", action="store_true", default=True, 
                       help="Enable memory growth for GPU (helps with OOM errors)")
    parser.add_argument("--memory-limit", type=int, default=None,
                       help="Limit GPU memory in MB (None = no limit)")
    
    # Add tuning-specific argument
    parser.add_argument("--tune-mode", type=str, default=None, 
                       help="If specified, output metrics to this JSON file for hyperparameter tuning")
    
    return parser.parse_args()

def configure_gpus(arglist):
    """Configure GPU settings based on command line arguments"""
    print("Configuring device settings...")
    
    if arglist.device == "cpu":
        print("Using CPU for computation")
        tf.config.set_visible_devices([], 'GPU')
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return
    
    # Check available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPU found. Falling back to CPU.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return
    
    # Configure which GPUs to use
    if arglist.gpu_id.lower() == 'all':
        # Using all GPUs
        visible_gpu_indices = list(range(len(gpus)))
        print(f"Using all available GPUs: {len(gpus)} found")
    else:
        try:
            gpu_ids = [int(id) for id in arglist.gpu_id.split(',')]
            visible_gpu_indices = [id for id in gpu_ids if id < len(gpus)]
            if not visible_gpu_indices:
                print(f"No valid GPU IDs specified. Using GPU 0.")
                visible_gpu_indices = [0] if gpus else []
            
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, visible_gpu_indices))
            print(f"Using GPU(s): {visible_gpu_indices}")
        except ValueError:
            print(f"Invalid GPU ID format. Using GPU 0.")
            visible_gpu_indices = [0] if gpus else []
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # IMPORTANT: Apply memory growth setting to ALL GPUs before limiting visible ones
    # This fixes the "Memory growth cannot differ between GPU devices" error
    if arglist.memory_growth:
        print("Enabling memory growth for ALL GPUs")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Memory growth setting error: {e}")
    
    # Set visible devices AFTER setting memory growth
    visible_gpus = [gpus[i] for i in visible_gpu_indices if i < len(gpus)]
    tf.config.set_visible_devices(visible_gpus, 'GPU')
    
    # Set memory limit if specified (only for visible GPUs)
    if arglist.memory_limit:
        print(f"Setting GPU memory limit to {arglist.memory_limit} MB")
        try:
            for gpu in visible_gpus:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=arglist.memory_limit)]
                )
        except Exception as e:
            print(f"Cannot set memory limit: {e}")
    
    try:
        # Print final TensorFlow device configuration
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"TensorFlow sees {len(logical_gpus)} logical GPU(s)")
    except ValueError as e:
        print(f"Error listing logical devices: {e}")
        print("This is expected if you modified GPU settings - continuing with available devices")

def get_trainers(env, obs_shape_n, act_space_n, lr, centralized_buffer, arglist):
    trainers = []
    
    AgentTrainerClass = MADDPGAgentTrainer
    
    for i in range(env.numAgents):
        trainers.append(AgentTrainerClass(
            name=f"agent_{i}",
            learning_rate=lr,
            obs_shape_n=obs_shape_n,
            act_space_n=act_space_n,
            centralized_buffer=centralized_buffer,
            agent_index=i,
            args=arglist,
            local_q_func=False  
        ))
    return trainers

@tf.numpy_function(Tout=[tf.int32, tf.float32, tf.int32, tf.float32])
def env_step(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    next_state, reward, done, infos = env.step(actions)
    return (np.array(next_state, dtype=np.int32), np.array(reward, dtype=np.float32), np.array(done, dtype=np.int32), np.array(infos, dtype=np.float32))

def save_models(trainers, save_dir, model_type):
    """Save all agent weights into a single file using pickle."""
    weights_dict = {}
    for i, agent in enumerate(trainers):
        weights_dict[f"agent_{i}_actor"] = agent.actor.get_weights()
        weights_dict[f"agent_{i}_critic"] = agent.critic.get_weights()
    
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Include model type in filename
    save_path = os.path.join(save_dir, f"final_weights_{model_type}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(weights_dict, f)
    print(f"All agent weights saved to {save_path}")

def load_models(trainers, save_dir, day, model_type):
    """Load all agent weights from a single file using pickle."""
    load_path = os.path.join(save_dir, f"day_{day}_weights_{model_type}.pkl")
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No checkpoint found at {load_path}")
    
    # Load weights from the file
    with open(load_path, "rb") as f:
        weights_dict = pickle.load(f)
    
    # Assign weights to each agent
    for i, agent in enumerate(trainers):
        agent.actor.set_weights(weights_dict[f"agent_{i}_actor"])
        agent.critic.set_weights(weights_dict[f"agent_{i}_critic"])
    print(f"All agent weights loaded from {load_path}")

# Add TensorBoard summary writer function
def create_summary_writer(log_dir, model_type):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(log_dir, f"{model_type}_{timestamp}")
    return tf.summary.create_file_writer(logdir)

# Create function to generate masks from state
def generate_action_masks(state, infos):
    action_masks = np.zeros((env.numAgents, env.numAgents + 1), dtype=np.float32)
    for i, agent in enumerate(env.agents):
        if infos[i] == 1:  # If agent is active
            nearby_grid = np.array(state[i][3:], dtype=np.int32)
            nearby_rider_ids = nearby_grid[nearby_grid != -1]
            valid_actions = np.concatenate((nearby_rider_ids, [env.numAgents]))
            action_masks[i][valid_actions] = 1.0
    return action_masks

def save_tune_metrics(metrics_dict, output_file):
    """Save metrics for hyperparameter tuning to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(metrics_dict, f)

def train(arglist, days):   
    obs_shape_n = [env.observation_space().shape[0] for i in range(env.numAgents)]
    act_space_n = [env.action_space() for i in range(env.numAgents)]
    
    # Use buffer size from argparse
    centralized_buffer = CentralizedReplayBuffer(
        num_agents=env.numAgents, 
        state_dim=obs_shape_n[0], 
        action_dim=act_space_n[0].n, 
        max_size=arglist.buffer_size,
        alpha_initial=arglist.alpha_initial,
        alpha_final=arglist.alpha_final,
        beta_initial=arglist.beta_initial,
        beta_final=arglist.beta_final,
        alpha_decay=arglist.alpha_decay,
        beta_decay=arglist.beta_decay,
        min_priority=arglist.min_priority
    )
    
    # Print which model type is being used
    print(f"Training with model type: {arglist.model.upper()}")
    
    # Initialize trainers only once
    trainers = get_trainers(env, obs_shape_n, act_space_n, arglist.lr, centralized_buffer, arglist)
    
    # Setup TensorBoard summary writer with model type in directory path
    summary_writer = create_summary_writer(arglist.log_dir, arglist.model)
    
    # Track metrics across all days
    global_episode = 0
    
    # Add these variables to track metrics
    all_episode_rewards = []
    evaluation_rewards = []
    
    for day in range(days):
        env.reset_day(day + 1, enable_dropout=False)  # Disable dropout during training
        print(f"Starting Day {day + 1}/{days}")
        
        # Load models if restoring
        if arglist.restore:
            load_models(trainers, arglist.save_dir, day, arglist.model)

        # Define the initial epsilon and decay parameters from args
        epsilon = arglist.epsilon_start
        min_epsilon = arglist.epsilon_min
        decay_start = 0
        decay_end = int(arglist.num_episodes * arglist.epsilon_decay_episodes)
        lambda_decay = -math.log(min_epsilon) / (decay_end - decay_start)

        episode_rewards = []
        for episode in range(arglist.num_episodes):
            state, infos = env.reset(enable_dropout=False)  # Disable dropout during training
            done = False
            steps = 0
            episode_rewards = np.zeros(env.numAgents)
            print(f"Episode {episode}: Epsilon = {epsilon:.4f}")
            
            # Metrics to track during episode
            train_metrics = {
                'critic_loss': [],
                'actor_loss': [],
                'td_errors': [],
                'valid_actions_count': []
            }
            
            # Disable exploration and training for the last episode (evaluation)
            is_evaluation = (episode == arglist.num_episodes - 1)
            if is_evaluation:
                epsilon = 0.0
                print("Evaluation mode: Exploration disabled (epsilon = 0).")
            
            while not done:
                actions = np.zeros(env.numAgents, dtype=np.int32)
                num_valid_actions = np.zeros(env.numAgents, dtype=np.int32)

                for i, agent in enumerate(env.agents):
                    if infos[i] == 1:
                        nearby_grid = np.array(state[i][3:], dtype=np.int32)
                        nearby_rider_ids = nearby_grid[nearby_grid != -1]
                        valid_actions = np.concatenate((nearby_rider_ids, [env.numAgents]))
                        num_valid_actions[i] = len(valid_actions)
                        
                        action_mask = np.zeros(env.numAgents + 1, dtype=np.int32)
                        action_mask[valid_actions] = 1
                        
                        if random.random() < epsilon:
                            actions[i] = np.random.choice(valid_actions)
                        else:
                            action_probs = trainers[i]._get_action_body(state[i], action_mask)
                            actions[i] = np.argmax(action_probs)
                
                train_metrics['valid_actions_count'].append(np.mean(num_valid_actions))
                
                if not is_evaluation and episode <= decay_end:
                    epsilon = 1.0 - (episode / decay_end) * (1.0 - min_epsilon)
                    epsilon = max(min_epsilon, epsilon)
                elif not is_evaluation and episode > decay_end:
                    epsilon = min_epsilon
                
                # Generate masks for current state
                state_action_masks = generate_action_masks(state, infos)

                next_state, reward, done, infos = env_step(actions)

                # Generate masks for next state
                next_state_action_masks = generate_action_masks(next_state, infos)

                episode_rewards += reward
                steps += 1

                one_hot_actions = np.zeros((len(actions), env.numAgents + 1), dtype=np.int32)
                for i, action in enumerate(actions):
                    one_hot_actions[i][action] = 1

                infos_reshaped = tf.reshape(infos, [-1, 1])
                infos_reshaped = tf.cast(infos_reshaped, dtype=state.dtype)
                masked_state = state * infos_reshaped
                masked_one_hot_actions = one_hot_actions * infos_reshaped
                masked_next_state = next_state * infos_reshaped
                
                if not is_evaluation:
                    store = 0
                    for i in num_valid_actions:
                        if i > 1:
                            store = 1
                    if store == 1:
                        centralized_buffer.add(
                            masked_state, 
                            masked_one_hot_actions, 
                            reward, 
                            masked_next_state, 
                            done, 
                            infos, 
                            state_action_masks,
                            next_state_action_masks
                        )
                
                state = next_state  

            centralized_buffer.update_alpha_beta(episode, max_episodes=arglist.num_episodes)

            if not is_evaluation:
                # Compute and apply gradients
                gradients_dict_list = []
                for agent_id, agent in enumerate(trainers):
                    if infos[agent_id] == 1:
                        gradients_dict = agent.compute_gradients(trainers, steps)
                        gradients_dict_list.append((agent_id, gradients_dict))
                        
                        # Collect training metrics
                        if 'critic_loss' in gradients_dict:
                            train_metrics['critic_loss'].append(gradients_dict['critic_loss'])
                        if 'actor_loss' in gradients_dict:
                            train_metrics['actor_loss'].append(gradients_dict['actor_loss'])
                        if 'td_errors' in gradients_dict:
                            train_metrics['td_errors'].append(tf.reduce_mean(gradients_dict['td_errors']))
                
                # Apply gradients for all agents
                for agent_id, gradients_dict in gradients_dict_list:
                    trainers[agent_id].apply_gradients(gradients_dict)

            # Log metrics to TensorBoard
            with summary_writer.as_default():
                # Log episode rewards
                tf.summary.scalar('rewards/total_reward', np.sum(episode_rewards), step=global_episode)
                tf.summary.scalar('rewards/mean_reward', np.mean(episode_rewards), step=global_episode)
                for i in range(env.numAgents):
                    tf.summary.scalar(f'rewards/agent_{i}', episode_rewards[i], step=global_episode)
                
                # Log training metrics
                for metric_name, values in train_metrics.items():
                    if values:  # Check if list is not empty
                        tf.summary.scalar(f'training/{metric_name}', np.mean(values), step=global_episode)
                
                # Log exploration parameters
                tf.summary.scalar('parameters/epsilon', epsilon, step=global_episode)
                tf.summary.scalar('parameters/buffer_beta', centralized_buffer.beta, step=global_episode)
                tf.summary.scalar('parameters/buffer_alpha', centralized_buffer.alpha, step=global_episode)
                tf.summary.scalar('parameters/buffer_size', centralized_buffer.size, step=global_episode)
                
                # Log day and episode information
                tf.summary.scalar('progress/day', day + 1, step=global_episode)
                tf.summary.scalar('progress/episode', episode, step=global_episode)
                tf.summary.scalar('progress/steps', steps, step=global_episode)

            print(f"Episode {episode}/{arglist.num_episodes}, Rewards: {', '.join([f'Agent {i}: {r:.2f}' for i, r in enumerate(episode_rewards)])}, Total Reward: {np.sum(episode_rewards):.2f}")
            global_episode += 1

    save_models(trainers, arglist.save_dir, arglist.model)
    print(f"Model saved")
    print(f"TensorBoard logs saved to {arglist.log_dir}. Run 'tensorboard --logdir={arglist.log_dir}' to view.")

    # After the training loop completes:
    if arglist.tune_mode:
        # Calculate metrics for tuning
        metrics = {
            "mean_reward": float(np.mean([np.sum(r) for r in all_episode_rewards])),
            "final_reward": float(np.sum(episode_rewards)),  # Last episode reward
            "mean_eval_reward": float(np.mean(evaluation_rewards)) if evaluation_rewards else 0.0,
        }
        save_tune_metrics(metrics, arglist.tune_mode)

if __name__ == "__main__":
    arglist = parse_args()
    
    # Configure GPU settings before creating the environment or any TensorFlow operations
    configure_gpus(arglist)
    
    env = Env(dataset_folder=arglist.dataset)
    train(arglist, days=arglist.days)