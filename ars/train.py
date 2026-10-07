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
from ars.env.env import Env
# Import both model implementations
from ars.models.maddpg import MADDPGTrainer
from ars.models.attention import AttentionTrainer
from ars.utils.prioritized_replaybuffer import CentralizedReplayBuffer
from ars.utils.vec_env import SubprocVecEnv
from datetime import datetime
import json
import wandb

# Add TensorBoard argument to parse_args
def parse_args():
    parser = argparse.ArgumentParser("Multiagent reinforcement learning with MADDPG")
    
    # Environment and scenario parameters
    parser.add_argument("--max-episode-len", type=int, default=1000, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=100, help="number of episodes")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size for optimization")
    parser.add_argument("--dataset", type=str, default="100_agents", help="dataset folder")
    parser.add_argument("--num-agents", type=int, default=100, help="number of agents in the environment")
    parser.add_argument("--initial-active-agents", type=int, default=100, help="number of initially active agents")
    parser.add_argument("--grid-size", type=int, default=15, help="side length of the square grid (height=width)")
    parser.add_argument("--perception-field", type=int, default=None,
                        help="egocentric perception field side length n (odd). Defaults to ~grid/3 snapped to odd")

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
    
    # Model type selection
    parser.add_argument("--model", type=str, default="maddpg", choices=["maddpg", "attention"], 
                       help="Model type: maddpg (default) or attention (Attention-MADDPG)")
    
    # Attention specific parameters
    parser.add_argument("--num-heads", type=int, default=4, help="number of attention heads (only for attention model)")
    
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
    
    # Weights & Biases logging
    parser.add_argument("--wandb-project", type=str, default="altruistic-ridesharing",
                        help="wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="wandb entity (team/user); None uses your default")
    parser.add_argument("--wandb-mode", type=str, default="online",
                        choices=["online", "offline", "disabled"],
                        help="wandb mode: online (sync to cloud), offline (local), or disabled")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="explicit wandb run name; defaults to an auto-generated config string")
    
    # GPU configuration parameters
    parser.add_argument("--device", type=str, default="cpu", choices=["gpu", "cpu"], 
                       help="Device to use: 'gpu' or 'cpu'")
    parser.add_argument("--gpu-id", type=str, default="0", 
                       help="GPU ID(s) to use (comma-separated for multiple GPUs, or 'all' for all available GPUs)")
    parser.add_argument("--memory-growth", action="store_true", default=True, 
                       help="Enable memory growth for GPU (helps with OOM errors)")
    parser.add_argument("--memory-limit", type=int, default=None,
                       help="Limit GPU memory in MB (None = no limit)")
    
    # Add tuning-specific argument
    parser.add_argument("--tune-mode", type=str, default=None, 
                       help="If specified, output metrics to this JSON file for hyperparameter tuning")

    # Add parallel environments argument
    parser.add_argument("--num-envs", type=int, default=1, help="number of parallel environments")
    
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
    
    # Choose agent trainer based on model argument
    if arglist.model == "attention":
        AgentTrainerClass = AttentionTrainer
        print("Using Attention MADDPG model")
    else:
        AgentTrainerClass = MADDPGTrainer
        print("Using standard MADDPG model")
    
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

def resolve_model_label(arglist):
    """Paper-facing model name. train.py runs agents with independent (unshared)
    networks, so attention here is MAAC (not ORACLE) and maddpg is MADDPG."""
    return "maac" if arglist.model == "attention" else "maddpg"


def build_save_dir(arglist):
    """Hierarchical checkpoint dir: <save_dir>/<model>/grid<N>/<agents>agents/."""
    path = os.path.join(arglist.save_dir, resolve_model_label(arglist),
                        f"grid{arglist.grid_size}", f"{arglist.num_agents}agents")
    os.makedirs(path, exist_ok=True)
    return path


def init_wandb(arglist):
    """Initialise a wandb run mirroring the previous TensorBoard config."""
    model_label = resolve_model_label(arglist)
    run_name = arglist.wandb_run_name or (
        f"{model_label}_grid{arglist.grid_size}_{arglist.num_agents}agents"
    )
    return wandb.init(
        project=arglist.wandb_project,
        entity=arglist.wandb_entity,
        mode=arglist.wandb_mode,
        name=run_name,
        config=vars(arglist),
    )

# Create function to generate masks from state
def generate_action_masks(state, infos, num_agents):
    action_masks = np.zeros((num_agents, num_agents + 1), dtype=np.float32)
    for i in range(num_agents):
        if infos[i] == 1:  # If agent is active
            nearby_grid = np.array(state[i][3:], dtype=np.int32)
            nearby_rider_ids = nearby_grid[nearby_grid != -1]
            valid_actions = np.concatenate((nearby_rider_ids, [num_agents]))
            action_masks[i][valid_actions] = 1.0
    return action_masks

def make_env(dataset_folder, num_agents, initial_active_agents, grid_size=15, perception_field=None):
    def _init():
        return Env(dataset_folder=dataset_folder, numAgents=num_agents, initial_active_agents=initial_active_agents,
                   height=grid_size, width=grid_size, perception_field=perception_field)
    return _init

def save_tune_metrics(metrics_dict, output_file):
    """Save metrics for hyperparameter tuning to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(metrics_dict, f)

def train(arglist, days):   
    if arglist.num_envs > 1:
        env_fns = [make_env(arglist.dataset, arglist.num_agents, arglist.initial_active_agents,
                            grid_size=arglist.grid_size, perception_field=arglist.perception_field)
                   for _ in range(arglist.num_envs)]
        vec_env = SubprocVecEnv(env_fns)
        print(f"Created {arglist.num_envs} parallel environments")
        
        # Scale batch size and learning rate for parallel environments
        arglist.batch_size = arglist.batch_size * arglist.num_envs
        arglist.lr = arglist.lr * math.sqrt(arglist.num_envs)
        print(f"Scaled batch size to {arglist.batch_size} and LR to {arglist.lr:.6f}")
    else:
        vec_env = Env(dataset_folder=arglist.dataset, numAgents=arglist.num_agents,
                      initial_active_agents=arglist.initial_active_agents,
                      height=arglist.grid_size, width=arglist.grid_size,
                      perception_field=arglist.perception_field)
        print("Using single environment (no parallelization)")

    episodes_per_day = max(1, arglist.num_episodes // max(1, arglist.num_envs))
    print(f"Episodes per day per environment: {episodes_per_day}")

    if arglist.num_envs > 1:
        obs_space = vec_env.get_attr('observation_space', indices=0)[0]()
        actual_obs_dim = obs_space.nvec.shape[0]
        action_space = vec_env.get_attr('action_space', indices=0)[0]()
        act_space_n = [action_space for _ in range(vec_env.numAgents)]
    else:
        obs_space = vec_env.observation_space()
        actual_obs_dim = obs_space.nvec.shape[0]
        act_space_n = [vec_env.action_space() for _ in range(vec_env.numAgents)]

    obs_shape_n = [actual_obs_dim for _ in range(vec_env.numAgents)]
    
    # Use buffer size from argparse
    centralized_buffer = CentralizedReplayBuffer(
        num_agents=vec_env.numAgents,
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
    trainers = get_trainers(vec_env, obs_shape_n, act_space_n, arglist.lr, centralized_buffer, arglist)
    
    # Setup TensorBoard summary writer with model type in directory path
    wandb_run = init_wandb(arglist)
    run_save_dir = build_save_dir(arglist)
    print(f"Checkpoints will be saved under: {run_save_dir}")
    
    # Track metrics across all days
    global_episode = 0
    
    # Add these variables to track metrics
    all_episode_rewards = []
    evaluation_rewards = []
    
    for day in range(days):
        vec_env.reset_day(day + 1, enable_birth_death=False)  # Disable dropout during training
        print(f"Starting Day {day + 1}/{days}")
        
        # Load models if restoring
        if arglist.restore:
            load_models(trainers, run_save_dir, day, arglist.model)

        # Define the initial epsilon and decay parameters from args
        epsilon = arglist.epsilon_start
        min_epsilon = arglist.epsilon_min
        decay_start = 0
        decay_end = int(episodes_per_day * arglist.epsilon_decay_episodes)
        lambda_decay = -math.log(min_epsilon) / (decay_end - decay_start)

        episode_rewards = []
        for episode in range(episodes_per_day):
            states, infos_all = vec_env.reset(enable_dropout=False)  # Disable dropout during training

            if arglist.num_envs > 1:
                num_envs = arglist.num_envs
            else:
                num_envs = 1
                states = states[np.newaxis, :]
                infos_all = infos_all[np.newaxis, :]

            dones = np.zeros(num_envs, dtype=bool)
            steps = 0
            episode_rewards_all = np.zeros((num_envs, vec_env.numAgents))
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
            
            while not np.all(dones):
                actions_all = []

                for env_idx in range(num_envs):
                    if dones[env_idx]:
                        actions_all.append(np.full(vec_env.numAgents, vec_env.numAgents, dtype=np.int32))
                        continue

                    state = states[env_idx]
                    infos = infos_all[env_idx]
                    actions = np.zeros(vec_env.numAgents, dtype=np.int32)
                    num_valid_actions = np.zeros(vec_env.numAgents, dtype=np.int32)

                    for i in range(vec_env.numAgents):
                        if infos[i] == 1:
                            nearby_grid = np.array(state[i][3:], dtype=np.int32)
                            nearby_rider_ids = nearby_grid[nearby_grid != -1]
                            valid_actions = np.concatenate((nearby_rider_ids, [vec_env.numAgents]))
                            num_valid_actions[i] = len(valid_actions)

                            action_mask = np.zeros(vec_env.numAgents + 1, dtype=np.int32)
                            action_mask[valid_actions] = 1

                            if random.random() < epsilon:
                                actions[i] = np.random.choice(valid_actions)
                            else:
                                action_probs = trainers[i]._get_action_body(state[i], action_mask)
                                actions[i] = np.argmax(action_probs)

                    actions_all.append(actions)
                    train_metrics['valid_actions_count'].append(np.mean(num_valid_actions))
                
                if not is_evaluation and episode <= decay_end:
                    epsilon = 1.0 - (episode / decay_end) * (1.0 - min_epsilon)
                    epsilon = max(min_epsilon, epsilon)
                elif not is_evaluation and episode > decay_end:
                    epsilon = min_epsilon
                
                if arglist.num_envs > 1:
                    next_states, rewards, dones_step, next_infos_all = vec_env.step(actions_all)
                else:
                    next_state, reward, done, next_infos = vec_env.step(actions_all[0])
                    next_states = next_state[np.newaxis, :]
                    rewards = reward[np.newaxis, :]
                    dones_step = np.array([done])
                    next_infos_all = next_infos[np.newaxis, :]

                dones = np.logical_or(dones, dones_step.astype(bool))

                for env_idx in range(num_envs):
                    if dones[env_idx]:
                        continue

                    state = states[env_idx]
                    actions = actions_all[env_idx]
                    reward = rewards[env_idx]
                    next_state = next_states[env_idx]
                    infos = infos_all[env_idx]
                    next_infos = next_infos_all[env_idx]

                    state_action_masks = generate_action_masks(state, infos, vec_env.numAgents)
                    next_state_action_masks = generate_action_masks(next_state, next_infos, vec_env.numAgents)

                    episode_rewards_all[env_idx] += reward
                    steps += 1

                    if not is_evaluation:
                        one_hot_actions = np.zeros((len(actions), vec_env.numAgents + 1), dtype=np.int32)
                        for i, action in enumerate(actions):
                            one_hot_actions[i][action] = 1

                        infos_reshaped = tf.reshape(infos, [-1, 1])
                        infos_reshaped = tf.cast(infos_reshaped, dtype=state.dtype)
                        masked_state = state * infos_reshaped
                        masked_one_hot_actions = one_hot_actions * infos_reshaped
                        masked_next_state = next_state * infos_reshaped

                        store = 0
                        for valid_count in np.sum(state_action_masks, axis=1):
                            if valid_count > 1:
                                store = 1
                        if store == 1:
                            centralized_buffer.add(
                                masked_state,
                                masked_one_hot_actions,
                                reward,
                                masked_next_state,
                                dones_step[env_idx],
                                infos,
                                state_action_masks,
                                next_state_action_masks
                            )

                states = next_states
                infos_all = next_infos_all

            centralized_buffer.update_alpha_beta(episode, max_episodes=arglist.num_episodes)

            if not is_evaluation and len(centralized_buffer) >= arglist.batch_size:
                # Compute and apply gradients
                gradients_dict_list = []
                for agent_id, agent in enumerate(trainers):
                    if np.any(infos_all[:, agent_id] == 1):
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

            # Log metrics to Weights & Biases
            avg_episode_rewards = np.mean(episode_rewards_all, axis=0)
            log_dict = {
                'parameters/epsilon': epsilon,
                'parameters/buffer_beta': centralized_buffer.beta,
                'parameters/buffer_alpha': centralized_buffer.alpha,
                'parameters/buffer_size': centralized_buffer.size,
                'progress/day': day + 1,
                'progress/episode': episode,
                'progress/steps': steps,
            }
            if is_evaluation:
                log_dict['val_rewards/total_reward'] = np.sum(avg_episode_rewards)
                log_dict['val_rewards/mean_reward'] = np.mean(avg_episode_rewards)
            else:
                log_dict['rewards/total_reward'] = np.sum(avg_episode_rewards)
                log_dict['rewards/mean_reward'] = np.mean(avg_episode_rewards)
            for i in range(vec_env.numAgents):
                log_dict[f'rewards/agent_{i}'] = avg_episode_rewards[i]
            for metric_name, values in train_metrics.items():
                if values:
                    log_dict[f'training/{metric_name}'] = np.mean(values)
            if arglist.model == "attention":
                log_dict['parameters/attention_heads'] = arglist.num_heads
            wandb.log(log_dict, step=global_episode)

            if is_evaluation:
                evaluation_rewards.append(float(np.sum(avg_episode_rewards)))
            else:
                all_episode_rewards.append(avg_episode_rewards.copy())

            print(f"Episode {episode}/{episodes_per_day}, Avg Rewards: {', '.join([f'Agent {i}: {r:.2f}' for i, r in enumerate(avg_episode_rewards)])}, Total Reward: {np.sum(avg_episode_rewards):.2f}")
            global_episode += 1

    save_models(trainers, run_save_dir, arglist.model)

    if arglist.num_envs > 1:
        vec_env.close()

    print(f"Model saved")
    if wandb_run is not None:
        wandb_run.finish()
        print(f"wandb run finished ({arglist.wandb_mode} mode).")

    # After the training loop completes:
    if arglist.tune_mode:
        # Calculate metrics for tuning
        metrics = {
            "mean_reward": float(np.mean([np.sum(r) for r in all_episode_rewards])),
            "final_reward": float(np.sum(episode_rewards)),  # Last episode reward
            "mean_eval_reward": float(np.mean(evaluation_rewards)) if evaluation_rewards else 0.0,
        }
        save_tune_metrics(metrics, arglist.tune_mode)

def main():
    arglist = parse_args()
    # Configure GPU settings before creating the environment or any TensorFlow operations
    configure_gpus(arglist)
    train(arglist, days=arglist.days)


if __name__ == "__main__":
    main()