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
from ars.models.maddpg_shared import MADDPGSharedTrainer
from ars.models.attention import AttentionTrainer
from ars.utils.prioritized_replaybuffer import CentralizedReplayBuffer
from ars.utils.vec_env import SubprocVecEnv
from ars.utils.gradient_estimators import create_exploration_strategy, EpsilonGreedy
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
    # Reward / altruism-economy knobs (must match evaluate_oracle so a policy is
    # trained and evaluated under the same reward). Defaults preserve NYC tuning;
    # e.g. Beijing's denser short-trip core needs a higher alpha_r (~0.6) for the
    # benefit term to outweigh detour — see the alpha_r sensitivity sweep.
    parser.add_argument("--alpha-r", type=float, default=0.4,
                        help="reward trade-off weight: rider benefit vs driver detour")
    parser.add_argument("--forced-driver-threshold", type=float, default=0.2,
                        help="altruism below which an agent is forced to drive")
    parser.add_argument("--alpha-s", type=float, default=0.5,
                        help="driver altruism-increase scaling")
    parser.add_argument("--beta-s", type=float, default=0.7,
                        help="rider altruism scaling")
    parser.add_argument("--selectivity-margin", type=float, default=-1.0,
                        help="net-savings margin (benefit-detour) a share must clear to earn reward; "
                             "<0 disables the selectivity gate (default, preserves prior behaviour)")
    parser.add_argument("--low-value-penalty", type=float, default=0.5,
                        help="penalty applied to shares that fail the selectivity margin")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--actor-lr", type=float, default=0.1, help="actor learning rate multiplier")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--reg-coef", type=float, default=1e-4, help="Regularization coefficient for actor loss")
    
    # Learning rate schedule
    parser.add_argument("--lr-schedule", type=str, default="cosine", choices=["constant", "cosine", "linear"],
                       help="Learning rate schedule: constant, cosine, or linear. Cosine decay "
                            "(default) reduces the late-training over-training collapse where a "
                            "constant LR knocks the converged policy off its peak.")
    parser.add_argument("--lr-min-ratio", type=float, default=0.1, 
                       help="Minimum LR as fraction of initial LR (floor for annealing)")
    
    # Network architecture
    parser.add_argument("--actor-hidden1", type=int, default=256, help="actor first hidden layer size")
    parser.add_argument("--actor-hidden2", type=int, default=512, help="actor second hidden layer size")
    parser.add_argument("--critic-hidden1", type=int, default=256, help="critic first hidden layer size")
    parser.add_argument("--critic-hidden2", type=int, default=512, help="critic second hidden layer size")
    
    # Model type selection
    parser.add_argument("--model", type=str, default="attention", choices=["maddpg", "attention"], 
                       help="Model type: maddpg or attention (Attention-MADDPG)")
    
    # Attention specific parameters
    parser.add_argument("--num-heads", type=int, default=4, help="number of attention heads (only for attention model)")
    parser.add_argument("--attn_temp", type=float, default=1.0, help="attention temperature (only for attention model)")
    parser.add_argument("--use-layer-norm", action="store_true",
                        help="Enable LayerNorm in the attention critic (stabilizes training).")
    parser.add_argument("--dropout-rate", type=float, default=0.0,
                        help="Dropout rate in the attention critic encoders (0 = off).")
    
    # Polyak averaging coefficient
    parser.add_argument("--tau", type=float, default=0.01, help="target network update rate (1-polyak)")
    
    # Exploration strategy selection
    parser.add_argument("--exploration", type=str, default="tags", 
                       choices=["epsilon_greedy", "tags", "gst"],
                       help="Exploration strategy: epsilon_greedy, tags (Temperature-Annealed Gumbel-Softmax), "
                            "or gst (Gapped Straight-Through)")
    
    # Exploration parameters (shared by epsilon_greedy and Gumbel methods)
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="starting epsilon for exploration")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="minimum epsilon value")
    parser.add_argument("--epsilon-decay-episodes", type=float, default=0.8, 
                        help="fraction of episodes over which to decay epsilon/temperature")
    
    # Gumbel-Softmax specific parameters
    parser.add_argument("--gumbel-temp-start", type=float, default=1.0,
                       help="Starting temperature for Gumbel-Softmax (high = more random)")
    parser.add_argument("--gumbel-temp-end", type=float, default=0.01,
                       help="Final temperature for Gumbel-Softmax (low = more greedy)")
    parser.add_argument("--gst-gap", type=float, default=1.0,
                       help="Starting gap parameter for GST estimator")
    parser.add_argument("--gst-gap-end", type=float, default=1.0,
                       help="Final gap parameter for GST estimator (annealed alongside temperature)")
    
    # Buffer parameters - UPDATED DEFAULTS
    parser.add_argument("--buffer-size", type=int, default=50000, help="size of the replay buffer")
    parser.add_argument("--alpha-initial", type=float, default=0.6, help="initial alpha for prioritized replay (PER paper default)")
    parser.add_argument("--alpha-final", type=float, default=0.6, help="final alpha - keep constant per PER paper")
    parser.add_argument("--beta-initial", type=float, default=0.4, help="initial beta for importance sampling (PER paper default)")
    parser.add_argument("--beta-final", type=float, default=1.0, help="final beta for importance sampling (must reach 1.0)")
    parser.add_argument("--alpha-decay", type=float, default=0.0, help="DEPRECATED - alpha/beta now use linear annealing")
    parser.add_argument("--beta-decay", type=float, default=0.0, help="DEPRECATED - alpha/beta now use linear annealing")
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
                       help="Device to use: 'gpu' or 'cpu' (default)")
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
    parser.add_argument("--num-envs", type=int, default=4, help="number of parallel environments")
    
    # Annealing schedule for temperature/gap decay
    parser.add_argument("--annealing", type=str, default="exponential",
                   choices=["exponential", "linear"],
                   help="Annealing schedule for temperature/gap decay")
    
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
    
    if arglist.memory_growth:
        print("Enabling memory growth for ALL GPUs")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Memory growth setting error: {e}")
    
    visible_gpus = [gpus[i] for i in visible_gpu_indices if i < len(gpus)]
    tf.config.set_visible_devices(visible_gpus, 'GPU')
    
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
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"TensorFlow sees {len(logical_gpus)} logical GPU(s)")
    except ValueError as e:
        print(f"Error listing logical devices: {e}")
        print("This is expected if you modified GPU settings - continuing with available devices")

def get_trainers(env, obs_shape_n, act_space_n, lr, centralized_buffer, arglist):
    trainers = []
    
    if arglist.model == "attention":
        AgentTrainerClass = AttentionTrainer
        print("Using Attention MADDPG model")
    else:
        AgentTrainerClass = MADDPGSharedTrainer
        print("Using standard MADDPG model")
    
    primary_trainer = AgentTrainerClass(
        name="agent_0",
        learning_rate=lr,
        obs_shape_n=obs_shape_n,
        act_space_n=act_space_n,
        centralized_buffer=centralized_buffer,
        agent_index=0,
        args=arglist,
        local_q_func=False  
    )
    trainers.append(primary_trainer)
    
    for i in range(1, env.numAgents):
        trainer = AgentTrainerClass(
            name=f"agent_{i}",
            learning_rate=lr,
            obs_shape_n=obs_shape_n,
            act_space_n=act_space_n,
            centralized_buffer=centralized_buffer,
            agent_index=i,
            args=arglist,
            local_q_func=False  
        )
        trainer.set_shared_networks(primary_trainer)
        trainers.append(trainer)
    
    return trainers

@tf.numpy_function(Tout=[tf.int32, tf.float32, tf.int32, tf.float32])
def env_step(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    next_state, reward, done, infos = env.step(actions)
    return (np.array(next_state, dtype=np.int32), np.array(reward, dtype=np.float32), np.array(done, dtype=np.int32), np.array(infos, dtype=np.float32))

def save_models(trainers, save_dir, model_type, num_agents, exploration, days=None,
                final=False, best=False):
    """Save all agent weights into a single file using pickle."""
    weights_dict = {}
    for i, agent in enumerate(trainers):
        weights_dict[f"agent_{i}_actor"] = agent.actor.get_weights()
        weights_dict[f"agent_{i}_critic"] = agent.critic.get_weights()

    os.makedirs(save_dir, exist_ok=True)

    if best:
        # Peak held-out-validation checkpoint (see training loop). Used for eval
        # because the final-day policy can be past its peak / collapsed.
        filename = f"best_weights_{model_type}_{num_agents}_{exploration}.pkl"
    elif final:
        filename = f"final_weights_{model_type}_{num_agents}_{exploration}.pkl"
    else:
        if days is None:
            raise ValueError("`days` must be provided when final=False")
        filename = f"{days}_days_weights_{model_type}_{num_agents}_{exploration}.pkl"

    save_path = os.path.join(save_dir, filename)
    with open(save_path, "wb") as f:
        pickle.dump(weights_dict, f)

    print(f"All agent weights saved to {save_path}")
    return save_path

def load_models(trainers, save_dir, day, model_type):
    """Load all agent weights from a single file using pickle."""
    load_path = os.path.join(save_dir, f"day_{day}_weights_{model_type}.pkl")
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No checkpoint found at {load_path}")
    
    with open(load_path, "rb") as f:
        weights_dict = pickle.load(f)
    
    for i, agent in enumerate(trainers):
        agent.actor.set_weights(weights_dict[f"agent_{i}_actor"])
        agent.critic.set_weights(weights_dict[f"agent_{i}_critic"])
    print(f"All agent weights loaded from {load_path}")

def resolve_model_label(arglist):
    """Paper-facing model name for this run. train_oracle always uses parameter
    sharing, so the attention architecture here is ORACLE (not MAAC)."""
    return "oracle" if arglist.model == "attention" else "maddpg_shared"


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
        f"{model_label}_grid{arglist.grid_size}_{arglist.num_agents}agents_{arglist.exploration}"
    )
    return wandb.init(
        project=arglist.wandb_project,
        entity=arglist.wandb_entity,
        mode=arglist.wandb_mode,
        name=run_name,
        config=vars(arglist),
    )

def generate_action_masks(state, infos):
    action_masks = np.zeros((env.numAgents, env.numAgents + 1), dtype=np.float32)
    for i, agent in enumerate(env.agents):
        if infos[i] == 1:
            nearby_grid = np.array(state[i][3:], dtype=np.int32)
            nearby_rider_ids = nearby_grid[nearby_grid != -1]
            valid_actions = np.concatenate((nearby_rider_ids, [env.numAgents]))
            action_masks[i][valid_actions] = 1.0
    return action_masks

def save_tune_metrics(metrics_dict, output_file):
    with open(output_file, 'w') as f:
        json.dump(metrics_dict, f)

def make_env(dataset_folder, num_agents, initial_active_agents, grid_size=15, perception_field=None,
             alpha_r=0.4, forced_driver_threshold=0.2, alpha_s=0.5, beta_s=0.7,
             selectivity_margin=-1.0, low_value_penalty=0.5):
    def _init():
        return Env(dataset_folder=dataset_folder, numAgents=num_agents, initial_active_agents=initial_active_agents,
                   height=grid_size, width=grid_size, perception_field=perception_field,
                   alpha_r=alpha_r, forced_driver_threshold=forced_driver_threshold,
                   alpha_s=alpha_s, beta_s=beta_s,
                   selectivity_margin=selectivity_margin, low_value_penalty=low_value_penalty)
    return _init

def compute_lr_scale(progress, schedule, min_ratio):
    """Compute learning rate scale factor based on schedule and training progress.
    
    Args:
        progress: float in [0, 1], fraction of training completed
        schedule: 'constant', 'cosine', or 'linear'
        min_ratio: minimum LR as fraction of initial LR
    
    Returns:
        lr_scale: float, multiplier for the base learning rate
    """
    if schedule == "constant":
        return 1.0
    elif schedule == "cosine":
        # Cosine annealing with floor
        lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_ratio, lr_scale)
    elif schedule == "linear":
        lr_scale = 1.0 - progress * (1.0 - min_ratio)
        return max(min_ratio, lr_scale)
    else:
        return 1.0

def train(arglist, days):
    # Create vectorized environments
    if arglist.num_envs > 1:
        env_fns = [make_env(arglist.dataset, arglist.num_agents, arglist.initial_active_agents,
                            grid_size=arglist.grid_size, perception_field=arglist.perception_field,
                            alpha_r=arglist.alpha_r, forced_driver_threshold=arglist.forced_driver_threshold,
                            alpha_s=arglist.alpha_s, beta_s=arglist.beta_s,
                            selectivity_margin=arglist.selectivity_margin,
                            low_value_penalty=arglist.low_value_penalty)
                   for _ in range(arglist.num_envs)]
        vec_env = SubprocVecEnv(env_fns)
        print(f"Created {arglist.num_envs} parallel environments")
    else:
        vec_env = Env(dataset_folder=arglist.dataset, numAgents=arglist.num_agents,
                      initial_active_agents=arglist.initial_active_agents,
                      height=arglist.grid_size, width=arglist.grid_size,
                      perception_field=arglist.perception_field,
                      alpha_r=arglist.alpha_r, forced_driver_threshold=arglist.forced_driver_threshold,
                      alpha_s=arglist.alpha_s, beta_s=arglist.beta_s,
                      selectivity_margin=arglist.selectivity_margin,
                      low_value_penalty=arglist.low_value_penalty)
        print("Using single environment (no parallelization)")
    
    episodes_per_day = arglist.num_episodes // arglist.num_envs
    print(f"Episodes per day per environment: {episodes_per_day}")
    
    # Compute total episodes for annealing schedules
    total_episodes = days * episodes_per_day
    print(f"Total training episodes (for annealing): {total_episodes}")
    
    if arglist.num_envs > 1:
        obs_space = vec_env.get_attr('observation_space', indices=0)[0]()
        actual_obs_dim = obs_space.nvec.shape[0]
        action_space = vec_env.get_attr('action_space', indices=0)[0]()
        act_space_n = [action_space for i in range(vec_env.numAgents)]
    else:
        obs_space = vec_env.observation_space()
        actual_obs_dim = obs_space.nvec.shape[0]
        act_space_n = [vec_env.action_space() for i in range(vec_env.numAgents)]
    
    obs_shape_n = [actual_obs_dim for i in range(vec_env.numAgents)]
    
    print(f"Observation dimension: {actual_obs_dim}, Number of agents: {vec_env.numAgents}")
    
    centralized_buffer = CentralizedReplayBuffer(
        num_agents=vec_env.numAgents, 
        state_dim=actual_obs_dim,
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
    
    print(f"Training with model type: {arglist.model.upper()}")
    print(f"Exploration strategy: {arglist.exploration.upper()}")
    print(f"PER parameters: alpha={arglist.alpha_initial}->{arglist.alpha_final}, beta={arglist.beta_initial}->{arglist.beta_final}")
    print(f"LR schedule: {arglist.lr_schedule}, min_ratio={arglist.lr_min_ratio}")
    
    if arglist.exploration in ['tags', 'gst']:
        print(f"Gumbel temperature: {arglist.gumbel_temp_start} -> {arglist.gumbel_temp_end}")
        if arglist.exploration == 'gst':
            print(f"GST gap: {arglist.gst_gap}")
    
    trainers = get_trainers(vec_env, obs_shape_n, act_space_n, arglist.lr, centralized_buffer, arglist)
    wandb_run = init_wandb(arglist)
    # Resolve hierarchical checkpoint dir: <save_dir>/<model>/grid<N>/<agents>agents/
    run_save_dir = build_save_dir(arglist)
    print(f"Checkpoints will be saved under: {run_save_dir}")
    
    # Create exploration strategy
    exploration = create_exploration_strategy(arglist.exploration, arglist, episodes_per_day)
    print(f"Exploration strategy initialized: {type(exploration).__name__}")
    
    global_episode = 0
    all_episode_rewards = []
    evaluation_rewards = []
    # Best-checkpoint tracking: the held-out validation reward peaks early and can
    # collapse with continued training (over-training). Save the peak-val policy
    # separately and evaluate THAT rather than the final-day (possibly collapsed)
    # weights. best_val_smoothed uses a short moving average so a single lucky
    # eval episode does not win over a genuinely better plateau.
    best_val_smoothed = -float("inf")
    best_val_day = 0
    
    for day in range(days):
        if arglist.num_envs > 1:
            vec_env.reset_day(day + 1, enable_birth_death=False)
        else:
            vec_env.reset_day(day + 1, enable_birth_death=False)
        print(f"Starting Day {day + 1}/{days}")
        
        if arglist.restore:
            load_models(trainers, run_save_dir, day, arglist.model)
        
        # Reset exploration strategy at start of each day (temperature/epsilon resets)
        exploration.reset()
        
        for episode in range(episodes_per_day):
            # === ANNEALING SCHEDULES (based on GLOBAL progress) ===
            current_global_episode = day * episodes_per_day + episode
            progress = min(1.0, current_global_episode / max(1, total_episodes - 1))
            
            # 1. Anneal alpha/beta linearly based on global progress
            centralized_buffer.update_alpha_beta(current_global_episode, max_episodes=total_episodes)
            
            # 2. Anneal learning rate
            lr_scale = compute_lr_scale(progress, arglist.lr_schedule, arglist.lr_min_ratio)
            current_critic_lr = arglist.lr * lr_scale
            current_actor_lr = arglist.lr * arglist.actor_lr * lr_scale
            
            # Apply LR to the primary trainer (shared mode means all agents use the same optimizers)
            primary_trainer = trainers[0]
            primary_trainer.critic_optimizer.learning_rate.assign(current_critic_lr)
            primary_trainer.actor_optimizer.learning_rate.assign(current_actor_lr)
            
            # Reset all environments
            states, infos_all = vec_env.reset(enable_dropout=False)
            
            if arglist.num_envs > 1:
                num_envs = arglist.num_envs
            else:
                num_envs = 1
                states = states[np.newaxis, :]
                infos_all = infos_all[np.newaxis, :]
            
            dones = np.zeros(num_envs, dtype=bool)
            episode_rewards_all = np.zeros((num_envs, vec_env.numAgents))
            steps = 0
            
            # Get current exploration temperature/epsilon for logging
            current_temp = exploration.get_temperature()
            
            print(f"Episode {episode}: Exploration({arglist.exploration}) = {current_temp:.4f}, "
                  f"Beta = {centralized_buffer.beta:.4f}, Alpha = {centralized_buffer.alpha:.4f}, "
                  f"LR_scale = {lr_scale:.4f}")
            
            train_metrics = {
                'critic_loss': [],
                'actor_loss': [],
                'td_errors': [],
                'valid_actions_count': []
            }
            
            is_evaluation = (episode == episodes_per_day - 1)
            if is_evaluation:
                print("Evaluation mode: Exploration disabled.")
            
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
                            valid_actions_arr = np.concatenate((nearby_rider_ids, [vec_env.numAgents]))
                            num_valid_actions[i] = len(valid_actions_arr)
                            
                            action_mask = np.zeros(vec_env.numAgents + 1, dtype=np.int32)
                            action_mask[valid_actions_arr] = 1
                            
                            if isinstance(exploration, EpsilonGreedy):
                                # Epsilon-greedy: use existing action selection
                                if not is_evaluation and random.random() < exploration.epsilon:
                                    actions[i] = np.random.choice(valid_actions_arr)
                                else:
                                    action_probs = trainers[i]._get_action_body(state[i], action_mask)
                                    actions[i] = np.argmax(action_probs)
                            else:
                                # Gumbel-based: use the estimator for action selection
                                if is_evaluation:
                                    # Eval: greedy
                                    action_probs = trainers[i]._get_action_body(state[i], action_mask)
                                    actions[i] = np.argmax(action_probs)
                                else:
                                    # Training: use Gumbel estimator for exploration
                                    logits = trainers[i]._get_raw_logits(state[i], action_mask)
                                    action_idx, _ = exploration(logits.numpy(), action_mask, training=True)
                                    actions[i] = action_idx
                    
                    actions_all.append(actions)
                    train_metrics['valid_actions_count'].append(np.mean(num_valid_actions))
                
                # Step all environments
                if arglist.num_envs > 1:
                    next_states, rewards, dones_step, next_infos_all = vec_env.step(actions_all)
                else:
                    # Single env: unwrap the list, call step, then re-wrap with batch dim
                    ns, rw, ds, ni = vec_env.step(actions_all[0])
                    next_states = ns[np.newaxis, :]
                    rewards = rw[np.newaxis, :]
                    dones_step = np.array([ds])
                    next_infos_all = ni[np.newaxis, :]
                
                dones = np.logical_or(dones, dones_step.astype(bool))
                
                for env_idx in range(num_envs):
                    if not dones[env_idx]:
                        state = states[env_idx]
                        actions = actions_all[env_idx]
                        reward = rewards[env_idx]
                        next_state = next_states[env_idx]
                        infos = infos_all[env_idx]
                        next_infos = next_infos_all[env_idx]
                        
                        episode_rewards_all[env_idx] += reward
                        
                        state_action_masks = generate_action_masks_parallel(state, infos, vec_env.numAgents)
                        next_state_action_masks = generate_action_masks_parallel(next_state, next_infos, vec_env.numAgents)
                        
                        if not is_evaluation:
                            one_hot_actions = np.zeros((len(actions), vec_env.numAgents + 1), dtype=np.int32)
                            for i, action in enumerate(actions):
                                one_hot_actions[i][action] = 1
                            
                            infos_reshaped = tf.reshape(infos, [-1, 1])
                            infos_reshaped = tf.cast(infos_reshaped, dtype=state.dtype)
                            masked_state = state * infos_reshaped
                            masked_one_hot_actions = one_hot_actions * infos_reshaped
                            masked_next_state = next_state * infos_reshaped
                            
                            store = False
                            for valid_count in np.sum(state_action_masks, axis=1):
                                if valid_count > 1:
                                    store = True
                                    break
                            if store:
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
                steps += 1
            
            # Update exploration strategy after each episode (decay temperature/epsilon)
            if not is_evaluation:
                exploration.update_state()
            
            # Training step - ONLY train primary agent (parameters are shared)
            if not is_evaluation and len(centralized_buffer) >= arglist.batch_size:
                primary_trainer = trainers[0]
                gradients_dict = primary_trainer.compute_gradients(
                    trainers, steps, exploration_strategy=exploration
                )
                
                if gradients_dict is not None:
                    primary_trainer.apply_gradients(gradients_dict)
                    
                    if 'critic_loss' in gradients_dict:
                        train_metrics['critic_loss'].append(gradients_dict['critic_loss'])
                    if 'actor_loss' in gradients_dict:
                        train_metrics['actor_loss'].append(gradients_dict['actor_loss'])
                    if 'td_errors' in gradients_dict:
                        train_metrics['td_errors'].append(tf.reduce_mean(gradients_dict['td_errors']))
            
            # Log metrics
            avg_episode_rewards = np.mean(episode_rewards_all, axis=0)
            current_temp = exploration.get_temperature()
            log_dict = {
                'parameters/exploration_temp': current_temp,
                'parameters/buffer_beta': centralized_buffer.beta,
                'parameters/buffer_alpha': centralized_buffer.alpha,
                'parameters/buffer_size': centralized_buffer.size,
                'parameters/critic_lr': current_critic_lr,
                'parameters/actor_lr': current_actor_lr,
                'parameters/lr_scale': lr_scale,
                'progress/day': day + 1,
                'progress/episode': episode,
                'progress/steps': steps,
            }
            if is_evaluation:
                log_dict['val_rewards/total_reward'] = np.sum(avg_episode_rewards)
                log_dict['val_rewards/mean_reward'] = np.mean(avg_episode_rewards)
                evaluation_rewards.append(float(np.sum(avg_episode_rewards)))
            else:
                log_dict['rewards/total_reward'] = np.sum(avg_episode_rewards)
                log_dict['rewards/mean_reward'] = np.mean(avg_episode_rewards)
            all_episode_rewards.append(float(np.sum(avg_episode_rewards)))
            for metric_name, values in train_metrics.items():
                if values:
                    log_dict[f'training/{metric_name}'] = np.mean(values)
            wandb.log(log_dict, step=global_episode)
            
            print(f"Episode {episode}/{episodes_per_day}, Avg Rewards: {', '.join([f'Agent {i}: {r:.2f}' for i, r in enumerate(avg_episode_rewards)])}, Total: {np.sum(avg_episode_rewards):.2f}")
            global_episode += 1

        # Best-checkpoint: if this day's held-out validation reward (smoothed over
        # the last few days to reject single-episode noise) is a new peak, save it
        # as best_weights. Eval restores best_weights, avoiding the over-training
        # collapse where the final-day policy is well past its peak.
        if evaluation_rewards:
            window = evaluation_rewards[-3:]
            val_smoothed = float(np.mean(window))
            if val_smoothed > best_val_smoothed:
                best_val_smoothed = val_smoothed
                best_val_day = day + 1
                save_models(
                    trainers,
                    run_save_dir,
                    arglist.model,
                    num_agents=arglist.num_agents,
                    exploration=arglist.exploration,
                    best=True,
                )
                print(f"[best] new peak val (smoothed)={val_smoothed:.3f} at day {day + 1}")

        # Save checkpoint every 10 days
        if (day + 1) % 10 == 0:
            save_models(
                trainers,
                run_save_dir,
                arglist.model,
                num_agents=arglist.num_agents,
                exploration=arglist.exploration,
                days=(day + 1),
                final=False
            )

    # Save both end-of-training variants
    save_models(
        trainers,
        run_save_dir,
        arglist.model,
        num_agents=arglist.num_agents,
        exploration=arglist.exploration,
        days=days,
        final=False
    )
    save_models(
        trainers,
        run_save_dir,
        arglist.model,
        num_agents=arglist.num_agents,
        exploration=arglist.exploration,
        final=True
    )

    print("Model saved")
    print(f"[best] best-val checkpoint from day {best_val_day} "
          f"(smoothed val reward {best_val_smoothed:.3f}); eval should restore "
          f"best_weights_{arglist.model}_{arglist.num_agents}_{arglist.exploration}.pkl")

    # Emit a metrics JSON for hyperparameter tuning. The objective is the mean
    # held-out evaluation reward (last episode of each day, exploration disabled),
    # which is what HPO maximises. We avoid a separate evaluation simulation to
    # keep each trial cheap.
    if arglist.tune_mode:
        metrics = {
            "mean_eval_reward": float(np.mean(evaluation_rewards)) if evaluation_rewards else 0.0,
            "final_eval_reward": float(evaluation_rewards[-1]) if evaluation_rewards else 0.0,
            "mean_reward": float(np.mean(all_episode_rewards)) if all_episode_rewards else 0.0,
            "final_reward": float(all_episode_rewards[-1]) if all_episode_rewards else 0.0,
        }
        save_tune_metrics(metrics, arglist.tune_mode)
        print(f"Tune metrics written to {arglist.tune_mode}: {metrics}")

    if wandb_run is not None:
        wandb_run.finish()
        print(f"wandb run finished ({arglist.wandb_mode} mode).")

def generate_action_masks_parallel(state, infos, num_agents):
    """Generate action masks for parallel environments"""
    action_masks = np.zeros((num_agents, num_agents + 1), dtype=np.float32)
    for i in range(num_agents):
        if infos[i] == 1:
            nearby_grid = np.array(state[i][3:], dtype=np.int32)
            nearby_rider_ids = nearby_grid[nearby_grid != -1]
            valid_actions = np.concatenate((nearby_rider_ids, [num_agents]))
            action_masks[i][valid_actions] = 1.0
    return action_masks

def main():
    arglist = parse_args()
    # Wire the CLI --attn_temp onto the attribute the model actually reads
    # (AttentionCritic is built with getattr(args, 'attention_temp', 1.0)).
    # Without this, --attn_temp was a no-op and every run used 1.0, so any
    # HPO over attention temperature had no effect. Fixed here.
    arglist.attention_temp = arglist.attn_temp
    configure_gpus(arglist)
    train(arglist, days=arglist.days)


if __name__ == "__main__":
    main()