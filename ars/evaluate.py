import numpy as np
import os
import argparse
import pickle
import random
import time
import tensorflow as tf
from ars.env.env import Env
from ars.models.maddpg import MADDPGTrainer
from ars.models.attention import AttentionTrainer
from tensorflow.python.keras.engine import data_adapter
from ars.env.agent import Agent, Driver, Rider, create_agent, switch_role
from ars.utils.prioritized_replaybuffer import CentralizedReplayBuffer
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import warnings
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio
from ars.metrics import *
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser("Multiagent reinforcement learning with MADDPG")
    # Environment and scenario parameters
    parser.add_argument("--max-episode-len", type=int, default=1000, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=100, help="number of episodes")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size for optimization")
    parser.add_argument("--dataset", type=str, default="100_agents", help="dataset folder")
    parser.add_argument("--num-agents", type=int, default=100, help="number of agents in the environment")
    parser.add_argument("--grid-size", type=int, default=15, help="side length of the square grid (height=width)")
    parser.add_argument("--perception-field", type=int, default=None,
                       help="egocentric perception field side length n (odd). Defaults to ~grid/3 snapped to odd")
    # Altruism distribution parameters
    parser.add_argument("--altruism-distribution", type=str, default="uniform", 
                       choices=["uniform", "gaussian"],
                       help="Type of altruism distribution: 'uniform' or 'gaussian'")
    parser.add_argument("--altruism-mean", type=float, default=0.5,
                       help="Mean altruism value (for uniform: fixed value, for gaussian: distribution mean)")
    parser.add_argument("--altruism-std", type=float, default=0.15,
                       help="Standard deviation for gaussian distribution (ignored for uniform)")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--actor-lr", type=float, default=1e-2, help="actor learning rate multiplier")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    # Network architecture
    parser.add_argument("--model-type", type=str, default="maddpg", choices=["maddpg", "attention"],
                        help="Which trainer/model architecture to use: 'maddpg' or 'attention'")
    parser.add_argument("--actor-hidden1", type=int, default=64, help="actor first hidden layer size")
    parser.add_argument("--actor-hidden2", type=int, default=128, help="actor second hidden layer size")
    parser.add_argument("--critic-hidden1", type=int, default=128, help="critic first hidden layer size")
    parser.add_argument("--critic-hidden2", type=int, default=256, help="critic second hidden layer size")
    # Attention specific parameters
    parser.add_argument("--num-heads", type=int, default=4, help="number of attention heads (only for attention model)")
    # Polyak averaging coefficient
    parser.add_argument("--tau", type=float, default=0.01, help="target network update rate (1-polyak)")
    # Exploration parameters
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="starting epsilon for exploration")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="minimum epsilon value")
    parser.add_argument("--epsilon-decay-episodes", type=float, default=0.8, help="fraction of episodes over which to decay epsilon")
    # Neutral Action Bias
    parser.add_argument("--neutral-bias-episodes", type=float, default=0.2, help="Fraction of episodes to bias towards neutral action at start")
    parser.add_argument("--neutral-bias-prob", type=float, default=0.7, help="Probability of taking neutral action during bias period")
    # Buffer parameters (match main.py defaults)
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
    parser.add_argument("--weights-file", type=str, default=None,
                       help="explicit checkpoint filename (under --save-dir, or an absolute path) to load. "
                            "Overrides the default final_weights_<model>_<num_agents>.pkl naming.")
    # Simulation parameters
    parser.add_argument("--days", type=int, default=1, help="number of days to simulate")
    parser.add_argument("--filter-neg-rew", action="store_false", default=True,
                        help="If set, filter out actions with reward below threshold (new method)")
    parser.add_argument("--reward-threshold", type=float, default=-0.75,
                        help="Threshold for filtering negative reward actions")
    parser.add_argument("--enable-birth-death", action="store_true", default=False,
                       help="Enable birth-death simulation (agents can join and leave)")
    parser.add_argument("--initial-active-agents", type=int, default=100,
                       help="Number of agents to start with (rest will be in never-joined state)")
    parser.add_argument("--new-agent-altruism", type=float, default=0.5,
                       help="Initial altruism score for newly joining agents")
    # Regularization coefficient (needed by attention model)
    parser.add_argument("--reg-coef", type=float, default=1e-4, help="Regularization coefficient for actor loss")
    # GPU configuration parameters
    parser.add_argument("--device", type=str, default="cpu", choices=["gpu", "cpu"], 
                       help="Device to use: 'gpu' (default) or 'cpu'")
    parser.add_argument("--gpu-id", type=str, default="1", 
                       help="GPU ID(s) to use (comma-separated for multiple GPUs, or 'all' for all available GPUs)")
    parser.add_argument("--memory-growth", action="store_true", default=True, 
                       help="Enable memory growth for GPU (helps with OOM errors)")
    parser.add_argument("--memory-limit", type=int, default=None,
                       help="Limit GPU memory in MB (None = no limit)")
    parser.add_argument("--results-dir", type=str, default=None,
                       help="Override the results output directory (e.g. to "
                            "isolate a second city); consumed by "
                            "ars.metrics.get_results_dir.")
    parser.add_argument("--alpha-r", type=float, default=0.4,
                       help="Reward trade-off weight (rider benefit vs driver "
                            "detour). Must match the value the policy trained "
                            "with (e.g. 0.6 for Beijing).")
    args, unknown = parser.parse_known_args()
    return args

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

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset

# NOTE: results dir is provided by ars.metrics.get_results_dir (single source of
# truth, grid-aware). evaluate.py runs agents with independent (unshared)
# networks, so it sets arglist.model_label = "maac" (or "maddpg") in main() and
# does NOT redefine get_results_dir, ensuring every plotting/saving function in
# metrics.py writes to the same results/grid<N>/<model>/ directory.

def setup_environment(height=15, width=15, numAgents=100, dataset_folder="100_agents",
                     initial_active_agents=100, altruism_distribution='uniform',
                     altruism_mean=0.5, altruism_std=0.15, perception_field=None):
    """
    Setup the environment with specified parameters.

    Args:
        altruism_distribution: 'uniform' or 'gaussian'
        altruism_mean: Mean altruism value (for uniform: the fixed value, for gaussian: distribution mean)
        altruism_std: Standard deviation for gaussian distribution (ignored for uniform)
    """
    env = Env(
        height=height,
        width=width,
        numAgents=numAgents,
        dataset_folder=dataset_folder,
        initial_active_agents=initial_active_agents,
        altruism_distribution=altruism_distribution,
        altruism_mean=altruism_mean,
        altruism_std=altruism_std,
        total_days=arglist.days,
        perception_field=perception_field,
        alpha_r=arglist.alpha_r
    )
    return env

def get_agent_trainer_class(arglist):
    """Return the appropriate agent trainer class based on model argument."""
    if arglist.model_type == "attention":
        print("Using Attention MADDPG model")
        return AttentionTrainer
    else:
        print("Using standard MADDPG model")
        return MADDPGTrainer

def load_models(agents, save_dir, arglist):
    """Load all agent weights from a single file using pickle."""
    weights_file = getattr(arglist, "weights_file", None)
    if weights_file:
        load_path = weights_file if os.path.isabs(weights_file) else os.path.join(save_dir, weights_file)
    else:
        load_path = os.path.join(save_dir, f"final_weights_{arglist.model_type}_{arglist.num_agents}.pkl")
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No checkpoint found at {load_path}")
    # Load weights from the file
    with open(load_path, "rb") as f:
        weights_dict = pickle.load(f)
    # Assign weights to each agent
    for i, agent in enumerate(agents):
        agent.actor.set_weights(weights_dict[f"agent_{i}_actor"])
        agent.critic.set_weights(weights_dict[f"agent_{i}_critic"])
    print(f"All agent weights loaded from {load_path}")

def run_simulation(env, agents, num_episodes=1, render=False, day=1, pickup_history=None, 
                  enable_dropout=False, enable_birth_death=False, arglist=None):
    """Run simulation and return both sharing and no-sharing paths."""
    if arglist is None:
        raise ValueError("arglist must be provided to run_simulation")

    # Set current day in environment for transaction tracking
    env.current_day = day
    
    episode_reward = {agent: 0 for agent in env.agents}
    paths = {agent: [env.agent_objects[i].start_position] for i, agent in enumerate(env.agents)}
    miles_given = {agent: 0 for agent in env.agents}
    miles_taken = {agent: 0 for agent in env.agents}
    ride_offers = 0
    ride_accepts = 0

    # Use the provided agents list instead of creating new ones
    if agents is None:
        obs_shape_n = [env.observation_space().shape[0] for i in range(env.numAgents)]
        act_space_n = [env.action_space() for i in range(env.numAgents)]
        AgentTrainerClass = get_agent_trainer_class(arglist)
        agents = []
        for i in range(env.numAgents):
            agents.append(AgentTrainerClass(
                name=f"agent_{i}",
                learning_rate=arglist.lr,
                obs_shape_n=obs_shape_n,
                act_space_n=act_space_n,
                centralized_buffer=None,
                agent_index=i,
                args=arglist,
                local_q_func=False  
            ))

    state, infos = env.reset(enable_dropout=enable_dropout, enable_birth_death=enable_birth_death)
    done = False
    positions = {}
    actions = {}
    vehicle_utilization_timesteps = {agent: [] for agent in env.agents if env.agent_role[agent] == "driver"}

    if render:
        env.render()
        time.sleep(0.4)

    # Calculate no-sharing paths at the beginning (only for active agents)
    paths_no_sharing = get_no_sharing_paths(env)

    while not done:
        actions = np.zeros(env.numAgents, dtype=np.int32)

        for i, agent in enumerate(env.agents):
            # Skip dropped out and never joined agents
            if env.agent_role[agent] in ['dropout', 'never_joined']:
                actions[i] = env.numAgents  # Default action for inactive agents
                continue
                
            nearby_grid = np.array(state[i][3:], dtype=np.int32)
            nearby_rider_ids = nearby_grid[nearby_grid != -1]
            valid_actions = np.concatenate((nearby_rider_ids, [env.numAgents]))

            if arglist.filter_neg_rew and env.agent_role[agent] == "driver":
                filtered_actions = []
                driver_obj = env.agent_objects[i]
                for rider_id in nearby_rider_ids:
                    # Skip if target rider is dropped out or never joined
                    if env.agent_role[env.agents[rider_id]] in ['dropout', 'never_joined']:
                        continue
                    rider_obj = env.agent_objects[rider_id]
                    reward_val = env.get_reward(driver_obj, rider_obj, 0)
                    if reward_val > arglist.reward_threshold:
                        filtered_actions.append(rider_id)
                filtered_actions.append(env.numAgents)
                valid_actions = np.array(filtered_actions, dtype=np.int32)

            # Count ride offers (for each valid rider action, not including no-op)
            if env.agent_role[agent] == "driver":
                ride_offers += len(valid_actions) - 1 # exclude no-op

            action_mask = np.zeros(env.numAgents + 1, dtype=np.int32)
            action_mask[valid_actions] = 1
            action_probs = agents[i]._get_action_body(state[i], action_mask)
            chosen_action = np.argmax(action_probs)
            actions[i] = chosen_action

            # Count ride accepts (if a rider is picked up)
            if env.agent_role[agent] == "driver" and chosen_action != env.numAgents:
                ride_accepts += 1

        # Track pickup interactions before stepping
        track_pickup_interactions(env, actions, pickup_history, day)
        
        next_state, reward, done, infos = env.step(actions)

        # Track paths for each agent (only active agents)
        for i, agent in enumerate(env.agents):
            if env.agent_role[agent] in ['dropout', 'never_joined']:
                continue
                
            agent_obj = env.agent_objects[i]
            if agent_obj.is_at_destination() and np.array_equal(paths[agent][-1], agent_obj.destination):
                continue
            paths[agent].append(tuple(agent_obj.position))  # Record position

        state = next_state

        for i, agent in enumerate(env.agents):
            episode_reward[agent] += reward[i]

        if render:
            env.render()
            # time.sleep(0.3)
  
        if done:
            # Calculate individual and shared distances (only for active agents)
            direct_distances = {}
            individual_distances = {}
            individual_times = {}
            direct_times = {}
            detour_distances = {agent: 0 for agent in env.agents}
            
            # Calculate vehicle utilization per driver for boxplot
            vehicle_utilization_per_driver = {}
            
            for agent in env.agents:
                if env.agent_role[agent] in ['dropout', 'never_joined']:
                    # Set default values for inactive agents
                    individual_distances[agent] = 0
                    direct_distances[agent] = 0
                    individual_times[agent] = 0
                    direct_times[agent] = 0
                    continue
                    
                if env.agent_role[agent] == 'driver':
                    # For each driver: (1 + number of riders picked) / 1
                    riders_picked = len(env.riders_drivers[agent])
                    utilization = 1.0 + riders_picked  # 1 (driver) + riders
                    vehicle_utilization_per_driver[agent] = utilization

            for agent in env.agents:
                if env.agent_role[agent] in ['dropout', 'never_joined']:
                    continue
                    
                agent_obj = env.agent_objects[env.agents.index(agent)]
                if isinstance(agent_obj, Driver):
                    # Calculate total distance (individual + shared)
                    total_time = calculate_time(paths[agent], env.time_weight_matrix)
                    individual_times[agent] = total_time
                    total_distance = calculate_distance(paths[agent], env.weight_matrix)
                    individual_distances[agent] = total_distance

                    direct_distance = agent_obj._calculate_path_length(
                        agent_obj._dijkstra_path(agent_obj.start_position, agent_obj.destination)
                    )
                    direct_distances[agent] = direct_distance

                    # Calculate shared distance (distance while carrying riders)
                    miles_given[agent] = 0
                    for rider in env.riders_drivers[agent]:
                        rider_path = agent_obj._dijkstra_path(rider.start_position, rider.destination)
                        miles_given[agent] += agent_obj._calculate_path_length(rider_path)
                    detour_distances[agent] = total_distance - direct_distance
                    direct_path = agent_obj._dijkstra_path(agent_obj.start_position, agent_obj.destination)
                    direct_time = calculate_time(direct_path, env.time_weight_matrix)
                    direct_times[agent] = direct_time
                else:  # Rider
                    if agent_obj.being_picked_up:
                        # Riders picked up by drivers have their distance included in the driver's shared distance
                        rider_path = agent_obj._dijkstra_path(agent_obj.start_position, agent_obj.destination)
                        # shared_distances[agent] = agent_obj._calculate_path_length(rider_path)
                        miles_taken[agent] = agent_obj._calculate_path_length(rider_path)
                        individual_distances[agent] = 0  # No individual distance if picked up
                        direct_distances[agent] = miles_taken[agent]
                        individual_times[agent] = 0
                        direct_times[agent] = calculate_time(rider_path, env.time_weight_matrix)
                    else:
                        # Riders not picked up travel individually
                        rider_path = agent_obj._dijkstra_path(agent_obj.start_position, agent_obj.destination)
                        individual_distances[agent] = agent_obj._calculate_path_length(rider_path)
                        miles_taken[agent] = 0
                        direct_distances[agent] = individual_distances[agent]
                        individual_times[agent] = calculate_time(rider_path, env.time_weight_matrix)
                        direct_times[agent] = individual_times[agent]
                        # shared_distances[agent] = 0  # No shared distance if not picked up

            break
    
    print(f"Day {day} Rewards: ")
    for agent_id, reward in episode_reward.items():
        if env.agent_role[agent_id] not in ['dropout', 'never_joined']:
            print(f"{agent_id}: {reward}")
    print()

    # Print dropout statistics
    dropout_stats = env.get_dropout_statistics(day)
    if dropout_stats['agents_dropped'] > 0:
        print(f"Day {day} Dropouts: {dropout_stats['agents_dropped']} agents dropped out")
        for agent in dropout_stats['dropped_agents']:
            print(f"  {agent} (altruism: {env.altruism_points_day[agent]:.3f})")

    acceptance_rate = ride_accepts / ride_offers if ride_offers > 0 else 0

    if enable_birth_death:
        bd_stats = env.get_birth_death_statistics(day)
        if bd_stats['births'] > 0 or bd_stats['dropouts'] > 0:
            print(f"Day {day} Birth-Death: {bd_stats['births']} births, {bd_stats['dropouts']} dropouts")
            if bd_stats['birth_agents']:
                print(f"  New agents: {bd_stats['birth_agents']}")
            if bd_stats['dropout_agents']:
                print(f"  Dropped agents: {bd_stats['dropout_agents']}")
    
    return episode_reward, individual_distances, direct_distances, detour_distances, miles_given, miles_taken, paths, paths_no_sharing, ride_offers, ride_accepts, acceptance_rate, individual_times, direct_times, vehicle_utilization_per_driver

# In the main simulation loop, update the call:
if __name__ == "__main__":
    arglist = parse_args()
    configure_gpus(arglist)
    # Model label drives the (grid-aware) results dir in ars.metrics.get_results_dir.
    # evaluate.py = unshared agents -> MAAC (attention) or MADDPG.
    arglist.model_label = "maac" if arglist.model_type == "attention" else "maddpg"
    # Print altruism distribution settings
    print(f"\n=== ALTRUISM DISTRIBUTION SETTINGS ===")
    print(f"Distribution type: {arglist.altruism_distribution}")
    print(f"Mean: {arglist.altruism_mean}")
    if arglist.altruism_distribution == 'gaussian':
        print(f"Standard deviation: {arglist.altruism_std}")
    
    print(f"Model type: {arglist.model_type.upper()}")
    
    # Setup environment with altruism distribution parameters
    env = setup_environment(
        height=arglist.grid_size,
        width=arglist.grid_size,
        numAgents=arglist.num_agents,
        dataset_folder=arglist.dataset,
        initial_active_agents=arglist.initial_active_agents,
        altruism_distribution=arglist.altruism_distribution,
        altruism_mean=arglist.altruism_mean,
        altruism_std=arglist.altruism_std,
        perception_field=arglist.perception_field
    )
    
    # Print initial distribution statistics
    stats = env.get_altruism_distribution_stats()
    print(f"\n=== INITIAL ALTRUISM STATISTICS ===")
    print(f"Mean: {stats['mean']:.3f}")
    print(f"Std: {stats['std']:.3f}")
    print(f"Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    print(f"Median: {stats['median']:.3f}")
    print(f"Q25-Q75: [{stats['q25']:.3f}, {stats['q75']:.3f}]")
    results = {}
    individual_distances_by_day = {}
    direct_distances_by_day = {}
    detour_distances_by_day = {}
    miles_given_by_day = {}
    miles_taken_by_day = {}
    altruism_points_by_day = {}
    roles_by_day = {}
    distance_total_sharing_by_day = {}
    distance_total_no_sharing_by_day = {}
    acceptance_rates_by_day = {}
    ride_offers_by_day = {}
    ride_accepts_by_day = {}
    individual_times_by_day = {}
    direct_times_by_day = {}
    vehicle_utilization_by_day = {}
    # Time series tracking for Granger causality
    daily_altruism_series = {}
    daily_benefit_series = {}
    daily_aggregate_altruism = []
    daily_aggregate_benefits = []

    # Build agents with correct model type
    obs_shape_n = [env.observation_space().shape[0] for _ in range(env.numAgents)]
    act_space_n = [env.action_space() for _ in range(env.numAgents)]
    
    AgentTrainerClass = get_agent_trainer_class(arglist)
    agents = []
    for i in range(env.numAgents):
        agents.append(AgentTrainerClass(
            name=f"agent_{i}",
            learning_rate=arglist.lr,
            obs_shape_n=obs_shape_n,
            act_space_n=act_space_n,
            centralized_buffer=None,
            agent_index=i,
            args=arglist,
            local_q_func=False  
        ))
    load_models(agents, save_dir=arglist.save_dir, arglist=arglist)
    print(f"Loaded final weights for all agents ({arglist.model_type} model).")

    paths_sharing_by_day = {}
    paths_no_sharing_by_day = {}

    # Add pickup history tracking
    pickup_history = {}
    
    for day in range(1, arglist.days + 1):
        if arglist.enable_birth_death:
            env.reset_day(day, enable_birth_death=True)
        else:
            env.reset_day(day)
        env.current_day = day
        print(f"Running simulation for Day {day}...")
        
        # Run simulation with arglist passed through
        (episode_reward, individual_distances, direct_distances, detour_distances, miles_given, 
         miles_taken, paths_sharing, paths_no_sharing, ride_offers, ride_accepts, acceptance_rate, 
         individual_times, direct_times, vehicle_utilization_per_driver) = run_simulation(
            env, agents, day=day, pickup_history=pickup_history, 
            enable_birth_death=arglist.enable_birth_death,
            enable_dropout=(not arglist.enable_birth_death),
            arglist=arglist
        )
        
        # Update pickup_history from environment after each day
        track_pickup_interactions(env, None, pickup_history, day)
        
        # Store both path types
        paths_sharing_by_day[day] = paths_sharing
        paths_no_sharing_by_day[day] = paths_no_sharing
        
        # Store time metrics
        individual_times_by_day[day] = copy.deepcopy(individual_times)
        direct_times_by_day[day] = copy.deepcopy(direct_times)
        acceptance_rates_by_day[day] = acceptance_rate
        ride_offers_by_day[day] = ride_offers
        ride_accepts_by_day[day] = ride_accepts
        results[day] = episode_reward
        individual_distances_by_day[day] = copy.deepcopy(individual_distances)
        direct_distances_by_day[day] = copy.deepcopy(direct_distances)
        detour_distances_by_day[day] = copy.deepcopy(detour_distances)

        miles_given_by_day[day] = copy.deepcopy(miles_given)
        miles_taken_by_day[day] = copy.deepcopy(miles_taken)

        # Store utilization values for boxplot
        vehicle_utilization_by_day[day] = list(vehicle_utilization_per_driver.values())

        # Store altruism points and roles for the current day
        altruism_points_by_day[day] = copy.deepcopy(env.altruism_points_day)
        roles_by_day[day] = copy.deepcopy(env.agent_role)

        # Calculate total distances only for active agents
        distance_total_sharing_by_day[day] = sum(
            dist for agent, dist in individual_distances_by_day[day].items() 
            if env.agent_role.get(agent, 'active') not in ['dropout', 'never_joined']
        )
        distance_total_no_sharing_by_day[day] = sum(
            dist for agent, dist in direct_distances_by_day[day].items() 
            if env.agent_role.get(agent, 'active') not in ['dropout', 'never_joined']
        )

        for agent in env.agents:
            if env.agent_role.get(agent, 'active') not in ['dropout', 'never_joined']:
                if agent not in daily_altruism_series:
                    daily_altruism_series[agent] = []
                daily_altruism_series[agent].append(altruism_points_by_day[day][agent])
        
        active_altruism_scores = [
            altruism_points_by_day[day][agent] 
            for agent in env.agents 
            if (env.agent_role.get(agent, 'active') not in ['dropout', 'never_joined'] and
                agent in altruism_points_by_day[day])
        ]
        
        if active_altruism_scores:
            daily_aggregate_altruism.append(np.mean(active_altruism_scores))
        else:
            daily_aggregate_altruism.append(0.0)
        
        print(f"Day {day}: Mean altruism = {daily_aggregate_altruism[-1]:.4f} (from {len(active_altruism_scores)} active agents)")

        if day > 1:
            for agent in env.agents:
                if (env.agent_role.get(agent, 'active') not in ['dropout', 'never_joined'] and 
                    agent in individual_distances_by_day[day] and 
                    agent in direct_distances_by_day[day]):
                    
                    if agent not in daily_benefit_series:
                        daily_benefit_series[agent] = []
                    
                    daily_benefit = (direct_distances_by_day[day][agent] - 
                                individual_distances_by_day[day][agent])
                    daily_benefit_series[agent].append(daily_benefit)

        if day > 1:
            active_benefits = []
            for agent in env.agents:
                if (env.agent_role.get(agent, 'active') not in ['dropout', 'never_joined'] and 
                    agent in individual_distances_by_day[day] and 
                    agent in direct_distances_by_day[day]):
                    
                    daily_benefit = (direct_distances_by_day[day][agent] - 
                                individual_distances_by_day[day][agent])
                    active_benefits.append(daily_benefit)
            
            if active_benefits:
                daily_aggregate_benefits.append(np.mean(active_benefits))
            else:
                daily_aggregate_benefits.append(0.0)
                
            print(f"Day {day}: Mean benefits = {daily_aggregate_benefits[-1]:.4f} (from {len(active_benefits)} active agents)")

    # After all simulations, analyze actual transactions
    analyze_altruism_transactions(env, arglist)

    plot_sharing_vs_non_sharing(distance_total_sharing_by_day, distance_total_no_sharing_by_day, env, arglist)
    plot_vehicle_utilization(vehicle_utilization_by_day, env, arglist)
    plot_ride_acceptance_rate(acceptance_rates_by_day, env, arglist)
    plot_agent_distance_boxplots(individual_distances_by_day, direct_distances_by_day, env, arglist)
    plot_altruism_boxplots(altruism_points_by_day, env, arglist)
    plot_detour_factor_and_trip_time(detour_distances_by_day, individual_times_by_day, direct_distances_by_day, direct_times_by_day, env, arglist)
    save_results(results, save_dir=f"results/{arglist.dataset}", altruism_points_by_day=altruism_points_by_day, roles_by_day=roles_by_day, individual_distances_by_day=individual_distances_by_day, direct_distances_by_day=direct_distances_by_day, detour_distances_by_day=detour_distances_by_day, miles_given_by_day=miles_given_by_day, miles_taken_by_day=miles_taken_by_day, distance_total_sharing_by_day=distance_total_sharing_by_day, distance_total_no_sharing_by_day=distance_total_no_sharing_by_day, arglist=arglist)
    generate_ieee_tikz_data(altruism_points_by_day, individual_distances_by_day, 
                        direct_distances_by_day, detour_distances_by_day,
                        distance_total_sharing_by_day, distance_total_no_sharing_by_day,
                        vehicle_utilization_by_day, acceptance_rates_by_day,
                        individual_times_by_day, direct_times_by_day, env, arglist)
    
    # 1. Agent Benefit Analysis (now multidimensional)
    agent_benefits, agent_total_altruism = calculate_agent_benefits(
        individual_distances_by_day, direct_distances_by_day, altruism_points_by_day,
        paths_sharing_by_day, paths_no_sharing_by_day, env, arglist
    )

    # 2. Create Agent Benefit Scatter Plot (updated for multidimensional)
    benefit_altruism_correlation = plot_agent_benefit_scatter(agent_benefits, env, arglist)

    # 2.5. Create Benefits vs Altruism Line Curves
    line_curve_stats = plot_benefits_vs_altruism_line_curves(agent_benefits, env, arglist)

    # 3. Winner/Loser Distribution Analysis (will use combined benefits)
    winner_loser_stats = analyze_winner_loser_distribution(agent_benefits, env, arglist)

    # 4. NEW: Multidimensional Benefit Distribution Analysis
    multidimensional_stats = analyze_multidimensional_benefit_distribution(agent_benefits, env, arglist)

    # 5. Original single-dimension analysis (for comparison)
    distribution_stats = analyze_benefit_distribution(agent_benefits, env, arglist)

    # 6. Save all metrics data for LaTeX
    save_new_metrics_data(agent_benefits, winner_loser_stats, 
                    distribution_stats, benefit_altruism_correlation, env, arglist, 
                    line_curve_stats)
    # 7. Save multidimensional metrics data
    save_multidimensional_metrics_data(agent_benefits, multidimensional_stats, get_results_dir(arglist), env, arglist)

    # Print comprehensive summary
    print(f"\n=== COMPREHENSIVE BENEFIT ANALYSIS SUMMARY ===")
    print(f"Distance Benefit Gini: {multidimensional_stats['gini_distance_benefits']:.4f}")
    print(f"Traffic Benefit Gini: {multidimensional_stats['gini_traffic_benefits']:.4f}")
    print(f"Combined Benefit Gini: {multidimensional_stats['gini_combined_benefits']:.4f}")
    print(f"Distance-Traffic Correlation: {multidimensional_stats['correlation_distance_traffic']:.4f}")
    print(f"Distance Benefit Concentration (Top 20%): {multidimensional_stats['concentration_ratio_distance_top_20']:.1f}%")
    print(f"Traffic Benefit Concentration (Top 20%): {multidimensional_stats['concentration_ratio_traffic_top_20']:.1f}%")
    print(f"Mean Distance Benefit: {multidimensional_stats['mean_distance_benefit']:.2f}")
    print(f"Mean Traffic Benefit: {multidimensional_stats['mean_traffic_benefit']:.2f}")
    print(f"Winners: {winner_loser_stats['winners']} ({winner_loser_stats['winner_percentage']:.1f}%)")
    print(f"Losers: {winner_loser_stats['losers']} ({winner_loser_stats['loser_percentage']:.1f}%)")

    # After all simulations complete, perform Granger causality analysis
    print("\n" + "="*60)
    print("GRANGER CAUSALITY ANALYSIS")
    print("="*60)
    
    granger_results = perform_granger_causality_analysis(
        daily_altruism_series, daily_benefit_series, 
        daily_aggregate_altruism, daily_aggregate_benefits,
        arglist.days, env, arglist
    )
    
    save_granger_causality_results(granger_results, get_results_dir(arglist), env, arglist)

    plot_time_series_and_causality(
        daily_altruism_series, daily_benefit_series,
        daily_aggregate_altruism, daily_aggregate_benefits,
        granger_results, get_results_dir(arglist), env, arglist
    )

    if arglist.enable_birth_death:
        bd_stats = plot_birth_death_analysis(env, arglist)
        print(f"\n=== FINAL BIRTH-DEATH SUMMARY ===")
        print(f"Growth Rate: {bd_stats['growth_rate']:.2f}%")
        print(f"Net Change: {bd_stats['net_change']} agents")
        reintegration_metrics = plot_reintegration_analysis(env, arglist)
        print(f"\n=== REINTEGRATION ANALYSIS ===")
        print(f"Final Reintegration Score: {reintegration_metrics['final_reintegration_score']:.1f}/100")
        print(f"Time-Weighted Rate: {reintegration_metrics['time_weighted_rate']:.3f}")
        print(f"Quick Return Rate: {reintegration_metrics['quick_return_rate']:.3f}")
        print(f"Stability Score: {reintegration_metrics['stability_score']:.3f}")
        print(f"Stable Returners: {reintegration_metrics['stable_returners']}")
        print(f"Instability Rate: {reintegration_metrics['instability_rate']:.3f}")
        print(f"Average Return Time: {reintegration_metrics['avg_return_time']:.1f} days")
    else:
        dropout_stats = plot_daily_dropouts(env, arglist)

    print("Creating improved traffic density analysis...")
    
    rho_threshold = 2.0
    
    traffic_metrics, avg_density_sharing, avg_density_no_sharing = calculate_traffic_density_metrics(
        paths_sharing_by_day, paths_no_sharing_by_day, 
        env.height, env.width, arglist.days, env, arglist, rho_threshold
    )
    
    density_diff = create_traffic_density_visualization(
        avg_density_sharing, avg_density_no_sharing, traffic_metrics, get_results_dir(arglist), env, arglist
    )
    
    save_traffic_density_data_for_latex(
        traffic_metrics, avg_density_sharing, avg_density_no_sharing, density_diff, get_results_dir(arglist), env, arglist
    )

    results_dir = get_results_dir(arglist)
    latex_dir = os.path.join(results_dir, "latex_data")
    
    with open(f"{latex_dir}/summary.dat", 'a') as f:
        f.write(f"dense_cell_reduction {traffic_metrics['dense_cell_reduction']}\n")
        f.write(f"dense_cell_reduction_percentage {traffic_metrics['dense_cell_reduction_percentage']:.2f}\n")
        f.write(f"traffic_reduction_percentage {traffic_metrics['traffic_reduction_percentage']:.2f}\n")
        f.write(f"peak_density_reduction_percentage {traffic_metrics['peak_density_reduction_percentage']:.2f}\n")
        f.write(f"hotspot_reduction {traffic_metrics['hotspot_reduction']}\n")
        f.write(f"congestion_threshold {traffic_metrics['rho_threshold']:.1f}\n")

    # Uniform aggregated summary for cross-model/grid comparison
    write_aggregated_results(
        results_dir, arglist.model_label, arglist,
        distance_total_sharing_by_day, distance_total_no_sharing_by_day,
        acceptance_rates_by_day=acceptance_rates_by_day,
        vehicle_utilization_by_day=vehicle_utilization_by_day,
        multidimensional_stats=multidimensional_stats,
        winner_loser_stats=winner_loser_stats,
        traffic_metrics=traffic_metrics,
        detour_distances_by_day=detour_distances_by_day,
        direct_distances_by_day=direct_distances_by_day,
        individual_times_by_day=individual_times_by_day,
        reintegration_metrics=(reintegration_metrics if arglist.enable_birth_death else None),
    )