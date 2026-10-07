import argparse
import numpy as np
import os
import json
import time
from datetime import datetime
import tensorflow as tf
from tqdm import tqdm

# Import your existing modules
from ars.env.env import Env
from ars.models.pso import BaselineTrainer

def parse_args():
    """Parse command line arguments for PSO baseline"""
    parser = argparse.ArgumentParser(description='PSO Baseline for Altruistic Ride Sharing')
    
    # Environment parameters
    parser.add_argument('--num-agents', type=int, default=100, 
                       help='Number of agents (100, 150, 200)')
    parser.add_argument('--altruism-dist', type=str, default='uniform', 
                       choices=['uniform', 'gaussian'], 
                       help='Altruism distribution type')
    parser.add_argument('--days', type=int, default=30, 
                       help='Number of simulation days')
    parser.add_argument('--grid-size', type=int, default=15, 
                       help='Grid size (15x15)')
    parser.add_argument('--max-capacity', type=int, default=4, 
                       help='Maximum vehicle capacity')
    parser.add_argument('--dataset', type=str, default='100_agents', 
                       help='Dataset folder name (e.g., 100_agents, 150_agents, 200_agents)')
    
    # Agent dynamics options (UPDATED - removed 'dropout')
    parser.add_argument('--agent-dynamics', type=str, default='fixed', 
                       choices=['fixed', 'birth_death'],
                       help='Agent population dynamics: fixed (no changes) or birth_death (agents can join/leave with daily role reassignment)')
    parser.add_argument('--initial-active-agents', type=int, default=None,
                       help='Number of initially active agents (default: same as num-agents for fixed, 80%% for birth_death)')
    
    # PSO parameters
    parser.add_argument('--pso-particles', type=int, default=50, 
                       help='Number of PSO particles')
    parser.add_argument('--pso-iterations', type=int, default=80, 
                       help='PSO optimization iterations')
    parser.add_argument('--pso-w', type=float, default=0.7, 
                       help='PSO inertia weight')
    parser.add_argument('--pso-c1', type=float, default=1.5, 
                       help='PSO cognitive coefficient')
    parser.add_argument('--pso-c2', type=float, default=1.5, 
                       help='PSO social coefficient')
    parser.add_argument('--alpha', type=float, default=0.4, 
                       help='Multi-objective balance parameter')
    parser.add_argument('--max-detour', type=float, default=6.0,
                       help='Maximum allowed detour for driver assignments')
    
    # Training parameters
    parser.add_argument('--episodes-per-day', type=int, default=1, 
                       help='Episodes per day (PSO typically uses 1)')
    parser.add_argument('--enable-evaluation', action='store_true', 
                       help='Enable evaluation mode')
    
    # Logging and saving
    parser.add_argument('--save-dir', type=str, default='./results/pso', 
                       help='Directory to save results')
    parser.add_argument('--tensorboard-dir', type=str, default='./runs/pso', 
                       help='Directory for tensorboard logs')
    parser.add_argument('--experiment-name', type=str, default=None, 
                       help='Experiment name for organizing results')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    
    parser.add_argument('--enable-benefit-analysis', action='store_true',
                       help='Enable comprehensive benefit analysis with Lorenz curves')
    
    return parser.parse_args()

def setup_experiment(args):
    """Setup experiment directory and configuration"""
    
    # Create experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f"pso_{args.num_agents}agents_{args.altruism_dist}_{args.agent_dynamics}_{timestamp}"
    
    # Create save directory
    save_dir = os.path.join(args.save_dir, args.experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Create tensorboard directory
    tensorboard_dir = os.path.join(args.tensorboard_dir, args.experiment_name)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Experiment: {args.experiment_name}")
    print(f"Results will be saved to: {save_dir}")
    print(f"Tensorboard logs: {tensorboard_dir}")
    print(f"Agent dynamics: {args.agent_dynamics}")
    if args.agent_dynamics == 'fixed':
        print("  → Fixed population with daily role reassignment based on altruism")
    else:
        print("  → Dynamic population with births/dropouts and daily role reassignment")
    
    return save_dir, tensorboard_dir

def initialize_environment(args):
    """Initialize the ARS environment with dynamic agent options"""
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Determine initial active agents based on dynamics mode
    if args.initial_active_agents is None:
        if args.agent_dynamics == 'birth_death':
            # Start with 80% of agents for birth-death dynamics
            args.initial_active_agents = int(args.num_agents * 0.8)
        else:  # fixed
            # Start with all agents for fixed dynamics
            args.initial_active_agents = args.num_agents
    
    # Create environment with correct parameters matching Env.__init__
    env = Env(
        height=args.grid_size,
        width=args.grid_size,
        numAgents=args.num_agents,
        dataset_folder=args.dataset,
        initial_active_agents=args.initial_active_agents,
        altruism_distribution=args.altruism_dist,
        altruism_mean=0.5,
        altruism_std=0.15,
        total_days=args.days
    )
    
    # Set alpha after initialization
    env.alpha = args.alpha
    
    if args.verbose:
        print(f"Environment initialized with {args.num_agents} total agents")
        print(f"Initial active agents: {args.initial_active_agents}")
        print(f"Grid size: {args.grid_size}x{args.grid_size}")
        print(f"Altruism distribution: {args.altruism_dist}")
        print(f"Agent dynamics: {args.agent_dynamics}")
        print(f"Dataset folder: {args.dataset}")
        print(f"Alpha: {args.alpha}")
    
    return env

def initialize_baseline_trainer(env, args):
    """Initialize PSO baseline trainer"""
    
    pso_params = {
        'num_particles': args.pso_particles,
        'max_iterations': args.pso_iterations,
        'w': args.pso_w,
        'c1': args.pso_c1,
        'c2': args.pso_c2,
        'alpha': args.alpha,
        'max_detour': args.max_detour,
        'verbose': args.verbose
    }
    
    trainer = BaselineTrainer(
        env=env,
        baseline_type='pso',
        **pso_params
    )
    
    if args.verbose:
        print(f"PSO trainer initialized:")
        print(f"  Particles: {args.pso_particles}")
        print(f"  Iterations: {args.pso_iterations}")
        print(f"  Parameters: w={args.pso_w}, c1={args.pso_c1}, c2={args.pso_c2}")
        print(f"  Alpha: {args.alpha}")
        print(f"  Max Detour: {args.max_detour}")
    
    return trainer

def log_to_tensorboard(writer, day, episode, result, env, global_step):
    """Log metrics to tensorboard using TensorFlow"""
    
    with writer.as_default():
        # Basic PSO metrics
        tf.summary.scalar('PSO/Total_Reward', result['total_reward'], step=global_step)
        tf.summary.scalar('PSO/Fitness_Score', result['pso_fitness'], step=global_step)
        tf.summary.scalar('PSO/Num_Assignments', result['num_assignments'], step=global_step)
        
        # PSO-specific metrics if available
        if 'pso_convergence' in result:
            tf.summary.scalar('PSO/Convergence_Ratio', result['pso_convergence'], step=global_step)
        if 'pso_diversity' in result:
            tf.summary.scalar('PSO/Swarm_Diversity', result['pso_diversity'], step=global_step)
        if 'pso_iterations_used' in result:
            tf.summary.scalar('PSO/Iterations_Used', result['pso_iterations_used'], step=global_step)
        
        # Detailed metrics if available
        if 'metrics' in result:
            metrics = result['metrics']
            
            # Distance and efficiency
            if 'distance_comparison' in metrics:
                dist_comp = metrics['distance_comparison']
                tf.summary.scalar('Metrics/Distance_Saved', dist_comp.get('distance_saved', 0), step=global_step)
                tf.summary.scalar('Metrics/Savings_Percentage', dist_comp.get('savings_percentage', 0), step=global_step)
            
            if 'detour_factors' in metrics:
                tf.summary.scalar('Metrics/Detour_Factor_Mean', metrics['detour_factors']['mean'], step=global_step)
            if 'avg_trip_time' in metrics:
                tf.summary.scalar('Metrics/Avg_Trip_Time', metrics['avg_trip_time']['mean'], step=global_step)
            if 'vehicle_utilization' in metrics:
                tf.summary.scalar('Metrics/Vehicle_Utilization', metrics['vehicle_utilization']['mean'], step=global_step)
            if 'per_agent_distances' in metrics:
                tf.summary.scalar('Metrics/Per_Agent_Distance_Mean', metrics['per_agent_distances']['mean'], step=global_step)
            
            # Altruism metrics
            if 'altruism_stats' in metrics:
                alt_dist = metrics['altruism_stats']
                if isinstance(alt_dist, dict):
                    tf.summary.scalar('Altruism/Mean', alt_dist.get('mean', 0), step=global_step)
                    tf.summary.scalar('Altruism/Std', alt_dist.get('std', 0), step=global_step)
                    tf.summary.scalar('Altruism/Min', alt_dist.get('min', 0), step=global_step)
                    tf.summary.scalar('Altruism/Max', alt_dist.get('max', 0), step=global_step)
        
        # Environment state metrics (including dynamic population tracking)
        active_agents = len([a for a in env.agents if env.agent_role.get(a, 'active') not in ['dropout', 'never_joined']])
        num_drivers = len([a for a in env.agents if env.agent_role.get(a, 'active') == 'driver'])
        num_riders = len([a for a in env.agents if env.agent_role.get(a, 'active') == 'rider'])
        
        tf.summary.scalar('Environment/Active_Agents', active_agents, step=global_step)
        tf.summary.scalar('Environment/Num_Drivers', num_drivers, step=global_step)
        tf.summary.scalar('Environment/Num_Riders', num_riders, step=global_step)
        tf.summary.scalar('Environment/Driver_Rider_Ratio', num_drivers / max(1, num_riders), step=global_step)
        
        # NEW: Dynamic population metrics
        if hasattr(env, 'dropped_out_agents'):
            tf.summary.scalar('Population/Dropped_Out_Agents', len(env.dropped_out_agents), step=global_step)
        if hasattr(env, 'never_joined_agents'):
            tf.summary.scalar('Population/Never_Joined_Agents', len(env.never_joined_agents), step=global_step)
        
        # Birth-death specific metrics
        if hasattr(env, 'daily_births') and day in env.daily_births:
            tf.summary.scalar('Population/Daily_Births', len(env.daily_births[day]), step=global_step)
        if hasattr(env, 'daily_dropouts') and day in env.daily_dropouts:
            tf.summary.scalar('Population/Daily_Dropouts', len(env.daily_dropouts[day]), step=global_step)
        
        # Daily progress
        tf.summary.scalar('Progress/Day', day, step=global_step)
        tf.summary.scalar('Progress/Episode', episode, step=global_step)
        
        # Flush the writer
        writer.flush()

def log_hyperparameters(writer, hparams):
    """Log hyperparameters to tensorboard"""
    with writer.as_default():
        # Convert hyperparameters to text summary
        hparam_text = "\n".join([f"{k}: {v}" for k, v in hparams.items()])
        tf.summary.text('Hyperparameters', hparam_text, step=0)
        
        # Log individual hyperparameters as scalars
        for key, value in hparams.items():
            if isinstance(value, (int, float)):
                tf.summary.scalar(f'Hyperparameters/{key}', value, step=0)
        
        writer.flush()

def _build_no_sharing_paths(env):
    """Direct Dijkstra cell-path for every active agent -- the no-sharing baseline
    the traffic-density metric rasterises. Mirrors what run_simulation records for
    the MARL evaluators so PSO traffic is on the SAME basis as ORACLE/MADDPG/MAAC."""
    from ars.metrics import _agent_index
    paths = {}
    for agent in env.agents:
        if env.agent_role.get(agent, 'active') in ('dropout', 'never_joined'):
            continue
        i = _agent_index(env, agent)
        if i is None:
            continue
        obj = env.agent_objects[i]
        try:
            p = obj._dijkstra_path(tuple(obj.start_position), tuple(obj.destination))
        except Exception:
            p = []
        paths[agent] = p or []
    return paths


def _gini_positive(values):
    """Gini over positive benefits only -- identical basis to the MARL evals'
    calculate_gini (0 = equal, 1 = maximally unequal)."""
    vals = sorted(v for v in values if v is not None and v > 0)
    if not vals:
        return 0.0
    n = len(vals); cum = np.cumsum(vals)
    return float((n + 1 - 2 * sum(cum) / cum[-1]) / n) if cum[-1] > 0 else 0.0


def _accumulate_pso_benefits(env, trainer):
    """Accumulate per-agent combined benefits across days (reusing the trainer's
    cached assignment, no re-optimization) so a Gini can be taken over the whole
    horizon -- the same combined distance+traffic benefit the MARL evals use."""
    try:
        day_ben = trainer.calculate_agent_benefits()
    except Exception:
        return
    acc = getattr(env, '_pso_agent_benefits', {})
    for ag, b in (day_ben or {}).items():
        e = acc.setdefault(ag, {'distance_benefit': 0.0, 'traffic_benefit': 0.0, 'combined_benefit': 0.0})
        e['distance_benefit'] += float(b.get('distance_benefit', 0.0) or 0.0)
        e['traffic_benefit'] += float(b.get('traffic_benefit', 0.0) or 0.0)
        e['combined_benefit'] += float(b.get('combined_benefit', 0.0) or 0.0)
    env._pso_agent_benefits = acc


def _record_pso_traffic_day(env, assignments, day):
    """Record this day's driver->rider pickups (so the traffic metric can rebuild
    shared routes) and stash the no-sharing paths on env, keyed by 1-indexed day."""
    for driver_name, rider_list in (assignments or {}).items():
        for entry in (rider_list or []):
            rider_name = entry[0] if isinstance(entry, (list, tuple)) else entry
            try:
                env.track_daily_pickup(driver_name, rider_name)
            except Exception:
                pass
    if not hasattr(env, '_pso_no_sharing_paths'):
        env._pso_no_sharing_paths = {}
    env._pso_no_sharing_paths[day] = _build_no_sharing_paths(env)


def run_daily_simulation(env, trainer, day, args, save_dir, writer):
    """Run simulation for a single day with role reassignment"""
    
    if args.verbose:
        print(f"\n{'='*50}")
        print(f"Day {day + 1} Simulation")
        print(f"{'='*50}")

    env.current_day = day + 1  
    
    # Reset environment for new day based on agent dynamics mode
    if args.agent_dynamics == 'fixed':
        # Fixed population - but roles change daily based on altruism
        env.reset_day(day + 1, enable_birth_death=False)
        if args.verbose:
            print("Fixed population mode: Same agents, roles reassigned by altruism")
    elif args.agent_dynamics == 'birth_death':
        # Full birth-death dynamics - agents can join/leave, roles reassigned
        env.reset_day(day + 1, enable_birth_death=True)
        env.reset(enable_birth_death=True)
        if args.verbose:
            print("Birth-death mode: Dynamic population with role reassignment")
    
    # Get active agent count after population dynamics and role assignment
    active_agents = len([a for a in env.agents if env.agent_role.get(a, 'active') not in ['dropout', 'never_joined']])
    num_drivers = len([a for a in env.agents if env.agent_role.get(a, 'active') == 'driver'])
    num_riders = len([a for a in env.agents if env.agent_role.get(a, 'active') == 'rider'])
    
    if args.verbose:
        print(f"After reset_day:")
        print(f"  Active agents: {active_agents}")
        print(f"  Drivers: {num_drivers}")
        print(f"  Riders: {num_riders}")
        
        # Show population changes if using birth_death mode
        if args.agent_dynamics == 'birth_death':
            if hasattr(env, 'daily_births') and (day + 1) in env.daily_births:
                births = len(env.daily_births[day + 1])
                print(f"  New births today: {births}")
            if hasattr(env, 'daily_dropouts') and (day + 1) in env.daily_dropouts:
                dropouts = len(env.daily_dropouts[day + 1])
                print(f"  Dropouts today: {dropouts}")
            if hasattr(env, 'never_joined_agents'):
                never_joined = len(env.never_joined_agents)
                print(f"  Never joined: {never_joined}")
        
        # Show role assignment rationale (for both modes)
        if day > 0 and args.verbose:
            driver_altruism = []
            rider_altruism = []
            for i, agent in enumerate(env.agents):
                role = env.agent_role.get(agent, 'active')
                if role == 'driver':
                    driver_altruism.append(env.agent_objects[i].altruism_points)
                elif role == 'rider':
                    rider_altruism.append(env.agent_objects[i].altruism_points)
            
            if driver_altruism and rider_altruism:
                print(f"  Role assignment (by altruism):")
                print(f"    Driver altruism range: {min(driver_altruism):.3f} - {max(driver_altruism):.3f}")
                print(f"    Rider altruism range: {min(rider_altruism):.3f} - {max(rider_altruism):.3f}")
    
    # Skip PSO if no active agents
    if active_agents == 0 or (num_drivers == 0 and num_riders == 0):
        if args.verbose:
            print("No active agents available for PSO optimization, skipping day")
        
        # Create empty result
        empty_result = {
            'total_reward': 0.0,
            'pso_fitness': 0.0,
            'num_assignments': 0,
            'assignments': {},
            'metrics': {
                'distance_comparison': {'distance_saved': 0.0, 'savings_percentage': 0.0},
                'detour_factors': {'mean': 0.0, 'std': 0.0},
                'avg_trip_time': {'mean': 0.0, 'std': 0.0},
                'vehicle_utilization': {'mean': 0.0, 'std': 0.0},
                'per_agent_distances': {'mean': 0.0, 'std': 0.0},
                'altruism_stats': env.get_altruism_distribution_stats()
            }
        }
        
        day_summary = {
            'day': day + 1,
            'active_agents': active_agents,
            'episodes': 1,
            'total_reward': 0.0,
            'avg_fitness': 0.0,
            'total_assignments': 0,
            'skipped': True
        }
        # no assignments today, but keep the day represented for traffic accounting
        _record_pso_traffic_day(env, {}, day + 1)
        return day_summary
    
    # Reset trainer for new day
    trainer.reset_for_new_day()
    
    # Run episodes for this day
    day_results = []
    for episode in range(args.episodes_per_day):
        if args.verbose and args.episodes_per_day > 1:
            print(f"  Episode {episode + 1}/{args.episodes_per_day}")
        
        # Train episode - this will create fresh PSO and calculate metrics
        result = trainer.train_episode(episode, args.enable_evaluation)
        
        if args.verbose:
            print(f"Episode {episode}: Assignments found: {result['num_assignments']}")
            
        day_results.append(result)
        
        # Update training history - this stores the result for JSON/dat output
        trainer._update_training_history(result)
        
        # Log to tensorboard
        global_step = day * args.episodes_per_day + episode
        log_to_tensorboard(writer, day + 1, episode + 1, result, env, global_step)
        
        if args.verbose:
            print(f"    Total Reward: {result['total_reward']:.2f}")
            print(f"    Assignments: {result['num_assignments']}")
            print(f"    PSO Fitness: {result['pso_fitness']:.4f}")
            
            # Print distance savings info
            metrics = result.get('metrics', {})
            dist_comp = metrics.get('distance_comparison', {})
            if dist_comp:
                print(f"    Distance Saved: {dist_comp.get('distance_saved', 0):.2f}")
                print(f"    Savings %: {dist_comp.get('savings_percentage', 0):.2f}%")
    
    # Lightweight save every day (JSON + .dat only); full save on the last day
    if (day + 1) == args.days:
        print(f"Saving full results for Day {day + 1}...")
        trainer.save_results(save_dir, day + 1)
    else:
        print(f"Saving daily results for Day {day + 1}...")
        trainer.save_results_lightweight(save_dir, day + 1)
    
    # Log daily summary to tensorboard
    total_reward = sum(r['total_reward'] for r in day_results)
    avg_fitness = np.mean([r['pso_fitness'] for r in day_results])
    total_assignments = sum(r['num_assignments'] for r in day_results)
    
    with writer.as_default():
        tf.summary.scalar('Daily/Total_Reward', total_reward, step=day + 1)
        tf.summary.scalar('Daily/Avg_Fitness', avg_fitness, step=day + 1)
        tf.summary.scalar('Daily/Total_Assignments', total_assignments, step=day + 1)
        writer.flush()
    
    # Aggregate per-episode metrics so the daily summary carries the headline
    # numbers (distance savings, vehicle utilization) needed for results_summary.txt.
    def _mean_metric(path_keys):
        vals = []
        for r in day_results:
            d = r.get('metrics', {})
            for k in path_keys:
                d = d.get(k, {}) if isinstance(d, dict) else {}
            if isinstance(d, (int, float)):
                vals.append(d)
        return float(np.mean(vals)) if vals else 0.0

    savings_percentage = _mean_metric(['distance_comparison', 'savings_percentage'])
    vehicle_utilization = _mean_metric(['vehicle_utilization', 'mean'])
    detour_factor = _mean_metric(['detour_factors', 'mean'])
    avg_trip_time = _mean_metric(['avg_trip_time', 'mean'])

    # Return summary of day
    day_summary = {
        'day': day + 1,
        'active_agents': active_agents,
        'num_drivers': num_drivers,
        'num_riders': num_riders,
        'episodes': len(day_results),
        'total_reward': total_reward,
        'avg_fitness': avg_fitness,
        'total_assignments': total_assignments,
        'savings_percentage': savings_percentage,
        'vehicle_utilization': vehicle_utilization,
        'detour_factor': detour_factor,
        'avg_trip_time': avg_trip_time,
        'skipped': False
    }

    # --- traffic bookkeeping: record the best episode's pickups + no-sharing paths ---
    try:
        best_r = max(day_results, key=lambda r: r.get('pso_fitness', float('-inf')))
        _record_pso_traffic_day(env, best_r.get('assignments', {}), day + 1)
    except Exception as e:
        print(f"  [traffic] day {day + 1} bookkeeping skipped: {e}")

    # --- benefit accumulation for the Gini (fairness) metric ---
    _accumulate_pso_benefits(env, trainer)

    return day_summary

def save_final_results(trainer, args, save_dir, daily_summaries, traffic_reduction_percent=None,
                       gini_combined=None, reintegration_score=None):
    """Save comprehensive final results"""
    
    # Get training statistics
    stats = trainer.get_training_statistics()
    
    # Compile comprehensive results
    final_results = {
        'experiment_config': vars(args),
        'training_statistics': stats,
        'daily_summaries': daily_summaries,
        'total_days': args.days,
        'baseline_type': 'pso',
        'agent_dynamics': args.agent_dynamics,
        'completion_time': datetime.now().isoformat()
    }
    
    # Save final results (this is the comprehensive summary, separate from daily results.json)
    results_path = os.path.join(save_dir, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\nFinal results saved to: {results_path}")

    # Uniform aggregated summary (same fields as the MARL evals) for cross-model
    # comparison in the grid ablation.
    active_days = [d for d in daily_summaries if not d.get('skipped', False)]
    savings = [d.get('savings_percentage', 0.0) for d in active_days]
    utils = [d.get('vehicle_utilization', 0.0) for d in active_days
             if d.get('vehicle_utilization', 0.0) > 0]
    detours = [d.get('detour_factor', 0.0) for d in active_days
               if d.get('detour_factor', 0.0) > 0]
    trips = [d.get('avg_trip_time', 0.0) for d in active_days
             if d.get('avg_trip_time', 0.0) > 0]
    dyn = 'birth_death' if getattr(args, 'agent_dynamics', 'fixed') == 'birth_death' else 'fixed'
    summary_lines = [
        "=== AGGREGATED RESULTS SUMMARY ===",
        "model: pso",
        f"grid_size: {getattr(args, 'grid_size', 'NA')}",
        f"num_agents: {getattr(args, 'num_agents', 'NA')}",
        f"altruism_distribution: {getattr(args, 'altruism_dist', 'NA')}",
        f"population_dynamics: {dyn}",
        f"days: {args.days}",
        "",
        f"distance_reduction_percent: {float(np.mean(savings)) if savings else 0.0:.2f}",
        f"mean_vehicle_utilization: {float(np.mean(utils)) if utils else 0.0:.3f}",
    ]
    # PSO also produces detour factor and trip time (from per-day metrics);
    # acceptance/gini/reintegration are not defined for the one-shot optimizer.
    if detours:
        summary_lines.append(f"avg_detour_factor: {float(np.mean(detours)):.4f}")
    if trips:
        summary_lines.append(f"avg_trip_time: {float(np.mean(trips)):.3f}")
    # Traffic-density reduction, computed with the SAME Dijkstra-rasterisation
    # logic as the MARL evals (only emitted when available).
    if traffic_reduction_percent is not None:
        summary_lines.append(f"traffic_reduction_percent: {float(traffic_reduction_percent):.2f}")
    # Benefit inequality + reintegration -- wireable for the optimizer even though
    # acceptance-rate is not (no offer/accept protocol in a one-shot assignment).
    if gini_combined is not None:
        summary_lines.append(f"gini_combined_benefits: {float(gini_combined):.4f}")
    if reintegration_score is not None:
        summary_lines.append(f"reintegration_score: {float(reintegration_score):.2f}")
    summary_path = os.path.join(save_dir, 'results_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"Aggregated summary saved to: {summary_path}")

    return final_results

def print_experiment_summary(final_results, args):
    """Print a summary of the experiment"""
    
    print(f"\n{'='*60}")
    print(f"PSO BASELINE EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    stats = final_results['training_statistics']
    daily_summaries = final_results['daily_summaries']
    
    print(f"Configuration:")
    print(f"  Agents: {args.num_agents} (initially active: {args.initial_active_agents})")
    print(f"  Days: {args.days}")
    print(f"  Agent Dynamics: {args.agent_dynamics}")
    if args.agent_dynamics == 'fixed':
        print(f"    → Fixed population with daily role reassignment")
    else:
        print(f"    → Dynamic population with births/dropouts and role reassignment")
    print(f"  Altruism Distribution: {args.altruism_dist}")
    print(f"  PSO Particles: {args.pso_particles}")
    print(f"  PSO Iterations: {args.pso_iterations}")
    
    print(f"\nOverall Performance:")
    print(f"  Total Episodes: {stats['training_episodes']}")
    print(f"  Mean Reward: {stats['rewards']['mean']:.2f} ± {stats['rewards']['std']:.2f}")
    print(f"  Mean Fitness: {stats['fitness_scores']['mean']:.4f} ± {stats['fitness_scores']['std']:.4f}")
    print(f"  Mean Assignments: {stats['assignments']['mean']:.1f} ± {stats['assignments']['std']:.1f}")
    
    # Population dynamics summary
    if args.agent_dynamics == 'birth_death':
        active_agents_over_time = [d['active_agents'] for d in daily_summaries]
        print(f"\nPopulation Dynamics:")
        print(f"  Starting Active Agents: {active_agents_over_time[0] if active_agents_over_time else 0}")
        print(f"  Final Active Agents: {active_agents_over_time[-1] if active_agents_over_time else 0}")
        print(f"  Peak Active Agents: {max(active_agents_over_time) if active_agents_over_time else 0}")
        print(f"  Min Active Agents: {min(active_agents_over_time) if active_agents_over_time else 0}")
    else:
        print(f"\nFixed Population Mode:")
        print(f"  Consistent {args.initial_active_agents} active agents with daily role reassignment")
    
    # Daily trend
    rewards = [d['total_reward'] for d in daily_summaries]
    assignments = [d['total_assignments'] for d in daily_summaries]
    
    print(f"\nDaily Trends:")
    print(f"  Best Day Reward: {max(rewards):.2f} (Day {rewards.index(max(rewards)) + 1})")
    print(f"  Best Day Assignments: {max(assignments)} (Day {assignments.index(max(assignments)) + 1})")
    print(f"  Final Day Reward: {rewards[-1]:.2f}")
    print(f"  Final Day Assignments: {assignments[-1]}")
    
    # Count skipped days (days with no active agents)
    skipped_days = sum(1 for d in daily_summaries if d.get('skipped', False))
    if skipped_days > 0:
        print(f"  Skipped Days (no active agents): {skipped_days}")
    
    print(f"\n{'='*60}")

def main():
    """Main function for PSO baseline experiment"""
    
    print("Altruistic Ride Sharing - PSO Baseline")
    print("=" * 50)
    
    # Parse arguments
    args = parse_args()
    
    # Setup experiment
    save_dir, tensorboard_dir = setup_experiment(args)
    
    # Initialize tensorboard writer (TensorFlow)
    writer = tf.summary.create_file_writer(tensorboard_dir)
    
    # Log hyperparameters to tensorboard
    hparams = {
        'num_agents': args.num_agents,
        'initial_active_agents': args.initial_active_agents,
        'days': args.days,
        'altruism_dist': args.altruism_dist,
        'agent_dynamics': args.agent_dynamics,
        'pso_particles': args.pso_particles,
        'pso_iterations': args.pso_iterations,
        'pso_w': args.pso_w,
        'pso_c1': args.pso_c1,
        'pso_c2': args.pso_c2,
        'alpha': args.alpha,
        'episodes_per_day': args.episodes_per_day,
        'seed': args.seed
    }
    log_hyperparameters(writer, hparams)
    
    # Initialize environment
    env = initialize_environment(args)
    
    # Initialize baseline trainer
    trainer = initialize_baseline_trainer(env, args)
    
    # Run multi-day simulation
    daily_summaries = []
    start_time = time.time()
    
    try:
        for day in range(args.days):
            day_summary = run_daily_simulation(env, trainer, day, args, save_dir, writer)
            daily_summaries.append(day_summary)
            
            if args.verbose:
                elapsed = time.time() - start_time
                print(f"Day {day + 1} completed. Elapsed time: {elapsed:.1f}s")
    
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        args.days = len(daily_summaries)  # Update days to actual completed days
    
    except Exception as e:
        print(f"\nError during experiment: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    finally:
        # Close tensorboard writer
        writer.close()
    
    # --- compute traffic-density reduction with the SAME logic as the MARL evals ---
    traffic_reduction_percent = None
    try:
        from ars.metrics import calculate_traffic_density_metrics
        paths_ns = getattr(env, '_pso_no_sharing_paths', {})
        if paths_ns and env.daily_pickups:
            tm, _, _ = calculate_traffic_density_metrics(
                paths_ns, paths_ns, env.height, env.width, args.days, env, args)
            traffic_reduction_percent = float(tm.get('traffic_reduction_percentage'))
            print(f"[traffic] PSO traffic_reduction_percent = {traffic_reduction_percent:.2f}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[traffic] PSO traffic calc failed: {e}")

    # --- Gini (benefit inequality) over the accumulated per-agent combined benefits ---
    gini_combined = None
    try:
        ab = getattr(env, '_pso_agent_benefits', {})
        if ab:
            gini_combined = _gini_positive([b['combined_benefit'] for b in ab.values()])
            print(f"[gini] PSO gini_combined_benefits = {gini_combined:.4f}")
    except Exception as e:
        print(f"[gini] PSO gini calc failed: {e}")

    # --- Reintegration score (birth-death only; env tracks the events) ---
    reintegration_score = None
    if getattr(args, 'agent_dynamics', 'fixed') == 'birth_death':
        try:
            from ars.metrics import calculate_reintegration_metrics
            rm = calculate_reintegration_metrics(env)
            reintegration_score = float(rm.get('final_reintegration_score'))
            print(f"[reintegration] PSO reintegration_score = {reintegration_score:.2f}")
        except Exception as e:
            print(f"[reintegration] PSO calc failed: {e}")

    # Save final results
    final_results = save_final_results(trainer, args, save_dir, daily_summaries,
                                       traffic_reduction_percent=traffic_reduction_percent,
                                       gini_combined=gini_combined,
                                       reintegration_score=reintegration_score)
    
    # Print summary
    print_experiment_summary(final_results, args)
    
    total_time = time.time() - start_time
    print(f"\nTotal experiment time: {total_time:.1f} seconds")
    print(f"Results saved to: {save_dir}")
    print(f"View tensorboard with: tensorboard --logdir {args.tensorboard_dir}")

if __name__ == "__main__":
    main()