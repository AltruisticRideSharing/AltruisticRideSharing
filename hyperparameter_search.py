import argparse
from env.env import Env
from models.pso.hyperparameter_optimizer import PSOHyperparameterOptimizer

def define_distance_focused_grid():
    """Define parameter grid focused on maximizing distance savings"""
    
    param_grid = {
        # PSO algorithm parameters - broader range for distance optimization
        'num_particles': [20, 30, 50],
        'max_iterations': [30, 50, 80],
        'w': [0.4, 0.5, 0.7, 0.9],        
        'c1': [1.0, 1.5, 2.0, 2.5],            
        'c2': [1.0, 1.5, 2.0, 2.5],            
        
        # Fitness function parameters - critical for distance savings
        'alpha': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],    # Distance vs detour balance
        'beta': [0.1, 0.2, 0.3, 0.4],                # Altruism weight
        'theta': [3.0, 5.0, 7.0, 10.0],              # Detour threshold
        
        # Bonus/penalty weights for distribution and efficiency
        'assignment_bonus_weight': [1.0, 2.0, 3.0, 4.0],
        'load_balance_weight': [0.5, 1.0, 2.0, 3.0],
        'unassigned_penalty_weight': [0.5, 1.0, 1.5, 2.0]
    }
    
    return param_grid

def define_quick_distance_grid():
    """Define a smaller grid for quick distance optimization testing"""
    
    param_grid = {
        'num_particles': [30, 50],
        'max_iterations': [50, 80],
        'w': [0.5, 0.7],
        'c1': [1.5, 2.0],
        'c2': [1.5, 2.0],
        'alpha': [0.3, 0.4, 0.5, 0.6],          # Key parameter for distance savings
        'theta': [5.0, 7.0],
        'assignment_bonus_weight': [2.0, 3.0],
        'load_balance_weight': [1.0, 2.0],
        'unassigned_penalty_weight': [1.0]
    }
    
    return param_grid

def main():
    parser = argparse.ArgumentParser(description='PSO Distance Savings Optimization')
    
    parser.add_argument('--num-agents', type=int, default=100, help='Number of agents')
    parser.add_argument('--grid-size', type=int, default=15, help='Grid size')
    parser.add_argument('--dataset', type=str, default='100_agents', help='Dataset folder')
    parser.add_argument('--evaluation-days', type=int, default=5, help='Days for evaluation')
    parser.add_argument('--grid-type', type=str, default='quick', choices=['full', 'quick'],
                       help='Type of parameter grid for distance optimization')
    parser.add_argument('--save-dir', type=str, default='./distance_optimization_results',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("🚗 PSO HYPERPARAMETER OPTIMIZATION FOR MAXIMUM DISTANCE SAVINGS 🚗")
    print("=" * 70)
    
    # Initialize environment
    env = Env(
        height=args.grid_size,
        width=args.grid_size,
        numAgents=args.num_agents,
        dataset_folder=args.dataset,
        initial_active_agents=args.num_agents,
        altruism_distribution='uniform',
        total_days=30
    )
    
    # Choose parameter grid focused on distance savings
    if args.grid_type == 'full':
        param_grid = define_distance_focused_grid()
        print("Using FULL parameter grid for comprehensive distance optimization")
    else:
        param_grid = define_quick_distance_grid()
        print("Using QUICK parameter grid for fast distance optimization")
    
    # Calculate total combinations
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    
    print(f"Total parameter combinations: {total_combinations}")
    print(f"Evaluation days per combination: {args.evaluation_days}")
    print(f"Estimated evaluation time: ~{total_combinations * args.evaluation_days * 0.5:.1f} minutes")
    
    # Setup evaluation config focused on distance savings
    evaluation_config = {
        'num_days': args.evaluation_days,
        'episodes_per_day': 1,
        'primary_metric': 'total_distance_saved'
    }
    
    # Run optimization
    optimizer = PSOHyperparameterOptimizer(
        env=env,
        param_grid=param_grid,
        evaluation_config=evaluation_config
    )
    
    best_params, all_results = optimizer.run_grid_search(save_dir=args.save_dir)
    
    print(f"\n🎉 DISTANCE SAVINGS OPTIMIZATION COMPLETED! 🎉")
    print(f"Results saved to: {args.save_dir}")
    print(f"Best parameters for maximum distance savings:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

if __name__ == "__main__":
    main()