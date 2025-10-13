import numpy as np
import itertools
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple
from .pso_baseline import PSOBaseline
from .baseline_trainer import BaselineTrainer
import time

class PSOHyperparameterOptimizer:
    """Grid search hyperparameter optimization for PSO focused on distance savings"""
    
    def __init__(self, env, param_grid: Dict, evaluation_config: Dict = None):
        self.env = env
        self.param_grid = param_grid
        self.evaluation_config = evaluation_config or {
            'num_days': 5,  # Shorter evaluation for speed
            'episodes_per_day': 1,
            'primary_metric': 'total_distance_saved'  # Focus on distance savings
        }
        self.results = []
        
    def run_grid_search(self, save_dir: str = "./hyperparameter_results"):
        """Run grid search optimizing for total distance saved"""
        
        # Create all parameter combinations
        param_combinations = self._generate_param_combinations()
        
        print(f"Starting grid search optimizing for TOTAL DISTANCE SAVED")
        print(f"Evaluating {len(param_combinations)} parameter combinations")
        print(f"Each combination tested for {self.evaluation_config['num_days']} days")
        
        os.makedirs(save_dir, exist_ok=True)
        
        for i, params in enumerate(param_combinations):
            print(f"\n{'='*60}")
            print(f"Combination {i+1}/{len(param_combinations)}")
            print(f"Parameters: {params}")
            print(f"{'='*60}")
            
            start_time = time.time()
            
            # Evaluate this parameter combination
            result = self._evaluate_params(params)
            result['combination_id'] = i
            result['params'] = params
            result['evaluation_time'] = time.time() - start_time
            
            self.results.append(result)
            
            print(f"Completed in {result['evaluation_time']:.1f}s")
            print(f"Total distance saved: {result['total_distance_saved']:.2f}")
            print(f"Average savings per day: {result['avg_distance_saved_per_day']:.2f}")
            print(f"Average savings percentage: {result['avg_savings_percentage']:.2f}%")
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"\n📊 Progress: {i+1}/{len(param_combinations)} combinations completed")
                current_best = max(self.results, key=lambda x: x.get('total_distance_saved', 0))
                print(f"Best so far: {current_best['total_distance_saved']:.2f} units saved")
        
        # Save final results and analysis focused on distance savings
        best_params = self._analyze_and_save_final_results(save_dir)
        
        return best_params, self.results
    
    def _generate_param_combinations(self) -> List[Dict]:
        """Generate all parameter combinations from grid"""
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        combinations = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def _evaluate_params(self, params: Dict) -> Dict:
        """Evaluate a single parameter combination"""
        
        # Properly initialize environment by calling reset_day first
        # This ensures agent_role dictionary is populated
        self.env.reset_day(1, enable_birth_death=False)
        
        # Create trainer with these parameters
        trainer = BaselineTrainer(
            env=self.env,
            baseline_type='pso',
            **params
        )
        
        # Track distance savings across days
        daily_results = []
        total_distance_saved = 0.0
        total_individual_distance = 0.0
        total_shared_distance = 0.0
        
        for day in range(self.evaluation_config['num_days']):
            # Reset day (day 1 is already done above, so start from day 1 for first iteration)
            current_day = day + 1
            if day > 0:  # Only reset for day 2 onwards
                self.env.reset_day(current_day, enable_birth_death=True)
            
            # Run episodes for this day
            day_metrics = []
            for episode in range(self.evaluation_config['episodes_per_day']):
                result = trainer.train_episode(episode)
                day_metrics.append(result)
                trainer._update_training_history(result)
            
            # Calculate day summary focused on distance savings
            day_summary = self._calculate_day_summary_distance_focused(day_metrics)
            daily_results.append(day_summary)
            
            # Accumulate total distance metrics
            total_distance_saved += day_summary['distance_saved']
            total_individual_distance += day_summary['individual_distance']
            total_shared_distance += day_summary['shared_distance']
        
        # Calculate overall distance-focused metrics
        overall_metrics = self._calculate_distance_focused_metrics(
            daily_results, total_distance_saved, total_individual_distance, total_shared_distance
        )
        
        return overall_metrics
    
    def _calculate_day_summary_distance_focused(self, day_metrics: List[Dict]) -> Dict:
        """Calculate summary for a single day focused on distance savings"""
        if not day_metrics:
            return {
                'distance_saved': 0.0,
                'individual_distance': 0.0,
                'shared_distance': 0.0,
                'savings_percentage': 0.0,
                'assignments': 0,
                'fitness': 0.0
            }
        
        # Extract distance comparison data
        distance_comparisons = [m.get('distance_comparison', {}) for m in day_metrics if 'distance_comparison' in m]
        
        if not distance_comparisons:
            return {
                'distance_saved': 0.0,
                'individual_distance': 0.0,
                'shared_distance': 0.0,
                'savings_percentage': 0.0,
                'assignments': 0,
                'fitness': 0.0
            }
        
        # Average across episodes (though typically 1 episode per day)
        individual_distances = [dc.get('total_individual_distance', 0) for dc in distance_comparisons]
        shared_distances = [dc.get('total_shared_distance', 0) for dc in distance_comparisons]
        distances_saved = [dc.get('distance_saved', 0) for dc in distance_comparisons]
        savings_percentages = [dc.get('savings_percentage', 0) for dc in distance_comparisons]
        
        return {
            'distance_saved': np.mean(distances_saved),
            'individual_distance': np.mean(individual_distances),
            'shared_distance': np.mean(shared_distances),
            'savings_percentage': np.mean(savings_percentages),
            'assignments': np.mean([m.get('num_assignments', 0) for m in day_metrics]),
            'fitness': np.mean([m.get('pso_fitness', 0) for m in day_metrics])
        }
    
    def _calculate_distance_focused_metrics(self, daily_results: List[Dict], 
                                          total_distance_saved: float,
                                          total_individual_distance: float, 
                                          total_shared_distance: float) -> Dict:
        """Calculate overall performance metrics focused on distance savings"""
        if not daily_results:
            return {'success': False}
        
        distance_saved_per_day = [d.get('distance_saved', 0) for d in daily_results]
        savings_percentages = [d.get('savings_percentage', 0) for d in daily_results]
        assignment_counts = [d.get('assignments', 0) for d in daily_results]
        fitness_scores = [d.get('fitness', 0) for d in daily_results]
        
        # Calculate efficiency metrics
        avg_distance_saved_per_day = np.mean(distance_saved_per_day)
        consistency_of_savings = 1.0 / (1.0 + np.std(distance_saved_per_day))  # Higher = more consistent savings
        overall_savings_percentage = (total_distance_saved / max(total_individual_distance, 1.0)) * 100
        
        # Distance savings efficiency (savings per assignment)
        total_assignments = sum(assignment_counts)
        savings_per_assignment = total_distance_saved / max(total_assignments, 1.0)
        
        return {
            'success': True,
            
            # PRIMARY METRICS (Distance Savings)
            'total_distance_saved': float(total_distance_saved),
            'avg_distance_saved_per_day': float(avg_distance_saved_per_day),
            'overall_savings_percentage': float(overall_savings_percentage),
            'savings_per_assignment': float(savings_per_assignment),
            'consistency_of_savings': float(consistency_of_savings),
            
            # Supporting metrics
            'total_individual_distance': float(total_individual_distance),
            'total_shared_distance': float(total_shared_distance),
            'total_assignments': int(total_assignments),
            'avg_assignments_per_day': float(np.mean(assignment_counts)),
            
            # Day-by-day breakdown
            'daily_distance_saved': distance_saved_per_day,
            'daily_savings_percentage': savings_percentages,
            'daily_assignments': assignment_counts,
            
            # Secondary metrics
            'avg_savings_percentage': float(np.mean(savings_percentages)),
            'std_savings_percentage': float(np.std(savings_percentages)),
            'avg_fitness': float(np.mean(fitness_scores)),
            'std_fitness': float(np.std(fitness_scores)),
            
            # Detailed daily results
            'daily_results': daily_results
        }
    
    def _analyze_and_save_final_results(self, save_dir: str) -> Dict:
        """Analyze results and save comprehensive report focused on distance savings"""
        
        if not self.results:
            return {}
        
        # Sort results by primary metric: total distance saved
        sorted_results = sorted(self.results, key=lambda x: x.get('total_distance_saved', 0), reverse=True)
        
        # Top performers by different distance-related metrics
        best_total_savings = sorted_results[0]  # Already sorted by total savings
        best_avg_daily_savings = max(self.results, key=lambda x: x.get('avg_distance_saved_per_day', 0))
        best_savings_percentage = max(self.results, key=lambda x: x.get('overall_savings_percentage', 0))
        best_efficiency = max(self.results, key=lambda x: x.get('savings_per_assignment', 0))
        best_consistency = max(self.results, key=lambda x: x.get('consistency_of_savings', 0))
        
        # Create comprehensive analysis focused on distance savings
        analysis = {
            'optimization_summary': {
                'optimization_objective': 'Total Distance Saved',
                'total_combinations': len(self.results),
                'evaluation_config': self.evaluation_config,
                'param_grid': self.param_grid,
                'timestamp': datetime.now().isoformat(),
                'total_evaluation_time': sum(r.get('evaluation_time', 0) for r in self.results)
            },
            'best_parameters': {
                'best_total_distance_saved': {
                    'params': best_total_savings['params'],
                    'total_distance_saved': best_total_savings['total_distance_saved'],
                    'avg_daily_savings': best_total_savings['avg_distance_saved_per_day'],
                    'savings_percentage': best_total_savings['overall_savings_percentage'],
                    'rank': 1
                },
                'best_daily_average_savings': {
                    'params': best_avg_daily_savings['params'],
                    'avg_daily_savings': best_avg_daily_savings['avg_distance_saved_per_day'],
                    'total_distance_saved': best_avg_daily_savings['total_distance_saved'],
                    'savings_percentage': best_avg_daily_savings['overall_savings_percentage']
                },
                'best_savings_percentage': {
                    'params': best_savings_percentage['params'],
                    'savings_percentage': best_savings_percentage['overall_savings_percentage'],
                    'total_distance_saved': best_savings_percentage['total_distance_saved'],
                    'avg_daily_savings': best_savings_percentage['avg_distance_saved_per_day']
                },
                'best_efficiency': {
                    'params': best_efficiency['params'],
                    'savings_per_assignment': best_efficiency['savings_per_assignment'],
                    'total_distance_saved': best_efficiency['total_distance_saved'],
                    'total_assignments': best_efficiency['total_assignments']
                },
                'best_consistency': {
                    'params': best_consistency['params'],
                    'consistency_score': best_consistency['consistency_of_savings'],
                    'total_distance_saved': best_consistency['total_distance_saved'],
                    'std_daily_savings': np.std(best_consistency['daily_distance_saved'])
                }
            },
            'top_10_by_distance_saved': [
                {
                    'rank': i+1,
                    'params': result['params'],
                    'total_distance_saved': result['total_distance_saved'],
                    'savings_percentage': result['overall_savings_percentage'],
                    'avg_daily_savings': result['avg_distance_saved_per_day']
                }
                for i, result in enumerate(sorted_results[:10])
            ],
            'parameter_analysis': self._analyze_parameter_effects_distance_focused(),
            'distance_savings_statistics': self._calculate_distance_savings_statistics(),
            'all_results': sorted_results  # Sorted by total distance saved
        }
        
        # Save comprehensive results - ONLY ONE FILE
        results_file = os.path.join(save_dir, 'pso_distance_optimization_complete.json')
        with open(results_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Save best parameters separately for easy access
        best_params_file = os.path.join(save_dir, 'best_parameters_distance_savings.json')
        with open(best_params_file, 'w') as f:
            json.dump(analysis['best_parameters']['best_total_distance_saved'], f, indent=2)
        
        # Print summary
        self._print_distance_optimization_summary(analysis)
        
        print(f"\n💾 Results saved to:")
        print(f"  📋 Complete analysis: {results_file}")
        print(f"  🏆 Best parameters: {best_params_file}")
        
        return best_total_savings['params']
    
    def _analyze_parameter_effects_distance_focused(self) -> Dict:
        """Analyze the effect of individual parameters on distance savings"""
        param_effects = {}
        
        for param_name in self.param_grid.keys():
            param_effects[param_name] = {}
            
            # Group results by parameter value
            value_groups = {}
            for result in self.results:
                param_value = result['params'][param_name]
                if param_value not in value_groups:
                    value_groups[param_value] = []
                value_groups[param_value].append(result)
            
            # Calculate distance-focused statistics for each value
            for value, group in value_groups.items():
                total_savings = [r['total_distance_saved'] for r in group]
                avg_daily_savings = [r['avg_distance_saved_per_day'] for r in group]
                savings_percentages = [r['overall_savings_percentage'] for r in group]
                
                param_effects[param_name][str(value)] = {
                    'count': len(group),
                    'mean_total_distance_saved': float(np.mean(total_savings)),
                    'std_total_distance_saved': float(np.std(total_savings)),
                    'mean_daily_savings': float(np.mean(avg_daily_savings)),
                    'mean_savings_percentage': float(np.mean(savings_percentages)),
                    'best_total_savings': float(max(total_savings))
                }
        
        return param_effects
    
    def _calculate_distance_savings_statistics(self) -> Dict:
        """Calculate overall statistics about distance savings across all combinations"""
        all_total_savings = [r['total_distance_saved'] for r in self.results]
        all_daily_savings = [r['avg_distance_saved_per_day'] for r in self.results]
        all_percentages = [r['overall_savings_percentage'] for r in self.results]
        
        return {
            'total_distance_saved': {
                'mean': float(np.mean(all_total_savings)),
                'std': float(np.std(all_total_savings)),
                'min': float(np.min(all_total_savings)),
                'max': float(np.max(all_total_savings)),
                'median': float(np.median(all_total_savings))
            },
            'daily_average_savings': {
                'mean': float(np.mean(all_daily_savings)),
                'std': float(np.std(all_daily_savings)),
                'min': float(np.min(all_daily_savings)),
                'max': float(np.max(all_daily_savings))
            },
            'savings_percentage': {
                'mean': float(np.mean(all_percentages)),
                'std': float(np.std(all_percentages)),
                'min': float(np.min(all_percentages)),
                'max': float(np.max(all_percentages))
            }
        }
    
    def _print_distance_optimization_summary(self, analysis: Dict):
        """Print optimization summary focused on distance savings"""
        print(f"\n{'='*80}")
        print("DISTANCE SAVINGS OPTIMIZATION SUMMARY")
        print(f"{'='*80}")
        
        summary = analysis['optimization_summary']
        best = analysis['best_parameters']['best_total_distance_saved']
        stats = analysis['distance_savings_statistics']
        
        print(f"Optimization objective: {summary['optimization_objective']}")
        print(f"Total combinations evaluated: {summary['total_combinations']}")
        print(f"Total evaluation time: {summary['total_evaluation_time']:.1f} seconds")
        
        print(f"\n🏆 BEST PARAMETERS (Maximum Total Distance Saved):")
        for param, value in best['params'].items():
            print(f"  {param}: {value}")
        
        print(f"\n📊 BEST PERFORMANCE:")
        print(f"  Total Distance Saved: {best['total_distance_saved']:.2f} units")
        print(f"  Average Daily Savings: {best['avg_daily_savings']:.2f} units/day")
        print(f"  Overall Savings Percentage: {best['savings_percentage']:.2f}%")
        
        print(f"\n📈 OPTIMIZATION STATISTICS:")
        print(f"  Best total savings found: {stats['total_distance_saved']['max']:.2f}")
        print(f"  Worst total savings: {stats['total_distance_saved']['min']:.2f}")
        print(f"  Average across all combinations: {stats['total_distance_saved']['mean']:.2f}")
        print(f"  Improvement potential: {(stats['total_distance_saved']['max'] - stats['total_distance_saved']['mean']):.2f} units")
        
        print(f"\n🔝 TOP 5 PARAMETER COMBINATIONS:")
        for i, combo in enumerate(analysis['top_10_by_distance_saved'][:5]):
            print(f"  #{combo['rank']}: {combo['total_distance_saved']:.2f} units saved ({combo['savings_percentage']:.1f}%)")
        
        print(f"\n💡 KEY PARAMETER INSIGHTS:")
        param_analysis = analysis['parameter_analysis']
        for param_name, effects in param_analysis.items():
            best_value = max(effects.items(), key=lambda x: x[1]['mean_total_distance_saved'])
            print(f"  {param_name}: {best_value[0]} performs best (avg: {best_value[1]['mean_total_distance_saved']:.2f} units saved)")