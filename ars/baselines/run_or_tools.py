import argparse
import os
import json
import time
from datetime import datetime

import numpy as np
import subprocess
import sys

from ars.env.env import Env
from ars.models.or_tools.baseline_trainer import ORToolsBaselineTrainer
from ars.models.or_tools import run_or_tools as rot


def _gini_positive(values):
    """Gini over positive benefits only -- same basis as the MARL evals."""
    vals = sorted(v for v in values if v is not None and v > 0)
    if not vals:
        return 0.0
    n = len(vals); cum = np.cumsum(vals)
    return float((n + 1 - 2 * sum(cum) / cum[-1]) / n) if cum[-1] > 0 else 0.0


def _ortools_day_benefits(env, assignments):
    """Per-agent combined benefit for one day's assignment, replicating the PSO
    accounting exactly (driver: route savings + absorbed rider distance; assigned
    rider: full distance; combined = distance + 0.5*traffic) so the two optimizers'
    Gini are directly comparable."""
    from ars.models.or_tools import run_or_tools as rot
    idx = {name: env.agent_objects[i] for i, name in enumerate(env.agents) if i < len(env.agent_objects)}
    assigned = set()
    for _, rl in (assignments or {}).items():
        for rn, _o in rl:
            assigned.add(rn)
    ben = {}
    for name in env.agents:
        role = env.agent_role.get(name, 'active')
        if role in ('dropout', 'never_joined') or name not in idx:
            continue
        obj = idx[name]
        indiv = rot.get_agent_individual_distance(obj)
        db = tb = 0.0
        if role == 'driver' and assignments.get(name):
            robjs = [r for _, r in assignments[name]]
            shared = rot.calculate_shared_route_distance(obj, robjs)
            db = max(0.0, indiv - shared)
            tb = sum(rot.get_agent_individual_distance(r) for _, r in assignments[name])
        elif role == 'rider' and name in assigned:
            db = indiv; tb = indiv
        ben[name] = {'combined_benefit': db + tb * 0.5}
    return ben


def parse_args():
    parser = argparse.ArgumentParser(description='OR-Tools Baseline for Altruistic Ride Sharing')
    parser.add_argument('--num-agents', type=int, default=100)
    parser.add_argument('--altruism-dist', type=str, default='uniform')
    parser.add_argument('--days', type=int, default=10)
    parser.add_argument('--grid-size', type=int, default=15)
    parser.add_argument('--max-capacity', type=int, default=4)
    parser.add_argument('--dataset', type=str, default='100_agents')
    parser.add_argument('--agent-dynamics', type=str, default='fixed', choices=['fixed', 'birth_death'])
    parser.add_argument('--initial-active-agents', type=int, default=None)
    parser.add_argument('--save-dir', type=str, default='./results/or_tools')
    parser.add_argument('--experiment-name', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def setup_experiment(args):
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f"or_tools_{args.num_agents}agents_{args.altruism_dist}_{timestamp}"

    save_dir = os.path.join(args.save_dir, args.experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    dat_dir = os.path.join(save_dir, 'dat')
    os.makedirs(dat_dir, exist_ok=True)

    # Save config
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"OR-Tools Experiment: {args.experiment_name}")
    print(f"Results will be saved to: {save_dir}")
    return save_dir


def initialize_environment(args):
    np.random.seed(args.seed)

    if args.initial_active_agents is None:
        args.initial_active_agents = args.num_agents if args.agent_dynamics == 'fixed' else 40

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

    env.max_capacity = args.max_capacity
    return env


def main():
    args = parse_args()
    save_dir = setup_experiment(args)
    env = initialize_environment(args)

    trainer = ORToolsBaselineTrainer(env)

    daily_summaries = []

    start_time = time.time()
    try:
        for day in range(args.days):
            print(f"Running day {day + 1}/{args.days}...")
            env.current_day = day + 1
            if args.agent_dynamics == 'fixed':
                env.reset_day(day + 1, enable_birth_death=False)
            else:
                env.reset_day(day + 1, enable_birth_death=True)

            result = trainer.train_episode(day, enable_evaluation=False)

            # Use metrics already computed inside train_episode
            metrics = result.get('metrics', {})
            population_stats = result.get('population_stats', {})

            daily_summaries.append({'day': day + 1, 'metrics': metrics, 'population_stats': population_stats})

            # accumulate per-agent combined benefits (for the Gini/fairness metric)
            try:
                acc = getattr(env, '_or_agent_benefits', {})
                for ag, b in _ortools_day_benefits(env, result.get('assignments', {})).items():
                    acc[ag] = acc.get(ag, 0.0) + b['combined_benefit']
                env._or_agent_benefits = acc
            except Exception:
                pass

    except KeyboardInterrupt:
        print('Interrupted by user')

    finally:
        total_time = time.time() - start_time
        print(f"Completed OR-Tools experiment in {total_time:.1f}s")

    # Final save
    trainer.save_results(save_dir, args.days, experiment_config=vars(args))

    # --- Aggregated results_summary.txt (same schema as PSO / MARL evals) so the
    # table collector can read OR-Tools. Without this, OR-Tools produced only
    # latex_data + final_results.json and never appeared in the result tables. ---
    try:
        active = [d for d in daily_summaries if d.get('metrics')]

        def _mean(path):
            vals = []
            for d in active:
                x = d['metrics']
                for k in path:
                    x = x.get(k, {}) if isinstance(x, dict) else {}
                if isinstance(x, (int, float)):
                    vals.append(x)
            return float(np.mean(vals)) if vals else 0.0

        dist = _mean(['distance_comparison', 'savings_percentage'])
        util = _mean(['vehicle_utilization', 'mean'])
        detour = _mean(['detour_factors', 'mean'])
        trip = _mean(['avg_trip_time', 'mean'])

        # Traffic-density reduction from the incrementally accumulated grids
        # (Dijkstra-rasterised, same basis as the MARL evals).
        traffic = None
        try:
            tm = trainer._compute_density_metrics_from_grids()
            traffic = float(tm.get('traffic_reduction_percentage'))
        except Exception:
            pass

        dyn = 'birth_death' if getattr(args, 'agent_dynamics', 'fixed') == 'birth_death' else 'fixed'
        lines = [
            "=== AGGREGATED RESULTS SUMMARY ===",
            "model: or_tools",
            f"grid_size: {getattr(args, 'grid_size', 'NA')}",
            f"num_agents: {getattr(args, 'num_agents', 'NA')}",
            f"altruism_distribution: {getattr(args, 'altruism_dist', 'NA')}",
            f"population_dynamics: {dyn}",
            f"days: {args.days}",
            "",
            f"distance_reduction_percent: {dist:.2f}",
            f"mean_vehicle_utilization: {util:.3f}",
        ]
        if detour:
            lines.append(f"avg_detour_factor: {detour:.4f}")
        if trip:
            lines.append(f"avg_trip_time: {trip:.3f}")
        if traffic is not None:
            lines.append(f"traffic_reduction_percent: {traffic:.2f}")
        # Gini (benefit inequality) over accumulated per-agent combined benefits
        ab = getattr(env, '_or_agent_benefits', {})
        if ab:
            lines.append(f"gini_combined_benefits: {_gini_positive(list(ab.values())):.4f}")
        # Reintegration (birth-death only; env tracks the events)
        if getattr(args, 'agent_dynamics', 'fixed') == 'birth_death':
            try:
                from ars.metrics import calculate_reintegration_metrics
                rs = float(calculate_reintegration_metrics(env).get('final_reintegration_score'))
                lines.append(f"reintegration_score: {rs:.2f}")
            except Exception:
                pass
        with open(os.path.join(save_dir, 'results_summary.txt'), 'w') as f:
            f.write("\n".join(lines) + "\n")
        print(f"Aggregated summary saved to: {os.path.join(save_dir, 'results_summary.txt')}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"Failed to write aggregated results_summary.txt: {e}")

    # Automatically generate plots from latex_data into a sibling `plots` folder
    latex_dir = os.path.join(save_dir, 'latex_data')
    plots_dir = os.path.join(save_dir, 'plots')
    if os.path.isdir(latex_dir):
        os.makedirs(plots_dir, exist_ok=True)
        out_path = os.path.join(plots_dir, 'density_heatmaps.png')
        try:
            subprocess.run([sys.executable, '-m', 'ars.models.or_tools.plot_heatmaps',
                            '--latex_dir', latex_dir, '--out', out_path], check=True)
            print(f"Saved plots to: {out_path}")
        except Exception as e:
            print(f"Failed to generate plots: {e}")


if __name__ == '__main__':
    main()
