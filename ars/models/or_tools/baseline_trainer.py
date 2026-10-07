import numpy as np
from typing import Dict, Any
from .or_tools_baseline import ORToolsBaseline
from ars.models.pso.pso_evaluator import PSOEvaluator
from . import run_or_tools as rot


class ORToolsBaselineTrainer:
    """Trainer wrapper to run OR-Tools baseline for one episode and compute
    metrics and maintain training history similar to the PSO trainer so that
    saved outputs (.dat and JSON summaries) follow the same structure.
    """

    def __init__(self, env, baseline_params: Dict[str, Any] = None):
        self.env = env
        self.baseline_params = baseline_params or {}
        # Training history compatible fields
        self.training_history = {'rewards': [], 'assignments': [], 'fitness_scores': []}
        self.daily_metrics = []
        self.daily_distance_comparison = []
        self.daily_population_stats = []
        self.daily_altruism_stats = []

        # Accumulate density grids incrementally to avoid re-running optimizations
        height = getattr(self.env, 'height', 15)
        width = getattr(self.env, 'width', 15)
        self._density_grid_sharing = np.zeros((height, width))
        self._density_grid_no_sharing = np.zeros((height, width))

    def train_episode(self, episode: int, enable_evaluation: bool = False) -> Dict[str, Any]:
        baseline = ORToolsBaseline(self.env, verbose=self.baseline_params.get('verbose', False))

        if not baseline.drivers and not baseline.riders:
            return {
                'total_reward': 0.0,
                'objective': 0.0,
                'num_assignments': 0,
                'assignments': {},
                'episode': episode
            }

        solution = baseline.optimize()
        assignments = baseline.get_assignments(solution)

        num_assignments = sum(len(v) for v in assignments.values())
        total_obj = solution.get('objective', 0.0)

        result = {
            'total_reward': -total_obj,  # objective is distance (minimized), flip sign to act like reward
            'objective': total_obj,
            'num_assignments': num_assignments,
            'assignments': assignments,
            'episode': episode
        }

        # --- Altruism updates (use env hook if available) ---
        drivers_updated = 0
        riders_updated = 0

        # Use PSO evaluator logic for altruism updates (keeps behavior consistent with PSO baseline)
        try:
            if assignments:
                evaluator = PSOEvaluator(self.env)
                drivers_updated, riders_updated = evaluator.update_altruism_scores_from_assignments(assignments)
            else:
                drivers_updated, riders_updated = 0, 0
        except Exception:
            drivers_updated, riders_updated = 0, 0

        result['drivers_updated'] = int(drivers_updated)
        result['riders_updated'] = int(riders_updated)

        # Compute metrics using shared helpers so OR-Tools matches PSO outputs
        drivers = solution.get('drivers', [])
        riders = solution.get('riders', [])
        try:
            distance_comp = rot.calculate_distance_comparison(assignments, drivers, riders)
            detour = rot.calculate_detour_factors(assignments, drivers)
            avg_trip = rot.calculate_average_trip_time(assignments, drivers, riders)
            util = rot.calculate_vehicle_utilization(assignments, drivers)
            per_agent = rot.calculate_per_agent_distances(assignments, drivers, riders)

            metrics = {
                'distance_comparison': distance_comp,
                'detour_factors': detour,
                'avg_trip_time': avg_trip,
                'vehicle_utilization': util,
                'per_agent_distances': per_agent,
                'altruism_stats': self.env.get_altruism_distribution_stats()
            }
        except Exception:
            metrics = {}
            distance_comp = {}

        result['metrics'] = metrics

        # --- Population dynamics snapshot ---
        try:
            active_agents = len([a for a in self.env.agents if self.env.agent_role.get(a, 'active') not in ['dropout', 'never_joined']])
            num_drivers = len([a for a in self.env.agents if self.env.agent_role.get(a, 'active') == 'driver'])
            num_riders = len([a for a in self.env.agents if self.env.agent_role.get(a, 'active') == 'rider'])
            num_dropouts = len([a for a in self.env.agents if self.env.agent_role.get(a, 'active') == 'dropout'])
            num_never_joined = len([a for a in self.env.agents if self.env.agent_role.get(a, 'active') == 'never_joined'])
            current_day = getattr(self.env, 'current_day', episode + 1)
            daily_births = len(self.env.daily_births.get(current_day, [])) if hasattr(self.env, 'daily_births') else 0
            daily_dropouts_count = len(self.env.daily_dropouts.get(current_day, [])) if hasattr(self.env, 'daily_dropouts') else 0
            result['population_stats'] = {
                'total_agents': self.env.numAgents,
                'active_agents': int(active_agents),
                'num_drivers': int(num_drivers),
                'num_riders': int(num_riders),
                'dropped_out_agents': int(num_dropouts),
                'never_joined_agents': int(num_never_joined),
                'daily_births': int(daily_births),
                'daily_dropouts': int(daily_dropouts_count),
            }
        except Exception:
            result['population_stats'] = None

        # --- Accumulate traffic density grids incrementally ---
        try:
            self._accumulate_density(assignments, drivers, riders)
        except Exception:
            pass

        # Update internal training history so final results include these metrics
        try:
            self._update_training_history(result, metrics=metrics, distance_comp=distance_comp if distance_comp else None, population_stats=result.get('population_stats', {}), altruism_stats=metrics.get('altruism_stats', {}))
        except Exception:
            pass

        return result

    def _accumulate_density(self, assignments, drivers, riders):
        """Accumulate traffic density grids from the current day's state (no re-optimization)."""
        height = getattr(self.env, 'height', 15)
        width = getattr(self.env, 'width', 15)

        agents_dict = {}
        for i, name in enumerate(self.env.agents):
            if i < len(self.env.agent_objects):
                agents_dict[name] = self.env.agent_objects[i]

        assigned_rider_names = set()
        for driver_name, rider_assignments in assignments.items():
            for rider_name, rider_obj in rider_assignments:
                assigned_rider_names.add(rider_name)

        # Sharing scenario
        for agent_name in self.env.agents:
            if self.env.agent_role.get(agent_name, 'active') in ['dropout', 'never_joined']:
                continue
            if agent_name not in agents_dict:
                continue
            agent_obj = agents_dict[agent_name]

            if self.env.agent_role.get(agent_name) == 'driver':
                if agent_name in assignments and assignments[agent_name]:
                    assigned_rider_objs = [r_obj for _, r_obj in assignments[agent_name]]
                    path = self._calculate_shared_route_path(agent_obj, assigned_rider_objs)
                else:
                    path = agent_obj._dijkstra_path(tuple(agent_obj.position), tuple(agent_obj.destination))

                if path:
                    for pos in path:
                        x, y = pos[0], pos[1]
                        if 0 <= x < height and 0 <= y < width:
                            self._density_grid_sharing[x, y] += 1

            elif self.env.agent_role.get(agent_name) == 'rider':
                if agent_name not in assigned_rider_names:
                    path = agent_obj._dijkstra_path(tuple(agent_obj.position), tuple(agent_obj.destination))
                    if path:
                        for pos in path:
                            x, y = pos[0], pos[1]
                            if 0 <= x < height and 0 <= y < width:
                                self._density_grid_sharing[x, y] += 1

        # No-sharing scenario: every active agent travels individually
        for agent_name in self.env.agents:
            if self.env.agent_role.get(agent_name, 'active') in ['dropout', 'never_joined']:
                continue
            if agent_name not in agents_dict:
                continue
            agent_obj = agents_dict[agent_name]
            path = agent_obj._dijkstra_path(tuple(agent_obj.position), tuple(agent_obj.destination))
            if path:
                for pos in path:
                    x, y = pos[0], pos[1]
                    if 0 <= x < height and 0 <= y < width:
                        self._density_grid_no_sharing[x, y] += 1

    def _update_training_history(self, result: Dict[str, Any], metrics: Dict[str, Any] = None,
                                 distance_comp: Dict[str, Any] = None,
                                 population_stats: Dict[str, Any] = None,
                                 altruism_stats: Dict[str, Any] = None):
        """Update internal accumulators used for final result generation."""
        # Reward/assignment/fitness placeholders
        self.training_history['rewards'].append(result.get('total_reward', 0.0))
        self.training_history['assignments'].append(result.get('num_assignments', 0))
        # use objective as a fitness proxy if provided
        fitness = result.get('objective', result.get('objective', 0.0))
        self.training_history['fitness_scores'].append(float(fitness))

        # Store blobs for later aggregation
        self.daily_metrics.append(metrics or result.get('metrics', {}))
        self.daily_distance_comparison.append(distance_comp or {})
        self.daily_population_stats.append(population_stats or result.get('population_stats', {}))
        self.daily_altruism_stats.append(altruism_stats or result.get('altruism_stats', {}))

    def reset_for_new_day(self):
        """Placeholder for API parity with PSO trainer."""
        return

    def get_training_statistics(self) -> Dict[str, Any]:
        import numpy as _np
        def stats(xs):
            arr = _np.array(xs, dtype=float) if xs else _np.array([])
            if arr.size == 0:
                return {'mean': 0.0, 'std': 0.0}
            return {'mean': float(_np.mean(arr)), 'std': float(_np.std(arr))}

        return {
            'training_episodes': len(self.training_history['rewards']),
            'rewards': stats(self.training_history['rewards']),
            'assignments': stats(self.training_history['assignments']),
            'fitness_scores': stats(self.training_history['fitness_scores'])
        }

    def save_results(self, save_dir: str, day: int = None, experiment_config: Dict[str, Any] = None):
        """Save a compact final_results.json and .dat tables similar to PSO outputs."""
        import os, json
        from datetime import datetime
        from . import run_or_tools as rot

        def _safe_mean(xs):
            import numpy as _np
            xs = [x for x in xs if x is not None]
            return float(_np.mean(xs)) if xs else 0.0

        os.makedirs(save_dir, exist_ok=True)
        results_path = os.path.join(save_dir, 'final_results.json')

        # Build daily summaries
        daily_summaries = []
        num_days = len(self.training_history['rewards'])
        for i in range(num_days):
            entry = {
                'day': i + 1,
                'total_reward': self.training_history['rewards'][i],
                'assignments': self.training_history['assignments'][i],
                'fitness': self.training_history['fitness_scores'][i]
            }
            # Flatten distance comparison into top-level entry for .dat generation
            if i < len(self.daily_distance_comparison) and self.daily_distance_comparison[i]:
                entry.update(self.daily_distance_comparison[i])
            elif i < len(self.daily_metrics):
                dc = (self.daily_metrics[i] or {}).get('distance_comparison', {})
                if dc:
                    entry.update(dc)
            if i < len(self.daily_metrics):
                entry['metrics'] = self.daily_metrics[i]
            if i < len(self.daily_population_stats):
                entry['population_stats'] = self.daily_population_stats[i]
            if i < len(self.daily_altruism_stats):
                entry['altruism_stats'] = self.daily_altruism_stats[i]
            daily_summaries.append(entry)

        summary = {
            'total_days': num_days,
            'average_savings_percentage': float(_safe_mean([d.get('savings_percentage', 0.0) for d in daily_summaries])),
            'average_assignment_rate': float(_safe_mean([d.get('assignment_rate', 0.0) for d in daily_summaries])),
            'total_distance_saved': float(sum(d.get('distance_saved', 0.0) for d in daily_summaries))
        }

        payload = {
            'experiment_config': experiment_config or {},
            'training_statistics': self.get_training_statistics(),
            'daily_summaries': daily_summaries,
            'total_days': num_days,
            'baseline_type': 'or_tools',
            'completion_time': datetime.now().isoformat()
        }

        with open(results_path, 'w') as f:
            json.dump(payload, f, indent=2)

        # Generate .dat files using helper save_table
        dat_dir = os.path.join(save_dir, 'dat')
        rot.ensure_dir(dat_dir)

        # Build table rows
        assign_rows = []
        dist_rows = []
        detour_rows = []
        time_rows = []
        util_rows = []
        per_agent_rows = []
        altruism_rows = []
        population_rows = []

        for i, entry in enumerate(daily_summaries, start=1):
            metrics = entry.get('metrics', {}) or {}
            det = metrics.get('detour_factors', {})
            tim = metrics.get('avg_trip_time', {})
            uti = metrics.get('vehicle_utilization', {})
            per_agent = metrics.get('per_agent_distances', {})

            assign_rows.append((i, int(entry.get('assignments', 0)), float(entry.get('total_reward', 0.0)), float(entry.get('fitness', 0.0))))
            dist_rows.append((i,
                              float(entry.get('total_individual_distance', 0.0)),
                              float(entry.get('total_shared_distance', 0.0)),
                              float(entry.get('distance_saved', 0.0)),
                              float(entry.get('savings_percentage', 0.0)),
                              float(entry.get('assigned_riders_count', 0)),
                              float(entry.get('total_riders_count', 0)),
                              float(entry.get('assignment_rate', 0.0))))

            detour_rows.append((i, float(det.get('mean', 0.0)), float(det.get('std', 0.0)), int(det.get('count', 0))))
            time_rows.append((i, float(tim.get('mean', 0.0)), float(tim.get('std', 0.0)), int(tim.get('count', 0))))
            util_rows.append((i, float(uti.get('mean', 0.0)), float(uti.get('std', 0.0)), int(uti.get('count', 0))))
            per_agent_rows.append((i, float(per_agent.get('mean', 0.0)), float(per_agent.get('std', 0.0)), int(per_agent.get('total_agents', 0)), int(per_agent.get('drivers_count', 0)), int(per_agent.get('riders_count', 0))))

            altru = entry.get('altruism_stats', {}) or {}
            n_agents = len(self.env.agents) if hasattr(self.env, 'agents') else 0
            altruism_rows.append((i, float(altru.get('mean', 0.0)), float(altru.get('std', 0.0)), int(altru.get('count', n_agents)), float(altru.get('min', 0.0)), float(altru.get('max', 0.0)), float(altru.get('median', 0.0))))

            pop = entry.get('population_stats', {}) or {}
            population_rows.append((i, int(pop.get('total_agents', 0)), int(pop.get('active_agents', 0)), int(pop.get('num_drivers', 0)), int(pop.get('num_riders', 0)), int(pop.get('dropped_out_agents', 0)), int(pop.get('never_joined_agents', 0)), int(pop.get('daily_births', 0)), int(pop.get('daily_dropouts', 0))))

        rot.save_table(os.path.join(dat_dir, 'assignments_over_time.dat'), 'day assignments reward fitness', assign_rows)
        rot.save_table(os.path.join(dat_dir, 'distance_over_time.dat'), 'day total_individual_distance total_shared_distance distance_saved savings_percentage assigned_riders_count total_riders_count assignment_rate', dist_rows)
        rot.save_table(os.path.join(dat_dir, 'detour_over_time.dat'), 'day mean std count', detour_rows)
        rot.save_table(os.path.join(dat_dir, 'avg_trip_time_over_time.dat'), 'day mean std count', time_rows)
        rot.save_table(os.path.join(dat_dir, 'vehicle_utilization_over_time.dat'), 'day mean std count', util_rows)
        rot.save_table(os.path.join(dat_dir, 'per_agent_distances_over_time.dat'), 'day mean std total_agents drivers_count riders_count', per_agent_rows)
        rot.save_table(os.path.join(dat_dir, 'altruism_evolution_over_time.dat'), 'day altruism_mean altruism_std count min max median # Population altruism statistics', altruism_rows)
        rot.save_table(os.path.join(dat_dir, 'population_dynamics_over_time.dat'), 'day total_agents active_agents drivers riders dropouts never_joined daily_births daily_dropouts', population_rows)

        print(f"OR-Tools final results saved to {results_path}")
        # --- Traffic density & heatmap data (LaTeX-friendly) ---
        try:
            print("Generating traffic density heatmap data from accumulated grids...")
            metrics = self._compute_density_metrics_from_grids()

            latex_dir = os.path.join(save_dir, 'latex_data')
            os.makedirs(latex_dir, exist_ok=True)

            # Traffic density metrics summary
            with open(f"{latex_dir}/traffic_density_metrics.dat", 'w') as f:
                f.write("metric value\n")
                for key, value in metrics.items():
                    f.write(f"{key} {value}\n")

            # Density grids
            avg_density_sharing = self._density_grid_sharing
            avg_density_no_sharing = self._density_grid_no_sharing
            height, width = avg_density_sharing.shape
            density_diff = avg_density_no_sharing - avg_density_sharing

            with open(f"{latex_dir}/density_no_sharing_grid.dat", 'w') as f:
                f.write("x y density\n")
                for r in range(height):
                    for c in range(width):
                        f.write(f"{c} {height-1-r} {avg_density_no_sharing[r,c]:.3f}\n")

            with open(f"{latex_dir}/density_sharing_grid.dat", 'w') as f:
                f.write("x y density\n")
                for r in range(height):
                    for c in range(width):
                        f.write(f"{c} {height-1-r} {avg_density_sharing[r,c]:.3f}\n")

            with open(f"{latex_dir}/density_reduction_grid.dat", 'w') as f:
                f.write("x y density\n")
                for r in range(height):
                    for c in range(width):
                        f.write(f"{c} {height-1-r} {density_diff[r,c]:.3f}\n")

            # Key metrics summary
            with open(f"{latex_dir}/traffic_summary.dat", 'w') as f:
                f.write("metric value unit description\n")
                f.write(f"dense_cell_reduction {metrics['dense_cell_reduction']} cells \"Dense cells eliminated\"\n")
                f.write(f"dense_cell_reduction_pct {metrics['dense_cell_reduction_percentage']:.1f} percent \"Percentage reduction in dense cells\"\n")
                f.write(f"traffic_reduction_pct {metrics['traffic_reduction_percentage']:.1f} percent \"Overall traffic reduction\"\n")
                f.write(f"peak_density_reduction_pct {metrics['peak_density_reduction_percentage']:.1f} percent \"Peak density reduction\"\n")
                f.write(f"hotspot_reduction {metrics['hotspot_reduction']} hotspots \"Congestion hotspots eliminated\"\n")
                f.write(f"threshold {metrics['rho_threshold']:.1f} vehicles \"Density threshold used\"\n")
                f.write(f"max_density_no_sharing {metrics['max_density_no_sharing']:.1f} vehicles \"Peak density without sharing\"\n")
                f.write(f"max_density_sharing {metrics['max_density_sharing']:.1f} vehicles \"Peak density with sharing\"\n")

            print(f"Saved traffic density LaTeX data to {latex_dir}")
        except Exception:
            print("Failed to calculate or save traffic density/heatmap data for OR-Tools")

    def _compute_density_metrics_from_grids(self, rho_threshold=2.0):
        """Compute traffic density metrics from the incrementally accumulated grids."""
        density_sharing = self._density_grid_sharing
        density_no_sharing = self._density_grid_no_sharing
        height, width = density_sharing.shape
        total_cells = height * width

        dense_cells_sharing = int(np.sum(density_sharing > rho_threshold))
        dense_cells_no_sharing = int(np.sum(density_no_sharing > rho_threshold))
        dense_percentage_sharing = (dense_cells_sharing / total_cells) * 100
        dense_percentage_no_sharing = (dense_cells_no_sharing / total_cells) * 100

        total_vehicle_passages_sharing = int(np.sum(density_sharing))
        total_vehicle_passages_no_sharing = int(np.sum(density_no_sharing))
        traffic_reduction_percentage = ((total_vehicle_passages_no_sharing - total_vehicle_passages_sharing) / total_vehicle_passages_no_sharing) * 100 if total_vehicle_passages_no_sharing > 0 else 0.0

        dense_cell_reduction = dense_cells_no_sharing - dense_cells_sharing
        dense_cell_reduction_percentage = (dense_cell_reduction / dense_cells_no_sharing) * 100 if dense_cells_no_sharing > 0 else 0.0

        max_density_sharing = float(np.max(density_sharing))
        max_density_no_sharing = float(np.max(density_no_sharing))
        peak_density_reduction = ((max_density_no_sharing - max_density_sharing) / max_density_no_sharing) * 100 if max_density_no_sharing > 0 else 0.0

        hotspots_sharing = int(np.sum(density_sharing > (rho_threshold * 2)))
        hotspots_no_sharing = int(np.sum(density_no_sharing > (rho_threshold * 2)))

        return {
            'rho_threshold': rho_threshold,
            'dense_cells_sharing': dense_cells_sharing,
            'dense_cells_no_sharing': dense_cells_no_sharing,
            'dense_percentage_sharing': dense_percentage_sharing,
            'dense_percentage_no_sharing': dense_percentage_no_sharing,
            'dense_cell_reduction': dense_cell_reduction,
            'dense_cell_reduction_percentage': dense_cell_reduction_percentage,
            'total_vehicle_passages_sharing': total_vehicle_passages_sharing,
            'total_vehicle_passages_no_sharing': total_vehicle_passages_no_sharing,
            'traffic_reduction_percentage': traffic_reduction_percentage,
            'avg_density_sharing': float(np.mean(density_sharing)),
            'avg_density_no_sharing': float(np.mean(density_no_sharing)),
            'max_density_sharing': max_density_sharing,
            'max_density_no_sharing': max_density_no_sharing,
            'peak_density_reduction_percentage': peak_density_reduction,
            'hotspots_sharing': hotspots_sharing,
            'hotspots_no_sharing': hotspots_no_sharing,
            'hotspot_reduction': int(hotspots_no_sharing - hotspots_sharing),
            'total_cells': int(total_cells),
            'avg_density_per_cell_sharing': float(np.mean(density_sharing)),
            'avg_density_per_cell_no_sharing': float(np.mean(density_no_sharing))
        }

    def _calculate_shared_route_path(self, driver_obj, assigned_rider_objs):
        """Greedy path builder for drivers serving assigned riders (returns list of positions)"""
        if not assigned_rider_objs:
            return driver_obj._dijkstra_path(tuple(driver_obj.position), tuple(driver_obj.destination))

        current_pos = tuple(driver_obj.position)
        full_path = [current_pos]
        remaining_riders = list(assigned_rider_objs)
        picked_up = []

        while remaining_riders or picked_up:
            best_distance = float('inf')
            best_action = None
            best_path = None

            # Consider pickups
            for rider in list(remaining_riders):
                pickup_pos = tuple(rider.position)
                path = driver_obj._dijkstra_path(current_pos, pickup_pos)
                if not path:
                    continue
                dist = driver_obj._calculate_path_length(path)
                if dist < best_distance:
                    best_distance = dist
                    best_action = ('pickup', rider)
                    best_path = path

            # Consider dropoffs
            for rider in list(picked_up):
                dropoff_pos = tuple(rider.destination)
                path = driver_obj._dijkstra_path(current_pos, dropoff_pos)
                if not path:
                    continue
                dist = driver_obj._calculate_path_length(path)
                if dist < best_distance:
                    best_distance = dist
                    best_action = ('dropoff', rider)
                    best_path = path

            if best_action is None or best_path is None:
                break

            # Add path (excluding starting position to avoid duplication)
            full_path.extend(best_path[1:])

            action_type, rider = best_action
            if action_type == 'pickup':
                remaining_riders.remove(rider)
                picked_up.append(rider)
                current_pos = tuple(rider.position)
            else:
                picked_up.remove(rider)
                current_pos = tuple(rider.destination)

        # Finally go to driver's destination
        final_path = driver_obj._dijkstra_path(current_pos, tuple(driver_obj.destination))
        if final_path and len(final_path) > 1:
            full_path.extend(final_path[1:])

        return full_path
