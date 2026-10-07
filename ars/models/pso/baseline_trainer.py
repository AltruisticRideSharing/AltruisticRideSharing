import os
import json
import numpy as np
from typing import Dict, Any
from .pso_baseline import PSOBaseline
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio

class BaselineTrainer:
    """PSO baseline trainer supporting fixed and birth_death population dynamics"""
    
    def __init__(self, env, baseline_type: str = "pso", **baseline_params):
        self.env = env
        self.baseline_type = baseline_type
        self.baseline_params = baseline_params
        self.training_history = {'rewards': [], 'assignments': [], 'fitness_scores': []}
        self.daily_distance_comparison = []  # Track sharing vs no sharing
        self.daily_metrics = []  # Store metrics per episode
        self.daily_altruism_stats = []  # NEW: Track daily altruism statistics

        # Population dynamics tracking (for both fixed and birth_death modes)
        self.daily_population_stats = []  # Track population changes
        
        # Add episode counter to track progress
        self.episode_counter = 0
        
        # Shared Dijkstra distance cache persists across days/PSO instances
        self.shared_distance_cache = {}
        
        # Store last optimization result to avoid redundant re-optimization
        self._last_pso = None
        self._last_assignments = None
    
    def _calculate_detour_factors(self, assignments, drivers):
        """Calculate detour factors for DRIVERS ONLY"""
        detour_factors = []
        drivers_dict = {name: obj for _, name, obj in drivers}
        
        for driver_name, driver_obj in drivers_dict.items():
            # Calculate detour factor = shared_distance / individual_distance
            individual_distance = self._get_agent_individual_distance(driver_obj)
            
            if driver_name in assignments:
                # Driver has assigned riders - calculate shared route
                assigned_riders = [r_obj for _, r_obj in assignments[driver_name]]
                shared_distance = self._calculate_shared_route_distance(driver_obj, assigned_riders)
            else:
                # Driver travels alone - no detour
                shared_distance = individual_distance
            
            if individual_distance > 0:
                detour_factor = shared_distance / individual_distance
                detour_factors.append(detour_factor)
        
        return {
            'mean': float(np.mean(detour_factors)) if detour_factors else 0.0,
            'std': float(np.std(detour_factors)) if detour_factors else 0.0,
            'all': detour_factors,
            'count': len(detour_factors)
        }

    def _calculate_average_trip_time(self, assignments, drivers, riders):
        """Calculate trip times for DRIVERS ONLY (riders don't have trip times in this metric)"""
        driver_trip_times = []
        drivers_dict = {name: obj for _, name, obj in drivers}
        
        # Calculate trip times for drivers only
        for driver_name, driver_obj in drivers_dict.items():
            if driver_name in assignments:
                # Driver with assigned riders - use shared route distance
                assigned_riders = [r_obj for _, r_obj in assignments[driver_name]]
                trip_distance = self._calculate_shared_route_distance(driver_obj, assigned_riders)
            else:
                # Driver traveling alone - use individual distance
                trip_distance = self._get_agent_individual_distance(driver_obj)
            
            # Convert distance to time (assuming 25 units/hour, convert to minutes)
            trip_time = (trip_distance / 25.0) * 60.0  # minutes
            driver_trip_times.append(trip_time)
        
        return {
            'mean': float(np.mean(driver_trip_times)) if driver_trip_times else 0.0,
            'std': float(np.std(driver_trip_times)) if driver_trip_times else 0.0,
            'all': driver_trip_times,
            'count': len(driver_trip_times)
        }

    def _calculate_vehicle_utilization(self, assignments, drivers):
        """Calculate vehicle utilization for DRIVERS ONLY"""
        utilizations = []
        drivers_dict = {name: obj for _, name, obj in drivers}
        
        for driver_name, driver_obj in drivers_dict.items():
            if driver_name in assignments:
                # Driver + assigned riders
                occupancy = 1 + len(assignments[driver_name])
            else:
                # Driver alone
                occupancy = 1
            
            utilizations.append(occupancy)
        
        return {
            'mean': float(np.mean(utilizations)) if utilizations else 0.0,
            'std': float(np.std(utilizations)) if utilizations else 0.0,
            'all': utilizations,
            'max_capacity': 4,  # Assuming max capacity is 4
            'count': len(utilizations)
        }

    def _calculate_per_agent_distances(self, assignments, drivers, riders):
        """Calculate individual distances for ALL agents (but separate drivers and riders)"""
        driver_distances = []
        rider_distances = []
        
        # Get all agents organized by type
        drivers_dict = {name: obj for _, name, obj in drivers}
        riders_dict = {name: obj for _, name, obj in riders}
        assigned_riders = set()
        
        # For drivers - calculate the distance they actually travel
        for driver_name, driver_obj in drivers_dict.items():
            if driver_name in assignments:
                # Driver with assigned riders - they travel the shared route distance
                assigned_rider_objs = [r_obj for _, r_obj in assignments[driver_name]]
                distance = self._calculate_shared_route_distance(driver_obj, assigned_rider_objs)
                
                # Track which riders were assigned
                for rider_name, _ in assignments[driver_name]:
                    assigned_riders.add(rider_name)
            else:
                # Driver traveling alone - individual distance
                distance = self._get_agent_individual_distance(driver_obj)
            
            driver_distances.append(distance)
        
        # For riders - always use their individual distance for comparison purposes
        # (This represents what they would have traveled if going alone)
        for rider_name, rider_obj in riders_dict.items():
            individual_distance = self._get_agent_individual_distance(rider_obj)
            rider_distances.append(individual_distance)
        
        # Combine all distances for overall statistics
        all_distances = driver_distances + rider_distances
        
        return {
            'mean': float(np.mean(all_distances)) if all_distances else 0.0,
            'std': float(np.std(all_distances)) if all_distances else 0.0,
            'all': all_distances,
            'driver_distances': {
                'mean': float(np.mean(driver_distances)) if driver_distances else 0.0,
                'std': float(np.std(driver_distances)) if driver_distances else 0.0,
                'all': driver_distances,
                'count': len(driver_distances)
            },
            'rider_distances': {
                'mean': float(np.mean(rider_distances)) if rider_distances else 0.0,
                'std': float(np.std(rider_distances)) if rider_distances else 0.0,
                'all': rider_distances,
                'count': len(rider_distances)
            },
            'total_agents': len(all_distances),
            'drivers_count': len(drivers_dict),
            'riders_count': len(riders_dict),
            'assigned_riders_count': len(assigned_riders)
        }

    def train_episode(self, episode: int, enable_evaluation: bool = False) -> Dict[str, Any]:
        """Train one episode using PSO and update altruism scores"""
        
        # Increment episode counter
        self.episode_counter += 1
        
        # NEW: Capture altruism stats BEFORE any changes
        altruism_stats_before = self._calculate_daily_altruism_stats()
        
        # Check if we have any active agents
        active_agents = [a for a in self.env.agents if self.env.agent_role.get(a, 'active') not in ['dropout', 'never_joined']]
        
        if not active_agents:
            # No active agents - return empty result but still track altruism
            self.daily_altruism_stats.append(altruism_stats_before)
            return {
                'total_reward': 0.0,
                'pso_fitness': 0.0,
                'num_assignments': 0,
                'assignments': {},
                'drivers_updated': 0,
                'riders_updated': 0,
                'metrics': {
                    'distance_comparison': {'distance_saved': 0.0, 'savings_percentage': 0.0, 'assignment_rate': 0.0},
                    'detour_factors': {'mean': 0.0, 'std': 0.0, 'count': 0},
                    'avg_trip_time': {'mean': 0.0, 'std': 0.0, 'count': 0},
                    'vehicle_utilization': {'mean': 0.0, 'std': 0.0, 'count': 0},
                    'per_agent_distances': {'mean': 0.0, 'std': 0.0, 'total_agents': 0, 'drivers_count': 0, 'riders_count': 0},
                    'altruism_stats': self.env.get_altruism_distribution_stats(),
                    'daily_altruism_stats': altruism_stats_before  # NEW
                },
                'episode': self.episode_counter,
                'population_stats': self._get_population_stats()
            }
        
        # Initialize fresh PSO for this episode/day (with shared distance cache)
        pso = PSOBaseline(env=self.env, shared_distance_cache=self.shared_distance_cache,
                          **self.baseline_params)
        
        # Check if PSO has any agents to work with
        if not pso.drivers and not pso.riders:
            # No drivers or riders available but still track altruism
            self.daily_altruism_stats.append(altruism_stats_before)
            return {
                'total_reward': 0.0,
                'pso_fitness': 0.0,
                'num_assignments': 0,
                'assignments': {},
                'drivers_updated': 0,
                'riders_updated': 0,
                'metrics': {
                    'distance_comparison': {'distance_saved': 0.0, 'savings_percentage': 0.0, 'assignment_rate': 0.0},
                    'detour_factors': {'mean': 0.0, 'std': 0.0, 'count': 0},
                    'avg_trip_time': {'mean': 0.0, 'std': 0.0, 'count': 0},
                    'vehicle_utilization': {'mean': 0.0, 'std': 0.0, 'count': 0},
                    'per_agent_distances': {'mean': 0.0, 'std': 0.0, 'total_agents': 0, 'drivers_count': 0, 'riders_count': 0},
                    'altruism_stats': self.env.get_altruism_distribution_stats(),
                    'daily_altruism_stats': altruism_stats_before  # NEW
                },
                'episode': self.episode_counter,
                'population_stats': self._get_population_stats()
            }
        
        # Run optimization
        solution = pso.optimize()
        
        # Get assignments
        assignments = pso.get_assignments(solution)

        constraint_stats = pso.evaluator.get_constraint_violation_stats()
        
        if self.baseline_params.get('verbose', False):
            print(f"Episode {episode}: Found {len(assignments)} driver assignments")
            total_riders = sum(len(riders) for riders in assignments.values())
            print(f"Episode {episode}: Total rider assignments: {total_riders}")
            print(f"Episode {episode}: Constraint violations: {constraint_stats['violation_rate']:.1f}%")
        
        # Calculate distance comparison (sharing vs no sharing) - FRESH calculation
        distance_comparison = self._calculate_distance_comparison(assignments, pso.drivers, pso.riders)
        
        # Update altruism scores based on assignments
        drivers_updated = 0
        riders_updated = 0
        if assignments:
            drivers_updated, riders_updated = pso.evaluator.update_altruism_scores_from_assignments(assignments)
            if self.baseline_params.get('verbose', False):
                print(f"Episode {episode}: Updated altruism for {drivers_updated} drivers, {riders_updated} riders")
        
        # NEW: Capture altruism stats AFTER updates (to see the effect of assignments)
        altruism_stats_after = self._calculate_daily_altruism_stats()
        self.daily_altruism_stats.append(altruism_stats_after)
        
        # Calculate metrics - FRESH calculations (drivers only for detour, time, utilization)
        total_reward = solution['best_fitness']
        num_assignments = len([r for driver_riders in assignments.values() for r in driver_riders])
        
        # Get altruism statistics after update
        altruism_stats = self.env.get_altruism_distribution_stats()

        # Calculate driver-specific metrics
        detour_factors = self._calculate_detour_factors(assignments, pso.drivers)
        avg_trip_time = self._calculate_average_trip_time(assignments, pso.drivers, pso.riders)
        vehicle_utilization = self._calculate_vehicle_utilization(assignments, pso.drivers)
        per_agent_distances = self._calculate_per_agent_distances(assignments, pso.drivers, pso.riders)

        metrics = {
            'detour_factors': detour_factors,
            'avg_trip_time': avg_trip_time,
            'vehicle_utilization': vehicle_utilization,
            'per_agent_distances': per_agent_distances,
            'distance_comparison': distance_comparison,
            'altruism_stats': altruism_stats,
            'daily_altruism_stats': altruism_stats_after,  # NEW
            'constraint_violations': constraint_stats,
        }

        result = {
            'total_reward': total_reward,
            'pso_fitness': solution['best_fitness'],
            'num_assignments': num_assignments,
            'assignments': assignments,
            'drivers_updated': drivers_updated,
            'riders_updated': riders_updated,
            'metrics': metrics,
            'episode': self.episode_counter,
            'population_stats': self._get_population_stats()
        }

        # Cache last optimization result for reuse in save_results
        self._last_pso = pso
        self._last_assignments = assignments

        return result

    def _calculate_distance_comparison(self, assignments, drivers, riders):
        """Calculate total distance with sharing vs without sharing - FRESH calculation each time"""
        total_individual_distance = 0.0
        total_shared_distance = 0.0
        
        # Get all active agents (drivers and riders)
        all_drivers = {name: obj for _, name, obj in drivers}
        all_riders = {name: obj for _, name, obj in riders}
        
        # Calculate individual distances (no sharing baseline) - RECALCULATE each time
        for _, driver_name, driver_obj in drivers:
            driver_individual = self._get_agent_individual_distance(driver_obj)
            total_individual_distance += driver_individual
        
        for _, rider_name, rider_obj in riders:
            rider_individual = self._get_agent_individual_distance(rider_obj)
            total_individual_distance += rider_individual
        
        # Calculate shared distances - RECALCULATE each time
        assigned_riders = set()
        
        # For drivers with assignments
        for driver_name, rider_assignments in assignments.items():
            driver_obj = all_drivers[driver_name]
            assigned_rider_objs = [rider_obj for _, rider_obj in rider_assignments]
            
            # Calculate optimal shared route for this driver
            shared_route_distance = self._calculate_shared_route_distance(driver_obj, assigned_rider_objs)
            total_shared_distance += shared_route_distance
            
            # Track assigned riders
            for rider_name, _ in rider_assignments:
                assigned_riders.add(rider_name)
        
        # For drivers without assignments (travel alone)
        for driver_name, driver_obj in all_drivers.items():
            if driver_name not in assignments:
                driver_alone_distance = self._get_agent_individual_distance(driver_obj)
                total_shared_distance += driver_alone_distance
        
        # For unassigned riders (travel alone)
        for rider_name, rider_obj in all_riders.items():
            if rider_name not in assigned_riders:
                rider_alone_distance = self._get_agent_individual_distance(rider_obj)
                total_shared_distance += rider_alone_distance
        
        # Calculate savings
        distance_saved = total_individual_distance - total_shared_distance
        savings_percentage = (distance_saved / max(total_individual_distance, 1.0)) * 100
        
        return {
            'total_individual_distance': float(total_individual_distance),
            'total_shared_distance': float(total_shared_distance),
            'distance_saved': float(distance_saved),
            'savings_percentage': float(savings_percentage),
            'assigned_riders_count': len(assigned_riders),
            'total_riders_count': len(all_riders),
            'assignment_rate': float(len(assigned_riders) / max(len(all_riders), 1) * 100)
        }
    
    def _get_agent_individual_distance(self, agent_obj):
        """Get distance for agent traveling alone"""
        start = tuple(agent_obj.position)
        dest = tuple(agent_obj.destination)
        path = agent_obj._dijkstra_path(start, dest)
        return agent_obj._calculate_path_length(path)
    
    def _calculate_shared_route_distance(self, driver_obj, assigned_riders):
        """Calculate optimal shared route distance for driver with assigned riders"""
        if not assigned_riders:
            return self._get_agent_individual_distance(driver_obj)
        
        # Use greedy approximation for speed (similar to PSO evaluator)
        current_pos = tuple(driver_obj.position)
        total_distance = 0.0
        remaining_riders = assigned_riders.copy()
        picked_up_riders = []
        
        while remaining_riders or picked_up_riders:
            best_distance = float('inf')
            best_action = None
            
            # Consider pickups
            for rider in remaining_riders:
                pickup_pos = tuple(rider.position)
                path = driver_obj._dijkstra_path(current_pos, pickup_pos)
                distance = driver_obj._calculate_path_length(path)
                if distance < best_distance:
                    best_distance = distance
                    best_action = ('pickup', rider)
            
            # Consider dropoffs
            for rider in picked_up_riders:
                dropoff_pos = tuple(rider.destination)
                path = driver_obj._dijkstra_path(current_pos, dropoff_pos)
                distance = driver_obj._calculate_path_length(path)
                if distance < best_distance:
                    best_distance = distance
                    best_action = ('dropoff', rider)
            
            # Execute best action
            if best_action:
                action_type, rider = best_action
                if action_type == 'pickup':
                    next_pos = tuple(rider.position)
                    remaining_riders.remove(rider)
                    picked_up_riders.append(rider)
                else:
                    next_pos = tuple(rider.destination)
                    picked_up_riders.remove(rider)
                
                path = driver_obj._dijkstra_path(current_pos, next_pos)
                total_distance += driver_obj._calculate_path_length(path)
                current_pos = next_pos
            else:
                break
        
        # Go to driver destination
        driver_dest = tuple(driver_obj.destination)
        final_path = driver_obj._dijkstra_path(current_pos, driver_dest)
        total_distance += driver_obj._calculate_path_length(final_path)
        
        return total_distance
    
    def _calculate_daily_altruism_stats(self):
        """Calculate daily altruism statistics for the whole population"""
        active_agents = [a for a in self.env.agents if self.env.agent_role.get(a, 'active') not in ['dropout', 'never_joined']]
        
        if not active_agents:
            print("DEBUG: No active agents found for altruism stats")
            return {
                'mean': 0.0,
                'std': 0.0,
                'count': 0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            }
        
        # Get altruism scores for all active agents
        altruism_scores = []
        debug_info = []
        
        for agent_name in active_agents:
            score = None
            source = None
            
            # Try multiple sources of altruism data
            if hasattr(self.env, 'altruism_points_day') and self.env.altruism_points_day and agent_name in self.env.altruism_points_day:
                score = self.env.altruism_points_day[agent_name]
                source = 'altruism_points_day'
            elif hasattr(self.env, 'agent_altruism') and self.env.agent_altruism and agent_name in self.env.agent_altruism:
                score = self.env.agent_altruism[agent_name]
                source = 'agent_altruism'
            elif hasattr(self.env, 'agents_dict') and agent_name in self.env.agents_dict:
                agent_obj = self.env.agents_dict[agent_name]
                if hasattr(agent_obj, 'altruism'):
                    score = agent_obj.altruism
                    source = 'agent_obj.altruism'
            else:
                # Try to find agent in agents list
                for agent_obj in self.env.agents_dict.values():
                    if hasattr(agent_obj, 'name') and agent_obj.name == agent_name:
                        if hasattr(agent_obj, 'altruism'):
                            score = agent_obj.altruism
                            source = 'agent_obj.altruism_by_name'
                        break
            
            if score is not None:
                altruism_scores.append(float(score))
                debug_info.append(f"{agent_name}: {score} ({source})")
            else:
                debug_info.append(f"{agent_name}: NOT FOUND")
        
        # Debug output (remove this after fixing)
        if self.baseline_params.get('verbose', False):
            print(f"DEBUG: Found {len(altruism_scores)} altruism scores out of {len(active_agents)} active agents")
            if len(debug_info) <= 5:  # Only show details for small numbers
                for info in debug_info:
                    print(f"  {info}")
            
            # Show what attributes the env has
            print(f"DEBUG: env attributes: {[attr for attr in dir(self.env) if 'altruism' in attr.lower()]}")
            if hasattr(self.env, 'altruism_points_day'):
                print(f"DEBUG: altruism_points_day keys: {list(self.env.altruism_points_day.keys())[:5] if self.env.altruism_points_day else 'None'}")
        
        if not altruism_scores:
            print("WARNING: No altruism scores found! Returning zero stats.")
            return {
                'mean': 0.0,
                'std': 0.0,
                'count': 0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            }
        
        altruism_array = np.array(altruism_scores, dtype=float)
        
        stats = {
            'mean': float(np.mean(altruism_array)),
            'std': float(np.std(altruism_array)),
            'count': len(altruism_scores),
            'min': float(np.min(altruism_array)),
            'max': float(np.max(altruism_array)),
            'median': float(np.median(altruism_array))
        }
        
        if self.baseline_params.get('verbose', False):
            print(f"DEBUG: Altruism stats calculated: mean={stats['mean']:.3f}, std={stats['std']:.3f}, count={stats['count']}")
        
        return stats
    
    def _calculate_traffic_density_metrics(self, rho_threshold=2.0):
        """Calculate traffic density metrics for PSO baseline using same method as simulation"""
        print(f"Calculating traffic density metrics with threshold ρ = {rho_threshold}")
        
        # Initialize density grids
        density_grid_sharing = np.zeros((self.env.height, self.env.width))
        density_grid_no_sharing = np.zeros((self.env.height, self.env.width))
        
        # Get active agents only
        active_agents = [a for a in self.env.agents if self.env.agent_role.get(a, 'active') not in ['dropout', 'never_joined']]
        
        if not active_agents:
            # No active agents - return zero metrics
            return self._create_empty_traffic_metrics(rho_threshold)
        
        # Create agents dictionary - check if it exists, otherwise create it
        if hasattr(self.env, 'agents_dict'):
            agents_dict = self.env.agents_dict
        else:
            # Create agents_dict from env.agents and env.agent_objects
            agents_dict = {}
            for i, agent_name in enumerate(self.env.agents):
                if i < len(self.env.agent_objects):
                    agents_dict[agent_name] = self.env.agent_objects[i]
            
            # Store it for future use
            self.env.agents_dict = agents_dict
        
        # Create fresh PSO instance for traffic density calculation
        pso = PSOBaseline(env=self.env, **self.baseline_params)
        
        if not pso.drivers and not pso.riders:
            # No drivers or riders available
            return self._create_empty_traffic_metrics(rho_threshold)
        
        # Run PSO optimization to get assignments
        solution = pso.optimize()
        assignments = pso.get_assignments(solution)
        assigned_riders = set()
        
        # Track assigned riders to avoid double counting
        for driver_name, assigned_list in assignments.items():
            for rider_name, rider_obj in assigned_list:
                assigned_riders.add(rider_name)
        
        # Count vehicles in sharing scenario
        for agent_name in active_agents:
            if agent_name not in agents_dict:
                print(f"Warning: Agent {agent_name} not found in agents_dict")
                continue
                
            agent_obj = agents_dict[agent_name]
            
            should_count = False
            if self.env.agent_role.get(agent_name) == "driver":
                should_count = True
            elif self.env.agent_role.get(agent_name) == "rider" and agent_name not in assigned_riders:
                should_count = True  # Unassigned riders travel individually
                
            if should_count:
                if agent_name in assignments and assignments[agent_name]:
                    # Driver with assigned riders - calculate shared route
                    assigned_rider_objs = [r_obj for _, r_obj in assignments[agent_name]]
                    shared_path = self._calculate_shared_route_path(agent_obj, assigned_rider_objs)
                else:
                    # Individual travel
                    shared_path = agent_obj._dijkstra_path(
                        tuple(agent_obj.position), tuple(agent_obj.destination)
                    )
                
                # Count path positions
                if shared_path:
                    for pos in shared_path:
                        x, y = pos[0], pos[1]
                        if 0 <= x < self.env.height and 0 <= y < self.env.width:
                            density_grid_sharing[x, y] += 1
        
        # Calculate no-sharing scenario density (all agents travel individually)
        for agent_name in active_agents:
            if agent_name not in agents_dict:
                continue
                
            agent_obj = agents_dict[agent_name]
            individual_path = agent_obj._dijkstra_path(
                tuple(agent_obj.position), tuple(agent_obj.destination)
            )
            
            if individual_path:
                for pos in individual_path:
                    x, y = pos[0], pos[1]
                    if 0 <= x < self.env.height and 0 <= y < self.env.width:
                        density_grid_no_sharing[x, y] += 1
        
        # For single-day PSO, density equals the raw counts
        avg_density_sharing = density_grid_sharing
        avg_density_no_sharing = density_grid_no_sharing
        
        # Calculate DENSE metrics
        dense_cells_sharing = np.sum(avg_density_sharing > rho_threshold)
        dense_cells_no_sharing = np.sum(avg_density_no_sharing > rho_threshold)
        
        # Additional metrics
        total_cells = self.env.height * self.env.width
        dense_percentage_sharing = (dense_cells_sharing / total_cells) * 100
        dense_percentage_no_sharing = (dense_cells_no_sharing / total_cells) * 100
        
        # Traffic reduction metrics
        total_vehicle_passages_sharing = np.sum(density_grid_sharing)
        total_vehicle_passages_no_sharing = np.sum(density_grid_no_sharing)
        traffic_reduction_percentage = ((total_vehicle_passages_no_sharing - total_vehicle_passages_sharing) / total_vehicle_passages_no_sharing) * 100 if total_vehicle_passages_no_sharing > 0 else 0
        
        # Dense cell reduction
        dense_cell_reduction = dense_cells_no_sharing - dense_cells_sharing
        dense_cell_reduction_percentage = (dense_cell_reduction / dense_cells_no_sharing) * 100 if dense_cells_no_sharing > 0 else 0
        
        # Peak density analysis
        max_density_sharing = np.max(avg_density_sharing)
        max_density_no_sharing = np.max(avg_density_no_sharing)
        peak_density_reduction = ((max_density_no_sharing - max_density_sharing) / max_density_no_sharing) * 100 if max_density_no_sharing > 0 else 0
        
        # Congestion hotspots
        high_congestion_threshold = rho_threshold * 2
        hotspots_sharing = np.sum(avg_density_sharing > high_congestion_threshold)
        hotspots_no_sharing = np.sum(avg_density_no_sharing > high_congestion_threshold)
        
        metrics = {
            'rho_threshold': rho_threshold,
            'dense_cells_sharing': int(dense_cells_sharing),
            'dense_cells_no_sharing': int(dense_cells_no_sharing),
            'dense_percentage_sharing': dense_percentage_sharing,
            'dense_percentage_no_sharing': dense_percentage_no_sharing,
            'dense_cell_reduction': int(dense_cell_reduction),
            'dense_cell_reduction_percentage': dense_cell_reduction_percentage,
            'total_vehicle_passages_sharing': int(total_vehicle_passages_sharing),
            'total_vehicle_passages_no_sharing': int(total_vehicle_passages_no_sharing),
            'traffic_reduction_percentage': traffic_reduction_percentage,
            'avg_density_sharing': float(np.mean(avg_density_sharing)),
            'avg_density_no_sharing': float(np.mean(avg_density_no_sharing)),
            'max_density_sharing': float(max_density_sharing),
            'max_density_no_sharing': float(max_density_no_sharing),
            'peak_density_reduction_percentage': peak_density_reduction,
            'hotspots_sharing': int(hotspots_sharing),
            'hotspots_no_sharing': int(hotspots_no_sharing),
            'hotspot_reduction': int(hotspots_no_sharing - hotspots_sharing),
            'total_cells': int(total_cells),
            'avg_density_per_cell_sharing': float(np.mean(avg_density_sharing)),
            'avg_density_per_cell_no_sharing': float(np.mean(avg_density_no_sharing))
        }
        
        print(f"\n=== PSO TRAFFIC DENSITY ANALYSIS (ρ = {rho_threshold}) ===")
        print(f"Dense cells - No sharing: {dense_cells_no_sharing} ({dense_percentage_no_sharing:.1f}%)")
        print(f"Dense cells - Sharing: {dense_cells_sharing} ({dense_percentage_sharing:.1f}%)")
        print(f"Dense cell reduction: {dense_cell_reduction} cells ({dense_cell_reduction_percentage:.1f}%)")
        print(f"Total traffic reduction: {traffic_reduction_percentage:.1f}%")
        print(f"Peak density reduction: {peak_density_reduction:.1f}%")
        print(f"Congestion hotspot reduction: {hotspots_no_sharing - hotspots_sharing} hotspots")
        
        return metrics, avg_density_sharing, avg_density_no_sharing

    def _create_empty_traffic_metrics(self, rho_threshold):
        """Create empty traffic metrics when no agents are available"""
        empty_grid = np.zeros((self.env.height, self.env.width))
        
        metrics = {
            'rho_threshold': rho_threshold,
            'dense_cells_sharing': 0,
            'dense_cells_no_sharing': 0,
            'dense_percentage_sharing': 0.0,
            'dense_percentage_no_sharing': 0.0,
            'dense_cell_reduction': 0,
            'dense_cell_reduction_percentage': 0.0,
            'total_vehicle_passages_sharing': 0,
            'total_vehicle_passages_no_sharing': 0,
            'traffic_reduction_percentage': 0.0,
            'avg_density_sharing': 0.0,
            'avg_density_no_sharing': 0.0,
            'max_density_sharing': 0.0,
            'max_density_no_sharing': 0.0,
            'peak_density_reduction_percentage': 0.0,
            'hotspots_sharing': 0,
            'hotspots_no_sharing': 0,
            'hotspot_reduction': 0,
            'total_cells': int(self.env.height * self.env.width),
            'avg_density_per_cell_sharing': 0.0,
            'avg_density_per_cell_no_sharing': 0.0
        }
        
        return metrics, empty_grid, empty_grid

    def _calculate_shared_route_path(self, driver_obj, assigned_rider_objs):
        """Calculate the actual path a driver takes when serving assigned riders"""
        if not assigned_rider_objs:
            return driver_obj._dijkstra_path(tuple(driver_obj.position), tuple(driver_obj.destination))
        
        current_pos = tuple(driver_obj.position)
        full_path = [current_pos]
        remaining_riders = list(assigned_rider_objs)
        picked_up = []
        
        # Greedy pickup/dropoff sequence (same logic as distance calculation)
        while remaining_riders or picked_up:
            best_distance = float('inf')
            best_action = None
            best_path = None
            
            # Consider pickups
            for rider in list(remaining_riders):
                pickup_pos = tuple(rider.position)
                path = driver_obj._dijkstra_path(current_pos, pickup_pos)
                dist = driver_obj._calculate_path_length(path)
                if dist < best_distance:
                    best_distance = dist
                    best_action = ('pickup', rider)
                    best_path = path
            
            # Consider dropoffs
            for rider in list(picked_up):
                dropoff_pos = tuple(rider.destination)
                path = driver_obj._dijkstra_path(current_pos, dropoff_pos)
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

    def _save_traffic_density_data_for_latex(self, metrics, avg_density_sharing, avg_density_no_sharing, save_dir):
        """Save PSO traffic density data in LaTeX-compatible formats"""
        os.makedirs(save_dir, exist_ok=True)
        latex_dir = os.path.join(save_dir, "latex_data")
        os.makedirs(latex_dir, exist_ok=True)
        
        # Traffic density metrics summary
        with open(f"{latex_dir}/traffic_density_metrics.dat", 'w') as f:
            f.write("metric value\n")
            for key, value in metrics.items():
                f.write(f"{key} {value}\n")
        
        # Density grids for heatmaps
        height, width = avg_density_sharing.shape
        density_diff = avg_density_no_sharing - avg_density_sharing
        
        # No-sharing density data
        with open(f"{latex_dir}/density_no_sharing_grid.dat", 'w') as f:
            f.write("x y density\n")
            for i in range(height):
                for j in range(width):
                    f.write(f"{j} {height-1-i} {avg_density_no_sharing[i,j]:.3f}\n")
        
        # Sharing density data
        with open(f"{latex_dir}/density_sharing_grid.dat", 'w') as f:
            f.write("x y density\n")
            for i in range(height):
                for j in range(width):
                    f.write(f"{j} {height-1-i} {avg_density_sharing[i,j]:.3f}\n")
        
        # Traffic reduction data
        with open(f"{latex_dir}/density_reduction_grid.dat", 'w') as f:
            f.write("x y density\n")
            for i in range(height):
                for j in range(width):
                    f.write(f"{j} {height-1-i} {density_diff[i,j]:.3f}\n")
        
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
        
        print("PSO traffic density data saved for LaTeX integration")

    def _create_traffic_density_visualization(self, avg_density_sharing, avg_density_no_sharing, metrics, save_dir):
        """Create traffic density visualization for PSO results (same as simulation)"""
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        
        # Set up the figure with proper sizing for paper inclusion
        plt.rcParams.update({
            'font.size': 14,
            'font.family': 'serif',
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'figure.dpi': 300
        })
        
        # Create custom colormaps
        density_colors = ['white', 'red']
        density_cmap = LinearSegmentedColormap.from_list('white_to_red', density_colors, N=256)
        
        reduction_colors = ['red', 'white', 'green']
        reduction_cmap = LinearSegmentedColormap.from_list('red_white_green', reduction_colors, N=256)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
        
        # Find unified color scale for first two plots
        vmin = 0
        vmax = max(avg_density_sharing.max(), avg_density_no_sharing.max())
        
        # 1. No-sharing scenario heatmap
        im1 = ax1.imshow(avg_density_no_sharing, cmap=density_cmap, vmin=vmin, vmax=vmax, 
                        interpolation='nearest', aspect='equal')
        ax1.set_title('(a) No Ride-Sharing', fontweight='bold', pad=15)
        ax1.set_xlabel('Grid X', fontweight='bold')
        ax1.set_ylabel('Grid Y', fontweight='bold')
        
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02)
        cbar1.set_label('Vehicle Density', fontweight='bold', rotation=270, labelpad=20)
        cbar1.ax.tick_params(labelsize=11)
        
        ax1.set_xticks(np.arange(0, avg_density_no_sharing.shape[1], 3))
        ax1.set_yticks(np.arange(0, avg_density_no_sharing.shape[0], 3))
        
        # 2. Sharing scenario heatmap
        im2 = ax2.imshow(avg_density_sharing, cmap=density_cmap, vmin=vmin, vmax=vmax, 
                        interpolation='nearest', aspect='equal')
        ax2.set_title('(b) With PSO Ride-Sharing', fontweight='bold', pad=15)
        ax2.set_xlabel('Grid X', fontweight='bold')
        ax2.set_ylabel('Grid Y', fontweight='bold')
        
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02)
        cbar2.set_label('Vehicle Density', fontweight='bold', rotation=270, labelpad=20)
        cbar2.ax.tick_params(labelsize=11)
        
        ax2.set_xticks(np.arange(0, avg_density_sharing.shape[1], 3))
        ax2.set_yticks(np.arange(0, avg_density_sharing.shape[0], 3))
        
        # 3. Traffic reduction heatmap
        density_diff = avg_density_no_sharing - avg_density_sharing
        vmax_diff = max(abs(density_diff.min()), abs(density_diff.max()))
        vmin_diff = -vmax_diff
        
        im3 = ax3.imshow(density_diff, cmap=reduction_cmap, vmin=vmin_diff, vmax=vmax_diff, 
                        interpolation='nearest', aspect='equal')
        ax3.set_title('(c) Traffic Density Reduction', fontweight='bold', pad=15)
        ax3.set_xlabel('Grid X', fontweight='bold')
        ax3.set_ylabel('Grid Y', fontweight='bold')
        
        cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8, pad=0.02)
        cbar3.set_label('Density Change', fontweight='bold', rotation=270, labelpad=20)
        cbar3.ax.tick_params(labelsize=11)
        
        ax3.set_xticks(np.arange(0, density_diff.shape[1], 3))
        ax3.set_yticks(np.arange(0, density_diff.shape[0], 3))
        
        # Adjust layout and add bottom text
        plt.tight_layout(pad=1.5)
        plt.subplots_adjust(bottom=0.15)
        
        traffic_reduction_text = f"Overall Traffic Reduction: {metrics['traffic_reduction_percentage']:.1f}%"
        fig.text(0.5, 0.02, traffic_reduction_text, ha='center', va='bottom', fontsize=16, 
                style='italic', color='darkblue', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.7))
        
        # Save in multiple formats
        heatmap_dir = os.path.join(save_dir, "heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)
        
        plt.savefig(os.path.join(heatmap_dir, "traffic_density_pso.png"), 
                    dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.savefig(os.path.join(heatmap_dir, "traffic_density_pso.pdf"), 
                    dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.savefig(os.path.join(heatmap_dir, "traffic_density_pso.eps"), 
                    dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        
        plt.close()
        plt.rcParams.update(plt.rcParamsDefault)
        
        print(f"PSO traffic density visualizations saved to {heatmap_dir}/")
        return density_diff
    
    def calculate_agent_benefits(self):
        """Calculate comprehensive agent benefits for PSO (similar to MADDPG)"""
        agent_benefits = {}
        
        # Get active agents
        active_agents = [a for a in self.env.agents if self.env.agent_role.get(a, 'active') not in ['dropout', 'never_joined']]
        
        if not active_agents:
            return agent_benefits
        
        # Reuse cached optimization result if available
        if self._last_pso is not None and self._last_assignments is not None:
            pso = self._last_pso
            assignments = self._last_assignments
        else:
            pso = PSOBaseline(env=self.env, shared_distance_cache=self.shared_distance_cache,
                              **self.baseline_params)
            if not pso.drivers and not pso.riders:
                return agent_benefits
            solution = pso.optimize()
            assignments = pso.get_assignments(solution)
        
        if not pso.drivers and not pso.riders:
            return agent_benefits
        
        # Calculate benefits for each agent
        for agent_name in active_agents:
            distance_benefit = 0.0
            traffic_benefit = 0.0
            combined_benefit = 0.0
            
            # Find agent object
            agent_obj = None
            for i, name in enumerate(self.env.agents):
                if name == agent_name:
                    agent_obj = self.env.agent_objects[i]
                    break
            
            if agent_obj is None:
                continue
            
            # Calculate distance benefit
            individual_distance = self._get_agent_individual_distance(agent_obj)
            
            if self.env.agent_role.get(agent_name) == 'driver':
                # Driver distance benefit
                if agent_name in assignments:
                    # Driver with assigned riders
                    assigned_rider_objs = [r_obj for _, r_obj in assignments[agent_name]]
                    shared_distance = self._calculate_shared_route_distance(agent_obj, assigned_rider_objs)
                    distance_benefit = max(0, individual_distance - shared_distance)  # Savings
                    
                    # Traffic benefit = total distance of assigned riders
                    traffic_benefit = sum(self._get_agent_individual_distance(r_obj) for _, r_obj in assignments[agent_name])
                else:
                    # Driver with no assignments
                    distance_benefit = 0.0
                    traffic_benefit = 0.0
            
            elif self.env.agent_role.get(agent_name) == 'rider':
                # Rider benefits
                is_assigned = False
                for driver_name, rider_assignments in assignments.items():
                    for rider_name, _ in rider_assignments:
                        if rider_name == agent_name:
                            is_assigned = True
                            break
                    if is_assigned:
                        break
                
                if is_assigned:
                    # Assigned rider gets their full distance as benefit
                    distance_benefit = individual_distance
                    traffic_benefit = individual_distance  # They don't drive, so this is their contribution
                else:
                    # Unassigned rider gets no benefit (travels alone)
                    distance_benefit = 0.0
                    traffic_benefit = 0.0
            
            # Combined benefit
            combined_benefit = distance_benefit + traffic_benefit * 0.5  # Weight traffic benefit
            
            agent_benefits[agent_name] = {
                'distance_benefit': distance_benefit,
                'traffic_benefit': traffic_benefit,
                'combined_benefit': combined_benefit,
                'role': self.env.agent_role.get(agent_name, 'unknown'),
                'individual_distance': individual_distance
            }
        
        return agent_benefits

    def analyze_multidimensional_benefit_distribution(self, agent_benefits, save_dir):
        """Analyze PSO benefit distribution with 3D Lorenz surface and Gini coefficients"""
        if not agent_benefits:
            print("No agent benefits to analyze")
            return {}
        
        distance_benefits = [data['distance_benefit'] for data in agent_benefits.values()]
        traffic_benefits = [data['traffic_benefit'] for data in agent_benefits.values()]
        
        # Separate positive benefits for Gini calculation
        positive_distance_benefits = [b for b in distance_benefits if b > 0]
        positive_traffic_benefits = [b for b in traffic_benefits if b > 0]
        
        def calculate_gini(values):
            """Calculate Gini coefficient (0 = perfect equality, 1 = perfect inequality)"""
            if not values or len(values) == 0:
                return 0
            values = sorted(values)
            n = len(values)
            cumsum = np.cumsum(values)
            return (n + 1 - 2 * sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
        
        # Calculate Gini coefficients
        gini_distance = calculate_gini(positive_distance_benefits) if positive_distance_benefits else 0
        gini_traffic = calculate_gini(positive_traffic_benefits) if positive_traffic_benefits else 0
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Create 3D Lorenz Surface plot using Plotly
        if len(agent_benefits) > 2:
            try:
                agents = list(agent_benefits.keys())
                n_agents = len(agents)
                
                # Extract benefits for all agents
                agent_distance_benefits = [agent_benefits[agent]['distance_benefit'] for agent in agents]
                agent_traffic_benefits = [agent_benefits[agent]['traffic_benefit'] for agent in agents]
                
                # Sort agents by benefits
                distance_sorted_indices = np.argsort(agent_distance_benefits)
                traffic_sorted_indices = np.argsort(agent_traffic_benefits)
                
                sorted_distance_benefits = [agent_distance_benefits[i] for i in distance_sorted_indices]
                sorted_traffic_benefits = [agent_traffic_benefits[i] for i in traffic_sorted_indices]
                
                # Calculate cumulative distributions
                cumulative_distance = np.cumsum(sorted_distance_benefits)
                cumulative_traffic = np.cumsum(sorted_traffic_benefits)
                
                # Normalize cumulative benefits
                total_distance = cumulative_distance[-1] if cumulative_distance[-1] > 0 else 1
                total_traffic = cumulative_traffic[-1] if cumulative_traffic[-1] > 0 else 1
                
                distance_cumulative_norm = cumulative_distance / total_distance
                traffic_cumulative_norm = cumulative_traffic / total_traffic
                
                # Create population shares
                population_shares = np.linspace(0, 1, n_agents)
                
                # Create interpolated values for smooth surface
                n_points = min(50, n_agents)
                interp_pop_shares = np.linspace(0, 1, n_points)
                
                # Interpolate cumulative benefits
                interp_distance_cumulative = np.interp(interp_pop_shares, population_shares, distance_cumulative_norm)
                interp_traffic_cumulative = np.interp(interp_pop_shares, population_shares, traffic_cumulative_norm)
                
                # Create meshgrid for 3D surface
                X, Y = np.meshgrid(interp_distance_cumulative, interp_traffic_cumulative)
                
                # Z represents cumulative population share
                Z = np.zeros_like(X)
                for i in range(n_points):
                    for j in range(n_points):
                        distance_threshold = X[i, j]
                        traffic_threshold = Y[i, j]
                        
                        qualifying_agents_distance = np.sum(distance_cumulative_norm <= distance_threshold)
                        qualifying_agents_traffic = np.sum(traffic_cumulative_norm <= traffic_threshold)
                        
                        qualifying_agents = min(qualifying_agents_distance, qualifying_agents_traffic)
                        Z[i, j] = qualifying_agents / n_agents
                
                # Create 3D Plotly figure
                fig = go.Figure()
                
                # Add main Lorenz surface
                fig.add_trace(go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    colorscale='Viridis',
                    opacity=0.9,
                    name='PSO Lorenz Surface',
                    colorbar=dict(
                        title=dict(
                            text="Cumulative Population Share",
                            side="right",
                            font=dict(size=14)
                        ),
                        tickfont=dict(size=12)
                    ),
                    hovertemplate=(
                        "Distance Benefits: %{x:.3f}<br>" +
                        "Traffic Benefits: %{y:.3f}<br>" +
                        "Population Share: %{z:.3f}<br>" +
                        "<extra></extra>"
                    )
                ))
                
                # Add equality plane
                X_eq, Y_eq = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
                Z_eq = (X_eq + Y_eq) / 2
                Z_eq = np.minimum(Z_eq, 1.0)
                
                fig.add_trace(go.Surface(
                    x=X_eq,
                    y=Y_eq,
                    z=Z_eq,
                    opacity=0.3,
                    colorscale=[[0, 'red'], [1, 'red']],
                    showscale=False,
                    name='Perfect Equality',
                    hovertemplate=(
                        "Perfect Equality Reference<br>" +
                        "Distance Benefits: %{x:.3f}<br>" +
                        "Traffic Benefits: %{y:.3f}<br>" +
                        "Population Share: %{z:.3f}<br>" +
                        "<extra></extra>"
                    )
                ))
                
                # Update layout
                fig.update_layout(
                    title={
                        'text': '3D Lorenz Surface - PSO Benefit Distribution',
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 18, 'family': 'Arial, sans-serif'}
                    },
                    scene=dict(
                        xaxis=dict(
                            title=dict(
                                text='Cumulative Share of Distance Benefits',
                                font=dict(size=14)
                            ),
                            tickfont=dict(size=12),
                            range=[0, 1]
                        ),
                        yaxis=dict(
                            title=dict(
                                text='Cumulative Share of Traffic Benefits',
                                font=dict(size=14)
                            ),
                            tickfont=dict(size=12),
                            range=[0, 1]
                        ),
                        zaxis=dict(
                            title=dict(
                                text='Cumulative Share of Population',
                                font=dict(size=14)
                            ),
                            tickfont=dict(size=12),
                            range=[0, 1]
                        ),
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.5)
                        ),
                        aspectmode='cube'
                    ),
                    width=1000,
                    height=800,
                    margin=dict(l=0, r=0, t=50, b=0),
                    annotations=[
                        dict(
                            text=f'PSO Analysis of {n_agents} agents',
                            x=0.5, y=0.02,
                            xref='paper', yref='paper',
                            showarrow=False,
                            font=dict(size=12, style='italic')
                        )
                    ]
                )
                
                # Save interactive HTML
                html_path = f"{save_dir}/PSO_3D_Lorenz_Surface.html"
                try:
                    pyo.plot(fig, filename=html_path, auto_open=False, config={'displayModeBar': True})
                    print(f"PSO Interactive 3D Lorenz Surface saved as: {html_path}")
                except Exception as html_error:
                    try:
                        fig.write_html(html_path, include_plotlyjs=True)
                        print(f"PSO Interactive 3D Lorenz Surface saved as: {html_path}")
                    except Exception as write_error:
                        print(f"Error saving PSO 3D Lorenz HTML: {write_error}")
                
                # Save static PNG
                png_path = f"{save_dir}/PSO_3D_Lorenz_Surface.png"
                try:
                    fig.write_image(png_path, width=1200, height=900, scale=2)
                    print(f"PSO 3D Lorenz Surface PNG saved as: {png_path}")
                except Exception as png_error:
                    print(f"Could not save PSO PNG (install kaleido): {png_error}")
                    
            except Exception as e:
                print(f"Error creating PSO 3D Lorenz surface: {e}")
        
        # Create comprehensive 2D visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Distance Benefits Lorenz Curve
        ax1 = axes[0, 0]
        if positive_distance_benefits:
            sorted_distance = sorted(positive_distance_benefits)
            cumulative_distance = np.cumsum(sorted_distance)
            cumulative_distance_norm = cumulative_distance / cumulative_distance[-1]
            x_distance = np.linspace(0, 1, len(cumulative_distance_norm))
            
            ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Equality', linewidth=2)
            ax1.plot(x_distance, cumulative_distance_norm, 'b-', linewidth=3, label='PSO Distribution')
            ax1.fill_between(x_distance, cumulative_distance_norm, x_distance, alpha=0.3, color='blue')
            
            ax1.set_xlabel('Cumulative Share of Agents')
            ax1.set_ylabel('Cumulative Share of Distance Benefits')
            ax1.set_title(f'PSO Distance Benefits Lorenz Curve\n(Gini: {gini_distance:.3f})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No Positive Distance Benefits', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=14)
        
        # 2. Traffic Benefits Lorenz Curve
        ax2 = axes[0, 1]
        if positive_traffic_benefits:
            sorted_traffic = sorted(positive_traffic_benefits)
            cumulative_traffic = np.cumsum(sorted_traffic)
            cumulative_traffic_norm = cumulative_traffic / cumulative_traffic[-1]
            x_traffic = np.linspace(0, 1, len(cumulative_traffic_norm))
            
            ax2.plot([0, 1], [0, 1], 'k--', label='Perfect Equality', linewidth=2)
            ax2.plot(x_traffic, cumulative_traffic_norm, 'g-', linewidth=3, label='PSO Distribution')
            ax2.fill_between(x_traffic, cumulative_traffic_norm, x_traffic, alpha=0.3, color='green')
            
            ax2.set_xlabel('Cumulative Share of Agents')
            ax2.set_ylabel('Cumulative Share of Traffic Benefits')
            ax2.set_title(f'PSO Traffic Benefits Lorenz Curve\n(Gini: {gini_traffic:.3f})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No Positive Traffic Benefits', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14)
        
        # 3. Benefit Scatter Plot
        ax3 = axes[0, 2]
        scatter = ax3.scatter(distance_benefits, traffic_benefits, alpha=0.7, s=60, 
                            c='purple', edgecolors='black', linewidth=0.5)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Distance Benefit')
        ax3.set_ylabel('Traffic Benefit')
        ax3.set_title('PSO Multidimensional Benefit Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Calculate correlation
        try:
            correlation = np.corrcoef(distance_benefits, traffic_benefits)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        except:
            correlation = 0.0
        
        ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax3.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"), fontsize=10)
        
        # 4. Distance Benefit Distribution
        ax4 = axes[1, 0]
        if distance_benefits:
            ax4.hist(distance_benefits, bins=min(20, len(set(distance_benefits))), 
                    alpha=0.7, color='blue', label='Distance Benefits', density=True)
            ax4.axvline(x=np.mean(distance_benefits), color='blue', linestyle='--', linewidth=2, 
                    label=f'Mean: {np.mean(distance_benefits):.3f}')
        ax4.set_xlabel('Distance Benefit')
        ax4.set_ylabel('Density')
        ax4.set_title('PSO Distance Benefit Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Traffic Benefit Distribution
        ax5 = axes[1, 1]
        if traffic_benefits:
            ax5.hist(traffic_benefits, bins=min(20, len(set(traffic_benefits))), 
                    alpha=0.7, color='green', label='Traffic Benefits', density=True)
            ax5.axvline(x=np.mean(traffic_benefits), color='green', linestyle='--', linewidth=2, 
                    label=f'Mean: {np.mean(traffic_benefits):.3f}')
        ax5.set_xlabel('Traffic Benefit')
        ax5.set_ylabel('Density')
        ax5.set_title('PSO Traffic Benefit Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Winner/Loser Analysis
        ax6 = axes[1, 2]
        combined_benefits = [data['combined_benefit'] for data in agent_benefits.values()]
        winners = sum(1 for b in combined_benefits if b > 1.0)
        losers = sum(1 for b in combined_benefits if b < -1.0)
        neutral = len(combined_benefits) - winners - losers
        
        categories = ['Winners', 'Neutral', 'Losers']
        counts = [winners, neutral, losers]
        colors = ['green', 'gray', 'red']
        
        bars = ax6.bar(categories, counts, color=colors, alpha=0.7)
        ax6.set_ylabel('Number of Agents')
        ax6.set_title('PSO Winner/Loser Analysis')
        
        # Add percentage labels
        total = len(combined_benefits)
        for i, (bar, count) in enumerate(zip(bars, counts)):
            percentage = (count / total) * 100 if total > 0 else 0
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/PSO_multidimensional_benefit_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate additional statistics
        combined_benefits = [data['combined_benefit'] for data in agent_benefits.values()]
        positive_combined_benefits = [b for b in combined_benefits if b > 0]
        combined_gini = calculate_gini(positive_combined_benefits) if positive_combined_benefits else 0
        
        # Return comprehensive statistics
        stats = {
            'gini_distance_benefits': gini_distance,
            'gini_traffic_benefits': gini_traffic,
            'gini_combined_benefits': combined_gini,
            'correlation_distance_traffic': correlation,
            'mean_distance_benefit': np.mean(distance_benefits) if distance_benefits else 0,
            'mean_traffic_benefit': np.mean(traffic_benefits) if traffic_benefits else 0,
            'std_distance_benefit': np.std(distance_benefits) if distance_benefits else 0,
            'std_traffic_benefit': np.std(traffic_benefits) if traffic_benefits else 0,
            'winners': winners,
            'losers': losers,
            'neutral': neutral,
            'total_agents_analyzed': len(agent_benefits)
        }
        
        print(f"\n=== PSO BENEFIT ANALYSIS SUMMARY ===")
        print(f"Distance Benefit Gini: {gini_distance:.3f}")
        print(f"Traffic Benefit Gini: {gini_traffic:.3f}")
        print(f"Combined Benefit Gini: {combined_gini:.3f}")
        print(f"Distance-Traffic Correlation: {correlation:.3f}")
        print(f"Winners: {winners} ({winners/total*100:.1f}%)")
        print(f"Losers: {losers} ({losers/total*100:.1f}%)")
        print(f"Neutral: {neutral} ({neutral/total*100:.1f}%)")
        
        return stats

    def analyze_pso_winner_loser_distribution(self, agent_benefits, save_dir):
        """Analyze winner/loser distribution for PSO results"""
        if not agent_benefits:
            return {}
        
        combined_benefits = [data['combined_benefit'] for data in agent_benefits.values()]
        
        # Classify agents
        winners = [b for b in combined_benefits if b > 1.0]
        losers = [b for b in combined_benefits if b < -1.0]
        neutral = [b for b in combined_benefits if -1.0 <= b <= 1.0]
        
        total_agents = len(combined_benefits)
        
        stats = {
            'winners': len(winners),
            'losers': len(losers), 
            'neutral': len(neutral),
            'winner_percentage': (len(winners) / total_agents * 100) if total_agents > 0 else 0,
            'loser_percentage': (len(losers) / total_agents * 100) if total_agents > 0 else 0,
            'neutral_percentage': (len(neutral) / total_agents * 100) if total_agents > 0 else 0,
            'average_winner_benefit': np.mean(winners) if winners else 0,
            'average_loser_benefit': np.mean(losers) if losers else 0,
            'total_agents': total_agents
        }
        
        return stats
    
    def _get_population_stats(self) -> Dict[str, Any]:
        """Get current population statistics (works for both fixed and birth_death modes)"""
        active_agents = [a for a in self.env.agents if self.env.agent_role.get(a, 'active') not in ['dropout', 'never_joined']]
        drivers = [a for a in self.env.agents if self.env.agent_role.get(a, 'active') == 'driver']
        riders = [a for a in self.env.agents if self.env.agent_role.get(a, 'active') == 'rider']
        
        stats = {
            'total_agents': len(self.env.agents),
            'active_agents': len(active_agents),
            'num_drivers': len(drivers),
            'num_riders': len(riders),
            'dropped_out_agents': len(getattr(self.env, 'dropped_out_agents', set())),
            'never_joined_agents': len(getattr(self.env, 'never_joined_agents', set())),
        }
        
        # Add current day's birth/death stats if available (birth_death mode only)
        current_day = getattr(self.env, 'current_day', 1)
        if hasattr(self.env, 'daily_births') and current_day in self.env.daily_births:
            stats['daily_births'] = len(self.env.daily_births[current_day])
        else:
            stats['daily_births'] = 0
            
        if hasattr(self.env, 'daily_dropouts') and current_day in self.env.daily_dropouts:
            stats['daily_dropouts'] = len(self.env.daily_dropouts[current_day])
        else:
            stats['daily_dropouts'] = 0
        
        return stats
    
    def _update_training_history(self, episode_metrics: Dict[str, Any]):
        # Append simple histories
        self.training_history['rewards'].append(episode_metrics.get('total_reward', 0))
        self.training_history['assignments'].append(episode_metrics.get('num_assignments', 0))
        self.training_history['fitness_scores'].append(episode_metrics.get('pso_fitness', 0))
        
        # Store distance comparison and metrics
        metrics = episode_metrics.get('metrics', {})
        if 'distance_comparison' in metrics and metrics['distance_comparison']:
            self.daily_distance_comparison.append(metrics['distance_comparison'])
        
        # Store full metrics blob
        if metrics:
            self.daily_metrics.append(metrics)
        
        # Store population stats
        if 'population_stats' in episode_metrics:
            self.daily_population_stats.append(episode_metrics['population_stats'])

    def get_training_statistics(self) -> Dict[str, Any]:
        rewards = np.array(self.training_history['rewards'], dtype=float) if self.training_history['rewards'] else np.array([])
        assigns = np.array(self.training_history['assignments'], dtype=float) if self.training_history['assignments'] else np.array([])
        fitness = np.array(self.training_history['fitness_scores'], dtype=float) if self.training_history['fitness_scores'] else np.array([])
        
        def stats(arr):
            return {
                'mean': float(arr.mean()) if arr.size else 0.0,
                'std': float(arr.std()) if arr.size else 0.0
            }
        
        return {
            'training_episodes': int(len(self.training_history['rewards'])),
            'rewards': stats(rewards),
            'assignments': stats(assigns),
            'fitness_scores': stats(fitness),
        }

    def _pgf_write_dat(self, filepath: str, header: str, rows: list):
        """Write a tab-separated .dat file with an optional header (commented)"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            if header:
                # Ensure the header is properly commented and on its own line
                f.write(f"% {header}\n")  # Use % instead of # for better pgfplots compatibility
            for row in rows:
                f.write("\t".join(str(x) for x in row) + "\n")

    def save_results(self, save_dir: str, day: int = None):
        """Full save: includes benefit analysis, traffic density, plots."""
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, "results.json")

        daily_summaries = self._build_daily_summaries()

        summary = self._build_summary(daily_summaries)

        # PERFORM BENEFIT ANALYSIS HERE
        print("Calculating PSO agent benefits...")
        agent_benefits = self.calculate_agent_benefits()
        
        benefit_stats = {}
        winner_loser_stats = {}
        
        if agent_benefits:
            print("Analyzing PSO multidimensional benefit distribution...")
            benefit_stats = self.analyze_multidimensional_benefit_distribution(agent_benefits, save_dir)
            
            print("Analyzing PSO winner/loser distribution...")
            winner_loser_stats = self.analyze_pso_winner_loser_distribution(agent_benefits, save_dir)
            
            print(f"PSO Benefit Analysis Complete:")
            print(f"  Distance Gini: {benefit_stats.get('gini_distance_benefits', 0):.3f}")
            print(f"  Traffic Gini: {benefit_stats.get('gini_traffic_benefits', 0):.3f}")
            print(f"  Winners: {winner_loser_stats.get('winners', 0)} ({winner_loser_stats.get('winner_percentage', 0):.1f}%)")
            print(f"  Losers: {winner_loser_stats.get('losers', 0)} ({winner_loser_stats.get('loser_percentage', 0):.1f}%)")
        else:
            print("No agent benefits calculated - insufficient data")

        results_payload = {
            'training_history': self.training_history,
            'statistics': self.get_training_statistics(),
            'altruism_stats': self.env.get_altruism_distribution_stats(),
            'daily_distance_comparisons': daily_summaries,
            'last_updated_day': day,
            'summary': summary,
            'population_dynamics': self.daily_population_stats,
            'daily_altruism_evolution': self.daily_altruism_stats,
            'agent_benefits': agent_benefits,
            'benefit_analysis': benefit_stats,
            'winner_loser_analysis': winner_loser_stats,
        }

        with open(filename, 'w') as f:
            json.dump(results_payload, f, indent=2)

        self._generate_dat_files(save_dir, daily_summaries)

        # Calculate and save traffic density metrics
        print("Calculating traffic density metrics for PSO...")
        traffic_metrics, avg_density_sharing, avg_density_no_sharing = self._calculate_traffic_density_metrics()
        
        self._create_traffic_density_visualization(
            avg_density_sharing, avg_density_no_sharing, traffic_metrics, save_dir
        )
        
        self._save_traffic_density_data_for_latex(
            traffic_metrics, avg_density_sharing, avg_density_no_sharing, save_dir
        )
        
        print(f"Full results saved to {save_dir}")

    def save_results_lightweight(self, save_dir: str, day: int = None):
        """Lightweight daily save: JSON + .dat files only, no heavy analysis."""
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, "results.json")

        daily_summaries = self._build_daily_summaries()

        summary = self._build_summary(daily_summaries)

        results_payload = {
            'training_history': self.training_history,
            'statistics': self.get_training_statistics(),
            'altruism_stats': self.env.get_altruism_distribution_stats(),
            'daily_distance_comparisons': daily_summaries,
            'last_updated_day': day,
            'summary': summary,
            'population_dynamics': self.daily_population_stats,
            'daily_altruism_evolution': self.daily_altruism_stats,
        }

        with open(filename, 'w') as f:
            json.dump(results_payload, f, indent=2)

        self._generate_dat_files(save_dir, daily_summaries)
        print(f"Lightweight results saved to {save_dir} (day {day})")

    def _build_daily_summaries(self):
        """Build per-day summaries aligned with existing schema."""
        daily_summaries = []
        num_days = len(self.training_history['rewards'])
        
        for i in range(num_days):
            entry = {
                'day': i + 1,
                'reward': self.training_history['rewards'][i],
                'fitness': self.training_history['fitness_scores'][i],
                'assignments': self.training_history['assignments'][i],
            }
            
            if i < len(self.daily_distance_comparison):
                dist_comp = self.daily_distance_comparison[i]
                entry.update(dist_comp)
            
            if i < len(self.daily_metrics):
                entry['metrics'] = self.daily_metrics[i]
            
            if i < len(self.daily_population_stats):
                entry['population_stats'] = self.daily_population_stats[i]
            
            if i < len(self.daily_altruism_stats):
                entry['altruism_stats'] = self.daily_altruism_stats[i]
                
            daily_summaries.append(entry)
        return daily_summaries

    def _build_summary(self, daily_summaries):
        """Build summary statistics from daily summaries."""
        def safe_mean(xs, key):
            vals = [d.get(key, 0.0) for d in xs if isinstance(d.get(key, None), (int, float)) and d.get(key, 0.0) != 0.0]
            return float(np.mean(vals)) if vals else 0.0

        def safe_sum(xs, key):
            vals = [d.get(key, 0.0) for d in xs if isinstance(d.get(key, None), (int, float))]
            return float(sum(vals)) if vals else 0.0

        summary = {
            'total_days': len(daily_summaries),
            'average_savings_percentage': safe_mean(daily_summaries, 'savings_percentage'),
            'average_assignment_rate': safe_mean(daily_summaries, 'assignment_rate'),
            'total_distance_saved': safe_sum(daily_summaries, 'distance_saved'),
        }
        
        if self.daily_population_stats:
            final_stats = self.daily_population_stats[-1]
            initial_stats = self.daily_population_stats[0]
            
            summary['population_dynamics'] = {
                'initial_active_agents': initial_stats.get('active_agents', 0),
                'final_active_agents': final_stats.get('active_agents', 0),
                'total_births': safe_sum([{'daily_births': p.get('daily_births', 0)} for p in self.daily_population_stats], 'daily_births'),
                'total_dropouts': safe_sum([{'daily_dropouts': p.get('daily_dropouts', 0)} for p in self.daily_population_stats], 'daily_dropouts'),
                'peak_active_agents': max([p.get('active_agents', 0) for p in self.daily_population_stats]),
                'min_active_agents': min([p.get('active_agents', 0) for p in self.daily_population_stats])
            }

        if self.daily_altruism_stats:
            initial_altruism = self.daily_altruism_stats[0]
            final_altruism = self.daily_altruism_stats[-1]
            all_means = [stats.get('mean', 0.0) for stats in self.daily_altruism_stats]
            all_stds = [stats.get('std', 0.0) for stats in self.daily_altruism_stats]
            
            summary['altruism_evolution'] = {
                'initial_mean': initial_altruism.get('mean', 0.0),
                'final_mean': final_altruism.get('mean', 0.0),
                'mean_change': final_altruism.get('mean', 0.0) - initial_altruism.get('mean', 0.0),
                'peak_mean': max(all_means) if all_means else 0.0,
                'lowest_mean': min(all_means) if all_means else 0.0,
                'average_std': float(np.mean(all_stds)) if all_stds else 0.0,
                'std_change': final_altruism.get('std', 0.0) - initial_altruism.get('std', 0.0)
            }
        
        return summary

    def _generate_dat_files(self, save_dir: str, daily_summaries: list):
        """Generate .dat files for plotting including population dynamics and altruism evolution"""
        dat_dir = os.path.join(save_dir, "dat")

        detour_rows = []   # day, mean, std
        time_rows = []     # day, mean, std  
        util_rows = []     # day, mean, std
        per_agent_dist_rows = []  # day, mean, std, total_agents, drivers_count, riders_count
        dist_rows = []     # day, indiv, shared, saved, savings_pct, assignment_rate
        assign_rows = []   # day, assignments, reward, fitness
        population_rows = []  # day, total_agents, active_agents, drivers, riders, dropouts, never_joined, births, daily_dropouts
        altruism_rows = []  # NEW: day, mean, std, count, min, max, median

        for i, entry in enumerate(daily_summaries, start=1):
            metrics = entry.get('metrics', {})
            
            # Extract metric data safely - these are now DRIVER-ONLY metrics
            det = metrics.get('detour_factors', {})
            tim = metrics.get('avg_trip_time', {})
            uti = metrics.get('vehicle_utilization', {})
            per_agent = metrics.get('per_agent_distances', {})

            detour_rows.append([i, float(det.get('mean', 0.0)), float(det.get('std', 0.0))])
            time_rows.append([i, float(tim.get('mean', 0.0)), float(tim.get('std', 0.0))])
            util_rows.append([i, float(uti.get('mean', 0.0)), float(uti.get('std', 0.0))])
            
            # Per-agent distance data (includes both drivers and riders)
            per_agent_dist_rows.append([
                i, 
                float(per_agent.get('mean', 0.0)), 
                float(per_agent.get('std', 0.0)),
                int(per_agent.get('total_agents', 0)),
                int(per_agent.get('drivers_count', 0)),
                int(per_agent.get('riders_count', 0))
            ])

            # Distance data
            indiv = float(entry.get('total_individual_distance', 0.0))
            shared = float(entry.get('total_shared_distance', 0.0))
            saved = float(entry.get('distance_saved', 0.0))
            sav_pct = float(entry.get('savings_percentage', 0.0))
            assign_rate = float(entry.get('assignment_rate', 0.0))
            dist_rows.append([i, indiv, shared, saved, sav_pct, assign_rate])

            # Assignment data
            assignments = int(entry.get('assignments', 0))
            reward = float(entry.get('reward', 0.0))
            fitness = float(entry.get('fitness', 0.0))
            assign_rows.append([i, assignments, reward, fitness])
            
            # Population data
            pop_stats = entry.get('population_stats', {})
            population_rows.append([
                i,  # day
                int(pop_stats.get('total_agents', 0)),
                int(pop_stats.get('active_agents', 0)), 
                int(pop_stats.get('num_drivers', 0)),
                int(pop_stats.get('num_riders', 0)),
                int(pop_stats.get('dropped_out_agents', 0)),
                int(pop_stats.get('never_joined_agents', 0)),
                int(pop_stats.get('daily_births', 0)),
                int(pop_stats.get('daily_dropouts', 0))
            ])
            
            # NEW: Altruism data
            altruism_stats = entry.get('altruism_stats', {})
            altruism_rows.append([
                i,  # day
                float(altruism_stats.get('mean', 0.0)),
                float(altruism_stats.get('std', 0.0)),
                int(altruism_stats.get('count', 0)),
                float(altruism_stats.get('min', 0.0)),
                float(altruism_stats.get('max', 0.0)),
                float(altruism_stats.get('median', 0.0))
            ])

        # Write .dat files with updated headers
        self._pgf_write_dat(
            os.path.join(dat_dir, "detour_over_time.dat"),
            "day detour_mean detour_std # Driver detour factors only",
            detour_rows
        )
        self._pgf_write_dat(
            os.path.join(dat_dir, "avg_trip_time_over_time.dat"),
            "day time_mean time_std # Driver trip times only (minutes)",
            time_rows
        )
        self._pgf_write_dat(
            os.path.join(dat_dir, "vehicle_utilization_over_time.dat"),
            "day util_mean util_std # Driver vehicle utilization only",
            util_rows
        )
        self._pgf_write_dat(
            os.path.join(dat_dir, "per_agent_distances_over_time.dat"),
            "day distance_mean distance_std total_agents drivers_count riders_count # All agents distances",
            per_agent_dist_rows
        )
        self._pgf_write_dat(
            os.path.join(dat_dir, "distance_over_time.dat"),
            "day individual_distance shared_distance distance_saved savings_pct assignment_rate",
            dist_rows
        )
        self._pgf_write_dat(
            os.path.join(dat_dir, "assignments_over_time.dat"),
            "day assignments reward fitness",
            assign_rows
        )
        self._pgf_write_dat(
            os.path.join(dat_dir, "population_dynamics_over_time.dat"),
            "day total_agents active_agents drivers riders dropouts never_joined daily_births daily_dropouts",
            population_rows
        )

        self._pgf_write_dat(
            os.path.join(dat_dir, "altruism_evolution_over_time.dat"),
            "day altruism_mean altruism_std count min max median # Population altruism statistics",
            altruism_rows
        )

    def reset_for_new_day(self):
        """Reset trainer state for a new day"""
        # Don't clear the accumulated history - we want to keep all days
        # Just ensure we're ready for the next day's calculations
        pass