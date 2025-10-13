import numpy as np
import itertools
from typing import List, Tuple

class PSOEvaluator:
    """Fitness evaluator aligned with RL reward objectives"""
    
    def __init__(self, env, alpha: float = 0.4, beta: float = 0.3, theta: float = 7.0, max_detour: float = 10.0):
        self.env = env
        self.alpha = alpha  # Balance between rider benefit and detour penalty
        self.beta = beta    # Weight for altruism component
        self.theta = theta  # Detour threshold (same as RL model)
        self.max_detour = max_detour  # NEW: Maximum allowed detour constraint
        
        # Caching for performance
        self.distance_cache = {}
        self.route_cache = {}
        
        # Track constraint violations for debugging
        self.constraint_violations = 0
        self.total_evaluations = 0
    
    def evaluate_particle(self, particle, drivers: List[Tuple], riders: List[Tuple]) -> float:
        """Calculate fitness using RL-aligned multi-objective function with max detour constraint"""
        if not drivers or not riders:
            return 0.0
        
        self.total_evaluations += 1
        
        total_rider_distance_served = 0.0
        total_weighted_detour_penalty = 0.0
        total_altruism_points = 0.0
        total_assignments = 0
        constraint_violations = 0
        
        # Evaluate each driver's assignments
        for driver_idx, (_, driver_name, driver_obj) in enumerate(drivers):
            assigned_riders = []
            for slot in range(self.env.max_capacity):
                rider_id = particle.position[driver_idx, slot]
                if 0 <= rider_id < len(riders):
                    rider_name = riders[rider_id][1]
                    rider_obj = riders[rider_id][2]
                    assigned_riders.append((rider_id, rider_name, rider_obj))
            
            if assigned_riders:
                # Check max detour constraint BEFORE calculating fitness
                valid_assignments, violated_assignments = self._filter_assignments_by_max_detour(
                    driver_obj, assigned_riders
                )
                
                if violated_assignments:
                    constraint_violations += len(violated_assignments)
                    # Remove violated assignments from particle (repair mechanism)
                    self._repair_particle_assignments(particle, driver_idx, violated_assignments, riders)
                
                if valid_assignments:
                    # Calculate metrics for valid assignments only
                    rider_benefits, detour_penalties, altruism_gained = self._evaluate_driver_assignments(
                        driver_name, driver_obj, valid_assignments
                    )
                    
                    total_rider_distance_served += rider_benefits
                    total_weighted_detour_penalty += detour_penalties
                    total_altruism_points += altruism_gained
                    total_assignments += len(valid_assignments)
        
        # Track violations
        if constraint_violations > 0:
            self.constraint_violations += 1
        
        # Multi-objective fitness function (aligned with RL reward)
        fitness = (self.alpha * total_rider_distance_served - 
                (1 - self.alpha) * total_weighted_detour_penalty)
        
        # Use configurable weights (with defaults if not set)
        assignment_bonus_weight = getattr(self, 'assignment_bonus_weight', 3.0)
        load_balance_weight = getattr(self, 'load_balance_weight', 2.0)
        unassigned_penalty_weight = getattr(self, 'unassigned_penalty_weight', 1.0)
        
        # Bonus for more assignments (encourage ride sharing)
        assignment_bonus = total_assignments * assignment_bonus_weight
        
        # Penalty for unassigned riders
        unassigned_penalty = (len(riders) - total_assignments) * unassigned_penalty_weight
        
        # Load balance bonus
        load_balance_bonus = self._calculate_load_balance_bonus(particle, len(drivers)) * load_balance_weight

        # NEW: Heavy penalty for constraint violations
        constraint_penalty = constraint_violations * 50.0  # Large penalty per violation

        total_fitness = fitness + assignment_bonus - unassigned_penalty + load_balance_bonus - constraint_penalty

        # Store metrics in particle for debugging
        particle.rider_distance_served = total_rider_distance_served
        particle.detour_penalty = total_weighted_detour_penalty
        particle.altruism_points = total_altruism_points
        particle.num_assignments = total_assignments
        particle.constraint_violations = constraint_violations  # NEW
        
        return total_fitness
    
    def _filter_assignments_by_max_detour(self, driver_obj, assigned_riders):
        """Filter assignments based on max detour constraint"""
        valid_assignments = []
        violated_assignments = []
        
        # Calculate individual detours for each rider
        for rider_id, rider_name, rider_obj in assigned_riders:
            detour = self._calculate_single_rider_detour(driver_obj, rider_obj)
            
            if detour <= self.max_detour:
                valid_assignments.append((rider_id, rider_name, rider_obj))
            else:
                violated_assignments.append((rider_id, rider_name, rider_obj))
        
        return valid_assignments, violated_assignments
    
    def _repair_particle_assignments(self, particle, driver_idx, violated_assignments, riders):
        """Remove violated assignments from particle (repair mechanism)"""
        violated_rider_ids = {rider_id for rider_id, _, _ in violated_assignments}
        
        for slot in range(self.env.max_capacity):
            rider_id = particle.position[driver_idx, slot]
            if rider_id in violated_rider_ids:
                particle.position[driver_idx, slot] = -1  # Remove assignment
    
    def get_constraint_violation_stats(self):
        """Get statistics about constraint violations"""
        if self.total_evaluations == 0:
            return {'violation_rate': 0.0, 'total_violations': 0, 'total_evaluations': 0}
        
        violation_rate = (self.constraint_violations / self.total_evaluations) * 100
        return {
            'violation_rate': violation_rate,
            'total_violations': self.constraint_violations,
            'total_evaluations': self.total_evaluations
        }
    
    def _calculate_load_balance_bonus(self, particle, num_drivers):
        """Bonus for well-distributed assignments"""
        driver_loads = []
        
        for driver_idx in range(num_drivers):
            load = np.sum(particle.position[driver_idx, :] >= 0)
            driver_loads.append(load)
        
        if not driver_loads or sum(driver_loads) == 0:
            return 0.0
        
        # Calculate how evenly distributed the load is
        total_assignments = sum(driver_loads)
        ideal_load = total_assignments / num_drivers
        
        # Bonus inversely proportional to deviation from ideal
        deviations = [abs(load - ideal_load) for load in driver_loads]
        avg_deviation = np.mean(deviations)
        
        # Higher bonus for lower average deviation
        max_possible_deviation = ideal_load  # Worst case scenario
        balance_score = 1.0 - (avg_deviation / max(max_possible_deviation, 1.0))
        
        return balance_score  # Scale the bonus appropriately
    
    def _evaluate_driver_assignments(self, driver_name, driver_obj, assigned_riders):
        """Evaluate a single driver's assignments using daily altruism scores"""
        if not assigned_riders:
            return 0.0, 0.0, 0.0
        
        total_rider_benefits = 0.0
        total_detour_penalties = 0.0
        total_altruism = 0.0
        
        # Get optimal route for this driver with assigned riders
        rider_objects = [rider_obj for _, _, rider_obj in assigned_riders]
        optimal_route_distance = self._find_optimal_route(driver_obj, rider_objects)
        
        # Calculate driver's solo distance (baseline)
        driver_solo_distance = self._get_cached_distance(
            tuple(driver_obj.position), 
            tuple(driver_obj.destination), 
            driver_obj
        )
        
        # Calculate per-rider metrics using daily altruism scores
        for rider_id, rider_name, rider_obj in assigned_riders:
            # 1. Rider distance benefit (distj)
            rider_travel_distance = self._get_cached_distance(
                tuple(rider_obj.position),
                tuple(rider_obj.destination),
                rider_obj
            )
            total_rider_benefits += rider_travel_distance
            
            # 2. Driver detour penalty (di(aj))
            detour = self._calculate_detour_for_rider(driver_obj, rider_obj, rider_objects)
            
            # Apply threshold-based weight (w)
            if detour <= self.theta:
                detour_weight = 1.0
            else:
                detour_weight = 2.0
            
            weighted_detour = detour_weight * detour
            total_detour_penalties += weighted_detour
            
            # 3. Altruism points gained (ΔA) - use daily altruism scores
            altruism_gain = self.env.altruism_points_day[rider_name]
            total_altruism += altruism_gain
        
        return total_rider_benefits, total_detour_penalties, total_altruism
    
    def update_altruism_scores_from_assignments(self, assignments):
        """Update daily altruism scores based on PSO assignments"""
        # Track which agents were involved in sharing
        drivers_involved = set()
        riders_involved = set()
        
        # Process each assignment
        for driver_name, rider_assignments in assignments.items():
            drivers_involved.add(driver_name)
            
            for rider_name, rider_obj in rider_assignments:
                riders_involved.add(rider_name)
                
                # Calculate altruism exchange (similar to environment's update_altruism_scores)
                driver_obj = None
                for i, agent_name in enumerate(self.env.agents):
                    if agent_name == driver_name:
                        driver_obj = self.env.agent_objects[i]
                        break
                
                if driver_obj:
                    # Calculate detour cost
                    detour_cost = self._calculate_single_rider_detour(driver_obj, rider_obj)
                    max_possible_detour = self.env.width + self.env.height
                    normalized_detour_cost = detour_cost / max_possible_detour
                    
                    # Driver's altruism increase
                    alpha = 0.5  # Scaling factor for driver's altruism increase
                    driver_altruism_increase = alpha * self.env.altruism_points_day[rider_name] * (1 - normalized_detour_cost)
                    
                    # Rider's altruism decrease
                    beta = 0.7  # Scaling factor for rider's altruism decrease
                    rider_altruism_decrease = beta * driver_altruism_increase
                    
                    # Update scores
                    self.env.altruism_points_day[driver_name] += driver_altruism_increase
                    self.env.altruism_points_day[rider_name] -= rider_altruism_decrease
                    
                    # Ensure altruism scores stay within bounds (0 to 1)
                    self.env.altruism_points_day[driver_name] = min(max(self.env.altruism_points_day[driver_name], 0), 1)
                    self.env.altruism_points_day[rider_name] = min(max(self.env.altruism_points_day[rider_name], 0), 1)
                    
                    # Store transaction for tracking
                    current_day = getattr(self.env, 'current_day', 1)
                    transaction = {
                        'day': current_day,
                        'driver': driver_name,
                        'rider': rider_name,
                        'driver_gain': driver_altruism_increase,
                        'rider_loss': rider_altruism_decrease,
                        'detour_cost': detour_cost,
                        'normalized_detour_cost': normalized_detour_cost,
                        'rider_initial_points': self.env.altruism_points_day[rider_name] + rider_altruism_decrease,
                        'driver_initial_points': self.env.altruism_points_day[driver_name] - driver_altruism_increase
                    }
                    
                    self.env.altruism_transactions.append(transaction)
                    if current_day in self.env.daily_transactions:
                        self.env.daily_transactions[current_day].append(transaction)
        
        print(f"Updated altruism scores for {len(drivers_involved)} drivers and {len(riders_involved)} riders")
        return len(drivers_involved), len(riders_involved)
    
    def _calculate_single_rider_detour(self, driver_obj, rider_obj):
        """Calculate detour for a single rider assignment"""
        # Driver's baseline distance
        baseline_distance = self._get_cached_distance(
            tuple(driver_obj.position),
            tuple(driver_obj.destination),
            driver_obj
        )
        
        # Distance with rider (pickup + dropoff + to destination)
        pickup_distance = self._get_cached_distance(
            tuple(driver_obj.position),
            tuple(rider_obj.position),
            driver_obj
        )
        
        dropoff_distance = self._get_cached_distance(
            tuple(rider_obj.position),
            tuple(rider_obj.destination),
            driver_obj
        )
        
        final_distance = self._get_cached_distance(
            tuple(rider_obj.destination),
            tuple(driver_obj.destination),
            driver_obj
        )
        
        total_with_rider = pickup_distance + dropoff_distance + final_distance
        
        return max(0, total_with_rider - baseline_distance)
    
    def _calculate_detour_for_rider(self, driver_obj, target_rider, all_assigned_riders):
        """Calculate detour incurred by driver to pick up a specific rider"""
        # Driver's baseline route (without any riders)
        baseline_distance = self._get_cached_distance(
            tuple(driver_obj.position),
            tuple(driver_obj.destination),
            driver_obj
        )
        
        # Driver's route with all assigned riders
        shared_distance = self._find_optimal_route(driver_obj, all_assigned_riders)
        
        # Driver's route without this specific rider
        other_riders = [r for r in all_assigned_riders if r != target_rider]
        if other_riders:
            without_rider_distance = self._find_optimal_route(driver_obj, other_riders)
        else:
            without_rider_distance = baseline_distance
        
        # Detour = additional distance caused by this rider
        detour = shared_distance - without_rider_distance
        return max(0, detour)  # Ensure non-negative
    
    def _find_optimal_route(self, driver_obj, assigned_riders):
        """Find optimal route with caching and complexity limits"""
        if not assigned_riders:
            return self._get_cached_distance(
                tuple(driver_obj.position),
                tuple(driver_obj.destination),
                driver_obj
            )
        
        # Create cache key
        driver_start = tuple(driver_obj.position)
        driver_dest = tuple(driver_obj.destination)
        rider_positions = tuple(sorted([
            (tuple(r.position), tuple(r.destination)) for r in assigned_riders
        ]))
        cache_key = (driver_start, driver_dest, rider_positions)
        
        if cache_key not in self.route_cache:
            if len(assigned_riders) <= 2:  # Use exact TSP for small cases
                self.route_cache[cache_key] = self._exact_tsp_route(driver_obj, assigned_riders)
            else:  # Use greedy approximation for larger cases
                self.route_cache[cache_key] = self._fast_greedy_route(driver_obj, assigned_riders)
        
        return self.route_cache[cache_key]
    
    def _exact_tsp_route(self, driver_obj, assigned_riders):
        """Exact TSP solution for small rider groups"""
        locations = []
        location_types = []
        rider_mapping = []
        
        for i, rider in enumerate(assigned_riders):
            # Pickup location
            locations.append(tuple(rider.position))
            location_types.append('pickup')
            rider_mapping.append(i)
            
            # Dropoff location
            locations.append(tuple(rider.destination))
            location_types.append('dropoff')
            rider_mapping.append(i)
        
        min_distance = float('inf')
        
        # Try all valid permutations
        for perm in itertools.permutations(range(len(locations))):
            if self._is_valid_sequence(perm, location_types, rider_mapping, len(assigned_riders)):
                distance = self._calculate_route_distance(driver_obj, locations, perm)
                min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _fast_greedy_route(self, driver_obj, assigned_riders):
        """Fast greedy approximation for larger rider groups"""
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
                distance = self._get_cached_distance(current_pos, pickup_pos, driver_obj)
                if distance < best_distance:
                    best_distance = distance
                    best_action = ('pickup', rider)
            
            # Consider dropoffs
            for rider in picked_up_riders:
                dropoff_pos = tuple(rider.destination)
                distance = self._get_cached_distance(current_pos, dropoff_pos, driver_obj)
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
                
                total_distance += self._get_cached_distance(current_pos, next_pos, driver_obj)
                current_pos = next_pos
            else:
                break
        
        # Go to driver destination
        driver_dest = tuple(driver_obj.destination)
        total_distance += self._get_cached_distance(current_pos, driver_dest, driver_obj)
        
        return total_distance
    
    def _get_cached_distance(self, start, dest, agent_obj):
        """Get distance with caching"""
        cache_key = (start, dest)
        if cache_key not in self.distance_cache:
            path = agent_obj._dijkstra_path(start, dest)
            self.distance_cache[cache_key] = agent_obj._calculate_path_length(path)
        return self.distance_cache[cache_key]
    
    def _is_valid_sequence(self, perm, location_types, rider_mapping, num_riders):
        """Check if sequence is valid (pickup before dropoff)"""
        picked_up = set()
        
        for idx in perm:
            rider_id = rider_mapping[idx]
            location_type = location_types[idx]
            
            if location_type == 'pickup':
                if rider_id in picked_up:
                    return False
                picked_up.add(rider_id)
            else:  # dropoff
                if rider_id not in picked_up:
                    return False
                picked_up.remove(rider_id)
        
        return True
    
    def _calculate_route_distance(self, driver_obj, locations, perm):
        """Calculate route distance using cached distances"""
        current_pos = tuple(driver_obj.position)
        total_distance = 0.0
        
        for idx in perm:
            next_pos = locations[idx]
            distance = self._get_cached_distance(current_pos, next_pos, driver_obj)
            total_distance += distance
            current_pos = next_pos
        
        # Go to driver destination
        driver_dest = tuple(driver_obj.destination)
        total_distance += self._get_cached_distance(current_pos, driver_dest, driver_obj)
        
        return total_distance