import numpy as np
import random
import copy
from typing import List

class PSOParticle:
    """Simple PSO particle for global ride assignments."""
    
    def __init__(self, num_drivers: int, num_riders: int, max_capacity: int, 
                 drivers_data=None, riders_data=None, max_detour: float = None):  # NEW
        self.num_drivers = num_drivers
        self.num_riders = num_riders
        self.max_capacity = max_capacity
        self.max_detour = max_detour  # NEW
        
        # Core PSO components
        if drivers_data is not None and riders_data is not None:
            # Use smart geographic initialization with detour constraint
            self.position = self._initialize_assignment_geographic_smart_with_detour(drivers_data, riders_data)
        else:
            # Use your current method
            self.position = self._initialize_assignment()
        
        self.velocity = np.zeros_like(self.position, dtype=float)
        
        # Best tracking
        self.best_position = copy.deepcopy(self.position)
        self.best_fitness = float('-inf')
        self.fitness = float('-inf')
        
        # NEW: Track constraint violations
        self.constraint_violations = 0

    def _initialize_assignment(self) -> np.ndarray:
        """Initialize by trying to assign all riders randomly"""
        assignment = np.full((self.num_drivers, self.max_capacity), -1, dtype=int)
        
        if self.num_riders == 0:
            return assignment
        
        # Shuffle riders and try to assign all of them
        available_riders = list(range(self.num_riders))
        random.shuffle(available_riders)
        
        rider_idx = 0
        # Go through each driver and assign riders
        for driver_idx in range(self.num_drivers):
            for slot in range(self.max_capacity):
                if rider_idx < len(available_riders):
                    assignment[driver_idx, slot] = available_riders[rider_idx]
                    rider_idx += 1
                else:
                    break
        
        return assignment
    
    def _initialize_assignment_geographic_smart(self, drivers_data, riders_data) -> np.ndarray:
        """Assign riders with load balancing: max 2 per driver unless rider has no top-3 options"""
        assignment = np.full((self.num_drivers, self.max_capacity), -1, dtype=int)
        
        if self.num_riders == 0:
            return assignment
        
        # Safety check for single driver case
        if self.num_drivers == 1:
            return self._initialize_assignment()  # Fall back to sequential
        
        distance_matrix = self._calculate_distance_matrix(drivers_data, riders_data)
        
        # Build rider preference profiles
        rider_profiles = []
        for rider_idx in range(self.num_riders):
            driver_distances = [(distance_matrix[driver_idx][rider_idx], driver_idx) 
                            for driver_idx in range(self.num_drivers)]
            driver_distances.sort()  # Sort by distance
            
            # Calculate preference difficulty (how much worse is 2nd choice vs 1st choice)
            if len(driver_distances) >= 2:
                preference_gap = driver_distances[1][0] - driver_distances[0][0]
            else:
                preference_gap = 0
            
            rider_profiles.append({
                'rider_idx': rider_idx,
                'preferences': driver_distances,
                'top_3': [d_idx for _, d_idx in driver_distances[:3]],
                'preference_gap': preference_gap
            })
        
        # Sort by preference difficulty (riders with fewer good options go first)
        rider_profiles.sort(key=lambda x: -x['preference_gap'])
        
        driver_loads = [0] * self.num_drivers
        
        # Assignment algorithm
        for profile in rider_profiles:
            rider_idx = profile['rider_idx']
            top_3 = profile['top_3']
            all_preferences = profile['preferences']
            
            assigned = False
            
            # Step 1: Try top 3 preferred drivers with < 2 riders
            for preferred_driver in top_3:
                if driver_loads[preferred_driver] < 2:
                    slot = driver_loads[preferred_driver]
                    assignment[preferred_driver, slot] = rider_idx
                    driver_loads[preferred_driver] += 1
                    assigned = True
                    break
            
            if assigned:
                continue
            
            # Step 2: Check if rider has NO available options in top 3
            top_3_available = [d for d in top_3 if driver_loads[d] < self.max_capacity]
            
            if not top_3_available:
                # No top 3 drivers have ANY capacity, assign to best available driver
                for distance, driver_idx in all_preferences:
                    if driver_loads[driver_idx] < self.max_capacity:
                        slot = driver_loads[driver_idx]
                        assignment[driver_idx, slot] = rider_idx
                        driver_loads[driver_idx] += 1
                        assigned = True
                        break
            else:
                # Top 3 drivers have capacity but already have 2+ riders
                # Only assign to them if this rider is particularly disadvantaged
                if profile['preference_gap'] > 5.0:  # Large gap means limited options
                    for preferred_driver in top_3_available:
                        slot = driver_loads[preferred_driver]
                        assignment[preferred_driver, slot] = rider_idx
                        driver_loads[preferred_driver] += 1
                        assigned = True
                        break
                else:
                    # Try any driver with capacity
                    for distance, driver_idx in all_preferences:
                        if driver_loads[driver_idx] < self.max_capacity:
                            slot = driver_loads[driver_idx]
                            assignment[driver_idx, slot] = rider_idx
                            driver_loads[driver_idx] += 1
                            assigned = True
                            break
            
            # Final fallback
            if not assigned:
                min_load = min(driver_loads)
                if min_load < self.max_capacity:
                    best_driver = driver_loads.index(min_load)
                    slot = driver_loads[best_driver]
                    assignment[best_driver, slot] = rider_idx
                    driver_loads[best_driver] += 1
        
        return assignment

    def _initialize_assignment_geographic_smart_with_detour(self, drivers_data, riders_data) -> np.ndarray:
        """Assign riders with load balancing and max detour constraint"""
        assignment = np.full((self.num_drivers, self.max_capacity), -1, dtype=int)
        
        if self.num_riders == 0 or self.max_detour is None:
            return self._initialize_assignment_geographic_smart(drivers_data, riders_data)
        
        # Safety check for single driver case
        if self.num_drivers == 1:
            return self._initialize_assignment()
        
        distance_matrix = self._calculate_distance_matrix(drivers_data, riders_data)
        
        # NEW: Calculate detour matrix (detour for each driver-rider pair)
        detour_matrix = self._calculate_detour_matrix(drivers_data, riders_data)
        
        # Build rider preference profiles with detour constraints
        rider_profiles = []
        for rider_idx in range(self.num_riders):
            feasible_drivers = []
            
            for driver_idx in range(self.num_drivers):
                detour = detour_matrix[driver_idx][rider_idx]
                distance = distance_matrix[driver_idx][rider_idx]
                
                # Only consider drivers that meet detour constraint
                if detour <= self.max_detour:
                    feasible_drivers.append((distance, driver_idx))
            
            if not feasible_drivers:
                # No feasible drivers for this rider - skip them
                continue
            
            feasible_drivers.sort()  # Sort by distance
            
            # Calculate preference difficulty among feasible drivers only
            if len(feasible_drivers) >= 2:
                preference_gap = feasible_drivers[1][0] - feasible_drivers[0][0]
            else:
                preference_gap = 0
            
            rider_profiles.append({
                'rider_idx': rider_idx,
                'feasible_drivers': feasible_drivers,
                'top_3_feasible': [d_idx for _, d_idx in feasible_drivers[:3]],
                'preference_gap': preference_gap
            })
        
        # Sort by preference difficulty (riders with fewer good options go first)
        rider_profiles.sort(key=lambda x: -x['preference_gap'])
        
        driver_loads = [0] * self.num_drivers
        
        # Assignment algorithm with detour constraints
        for profile in rider_profiles:
            rider_idx = profile['rider_idx']
            top_3_feasible = profile['top_3_feasible']
            all_feasible = profile['feasible_drivers']
            
            assigned = False
            
            # Step 1: Try top 3 feasible drivers with < 2 riders
            for preferred_driver in top_3_feasible:
                if driver_loads[preferred_driver] < 2:
                    slot = driver_loads[preferred_driver]
                    assignment[preferred_driver, slot] = rider_idx
                    driver_loads[preferred_driver] += 1
                    assigned = True
                    break
            
            if assigned:
                continue
            
            # Step 2: Try any feasible driver with capacity
            for distance, driver_idx in all_feasible:
                if driver_loads[driver_idx] < self.max_capacity:
                    slot = driver_loads[driver_idx]
                    assignment[driver_idx, slot] = rider_idx
                    driver_loads[driver_idx] += 1
                    assigned = True
                    break
        
        return assignment
    
    def _calculate_detour_matrix(self, drivers_data, riders_data):
        """Calculate detour matrix between all drivers and riders"""
        detour_matrix = np.zeros((len(drivers_data), len(riders_data)))
        
        for driver_idx, (_, _, driver_obj) in enumerate(drivers_data):
            for rider_idx, (_, _, rider_obj) in enumerate(riders_data):
                # Calculate detour for this driver-rider pair
                detour = self._calculate_single_rider_detour(driver_obj, rider_obj)
                detour_matrix[driver_idx][rider_idx] = detour
        
        return detour_matrix
    
    def _calculate_single_rider_detour(self, driver_obj, rider_obj):
        """Calculate detour for a single rider assignment"""
        # Driver's baseline distance
        driver_pos = tuple(driver_obj.position)
        driver_dest = tuple(driver_obj.destination)
        baseline_path = driver_obj._dijkstra_path(driver_pos, driver_dest)
        baseline_distance = driver_obj._calculate_path_length(baseline_path)
        
        # Distance with rider (pickup + dropoff + to destination)
        rider_pos = tuple(rider_obj.position)
        rider_dest = tuple(rider_obj.destination)
        
        pickup_path = driver_obj._dijkstra_path(driver_pos, rider_pos)
        pickup_distance = driver_obj._calculate_path_length(pickup_path)
        
        dropoff_path = driver_obj._dijkstra_path(rider_pos, rider_dest)
        dropoff_distance = driver_obj._calculate_path_length(dropoff_path)
        
        final_path = driver_obj._dijkstra_path(rider_dest, driver_dest)
        final_distance = driver_obj._calculate_path_length(final_path)
        
        total_with_rider = pickup_distance + dropoff_distance + final_distance
        
        return max(0, total_with_rider - baseline_distance)

    def _calculate_distance_matrix(self, drivers_data, riders_data):
        """Calculate distance matrix between all drivers and riders"""
        distance_matrix = np.zeros((len(drivers_data), len(riders_data)))
        
        for driver_idx, (_, _, driver_obj) in enumerate(drivers_data):
            for rider_idx, (_, _, rider_obj) in enumerate(riders_data):
                # Calculate distance from driver to rider pickup location
                driver_pos = tuple(driver_obj.position)
                rider_pos = tuple(rider_obj.position)
                
                # Use Dijkstra for accurate distance
                path = driver_obj._dijkstra_path(driver_pos, rider_pos)
                distance = driver_obj._calculate_path_length(path)
                distance_matrix[driver_idx][rider_idx] = distance
        
        return distance_matrix
    
    def update_velocity(self, global_best_position: np.ndarray, w: float, c1: float, c2: float):
        """Standard PSO velocity update"""
        if global_best_position is None:
            return
            
        r1, r2 = random.random(), random.random()
        
        # Standard PSO velocity update
        self.velocity = (w * self.velocity + 
                        c1 * r1 * (self.best_position - self.position) +
                        c2 * r2 * (global_best_position - self.position))
        
        # Clip velocity to reasonable bounds
        self.velocity = np.clip(self.velocity, -3.0, 3.0)
    
    def update_position(self):
        """Standard PSO position update"""
        # Update position with velocity
        self.position = self.position + self.velocity
        
        # Round to integers and clip to valid range
        self.position = np.round(self.position).astype(int)
        self.position = np.clip(self.position, -1, self.num_riders - 1)
        
        # Remove duplicate assignments
        self._remove_duplicates()
    
    def _remove_duplicates(self):
        """Remove duplicate rider assignments"""
        assigned_riders = set()
        
        for driver_idx in range(self.num_drivers):
            for slot in range(self.max_capacity):
                rider_id = self.position[driver_idx, slot]
                
                if rider_id >= 0:
                    if rider_id in assigned_riders:
                        self.position[driver_idx, slot] = -1
                    else:
                        assigned_riders.add(rider_id)
    
    def update_best(self, fitness: float):
        """Update personal best if current fitness is better"""
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_position = copy.deepcopy(self.position)
            return True
        return False
    
    def get_assignment_count(self) -> int:
        """Get current number of assignments"""
        return np.sum(self.position >= 0)