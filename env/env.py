import functools
from PIL import Image
import copy
import numpy as np
import math
import json
import random
import pygame
import itertools
import gym
from gym.spaces import MultiDiscrete, Discrete
from env.agent import Agent, Driver, Rider, create_agent, switch_role

class Env(gym.Env):
    def __init__(self, height=15, width=15, numAgents=100, dataset_folder=None, initial_active_agents=100,
                 altruism_distribution='uniform', altruism_mean=0.5, altruism_std=0.15, total_days=30):
        self.height = height
        self.width = width
        self.numAgents = numAgents
        self.initial_active_agents = initial_active_agents
        self.agents = ["agent_" + str(i) for i in range(numAgents)]
        self.n = 5
        self.rider_destinations_colors = {}
        self.no_op_reward = 0.1
        self.total_days = total_days
        
        # Altruism distribution parameters
        self.altruism_distribution = altruism_distribution
        self.altruism_mean = altruism_mean
        self.altruism_std = altruism_std
        
        # Load weight matrix
        self.weight_matrix = np.load(f"dataset/{dataset_folder}/weight_matrix.npy")
        self.time_weight_matrix = np.load(f"dataset/{dataset_folder}/time_weight_matrix.npy")

        # Load fixed positions from NY dataset
        try:
            self.fixed_positions = np.load(f"dataset/{dataset_folder}/fixed_positions.npy")
            self.fixed_destinations = np.load(f"dataset/{dataset_folder}/fixed_destinations.npy")
            print("Successfully loaded positions from NY dataset")
        except:
            print("Could not load position files, using random positions")
            
        self.agent_locations = {
            agent: [self.fixed_positions[i], self.fixed_destinations[i]]
            for i, agent in enumerate(self.agents)
        }

        self.alpha = 0.4
        self.agent_role = {}
        self.max_capacity = 4

        self.detour = {agent: 0 for agent in self.agents}
        self.riders_drivers = {agent: [] for agent in self. agents}

        # Initialize altruism points based on distribution
        self.altruism_points = self._initialize_altruism_distribution()
        self.altruism_points_day = copy.deepcopy(self.altruism_points)

        self.active_riders_list = []
        self.agent_objects = None
        self.picking_up = None
        self.rejected_riders = None
        self.terminated_agents = set()

        # Dropout functionality
        self.dropped_out_agents = set()
        self.daily_dropouts = {}
        self.dropout_history = {}

        # NEW: Birth functionality
        self.never_joined_agents = set()  # Agents who have never joined the system
        self.daily_births = {}  # Track births per day {day: [agent_list]}
        self.birth_history = {}  # Track agent birth history {agent: day_joined}
        self.new_agent_altruism = 0.5  # Default altruism for new agents
        
        # Initialize agents based on initial_active_agents
        if initial_active_agents < numAgents:
            # Start with some agents in never_joined state
            agents_to_exclude = random.sample(self.agents, numAgents - initial_active_agents)
            for agent in agents_to_exclude:
                self.never_joined_agents.add(agent)
                self.agent_role[agent] = 'never_joined'
            print(f"Starting with {initial_active_agents} active agents, {len(agents_to_exclude)} never joined")

        self.renderStarted = False
        self.gridSize = 65
        self.frames = list()

        # Add altruism transaction tracking
        self.altruism_transactions = []
        self.daily_transactions = {}
        
        # Add day-wise pickup tracking
        self.daily_pickups = {}
        self.current_day_pickups = set()

        # NEW: Reintegration tracking (only data storage)
        self.reintegration_events = []  # [(agent_id, dropout_day, return_day, altruism_at_dropout), ...]
        self.agent_dropout_events = {}  # {agent_id: [dropout_days]}
        self.agent_return_events = {}   # {agent_id: [return_days]}

    def reset_day(self, day, enable_birth_death=False):
        """
        Reset day with optional dropout or birth-death functionality.
        enable_dropout: Set to True for dropout-only simulation
        enable_birth_death: Set to True for birth-death simulation (overrides enable_dropout)
        """
        self.altruism_points = copy.deepcopy(self.altruism_points_day)
        
        if day == 1:
            # Use altruism-based role assignment for day 1 with 1:1 ratio constraint
            self.decide_agent_roles_day1()
        else:
            if enable_birth_death:
                self.decide_agent_roles_with_birth_death(day)
            else:
                self.decide_agent_roles()

        if day == 1:
            self.agent_objects = [
                create_agent(
                    self.agent_role[agent] if agent not in self.never_joined_agents else 'never_joined',
                    self.agent_locations[agent][0],
                    self.agent_locations[agent][1],
                    grid_world=(self.height, self.width),
                    weight_matrix=self.weight_matrix,
                    altruism_points=self.altruism_points_day[agent],
                    **({'max_capacity': self.max_capacity} if self.agent_role[agent] == "driver" else {})
                )
                for agent in self.agents
            ]
        else:
            for i, agent in enumerate(self.agents):
                current_role = self.agent_role.get(agent, 'active')
                if current_role in ['never_joined', 'dropout']:
                    # Keep these agents inactive
                    continue
                switch_role(self.agent_objects[i], current_role, max_capacity=self.max_capacity)

        # Initialize daily transaction tracking
        self.daily_transactions[day] = []
        
        # Reset current day pickups tracking
        self.current_day_pickups = set()

    def reset(self, enable_dropout=False, enable_birth_death=False):
        """
        Reset environment with optional dropout or birth-death functionality.
        """
        self.altruism_points_day = copy.deepcopy(self.altruism_points)

        self.agent_objects = [
            create_agent(
                self.agent_role[agent] if agent not in self.never_joined_agents else 'never_joined',
                self.agent_locations[agent][0],
                self.agent_locations[agent][1],
                grid_world=(self.height, self.width),
                weight_matrix=self.weight_matrix,
                altruism_points=self.altruism_points_day[agent],
                **({'max_capacity': self.max_capacity} if self.agent_role[agent] == "driver" else {})
            )
            for agent in self.agents
        ]

        self.active_riders_list = []

        for i, agent in enumerate(self.agents): 
            agent_obj = self.agent_objects[i]
            if isinstance(agent_obj, Rider) and self.agent_role[agent] not in ['dropout', 'never_joined']:
                self.active_riders_list.append(agent_obj)

        self.picking_up = {agent: None for agent in self.agents}
        self.rejected_riders = {agent: [] for agent in self.agents}
        self.terminated_agents = set()
        self.detour = {agent: 0 for agent in self.agents}
        self.riders_drivers = {agent: [] for agent in self.agents}

        self.frames = list()

        observations = [self.get_observation(agent) for agent in self.agents]
        infos = [0 if self.agent_role[agent] in ['rider', 'dropout', 'never_joined'] else 1 for agent in self.agents]

        return np.array(observations, dtype=np.int32), np.array(infos, dtype=np.float32)
    
    def step(self, actions):
        rewards = [0 for agent in self.agents]
        done = 0

        for i, agent in enumerate(self.agents):
            # Skip dropped out and never joined agents
            if self.agent_role[agent] in ['dropout', 'never_joined'] or agent in self.terminated_agents:
                continue
                
            agent_obj = self.agent_objects[i]
            action = actions[i]

            if isinstance(agent_obj, Driver):
                nearby_riders = self.get_nearby_riders(agent_obj, i)

                if nearby_riders and not self.picking_up[agent] and len(agent_obj.riders) < agent_obj.max_capacity:
                    if action < self.numAgents:
                        for idx, rider in enumerate(self.agent_objects):
                            if (rider in nearby_riders and idx == action and 
                                rider not in self.rejected_riders[agent] and 
                                rider.altruism_points > 0 and
                                self.agent_role[self.agents[idx]] not in ['dropout', 'never_joined']):  # NEW: Check rider status
                                
                                self.detour[agent] += agent_obj._calculate_detour_cost(rider)
                                for j, agent_2 in enumerate(self.agents):
                                    if self.agent_objects[j] == rider:
                                        break

                                driver_altruism_increase = self.update_altruism_scores(agent_obj, agent, rider, agent_2)

                                rewards[i] += self.get_reward(agent_obj, rider, driver_altruism_increase)

                                if np.array_equal(rider.position, agent_obj.position):
                                    agent_obj.add_rider(rider)
                                    self.riders_drivers[agent].append(rider)
                                    self.active_riders_list.remove(rider)
                                    target_pos = self.get_optimal_route_target(agent_obj)
                                    
                                    self.track_daily_pickup(agent, agent_2)
                                else:
                                    self.picking_up[agent] = rider
                                    rider.being_picked_up = True
                                    target_pos = rider.start_position
                                break
                            else:
                                target_pos = agent_obj.destination
                    else:
                        for rider in nearby_riders:
                            for j in range(len(self.agents)):
                                if self.agent_objects[j] == rider:
                                    break
                            if (self.agents[j] not in self.rejected_riders[self.agents[i]] and
                                self.agent_role[self.agents[j]] not in ['dropout', 'never_joined']):  # NEW: Check rider status
                                if agent_obj.riders:
                                    target_pos = self.get_optimal_route_target(agent_obj)
                                else:
                                    target_pos = agent_obj.destination
                                self.rejected_riders[self.agents[i]].append(self.agents[j])
                else:
                    if nearby_riders:
                        rewards[i] += np.tanh(self.no_op_reward)

                    if self.picking_up[agent]:
                        rider = self.picking_up[agent]
                        if np.array_equal(rider.start_position, agent_obj.position):
                            agent_obj.add_rider(rider)
                            self.riders_drivers[agent].append(rider)
                            agent_obj.riders[0].altruism_points -= 1
                            self.active_riders_list.remove(rider)
                            target_pos = self.get_optimal_route_target(agent_obj)
                            self.picking_up[agent] = None
                            
                            for j, agent_name in enumerate(self.agents):
                                if self.agent_objects[j] == rider:
                                    self.track_daily_pickup(agent, agent_name)
                                    break
                        else:
                            self.picking_up[agent] = rider
                            target_pos = rider.start_position
                    elif agent_obj.riders:
                        riders_to_remove = []
                        for idx, rider in enumerate(agent_obj.riders):
                            if np.array_equal(agent_obj.position, rider.destination):
                                riders_to_remove.append(idx)
                        
                        for idx in sorted(riders_to_remove, reverse=True):
                            agent_obj.riders.pop(idx)
                        
                        if agent_obj.riders:
                            target_pos = self.get_optimal_route_target(agent_obj)
                        else:
                            target_pos = agent_obj.destination
                    else:
                        target_pos = agent_obj.destination

                agent_obj.move_towards(target_pos)

                if agent_obj.is_at_destination() and not agent_obj.riders:
                    self.terminated_agents.add(agent)
            else:
                rewards[i] = 0
                if agent not in self.terminated_agents:
                    self.terminated_agents.add(agent)

        # Check if all active agents are terminated
        active_agents = [agent for agent in self.agents 
                        if self.agent_role[agent] not in ['dropout', 'never_joined']]
        terminated_active_agents = [agent for agent in active_agents if agent in self.terminated_agents]
        
        if len(terminated_active_agents) == len(active_agents):
            done = 1

        observations = [self.get_observation(agent) for agent in self.agents]
        infos = [0 if self.agent_role[agent] in ['rider', 'dropout', 'never_joined'] else 1 for agent in self.agents]

        return np.array(observations, dtype=np.int32), np.array(rewards, dtype=np.float32), np.array(done, dtype=np.int32), np.array(infos, dtype=np.float32)

    def get_reward(self, driver, rider, driver_altruism_increase):
        reward = self.alpha * driver._calculate_positive_reward(rider) - (1 - self.alpha) * driver._calculate_detour_cost(rider) + driver_altruism_increase
        return np.tanh(reward)
            
    @functools.lru_cache(maxsize=None)
    def observation_space(self):
        # Two elements for the agent's position (x and y), one for its role, and nine elements for the 3x3 grid
        return MultiDiscrete([self.width, self.height, 2] + [self.numAgents] * self.n * self.n)

    @functools.lru_cache(maxsize=None)
    def action_space(self):
        return Discrete(self.numAgents + 1)
    
    def get_observation(self, agent):
        idx = int(agent.split("_")[1])
        agent_obj = self.agent_objects[idx]

        agent_x, agent_y = agent_obj.position[0], agent_obj.position[1]
        agent_position = np.array([agent_x, agent_y], dtype=np.int32)
        agent_role = np.array([1 if isinstance(agent_obj, Driver) else 0], dtype=np.int32)

        grid_radius = self.n // 2
        nearby_grid = np.full((self.n, self.n), -1, dtype=np.int32)

        nearby_riders = self.get_nearby_riders(agent_obj, idx)

        for rider in nearby_riders:
            rider_x, rider_y = rider.position[0], rider.position[1]
            if agent_x - grid_radius <= rider_x <= agent_x + grid_radius and agent_y - grid_radius <= rider_y <= agent_y + grid_radius:
                grid_x = rider_x - (agent_x - grid_radius)
                grid_y = rider_y - (agent_y - grid_radius)
                nearby_grid[grid_y, grid_x] = self.agent_objects.index(rider)

        nearby_grid_flat = nearby_grid.flatten()
        observation = np.concatenate([agent_position, agent_role, nearby_grid_flat])

        return observation

    # def get_agent_id_at_position(self, x, y, idx):
    #     for agent_id, agent in enumerate(self.agent_objects):
    #         if np.array_equal(agent.get_position(), np.array([x, y])) and isinstance(agent, Rider) and agent_id != idx:
    #             return agent_id
    #     return -1  # No agent at this position

    def get_nearby_riders(self, driver, idx):
        grid_radius = self.n // 2
        nearby_riders = []

        min_x = max(driver.position[0] - grid_radius, 0)
        max_x = min(driver.position[0] + grid_radius + 1, self.width)
        min_y = max(driver.position[1] - grid_radius, 0)
        max_y = min(driver.position[1] + grid_radius + 1, self.height)

        for i, agent in enumerate(self.agent_objects):
            # Skip dropped out and never joined agents
            if self.agent_role[self.agents[i]] in ['dropout', 'never_joined']:
                continue
                
            if (isinstance(agent, Rider) and not agent.being_picked_up and 
                self.agents[i] not in self.rejected_riders[self.agents[idx]]):
                if min_x <= agent.position[0] < max_x and min_y <= agent.position[1] < max_y:
                    nearby_riders.append(agent)

        return nearby_riders
    
    def get_optimal_route_target(self, driver_obj):
        """Find the next destination in the globally optimal route."""
        if not driver_obj.riders:
            return driver_obj.destination
        
        # For a single rider, there's only one possible route
        if len(driver_obj.riders) == 1:
            return driver_obj.riders[0].destination
        
        # For multiple riders, we need to find the optimal ordering
        rider_destinations = [rider.destination for rider in driver_obj.riders]
        
        # Generate all possible permutations of drop-off order
        all_permutations = list(itertools.permutations(range(len(rider_destinations))))
        
        min_total_distance = float('inf')
        best_permutation = None
        
        # Evaluate each possible permutation
        for perm in all_permutations:
            total_distance = 0
            current_pos = driver_obj.position
            
            # Calculate distance for each leg in this permutation
            for idx in perm:
                dest = rider_destinations[idx]
                path = driver_obj._dijkstra_path(current_pos, dest)
                distance = driver_obj._calculate_path_length(path)
                total_distance += distance
                current_pos = dest
            
            # Add final leg to driver's destination
            final_path = driver_obj._dijkstra_path(current_pos, driver_obj.destination)
            final_distance = driver_obj._calculate_path_length(final_path)
            total_distance += final_distance
            
            # Update if this is the best route found
            if total_distance < min_total_distance:
                min_total_distance = total_distance
                best_permutation = perm
        
        # Return the first destination in the optimal route
        if best_permutation:
            return rider_destinations[best_permutation[0]]
        else:
            # Fallback to nearest destination if something goes wrong
            return self.get_nearest_rider_destination(driver_obj)
    
    def decide_agent_roles_day1(self):
        """
        Decide agent roles for day 1 using altruism-based assignment with 1:1 ratio constraint.
        """
        # Get active agents (exclude never_joined)
        active_agents = [agent for agent in self.agents if agent not in self.never_joined_agents]
        
        if not active_agents:
            return
        
        # Find the maximum altruism score among active agents
        max_altruism_score = max(self.altruism_points_day[agent] for agent in active_agents)
        print(f"Day 1 - Max altruism score: {max_altruism_score}")
        
        # Calculate target counts for 1:1 ratio
        total_active = len(active_agents)
        target_drivers = total_active // 2
        target_riders = total_active - target_drivers
        
        print(f"Day 1 - Target: {target_drivers} drivers, {target_riders} riders from {total_active} active agents")
        
        # Step 1: Apply altruism-based role assignment logic
        initial_assignments = {}
        
        for agent in active_agents:
            agent_altruism = self.altruism_points_day[agent]
            
            # If the agent's altruism score is very low, force them to become a driver
            if agent_altruism <= 0.2:
                initial_assignments[agent] = 'driver'
            else:
                # Introduce a small random chance (10%) to switch roles regardless of altruism score
                if random.random() < 0.1:
                    initial_assignments[agent] = random.choice(['driver', 'rider'])
                else:
                    # Normalize the agent's altruism score relative to the maximum score
                    normalized_altruism = agent_altruism / max_altruism_score
                    
                    # Higher altruism score increases the probability of becoming a rider
                    if random.random() < normalized_altruism:
                        initial_assignments[agent] = 'rider'
                    else:
                        initial_assignments[agent] = 'driver'
        
        # Step 2: Count initial assignments
        initial_drivers = [agent for agent, role in initial_assignments.items() if role == 'driver']
        initial_riders = [agent for agent, role in initial_assignments.items() if role == 'rider']
        
        print(f"Day 1 - Initial assignment: {len(initial_drivers)} drivers, {len(initial_riders)} riders")
        
        # Step 3: Adjust to achieve 1:1 ratio
        # Create lists sorted by altruism for intelligent swapping
        drivers_by_altruism = sorted(initial_drivers, key=lambda x: self.altruism_points_day[x], reverse=True)
        riders_by_altruism = sorted(initial_riders, key=lambda x: self.altruism_points_day[x])
        
        # Case 1: Too many drivers, need more riders
        if len(initial_drivers) > target_drivers:
            excess_drivers = len(initial_drivers) - target_drivers
            # Convert highest altruism drivers to riders (they're more likely to be riders anyway)
            for i in range(excess_drivers):
                if i < len(drivers_by_altruism):
                    agent_to_convert = drivers_by_altruism[i]
                    # Only convert if altruism > 0.2 (don't convert forced drivers)
                    if self.altruism_points_day[agent_to_convert] > 0.2:
                        initial_assignments[agent_to_convert] = 'rider'
        
        # Case 2: Too many riders, need more drivers
        elif len(initial_riders) > target_riders:
            excess_riders = len(initial_riders) - target_riders
            # Convert lowest altruism riders to drivers (they're more likely to be drivers anyway)
            for i in range(excess_riders):
                if i < len(riders_by_altruism):
                    agent_to_convert = riders_by_altruism[i]
                    initial_assignments[agent_to_convert] = 'driver'
        
        # Step 4: Apply final assignments
        for agent in active_agents:
            self.agent_role[agent] = initial_assignments[agent]
        
        # Set roles for never_joined agents
        for agent in self.never_joined_agents:
            self.agent_role[agent] = 'never_joined'
        
        # Step 5: Verify final counts
        final_drivers = [agent for agent in active_agents if self.agent_role[agent] == 'driver']
        final_riders = [agent for agent in active_agents if self.agent_role[agent] == 'rider']
        
        print(f"Day 1 - Final assignment: {len(final_drivers)} drivers, {len(final_riders)} riders")
        print(f"Day 1 - Driver altruism range: {min([self.altruism_points_day[a] for a in final_drivers]):.3f} - {max([self.altruism_points_day[a] for a in final_drivers]):.3f}")
        print(f"Day 1 - Rider altruism range: {min([self.altruism_points_day[a] for a in final_riders]):.3f} - {max([self.altruism_points_day[a] for a in final_riders]):.3f}")
    
    def get_dropout_statistics(self, day=None):
        """
        Get statistics about agent dropouts.
        """
        if day is None:
            # Overall statistics
            total_dropouts = len(self.dropout_history)
            dropout_rates_by_altruism = {}
            
            for agent, dropout_days in self.dropout_history.items():
                current_altruism = self.altruism_points_day[agent]
                altruism_bin = round(current_altruism * 10) / 10  # Round to nearest 0.1
                
                if altruism_bin not in dropout_rates_by_altruism:
                    dropout_rates_by_altruism[altruism_bin] = {'agents': 0, 'dropouts': 0}
                
                dropout_rates_by_altruism[altruism_bin]['agents'] += 1
                dropout_rates_by_altruism[altruism_bin]['dropouts'] += len(dropout_days)
            
            return {
                'total_agents_dropped': total_dropouts,
                'currently_dropped_out': len(self.dropped_out_agents),
                'dropout_rates_by_altruism': dropout_rates_by_altruism,
                'daily_dropouts': self.daily_dropouts
            }
        else:
            # Day-specific statistics
            day_dropouts = self.daily_dropouts.get(day, [])
            return {
                'day': day,
                'agents_dropped': len(day_dropouts),
                'dropped_agents': day_dropouts
            }
        
    def _initialize_altruism_distribution(self):
        """
        Initialize altruism points based on the specified distribution.
        
        Returns:
            Dictionary mapping agents to their initial altruism scores
        """
        altruism_points = {}
        
        if self.altruism_distribution == 'uniform':
            # All agents get the same altruism value
            for agent in self.agents:
                altruism_points[agent] = self.altruism_mean
            print(f"Initialized {len(self.agents)} agents with uniform altruism: {self.altruism_mean}")
            
        elif self.altruism_distribution == 'gaussian':
            # Generate altruism values from Gaussian distribution
            np.random.seed(42)  # For reproducibility
            altruism_values = np.random.normal(self.altruism_mean, self.altruism_std, len(self.agents))
            
            # Clip values to [0, 1] range
            altruism_values = np.clip(altruism_values, 0.0, 1.0)
            
            for i, agent in enumerate(self.agents):
                altruism_points[agent] = altruism_values[i]
                
            # Print distribution statistics
            print(f"Initialized {len(self.agents)} agents with Gaussian altruism distribution:")
            print(f"  Mean: {np.mean(altruism_values):.3f} (target: {self.altruism_mean})")
            print(f"  Std: {np.std(altruism_values):.3f} (target: {self.altruism_std})")
            print(f"  Range: [{np.min(altruism_values):.3f}, {np.max(altruism_values):.3f}]")
            
        else:
            raise ValueError(f"Unknown altruism distribution: {self.altruism_distribution}. "
                           f"Supported values: 'uniform', 'gaussian'")
        
        return altruism_points

    def set_new_agent_altruism(self, altruism_value):
        """Set the altruism score for newly joining agents."""
        self.new_agent_altruism = altruism_value
        print(f"New agent altruism set to: {altruism_value}")
        
    def get_altruism_distribution_stats(self):
        """
        Get statistics about the current altruism distribution.
        
        Returns:
            Dictionary with distribution statistics
        """
        values = list(self.altruism_points_day.values())
        return {
            'distribution_type': self.altruism_distribution,
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75)
        }
    
    def calculate_new_agent_altruism(self, day, current_active_agents):
        """
        Calculate altruism score for newly joining agents based on early-bird benefits model.
        
        Args:
            day: Current simulation day
            current_active_agents: Number of currently active agents
            
        Returns:
            Altruism score (0.0 to 1.0) for new agent
        """
        total_days = getattr(self, 'total_days', 30)
        total_possible_agents = self.numAgents
        current_adoption_rate = (total_possible_agents - len(self.never_joined_agents)) / total_possible_agents
        
        # BASE ALTRUISM SCORE (reduced from 0.5)
        base_altruism = 0.3
        
        # FACTOR 1: ADOPTION PHASE (reduced ranges)
        adoption_factor = 0.0
        
        if current_adoption_rate < 0.025:  # Innovators (2.5%) -> Target: 0.4-0.6
            adoption_factor = 0.15  # Base 0.3 + 0.15 = 0.45 (center of range)
        elif current_adoption_rate < 0.16:  # Early adopters (13.5%) -> Target: 0.3-0.6
            adoption_factor = 0.10  # Base 0.3 + 0.10 = 0.40 (center of range)
        elif current_adoption_rate < 0.50:  # Early majority (34%) -> Target: 0.3-0.4
            adoption_factor = 0.05  # Base 0.3 + 0.05 = 0.35 (center of range)
        elif current_adoption_rate < 0.84:  # Late majority (34%) -> Target: 0.2-0.4
            adoption_factor = 0.0   # Base 0.3 + 0.0 = 0.30 (center of range)
        else:  # Laggards (16%) -> Target: 0.1-0.3
            adoption_factor = -0.1  # Base 0.3 - 0.1 = 0.20 (center of range)
        
        # FACTOR 2: NETWORK EFFECTS (reduced)
        network_factor = 0.0
        
        # Critical mass bonus (30% adoption threshold)
        critical_mass_threshold = total_possible_agents * 0.3
        if current_active_agents >= critical_mass_threshold:
            network_factor += 0.05  # Reduced from 0.15 to 0.05
        
        # Transaction activity bonus (based on recent activity)
        if hasattr(self, 'daily_transactions') and self.daily_transactions:
            recent_days = range(max(1, day-3), day+1)
            recent_transaction_count = sum(len(self.daily_transactions.get(d, [])) for d in recent_days)
            
            if current_active_agents > 0:
                transaction_rate = recent_transaction_count / (current_active_agents * 3)  # Per agent per day
                network_factor += min(0.03, transaction_rate * 0.3)  # Reduced from 0.10 to 0.03
        
        # FACTOR 3: SCARCITY INCENTIVES (FOMO) (reduced)
        scarcity_factor = 0.0
        
        remaining_spots = len(self.never_joined_agents)
        remaining_ratio = remaining_spots / total_possible_agents
        
        if remaining_ratio < 0.3:  # Less than 30% spots remaining
            # FOMO bonus increases as spots become scarce
            fomo_intensity = (0.3 - remaining_ratio) / 0.3
            scarcity_factor = fomo_intensity * 0.08  # Reduced from 0.20 to 0.08
        
        # Additional urgency based on time pressure
        time_pressure_threshold = total_days * 0.7  # Last 30% of simulation
        if day >= time_pressure_threshold:
            time_urgency = (day - time_pressure_threshold) / (total_days - time_pressure_threshold)
            scarcity_factor += time_urgency * 0.05  # Reduced from 0.15 to 0.05
        
        # FACTOR 4: SYSTEM REPUTATION (reduced)
        reputation_factor = 0.0
        
        # Recent dropout patterns (last 5 days)
        recent_dropout_count = 0
        recent_days_to_check = min(5, day)
        
        for check_day in range(max(1, day - recent_days_to_check + 1), day + 1):
            recent_dropout_count += len(self.daily_dropouts.get(check_day, []))
        
        if current_active_agents > 0 and recent_days_to_check > 0:
            dropout_rate = recent_dropout_count / (current_active_agents * recent_days_to_check)
            
            if dropout_rate < 0.02:  # Very low dropout rate (< 2% per day)
                reputation_factor = 0.05  # Reduced from 0.15 to 0.05
            elif dropout_rate < 0.05:  # Low dropout rate (< 5% per day)
                reputation_factor = 0.03  # Reduced from 0.10 to 0.03
            elif dropout_rate < 0.10:  # Moderate dropout rate (< 10% per day)
                reputation_factor = 0.00  # Neutral reputation
            elif dropout_rate < 0.20:  # High dropout rate (< 20% per day)
                reputation_factor = -0.03  # Reduced from -0.10 to -0.03
            else:  # Very high dropout rate (>= 20% per day)
                reputation_factor = -0.05  # Reduced from -0.20 to -0.05
        
        # Average altruism of current active agents (system health indicator)
        if current_active_agents > 0:
            active_agents = [a for a in self.agents if a not in self.never_joined_agents 
                            and a not in self.dropped_out_agents]
            if active_agents:
                avg_system_altruism = np.mean([self.altruism_points_day[a] for a in active_agents])
                # Adjust based on system health (reduced impact)
                health_adjustment = (avg_system_altruism - 0.3) * 0.1  # Reduced from 0.3 to 0.1
                reputation_factor += health_adjustment
        
        # COMBINE ALL FACTORS
        final_altruism = base_altruism + adoption_factor + network_factor + scarcity_factor + reputation_factor
        
        # PHASE-SPECIFIC BOUNDED LIMITS (ensure target ranges)
        if current_adoption_rate < 0.025:  # Innovators: 0.4-0.6
            final_altruism = max(0.4, min(0.6, final_altruism))
        elif current_adoption_rate < 0.16:  # Early adopters: 0.3-0.6
            final_altruism = max(0.3, min(0.6, final_altruism))
        elif current_adoption_rate < 0.50:  # Early majority: 0.3-0.4
            final_altruism = max(0.3, min(0.4, final_altruism))
        elif current_adoption_rate < 0.84:  # Late majority: 0.2-0.4
            final_altruism = max(0.2, min(0.4, final_altruism))
        else:  # Laggards: 0.1-0.3
            final_altruism = max(0.1, min(0.3, final_altruism))
        
        # Small random variation within bounds (±2%)
        phase_range = 0.0
        if current_adoption_rate < 0.025:  # Innovators
            phase_range = 0.2  # 0.6 - 0.4 = 0.2
        elif current_adoption_rate < 0.16:  # Early adopters
            phase_range = 0.3  # 0.6 - 0.3 = 0.3
        elif current_adoption_rate < 0.50:  # Early majority
            phase_range = 0.1  # 0.4 - 0.3 = 0.1
        elif current_adoption_rate < 0.84:  # Late majority
            phase_range = 0.2  # 0.4 - 0.2 = 0.2
        else:  # Laggards
            phase_range = 0.2  # 0.3 - 0.1 = 0.2
        
        random_variation = (random.random() - 0.5) * 0.04 * phase_range  # ±2% of range
        final_altruism += random_variation
        
        # Final clamp to ensure we stay within phase bounds
        if current_adoption_rate < 0.025:  # Innovators: 0.4-0.6
            final_altruism = max(0.4, min(0.6, final_altruism))
        elif current_adoption_rate < 0.16:  # Early adopters: 0.3-0.6
            final_altruism = max(0.3, min(0.6, final_altruism))
        elif current_adoption_rate < 0.50:  # Early majority: 0.3-0.4
            final_altruism = max(0.3, min(0.4, final_altruism))
        elif current_adoption_rate < 0.84:  # Late majority: 0.2-0.4
            final_altruism = max(0.2, min(0.4, final_altruism))
        else:  # Laggards: 0.1-0.3
            final_altruism = max(0.1, min(0.3, final_altruism))
        
        return final_altruism
    
    def update_altruism_scores(self, agent_obj, driver, rider_obj, rider):
        # Calculate detour cost (normalized between 0 and 1)
        detour_cost = agent_obj._calculate_detour_cost(rider_obj)
        max_possible_detour = self.width + self.height  # Example: maximum possible detour in the grid
        normalized_detour_cost = detour_cost / max_possible_detour

        # Driver's altruism increase
        alpha = 0.5  # Scaling factor for driver's altruism increase
        driver_altruism_increase = alpha * self.altruism_points_day[rider] * (1 - normalized_detour_cost)
        
        # Rider's altruism decrease
        beta = 0.7  # Scaling factor for rider's altruism decrease
        rider_altruism_decrease = beta * driver_altruism_increase
        
        # Store the transaction before updating scores
        current_day = getattr(self, 'current_day', 1)  # Default to day 1 if not set
        transaction = {
            'day': current_day,
            'driver': driver,
            'rider': rider,
            'driver_gain': driver_altruism_increase,
            'rider_loss': rider_altruism_decrease,
            'detour_cost': detour_cost,
            'normalized_detour_cost': normalized_detour_cost,
            'rider_initial_points': self.altruism_points_day[rider],
            'driver_initial_points': self.altruism_points_day[driver]
        }
        
        self.altruism_transactions.append(transaction)
        if current_day in self.daily_transactions:
            self.daily_transactions[current_day].append(transaction)

        # Update scores as before
        self.altruism_points_day[driver] += driver_altruism_increase
        self.altruism_points_day[rider] -= rider_altruism_decrease

        # Ensure altruism scores stay within bounds (0 to 1)
        self.altruism_points_day[driver] = min(max(self.altruism_points_day[driver], 0), 1)
        self.altruism_points_day[rider] = min(max(self.altruism_points_day[rider], 0), 1)

        return driver_altruism_increase

    def decide_agent_roles(self):
        # Find the maximum altruism score in the current day
        max_altruism_score = max(self.altruism_points_day[agent] for agent in self.agents)
        print("Max altruism score: ", max_altruism_score)

        for i, agent_obj in enumerate(self.agent_objects):
            agent_id = self.agents[i]
            agent_altruism = self.altruism_points_day[agent_id]

            # If the agent's altruism score is very low, force them to become a driver
            if agent_altruism <= 0.2:
                self.agent_role[agent_id] = 'driver'
            else:
                # Introduce a small random chance (10%) to switch roles regardless of altruism score
                if random.random() < 0.1:
                    self.agent_role[agent_id] = random.choice(['driver', 'rider'])
                else:
                    # Normalize the agent's altruism score relative to the maximum score
                    normalized_altruism = agent_altruism / max_altruism_score

                    # Higher altruism score increases the probability of becoming a rider
                    if random.random() < normalized_altruism:
                        self.agent_role[agent_id] = 'rider'
                    else:
                        self.agent_role[agent_id] = 'driver'

    def track_daily_pickup(self, driver_agent, rider_agent):
        """
        Track a successful pickup for the current day.
        Only counts once per day per driver-rider pair.
        """
        current_day = getattr(self, 'current_day', 1)
        
        # Initialize day if not exists
        if current_day not in self.daily_pickups:
            self.daily_pickups[current_day] = {}
        
        pair = (driver_agent, rider_agent)
        daily_key = (current_day, pair)
        
        # Only count once per day per pair
        if daily_key not in self.current_day_pickups:
            self.current_day_pickups.add(daily_key)
            
            if pair not in self.daily_pickups[current_day]:
                self.daily_pickups[current_day][pair] = 0
            self.daily_pickups[current_day][pair] += 1

    def calculate_birth_probability(self, day, total_active_agents, total_days=None):
        """
        Calculate sophisticated birth probability using multi-factor adoption model.
        
        Factors considered:
        1. S-curve adoption pattern (technology adoption lifecycle)
        2. Network effects with early adopter vs mainstream dynamics
        3. Market saturation and diminishing returns
        4. Urgency deadlines and FOMO (Fear of Missing Out)
        5. System reputation based on active user satisfaction
        
        Args:
            day: Current simulation day
            total_active_agents: Number of currently active agents
            total_days: Total simulation days (for deadline calculations)
            
        Returns:
            Probability of joining (0.0 to 1.0)
        """
        if total_days is None:
            total_days = getattr(self, 'total_simulation_days', 30)
        
        never_joined_count = len(self.never_joined_agents)
        if never_joined_count == 0:
            return 0.0
        
        # 1. S-CURVE ADOPTION PATTERN (Innovation Diffusion Theory)
        # Based on Rogers' adoption curve: Innovators -> Early Adopters -> Early Majority -> Late Majority -> Laggards
        
        # Calculate adoption phase based on current market penetration
        current_adoption_rate = (self.numAgents - never_joined_count) / self.numAgents
        
        if current_adoption_rate < 0.025:  # Innovators phase (2.5%)
            phase_multiplier = 1.2  # High enthusiasm from innovators
            base_prob = 0.04
        elif current_adoption_rate < 0.16:  # Early adopters phase (13.5%)
            phase_multiplier = 1.1
            base_prob = 0.05  # Slightly higher
        elif current_adoption_rate < 0.50:  # Early majority phase (34%)
            phase_multiplier = 1.0  # Peak adoption rate
            base_prob = 0.06  # Slightly higher
        elif current_adoption_rate < 0.84:  # Late majority phase (34%)
            phase_multiplier = 0.8  # Standard adoption
            base_prob = 0.04  # Slightly higher
        else:  # Laggards phase (16%)
            phase_multiplier = 0.5  # Reluctant adopters
            base_prob = 0.02  # Slightly higher
        
        # 2. TIME-BASED URGENCY (Multiple deadlines create urgency waves)
        urgency_factor = 1.0
        
        # Early bird bonus (first 20% of time)
        if day <= total_days * 0.2:
            urgency_factor += 0.3  # Early adoption bonus
        
        # Mid-simulation pressure (40-60% of time)
        elif total_days * 0.4 <= day <= total_days * 0.6:
            urgency_factor += 0.5  # FOMO kicks in
        
        # Final deadline pressure (last 30% of time)
        elif day >= total_days * 0.7:
            deadline_pressure = (day - total_days * 0.7) / (total_days * 0.3)
            urgency_factor += deadline_pressure * 2.0  # Exponential urgency
        
        # 3. SOPHISTICATED NETWORK EFFECTS
        network_factor = 1.0
        
        # Critical mass effect (system becomes viable)
        critical_mass_threshold = self.numAgents * 0.3  # 30% adoption creates viability
        if total_active_agents >= critical_mass_threshold:
            network_factor += 0.8  # Strong positive network effect
        
        # Social proof effect (seeing others succeed)
        if hasattr(self, 'daily_transactions') and self.daily_transactions:
            # More transactions = more visible success
            recent_transaction_count = sum(len(self.daily_transactions.get(d, [])) 
                                        for d in range(max(1, day-3), day+1))
            social_proof = min(0.5, recent_transaction_count / (total_active_agents + 1) * 0.1)
            network_factor += social_proof
        
        # Congestion effect (too many users can be negative)
        congestion_threshold = self.numAgents * 0.85  # 85% adoption creates congestion
        if total_active_agents >= congestion_threshold:
            congestion_penalty = (total_active_agents - congestion_threshold) / (self.numAgents - congestion_threshold)
            network_factor -= congestion_penalty * 0.3
        
        # 4. MARKET SATURATION
        remaining_capacity = never_joined_count / self.numAgents
        saturation_factor = 0.5 + remaining_capacity * 0.5  # Less attractive as fewer spots remain
        
        # 5. SYSTEM REPUTATION (based on current user experience)
        reputation_factor = 1.0
        
        if hasattr(self, 'altruism_points_day'):
            # High average altruism = good reputation
            active_agents = [a for a in self.agents if a not in self.never_joined_agents 
                            and a not in self.dropped_out_agents]
            if active_agents:
                avg_altruism = np.mean([self.altruism_points_day[a] for a in active_agents])
                reputation_factor += (avg_altruism - 0.5) * 0.4  # ±20% based on altruism
        
        # COMBINE ALL FACTORS
        final_probability = (base_prob * 
                            phase_multiplier * 
                            urgency_factor * 
                            network_factor * 
                            saturation_factor * 
                            reputation_factor)
        
        # FINAL BOUNDS AND ADJUSTMENTS
        # Cap between 0.001 and 0.95 (0.1% to 95% per day)
        final_probability = max(0.001, min(0.95, final_probability))
        
        # Log detailed breakdown for debugging (optional)
        if day % 5 == 1:  # Log every 5 days
            print(f"Day {day} Birth Probability Breakdown:")
            print(f"  Base prob: {base_prob:.3f}, Phase: {phase_multiplier:.2f}")
            print(f"  Urgency: {urgency_factor:.2f}, Network: {network_factor:.2f}")
            print(f"  Saturation: {saturation_factor:.2f}, Reputation: {reputation_factor:.2f}")
            print(f"  Final probability: {final_probability:.3f}")
        
        return final_probability
    
    def calculate_dropout_probability(self, altruism_score):
        """
        Calculate dropout probability based on altruism score.
        Higher altruism = lower dropout probability.
        Below 0.2 altruism = much higher dropout probability.
        """
        if altruism_score < 0.0:
            altruism_score = 0.0
        elif altruism_score > 1.0:
            altruism_score = 1.0
            
        if altruism_score < 0.2:
            # Very high dropout probability for low altruism (exponential decay)
            # At 0.0 altruism: ~90% dropout probability
            # At 0.2 altruism: ~20% dropout probability
            dropout_prob = 0.9 * np.exp(-10 * altruism_score)
        else:
            # Lower dropout probability for higher altruism (linear decay)
            # At 0.2 altruism: ~5% dropout probability (reduced from 15%)
            # At 1.0 altruism: ~1% dropout probability (reduced from 2%)
            dropout_prob = 0.05 - 0.04 * (altruism_score - 0.2) / 0.8
            
        return max(0.005, min(0.95, dropout_prob))  # Clamp between 0.5% and 95%

    def decide_agent_roles_with_birth_death(self, day):
        """
        Updated version with dynamic altruism assignment for new agents.
        """
        # Initialize daily tracking
        if day not in self.daily_dropouts:
            self.daily_dropouts[day] = []
        if day not in self.daily_births:
            self.daily_births[day] = []
        
        # Count currently active agents (excluding dropout and never_joined)
        active_agents = [agent for agent in self.agents 
                        if self.agent_role.get(agent, 'active') not in ['dropout', 'never_joined']]
        total_active = len(active_agents)
        
        print(f"Day {day}: Starting with {total_active} active agents")
        
        # PHASE 1: Handle existing agents (dropout decisions) - unchanged
        max_altruism_score = max(self.altruism_points_day[agent] for agent in active_agents) if active_agents else 1.0

        dropout_count = 0
        for i, agent_obj in enumerate(self.agent_objects):
            agent_id = self.agents[i]

            if agent_id in self.never_joined_agents:
                continue

            agent_altruism = self.altruism_points_day[agent_id]
            
            # Calculate dropout probability
            dropout_prob = self.calculate_dropout_probability(agent_altruism)
            
            # Decide if agent drops out
            dropout_decision = random.random()
            if dropout_decision < dropout_prob:
                # Track dropout event
                if self.agent_role[agent_id] != 'dropout':
                    if agent_id not in self.agent_dropout_events:
                        self.agent_dropout_events[agent_id] = []
                    self.agent_dropout_events[agent_id].append(day)
                # Agent drops out
                self.agent_role[agent_id] = 'dropout'
                self.dropped_out_agents.add(agent_id)
                self.daily_dropouts[day].append(agent_id)
                
                if agent_id not in self.dropout_history:
                    self.dropout_history[agent_id] = []
                self.dropout_history[agent_id].append(day)
                dropout_count += 1
                continue
            
            # If agent didn't drop out, remove from dropped out set (they're back)
            if agent_id in self.dropped_out_agents:
                self.dropped_out_agents.remove(agent_id)
                
                # Track return event
                if agent_id not in self.agent_return_events:
                    self.agent_return_events[agent_id] = []
                self.agent_return_events[agent_id].append(day)
                
                # Find corresponding dropout and record reintegration
                if agent_id in self.agent_dropout_events:
                    recent_dropouts = [d for d in self.agent_dropout_events[agent_id] if d < day]
                    if recent_dropouts:
                        dropout_day = max(recent_dropouts)
                        self.reintegration_events.append((agent_id, dropout_day, day, agent_altruism))
            
            # Assign role based on altruism (existing logic)
            if agent_altruism <= 0.2:
                self.agent_role[agent_id] = 'driver'
            else:
                if random.random() < 0.1:
                    self.agent_role[agent_id] = random.choice(['driver', 'rider'])
                else:
                    normalized_altruism = agent_altruism / max_altruism_score
                    if random.random() < normalized_altruism:
                        self.agent_role[agent_id] = 'rider'
                    else:
                        self.agent_role[agent_id] = 'driver'
        
        # PHASE 2: Handle potential new agents (birth decisions) - UPDATED
        birth_count = 0
        if self.never_joined_agents:
            # Update active count after dropouts
            current_active = len([agent for agent in self.agents 
                                if self.agent_role.get(agent, 'active') not in ['dropout', 'never_joined']])
            
            birth_prob = self.calculate_birth_probability(day, current_active)
            
            # Check each never-joined agent for potential joining
            agents_to_join = []
            for agent_id in list(self.never_joined_agents):
                join_decision = random.random()
                
                if join_decision < birth_prob:
                    agents_to_join.append(agent_id)
            
            # Process agents that decided to join
            for agent_id in agents_to_join:
                # Remove from never_joined set
                self.never_joined_agents.remove(agent_id)
                
                # CALCULATE DYNAMIC ALTRUISM SCORE
                dynamic_altruism = self.calculate_new_agent_altruism(day, current_active)
                
                # Set altruism scores
                self.altruism_points_day[agent_id] = dynamic_altruism
                self.altruism_points[agent_id] = dynamic_altruism
                
                # Assign initial role based on dynamic altruism
                if dynamic_altruism <= 0.2:
                    self.agent_role[agent_id] = 'driver'
                else:
                    # Higher altruism agents get random assignment (50-50 chance)
                    self.agent_role[agent_id] = random.choice(['driver', 'rider'])
                
                # Track birth
                self.daily_births[day].append(agent_id)
                self.birth_history[agent_id] = day
                birth_count += 1
                
                print(f"Agent {agent_id} joined system (dynamic altruism: {dynamic_altruism:.3f}, role: {self.agent_role[agent_id]})")
        
        # Print summary
        final_active = len([agent for agent in self.agents 
                        if self.agent_role.get(agent, 'active') not in ['dropout', 'never_joined']])
        print(f"Day {day}: {dropout_count} dropouts, {birth_count} births, {final_active} final active agents, {len(self.never_joined_agents)} never joined agents")

    def get_birth_death_statistics(self, day=None):
        """
        Get statistics about agent births and deaths.
        """
        if day is None:
            # Overall statistics
            total_births = len(self.birth_history)
            total_dropouts = len(self.dropout_history)
            currently_never_joined = len(self.never_joined_agents)
            currently_dropped_out = len(self.dropped_out_agents)
            currently_active = self.numAgents - currently_never_joined - currently_dropped_out
            
            return {
                'total_births': total_births,
                'total_dropouts': total_dropouts,
                'currently_active': currently_active,
                'currently_never_joined': currently_never_joined,
                'currently_dropped_out': currently_dropped_out,
                'daily_births': self.daily_births,
                'daily_dropouts': self.daily_dropouts,
                'birth_history': self.birth_history,
                'dropout_history': self.dropout_history
            }
        else:
            # Day-specific statistics
            day_births = self.daily_births.get(day, [])
            day_dropouts = self.daily_dropouts.get(day, [])
            return {
                'day': day,
                'births': len(day_births),
                'dropouts': len(day_dropouts),
                'birth_agents': day_births,
                'dropout_agents': day_dropouts
            }

    def render(self):

        def tint_image(image, color):
            tinted_image = image.copy()
            tinted_image.fill(color[0:3], special_flags=pygame.BLEND_RGBA_ADD)
            return tinted_image

        def draw_dotted_line(
            screen, color, start_pos, end_pos, dot_radius=2, spacing=5
        ):
            distance = math.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
            dots_count = int(distance / spacing)

            for i in range(dots_count):
                # Interpolate between the start and end position
                x = start_pos[0] + (end_pos[0] - start_pos[0]) * i / dots_count
                y = start_pos[1] + (end_pos[1] - start_pos[1]) * i / dots_count

                # Draw a small circle or rect for each dot
                pygame.draw.circle(screen, color, (x, y), dot_radius)

        pink = (255, 0, 160)
        red = (255, 0, 0)
        peach = (255, 118, 95)
        blue = (0, 0, 255)
        blue_1 = (38, 0, 160)
        dark_yellow = (255, 174, 0)
        green = (0, 128, 0)
        orange = (255, 81, 0)
        color = [dark_yellow, pink, blue, green, peach, orange, red, blue_1]

        
        if not self.renderStarted:
            pygame.init()
            window_size = (self.width * self.gridSize, self.height * self.gridSize)
            self.screen = pygame.display.set_mode(window_size)
            pygame.display.set_caption("Altruistic Ride Sharing")
            self.renderStarted = True

        self.screen.fill((255, 255, 255))

        # Draw the grid
        for x in range(self.width):
            for y in range(self.height):
                rect = pygame.Rect(
                    x * self.gridSize,
                    y * self.gridSize,
                    self.gridSize,
                    self.gridSize,
                )
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        # Filter the drivers from the agent objects
        drivers = [agent for agent in self.agent_objects if isinstance(agent, Driver)]
        riders_picked = list()

        car1 = pygame.transform.scale(
            pygame.image.load("Images/Car-1.png"),
            (self.gridSize // 2, self.gridSize // 2),
        )
        car2 = pygame.transform.scale(
            pygame.image.load("Images/Car-2.png"),
            (self.gridSize // 2, self.gridSize // 2),
        )
        pin = pygame.transform.scale(
            pygame.image.load("Images/Pin.png"),
            (self.gridSize // 2, self.gridSize // 1.5),
        )
        man = pygame.transform.scale(
            pygame.image.load("Images/Man.png"),
            (self.gridSize // 2, self.gridSize // 1.5),
        )
        manpin = pygame.transform.scale(
            pygame.image.load("Images/Man-Pin.png"),
            (self.gridSize // 2, self.gridSize // 2),
        )
        carpin = pygame.transform.scale(
            pygame.image.load("Images/Car-Pin.png"),
            (self.gridSize // 1.5, self.gridSize // 2),
        )
        count = -1
        #draw drivers and their destinations
        for driver in drivers:
            count += 1
            driver.past_positions.append(driver.position)

            driver_color = color[(count) % (len(color))]

            # Draw Car2 if the driver has any rider

            if driver.riders:
                riders_picked.append(driver.riders[0])
                car2_tint = tint_image(car2, driver_color)
                self.screen.blit(
                    car2_tint,
                    (
                        driver.position[0] * self.gridSize + self.gridSize // 4,
                        driver.position[1] * self.gridSize + self.gridSize // 4,
                    ),
                )
                # Draw driver's final destination
                pin_tint = tint_image(pin, driver_color)
                self.screen.blit(
                    pin_tint,
                    (
                        driver.destination[0] * self.gridSize + self.gridSize // 4,
                        driver.destination[1] * self.gridSize + self.gridSize // 4,
                    ),
                )
                rider = driver.riders[0]
                pin_tint = tint_image(pin, self.rider_destinations_colors[rider])
                self.screen.blit(
                    pin_tint,
                    (
                        rider.destination[0] * self.gridSize + self.gridSize // 4,
                        rider.destination[1] * self.gridSize + self.gridSize // 4,
                    ),
                )

            # if driver is alone and at destination
            elif (
                driver.position[0] == driver.destination[0]
                and driver.position[1] == driver.destination[1]
            ):
                carpin_tint = tint_image(carpin, driver_color)
                self.screen.blit(
                    carpin_tint,
                    (
                        driver.position[0] * self.gridSize + self.gridSize // 4,
                        driver.position[1] * self.gridSize + self.gridSize // 4,
                    ),
                )
            # Else Car1
            else:
                car1_tint = tint_image(car1, driver_color)
                self.screen.blit(
                    car1_tint,
                    (
                        driver.position[0] * self.gridSize + self.gridSize // 4,
                        driver.position[1] * self.gridSize + self.gridSize // 4,
                    ),
                )
                # Draw driver's final destination
                pin_tint = tint_image(pin, driver_color)
                self.screen.blit(
                    pin_tint,
                    (
                        driver.destination[0] * self.gridSize + self.gridSize // 4,
                        driver.destination[1] * self.gridSize + self.gridSize // 4,
                    ),
                )

                #drawing dottes line
            if len(driver.past_positions) > 1:
                for i in range(len(driver.past_positions) - 1):
                    prev_x = driver.past_positions[i][0] * self.gridSize + self.gridSize // 2
                    prev_y = driver.past_positions[i][1] * self.gridSize + self.gridSize // 2
                    curr_x = (
                        driver.past_positions[i + 1][0] * self.gridSize + self.gridSize // 2
                    )
                    curr_y = (
                        driver.past_positions[i + 1][1] * self.gridSize + self.gridSize // 2
                    )
                    draw_dotted_line(
                        self.screen,
                        driver_color,
                        (prev_x, prev_y),
                        (curr_x, curr_y),
                        dot_radius=2,  # Size of each dot
                        spacing=10,  # Space between the dots
                    )


        # Draw riders not picked up yet (and their destinations)
        riders = [agent for agent in self.active_riders_list]
        for rider in riders:
            count += 1
            rider_color = color[(count) % (len(color))]
            self.rider_destinations_colors[rider] = rider_color
            if rider in riders_picked:
                pin_tint = tint_image(pin, rider_color)
                self.screen.blit(
                    pin_tint,
                    (
                        rider.destination[0] * self.gridSize + self.gridSize // 4,
                        rider.destination[1] * self.gridSize + self.gridSize // 4,
                    ),
                )
            elif (
                rider.position[0] == rider.destination[0]
                and rider.position[1] == rider.destination[1]
            ):
                manpin_tint = tint_image(manpin, rider_color)
                self.screen.blit(
                    manpin_tint,
                    (
                        rider.destination[0] * self.gridSize + self.gridSize // 4,
                        rider.destination[1] * self.gridSize + self.gridSize // 4,
                    ),
                )
            else:
                man_tint = tint_image(man, rider_color)
                self.screen.blit(
                    man_tint,
                    (
                        rider.position[0] * self.gridSize + self.gridSize // 4,
                        rider.position[1] * self.gridSize + self.gridSize // 4,
                    ),
                )
                pin_tint = tint_image(pin, rider_color)
                self.screen.blit(
                    pin_tint,
                    (
                        rider.destination[0] * self.gridSize + self.gridSize // 4,
                        rider.destination[1] * self.gridSize + self.gridSize // 4,
                    ),
                )
        pygame.event.get()
        pygame.display.update()

        # Capture the current frame
        data = pygame.image.tostring(self.screen, "RGBA")
        image = Image.frombytes("RGBA", self.screen.get_size(), data).convert("RGB")
        self.frames.append(image)

    def save(self, day=None):
        if day:
            save_dir = f"./savedexamples/animation{day}.gif"
        else:
            save_dir = "./savedexamples/animation.gif"

        if self.frames:
            # Ensure all frames are compatible with GIF format
            processed_frames = [frame.convert("RGB").quantize(method=Image.FASTOCTREE) for frame in self.frames]

            # Save as optimized GIF
            processed_frames[0].save(
                save_dir,
                format="GIF",
                append_images=processed_frames[1:],
                save_all=True,
                duration=200,
                loop=0,
                optimize=True,
            )
            print(f"Animation saved at {save_dir}")
        else:
            print("No frames to save. Make sure to call render() before save().")
