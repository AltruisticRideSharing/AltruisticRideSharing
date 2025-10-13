import numpy as np
import copy
from typing import Dict, List, Tuple
from tqdm import tqdm
from .pso_particle import PSOParticle
from .pso_evaluator import PSOEvaluator

class PSOBaseline:
    """Simplified PSO for global ride assignments"""
    
    def __init__(self, env, num_particles: int = 30, max_iterations: int = 50,
                 w: float = 0.5, c1: float = 2.0, c2: float = 1.5, 
                 alpha: float = 0.5, beta: float = 0.3, theta: float = 5.0,
                 assignment_bonus_weight: float = 2.0, load_balance_weight: float = 1.0,
                 unassigned_penalty_weight: float = 1.0, max_detour: float = 10.0,
                 verbose: bool = True):
        
        self.env = env
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.w, self.c1, self.c2 = w, c1, c2
        self.alpha, self.beta, self.theta = alpha, beta, theta
        self.assignment_bonus_weight = assignment_bonus_weight
        self.load_balance_weight = load_balance_weight
        self.unassigned_penalty_weight = unassigned_penalty_weight
        self.max_detour = max_detour  # NEW: Maximum allowed detour
        
        # Get active agents
        self.drivers, self.riders = self._get_active_agents()
        
        if self.verbose:
            print(f"PSO: {len(self.drivers)} drivers, {len(self.riders)} riders")
            print(f"Parameters: α={alpha}, β={beta}, θ={theta}, max_detour={max_detour}")
        
        # Initialize evaluator with all parameters including max_detour
        self.evaluator = PSOEvaluator(env, alpha, beta, theta, max_detour)  # UPDATED
        
        # Pass weights to evaluator
        self.evaluator.assignment_bonus_weight = assignment_bonus_weight
        self.evaluator.load_balance_weight = load_balance_weight
        self.evaluator.unassigned_penalty_weight = unassigned_penalty_weight
        
        self.particles = []
        self.global_best_position = None
        self.global_best_fitness = float('-inf')
        
        self._initialize_swarm()
    
    def _get_active_agents(self) -> Tuple[List, List]:
        """Get active drivers and riders"""
        drivers, riders = [], []
        
        for i, agent in enumerate(self.env.agents):
            if self.env.agent_role.get(agent, 'active') in ['dropout', 'never_joined']:
                continue
                
            agent_obj = self.env.agent_objects[i]
            
            if self.env.agent_role.get(agent, 'active') == 'driver':
                drivers.append((i, agent, agent_obj))
            elif self.env.agent_role.get(agent, 'active') == 'rider':
                riders.append((i, agent, agent_obj))

        return drivers, riders
    
    def _initialize_swarm(self):
        """Initialize particle swarm with smart geographic initialization"""
        if not self.drivers or not self.riders:
            return
            
        for _ in range(self.num_particles):
            # Pass driver and rider data for geographic initialization
            particle = PSOParticle(
                len(self.drivers), 
                len(self.riders), 
                self.env.max_capacity,
                drivers_data=self.drivers,    # Pass the actual data
                riders_data=self.riders       # Pass the actual data
            )
            
            self.particles.append(particle)
        
        if self.verbose and self.particles:
            init_assignments = [p.get_assignment_count() for p in self.particles]
            print(f"Initial assignments (smart geographic): {np.mean(init_assignments):.1f} ± {np.std(init_assignments):.1f}")
        
        # Initialize global best
        self.global_best_position = np.full((len(self.drivers), self.env.max_capacity), -1, dtype=int)
    
    def optimize(self) -> Dict:
        """Run PSO optimization"""
        if not self.particles:
            return {'success': False, 'best_assignment': None, 'best_fitness': 0.0}
        
        best_assignments_over_time = []
        best_fitness_over_time = []
        
        # Evaluate initial population
        for particle in self.particles:
            fitness = self.evaluator.evaluate_particle(particle, self.drivers, self.riders)
            particle.fitness = fitness
            particle.update_best(fitness)
            
            if particle.best_fitness > self.global_best_fitness:
                self.global_best_fitness = particle.best_fitness
                self.global_best_position = copy.deepcopy(particle.best_position)
        
        # Create progress bar
        pbar = tqdm(range(self.max_iterations), desc="PSO Optimization", leave=False)
        
        for iteration in pbar:
            # Update swarm first
            for particle in self.particles:
                particle.update_velocity(self.global_best_position, self.w, self.c1, self.c2)
                particle.update_position()
            
            # Then evaluate
            for particle in self.particles:
                fitness = self.evaluator.evaluate_particle(particle, self.drivers, self.riders)
                particle.fitness = fitness
                particle.update_best(fitness)
                
                # Update global best
                if particle.best_fitness > self.global_best_fitness:
                    self.global_best_fitness = particle.best_fitness
                    self.global_best_position = copy.deepcopy(particle.best_position)
            
            # Track progress
            current_assignments = [p.get_assignment_count() for p in self.particles]
            avg_assignments = np.mean(current_assignments)
            best_assignments = np.sum(self.global_best_position >= 0)
            
            # Update tqdm description
            pbar.set_description(f"PSO - Assignments: {avg_assignments:.1f} | "
                               f"Best Fitness: {self.global_best_fitness:.1f}")
            
            best_assignments_over_time.append(avg_assignments)
            best_fitness_over_time.append(self.global_best_fitness)
        
        pbar.close()
        
        if self.verbose:
            print(f"Final: Best solution has {np.sum(self.global_best_position >= 0)} assignments")
            print(f"Final: Best fitness = {self.global_best_fitness:.2f}")
        
        return {
            'success': True,
            'best_assignment': self.global_best_position,
            'best_fitness': self.global_best_fitness,
            'drivers': self.drivers,
            'riders': self.riders,
            'assignment_trend': best_assignments_over_time,
            'fitness_trend': best_fitness_over_time
        }
    
    def get_assignments(self, solution: Dict) -> Dict:
        """Convert PSO solution to driver-rider assignments"""
        assignments = {}
        
        if not solution['success']:
            return assignments
        
        best_assignment = solution['best_assignment']
        drivers = solution['drivers']
        riders = solution['riders']
        
        for driver_idx, (driver_env_idx, driver_name, driver_obj) in enumerate(drivers):
            driver_assignments = []
            
            for slot in range(self.env.max_capacity):
                rider_id = best_assignment[driver_idx, slot]
                
                if 0 <= rider_id < len(riders):
                    rider_env_idx, rider_name, rider_obj = riders[rider_id]
                    driver_assignments.append((rider_name, rider_obj))
            
            if driver_assignments:
                assignments[driver_name] = driver_assignments
        
        return assignments