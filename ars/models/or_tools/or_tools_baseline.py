import numpy as np
from typing import Dict, List, Tuple
from ortools.linear_solver import pywraplp
import concurrent.futures
import os


class ORToolsBaseline:
    """Simple OR-Tools baseline: assign riders to drivers using a linear assignment
    approximation. Each rider can be assigned to at most one driver. Each driver
    has a capacity (from env.max_capacity). Cost is approximated as the extra
    distance a driver would travel to pick up and drop a single rider (pairwise
    approximation, interactions ignored).
    """

    def __init__(self, env, verbose: bool = False):
        self.env = env
        self.verbose = verbose
        self.drivers, self.riders = self._get_active_agents()

    def _get_active_agents(self) -> Tuple[List, List]:
        drivers, riders = [], []
        for i, agent in enumerate(self.env.agents):
            if self.env.agent_role.get(agent, 'active') in ['dropout', 'never_joined']:
                continue

            agent_obj = self.env.agent_objects[i]
            role = self.env.agent_role.get(agent, 'active')
            if role == 'driver':
                drivers.append((i, agent, agent_obj))
            elif role == 'rider':
                riders.append((i, agent, agent_obj))

        return drivers, riders

    def _pairwise_cost(self, driver_obj, rider_obj) -> float:
        """Approximate incremental cost of assigning `rider_obj` to `driver_obj`.
        Uses dijkstra path length helpers on agent objects. If any path is missing
        or an error occurs, returns a large cost.
        """
        try:
            start = tuple(driver_obj.position)
            driver_dest = tuple(driver_obj.destination)
            rider_start = tuple(rider_obj.position)
            rider_dest = tuple(rider_obj.destination)

            # Distance driver would travel alone
            path_driver_alone = driver_obj._dijkstra_path(start, driver_dest)
            dist_driver_alone = driver_obj._calculate_path_length(path_driver_alone)

            # Approximate the detour for taking this rider alone (pickup then to driver's dest)
            path_to_pickup = driver_obj._dijkstra_path(start, rider_start)
            path_pickup_to_driver_dest = driver_obj._dijkstra_path(rider_start, driver_dest)
            detour = driver_obj._calculate_path_length(path_to_pickup) + driver_obj._calculate_path_length(path_pickup_to_driver_dest) - dist_driver_alone

            # If detour is negative or tiny due to graph quirks, clip to zero
            detour = max(0.0, detour)

            # Rider travel distance (benefit to be assigned)
            path_rider = driver_obj._dijkstra_path(rider_start, rider_dest)
            rider_travel = driver_obj._calculate_path_length(path_rider) if path_rider is not None else 0.0

            # Use environment alpha (fallback to 0.4) to balance rider benefit vs driver detour
            alpha = getattr(self.env, 'alpha', 0.4)

            # Net cost: weighted detour minus weighted rider benefit. Minimizer will
            # prefer assignments that reduce this net cost (can become negative).
            net_cost = (1.0 - alpha) * detour - alpha * rider_travel

            return float(net_cost)
        except Exception as e:
            # On failure return a large cost so solver avoids this pair
            return 1e6

    def _build_pairwise_costs(self, drivers, riders) -> Dict[Tuple[int, int], float]:
        """Compute pairwise costs for all driver-rider index pairs in parallel.

        Returns a dict keyed by (driver_index, rider_index) with float costs.
        """
        costs: Dict[Tuple[int, int], float] = {}
        pairs = []
        for d_idx, (env_idx_d, d_name, d_obj) in enumerate(drivers):
            for r_idx, (env_idx_r, r_name, r_obj) in enumerate(riders):
                pairs.append((d_idx, r_idx, d_obj, r_obj))

        def _compute(pair):
            d_i, r_i, d_obj, r_obj = pair
            try:
                c = self._pairwise_cost(d_obj, r_obj)
            except Exception:
                c = 1e6
            return (d_i, r_i, c)

        max_workers = min(16, (os.cpu_count() or 2) * 2)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            for d_i, r_i, c in ex.map(_compute, pairs):
                costs[(d_i, r_i)] = c

        return costs

    def optimize(self) -> Dict:
        """Build and solve an integer linear program to assign riders to drivers."""
        if not self.drivers or not self.riders:
            return {'success': False, 'assignment_matrix': None, 'drivers': self.drivers, 'riders': self.riders, 'objective': 0.0}

        num_drivers = len(self.drivers)
        num_riders = len(self.riders)

        solver = pywraplp.Solver.CreateSolver('CBC')
        if solver is None:
            raise RuntimeError('ORTools solver not available')

        # Variables x[d][r] in {0,1}
        x = {}
        for d in range(num_drivers):
            for r in range(num_riders):
                x[d, r] = solver.IntVar(0, 1, f'x_{d}_{r}')

        # Constraints: each rider assigned at most once
        for r in range(num_riders):
            solver.Add(sum(x[d, r] for d in range(num_drivers)) <= 1)

        # Each driver capacity constraint
        capacity = getattr(self.env, 'max_capacity', 4)
        for d in range(num_drivers):
            solver.Add(sum(x[d, r] for r in range(num_riders)) <= capacity)

        # Objective: minimize total detour (approximate)
        costs = self._build_pairwise_costs(self.drivers, self.riders)

        # Debug: inspect costs distribution to diagnose why solver returns no assignments
        try:
            cost_vals = list(costs.values())
            min_cost = min(cost_vals) if cost_vals else None
            max_cost = max(cost_vals) if cost_vals else None
            neg_count = sum(1 for v in cost_vals if v < 0)
            print(f"ORTools pairwise cost stats: min={min_cost}, max={max_cost}, negative_count={neg_count}, total_pairs={len(cost_vals)}")
        except Exception:
            print("Failed to compute cost stats for ORTools baseline")

        objective = solver.Objective()
        for (d, r), var in x.items():
            # if a cost wasn't computed for this pair, fall back to large cost
            objective.SetCoefficient(var, costs.get((d, r), 1e6))
        objective.SetMinimization()

        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL and status != pywraplp.Solver.FEASIBLE:
            return {'success': False, 'assignment_matrix': None, 'drivers': self.drivers, 'riders': self.riders, 'objective': None}

        # Debug: report solver status, objective and selected variable count
        try:
            total_obj = objective.Value()
        except Exception:
            total_obj = None

        try:
            selected_count = sum(1 for (d, r), var in x.items() if var.solution_value() > 0.5)
        except Exception:
            selected_count = 0

        print(f"ORTools solver status: {status}, drivers={num_drivers}, riders={num_riders}, objective={total_obj}, selected_assignments={selected_count}")

        # Build assignment matrix: fill slots sequentially up to capacity
        assignment_matrix = np.full((num_drivers, capacity), -1, dtype=int)
        for d in range(num_drivers):
            assigned = []
            for r in range(num_riders):
                if x[d, r].solution_value() > 0.5:
                    assigned.append(r)

            for slot_idx, r_idx in enumerate(assigned[:capacity]):
                assignment_matrix[d, slot_idx] = int(r_idx)

        total_obj = objective.Value()

        return {
            'success': True,
            'assignment_matrix': assignment_matrix,
            'drivers': self.drivers,
            'riders': self.riders,
            'objective': float(total_obj)
        }

    def get_assignments(self, solution: Dict) -> Dict:
        assignments = {}
        if not solution.get('success'):
            return assignments

        matrix = solution['assignment_matrix']
        drivers = solution['drivers']
        riders = solution['riders']

        for d_idx, (env_idx, d_name, d_obj) in enumerate(drivers):
            assigned_list = []
            for slot in range(matrix.shape[1]):
                r_idx = int(matrix[d_idx, slot])
                if 0 <= r_idx < len(riders):
                    r_env_idx, r_name, r_obj = riders[r_idx]
                    assigned_list.append((r_name, r_obj))

            if assigned_list:
                assignments[d_name] = assigned_list

        return assignments
