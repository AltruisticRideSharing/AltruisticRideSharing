#!/usr/bin/env python3
"""Standalone CLI runner for the OR-Tools baseline.
Place this file under models/or_tools and run directly to execute simulations.
"""
import os
import sys
import time
from datetime import datetime
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ars.env.env import Env
from ars.models.or_tools.or_tools_baseline import ORToolsBaseline


# Metric helpers (adapted from scripts/run_or_tools_3day.py)
def get_agent_individual_distance(agent_obj):
    start = tuple(agent_obj.position)
    dest = tuple(agent_obj.destination)
    try:
        path = agent_obj._dijkstra_path(start, dest)
        return agent_obj._calculate_path_length(path)
    except Exception:
        # Fallback: use straight-line Euclidean distance if pathing fails
        try:
            a = np.array(agent_obj.position, dtype=float)
            b = np.array(agent_obj.destination, dtype=float)
            return float(np.linalg.norm(a - b))
        except Exception:
            return 0.0


def calculate_shared_route_distance(driver_obj, assigned_riders):
    if not assigned_riders:
        return get_agent_individual_distance(driver_obj)

    current_pos = tuple(driver_obj.position)
    total_distance = 0.0
    remaining_riders = list(assigned_riders)
    picked_up_riders = []

    while remaining_riders or picked_up_riders:
        best_distance = float('inf')
        best_action = None

        for rider in list(remaining_riders):
            pickup_pos = tuple(rider.position)
            try:
                path = driver_obj._dijkstra_path(current_pos, pickup_pos)
                distance = driver_obj._calculate_path_length(path)
            except Exception:
                # Fallback to Euclidean distance
                try:
                    distance = float(np.linalg.norm(np.array(current_pos, dtype=float) - np.array(pickup_pos, dtype=float)))
                except Exception:
                    distance = 0.0
            if distance < best_distance:
                best_distance = distance
                best_action = ('pickup', rider, pickup_pos, path)

        for rider in list(picked_up_riders):
            dropoff_pos = tuple(rider.destination)
            try:
                path = driver_obj._dijkstra_path(current_pos, dropoff_pos)
                distance = driver_obj._calculate_path_length(path)
            except Exception:
                try:
                    distance = float(np.linalg.norm(np.array(current_pos, dtype=float) - np.array(dropoff_pos, dtype=float)))
                except Exception:
                    distance = 0.0
            if distance < best_distance:
                best_distance = distance
                best_action = ('dropoff', rider, dropoff_pos, path)

        if best_action is None:
            break

        action_type, rider, next_pos, path = best_action
        total_distance += driver_obj._calculate_path_length(path)
        current_pos = next_pos

        if action_type == 'pickup':
            remaining_riders.remove(rider)
            picked_up_riders.append(rider)
        else:
            picked_up_riders.remove(rider)

    try:
        final_path = driver_obj._dijkstra_path(current_pos, tuple(driver_obj.destination))
        total_distance += driver_obj._calculate_path_length(final_path)
    except Exception:
        try:
            total_distance += float(np.linalg.norm(np.array(current_pos, dtype=float) - np.array(driver_obj.destination, dtype=float)))
        except Exception:
            total_distance += 0.0

    return total_distance


def calculate_detour_factors(assignments, drivers):
    detour_factors = []
    drivers_dict = {name: obj for _, name, obj in drivers}

    for driver_name, driver_obj in drivers_dict.items():
        individual_distance = get_agent_individual_distance(driver_obj)

        if driver_name in assignments:
            assigned_riders = [r_obj for _, r_obj in assignments[driver_name]]
            shared_distance = calculate_shared_route_distance(driver_obj, assigned_riders)
        else:
            shared_distance = individual_distance

        if individual_distance > 0:
            detour_factors.append(shared_distance / individual_distance)

    return {
        'mean': float(np.mean(detour_factors)) if detour_factors else 0.0,
        'std': float(np.std(detour_factors)) if detour_factors else 0.0,
        'all': detour_factors,
        'count': len(detour_factors)
    }


def calculate_average_trip_time(assignments, drivers, riders):
    driver_trip_times = []
    drivers_dict = {name: obj for _, name, obj in drivers}

    for driver_name, driver_obj in drivers_dict.items():
        if driver_name in assignments:
            assigned_riders = [r_obj for _, r_obj in assignments[driver_name]]
            trip_distance = calculate_shared_route_distance(driver_obj, assigned_riders)
        else:
            trip_distance = get_agent_individual_distance(driver_obj)

        trip_time = (trip_distance / 25.0) * 60.0
        driver_trip_times.append(trip_time)

    return {
        'mean': float(np.mean(driver_trip_times)) if driver_trip_times else 0.0,
        'std': float(np.std(driver_trip_times)) if driver_trip_times else 0.0,
        'all': driver_trip_times,
        'count': len(driver_trip_times)
    }


def calculate_vehicle_utilization(assignments, drivers):
    utilizations = []
    drivers_dict = {name: obj for _, name, obj in drivers}

    for driver_name, driver_obj in drivers_dict.items():
        if driver_name in assignments:
            occupancy = 1 + len(assignments[driver_name])
        else:
            occupancy = 1
        utilizations.append(occupancy)

    return {
        'mean': float(np.mean(utilizations)) if utilizations else 0.0,
        'std': float(np.std(utilizations)) if utilizations else 0.0,
        'all': utilizations,
        'max_capacity': 4,
        'count': len(utilizations)
    }


def calculate_per_agent_distances(assignments, drivers, riders):
    driver_distances = []
    rider_distances = []
    drivers_dict = {name: obj for _, name, obj in drivers}
    riders_dict = {name: obj for _, name, obj in riders}
    assigned_riders = set()

    for driver_name, driver_obj in drivers_dict.items():
        if driver_name in assignments:
            assigned_rider_objs = [r_obj for _, r_obj in assignments[driver_name]]
            distance = calculate_shared_route_distance(driver_obj, assigned_rider_objs)
            for rider_name, _ in assignments[driver_name]:
                assigned_riders.add(rider_name)
        else:
            distance = get_agent_individual_distance(driver_obj)

        driver_distances.append(distance)

    for rider_name, rider_obj in riders_dict.items():
        individual_distance = get_agent_individual_distance(rider_obj)
        rider_distances.append(individual_distance)

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


def calculate_distance_comparison(assignments, drivers, riders):
    total_individual_distance = 0.0
    total_shared_distance = 0.0

    all_drivers = {name: obj for _, name, obj in drivers}
    all_riders = {name: obj for _, name, obj in riders}

    for _, driver_name, driver_obj in drivers:
        driver_individual = get_agent_individual_distance(driver_obj)
        total_individual_distance += driver_individual

    for _, rider_name, rider_obj in riders:
        rider_individual = get_agent_individual_distance(rider_obj)
        total_individual_distance += rider_individual

    assigned_rider_names = set()
    for driver_name, rider_assignments in assignments.items():
        driver_obj = all_drivers[driver_name]
        assigned_rider_objs = [r_obj for _, r_obj in rider_assignments]
        shared_route_distance = calculate_shared_route_distance(driver_obj, assigned_rider_objs)
        total_shared_distance += shared_route_distance
        for rider_name, _ in rider_assignments:
            assigned_rider_names.add(rider_name)

    for driver_name, driver_obj in all_drivers.items():
        if driver_name not in assignments:
            total_shared_distance += get_agent_individual_distance(driver_obj)

    for rider_name, rider_obj in all_riders.items():
        if rider_name not in assigned_rider_names:
            total_shared_distance += get_agent_individual_distance(rider_obj)

    distance_saved = total_individual_distance - total_shared_distance
    savings_percentage = (distance_saved / max(total_individual_distance, 1.0)) * 100

    return {
        'total_individual_distance': float(total_individual_distance),
        'total_shared_distance': float(total_shared_distance),
        'distance_saved': float(distance_saved),
        'savings_percentage': float(savings_percentage),
        'assigned_riders_count': len(assigned_rider_names),
        'total_riders_count': len(all_riders),
        'assignment_rate': float(len(assigned_rider_names) / max(len(all_riders), 1) * 100)
    }


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_table(path, header, rows):
    with open(path, 'w') as f:
        f.write(header + '\n')
        for r in rows:
            f.write('\t'.join(str(x) for x in r) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Run OR-Tools baseline simulation')
    parser.add_argument('--num-agents', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='100_agents')
    parser.add_argument('--days', type=int, default=3)
    parser.add_argument('--height', type=int, default=15)
    parser.add_argument('--width', type=int, default=15)
    parser.add_argument('--altruism-distribution', type=str, default='uniform')
    parser.add_argument('--altruism-mean', type=float, default=0.5)
    parser.add_argument('--altruism-std', type=float, default=0.15)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    env = Env(
        height=args.height,
        width=args.width,
        numAgents=args.num_agents,
        dataset_folder=args.dataset,
        initial_active_agents=args.num_agents,
        altruism_distribution=args.altruism_distribution,
        altruism_mean=args.altruism_mean,
        altruism_std=args.altruism_std,
        total_days=args.days
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"or_tools_{args.num_agents}agents_{args.altruism_distribution}_{timestamp}"
    save_dir = args.save_dir or os.path.join('results', 'or_tools', exp_name)
    dat_dir = os.path.join(save_dir, 'dat')
    ensure_dir(dat_dir)

    assignments_rows = []
    distance_rows = []
    detour_rows = []
    avg_trip_time_rows = []
    vehicle_util_rows = []
    per_agent_dist_rows = []
    altruism_rows = []
    population_rows = []

    for day in range(1, args.days + 1):
        print(f"Running OR-Tools baseline for day {day}...")
        env.reset_day(day, enable_birth_death=False)

        solver = ORToolsBaseline(env, verbose=args.verbose)
        solution = solver.optimize()
        assignments = solver.get_assignments(solution)

        drivers = solution.get('drivers', [])
        riders = solution.get('riders', [])
        objective = solution.get('objective', 0.0) or 0.0
        total_reward = -objective

        num_assigned = sum(len(v) for v in assignments.values())

        distance_comp = calculate_distance_comparison(assignments, drivers, riders)
        detour = calculate_detour_factors(assignments, drivers)
        avg_trip = calculate_average_trip_time(assignments, drivers, riders)
        util = calculate_vehicle_utilization(assignments, drivers)
        per_agent = calculate_per_agent_distances(assignments, drivers, riders)
        altruism_stats = env.get_altruism_distribution_stats()

        assignments_rows.append((day, num_assigned, total_reward, -total_reward))

        distance_rows.append((day,
                              distance_comp['total_individual_distance'],
                              distance_comp['total_shared_distance'],
                              distance_comp['distance_saved'],
                              distance_comp['savings_percentage'],
                              distance_comp['assigned_riders_count'],
                              distance_comp['total_riders_count'],
                              distance_comp['assignment_rate']))

        detour_rows.append((day, detour['mean'], detour['std'], detour['count']))
        avg_trip_time_rows.append((day, avg_trip['mean'], avg_trip['std'], avg_trip['count']))
        vehicle_util_rows.append((day, util['mean'], util['std'], util['count']))
        per_agent_dist_rows.append((day, per_agent['mean'], per_agent['std'], per_agent['total_agents']))

        altruism_rows.append((day, altruism_stats['mean'], altruism_stats['std'], altruism_stats['min'], altruism_stats['max']))

        active_agents = len([a for a in env.agents if env.agent_role.get(a, 'active') not in ['dropout', 'never_joined']])
        num_drivers = len([a for a in env.agents if env.agent_role.get(a, 'active') == 'driver'])
        num_riders = len([a for a in env.agents if env.agent_role.get(a, 'active') == 'rider'])
        population_rows.append((day, active_agents, num_drivers, num_riders))

        time.sleep(0.05)

    save_table(os.path.join(dat_dir, 'assignments_over_time.dat'), 'day assignments reward fitness', assignments_rows)
    save_table(os.path.join(dat_dir, 'distance_over_time.dat'), 'day total_individual_distance total_shared_distance distance_saved savings_percentage assigned_riders_count total_riders_count assignment_rate', distance_rows)
    save_table(os.path.join(dat_dir, 'detour_over_time.dat'), 'day mean std count', detour_rows)
    save_table(os.path.join(dat_dir, 'avg_trip_time_over_time.dat'), 'day mean std count', avg_trip_time_rows)
    save_table(os.path.join(dat_dir, 'vehicle_utilization_over_time.dat'), 'day mean std count', vehicle_util_rows)
    save_table(os.path.join(dat_dir, 'per_agent_distances_over_time.dat'), 'day mean std total_agents', per_agent_dist_rows)
    save_table(os.path.join(dat_dir, 'altruism_evolution_over_time.dat'), 'day mean std min max', altruism_rows)
    save_table(os.path.join(dat_dir, 'population_dynamics_over_time.dat'), 'day active_agents drivers riders', population_rows)

    print(f"Saved OR-Tools results to {dat_dir}")


if __name__ == '__main__':
    main()
