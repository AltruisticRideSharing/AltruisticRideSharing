import numpy as np
import os
import argparse
import pickle
import random
import time
from ars.env.agent import Driver, Rider
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import warnings
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio
warnings.filterwarnings('ignore')

def get_results_dir(arglist):
    """Single source of truth for the per-run results directory.

    Grid-first layout:  results/grid<N>/<model>/<agents>agents_<dist>_<mode>/

    <model> comes from arglist.model_label when set by the calling eval script
    (maac / maddpg / oracle / maddpg_shared); otherwise it falls back to the raw
    model_type. grid_size defaults to 15 if absent (legacy single-grid runs).
    """
    # Explicit override (used by ablation/sensitivity sweeps to isolate runs).
    override = getattr(arglist, "results_dir", None)
    if override:
        return override
    mode = "bd" if getattr(arglist, "enable_birth_death", False) else "fixed"
    model = getattr(arglist, "model_label", None)
    if not model:
        model = "maac" if getattr(arglist, "model_type", None) == "attention" else "maddpg"
    grid = getattr(arglist, "grid_size", 15)
    config = f"{arglist.num_agents}agents_{arglist.altruism_distribution}_{mode}"
    return os.path.join("results", f"grid{grid}", model, config)

def calculate_distance(path, weight_matrix):
    """Calculate the total distance of a path using the weight matrix."""
    total_distance = 0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i+1]
        
        # Determine direction index
        if x2 - x1 == -1:  # Moving up
            dir_idx = 0
        elif x2 - x1 == 1:  # Moving down
            dir_idx = 1
        elif y2 - y1 == -1:  # Moving left
            dir_idx = 2
        elif y2 - y1 == 1:  # Moving right
            dir_idx = 3
        else:
            continue  # Skip if not a valid direction (should not happen in a proper path)
        
        total_distance += float(weight_matrix[x1, y1, dir_idx])
    return total_distance

def calculate_time(path, time_weight_matrix):
    """Calculate the total time of a path using the time weight matrix."""
    total_time = 0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i+1]
        
        # Determine direction index
        if x2 - x1 == -1:  # Moving up
            dir_idx = 0
        elif x2 - x1 == 1:  # Moving down
            dir_idx = 1
        elif y2 - y1 == -1:  # Moving left
            dir_idx = 2
        elif y2 - y1 == 1:  # Moving right
            dir_idx = 3
        else:
            continue  # Skip if not a valid direction
        
        total_time += float(time_weight_matrix[x1, y1, dir_idx])
    return total_time

def get_no_sharing_paths(env):
    """Calculate direct paths for all agents without any ride-sharing."""
    paths_no_sharing = {}
    
    for i, agent in enumerate(env.agents):
        # Skip dropped out and never joined agents
        if env.agent_role[agent] in ['dropout', 'never_joined']:
            paths_no_sharing[agent] = []
            continue
            
        agent_obj = env.agent_objects[i]
        # Calculate direct path from start to destination
        direct_path = agent_obj._dijkstra_path(agent_obj.start_position, agent_obj.destination)
        paths_no_sharing[agent] = direct_path
    
    return paths_no_sharing

def plot_agent_distance_boxplots(individual_distances_by_day, direct_distances_by_day, env, arglist):
    """
    Boxplots of per-agent distance travelled per day (sharing vs non-sharing), with mean lines.
    Only includes active agents.
    """
    days = sorted(individual_distances_by_day.keys())
    sharing_data = []
    non_sharing_data = []
    sharing_means = []
    non_sharing_means = []
    for day in days:
        # Filter out inactive agents
        sharing_vals = [dist for agent, dist in individual_distances_by_day[day].items() 
                       if env.agent_role.get(agent, 'active') not in ['dropout', 'never_joined']]
        non_sharing_vals = [dist for agent, dist in direct_distances_by_day[day].items() 
                           if env.agent_role.get(agent, 'active') not in ['dropout', 'never_joined']]
        
        sharing_data.append(sharing_vals)
        non_sharing_data.append(non_sharing_vals)
        sharing_means.append(np.mean(sharing_vals) if sharing_vals else 0)
        non_sharing_means.append(np.mean(non_sharing_vals) if non_sharing_vals else 0)

    plt.figure(figsize=(14, 7))
    b1 = plt.boxplot(sharing_data, positions=np.array(days)-0.15, widths=0.25, patch_artist=True,
                     boxprops=dict(facecolor="blue", alpha=0.5), medianprops=dict(color="black"))
    b2 = plt.boxplot(non_sharing_data, positions=np.array(days)+0.15, widths=0.25, patch_artist=True,
                     boxprops=dict(facecolor="red", alpha=0.5), medianprops=dict(color="black"))
    # Plot mean lines
    plt.plot(days, sharing_means, color="blue", marker="o", linestyle="--", label="Sharing Mean")
    plt.plot(days, non_sharing_means, color="red", marker="o", linestyle="--", label="Non-Sharing Mean")
    plt.xticks(days)
    plt.xlabel("Day")
    plt.ylabel("Distance")
    plt.title("Per-Agent Distance Travelled per Day (Sharing vs Non-Sharing) - Active Agents Only")
    plt.legend([b1["boxes"][0], b2["boxes"][0], plt.Line2D([], [], color="blue", linestyle="--"),
                plt.Line2D([], [], color="red", linestyle="--")],
               ["Sharing", "Non-Sharing", "Sharing Mean", "Non-Sharing Mean"])
    plt.tight_layout()
    results_dir = get_results_dir(arglist)
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "per_agent_distance_boxplot.png"))
    plt.close()

def plot_altruism_boxplots(altruism_points_by_day, env, arglist):
    """
    Boxplots of per-agent altruism scores per day.
    Only includes active agents.
    """
    days = sorted(altruism_points_by_day.keys())
    altruism_data = []
    for day in days:
        # Filter out inactive agents
        day_altruism = [score for agent, score in altruism_points_by_day[day].items() 
                       if env.agent_role.get(agent, 'active') not in ['dropout', 'never_joined']]
        altruism_data.append(day_altruism)

    plt.figure(figsize=(14, 7))
    b = plt.boxplot(altruism_data, positions=days, widths=0.5, patch_artist=True,
                    boxprops=dict(facecolor="green", alpha=0.5), medianprops=dict(color="black"))
    plt.xticks(days)
    plt.xlabel("Day")
    plt.ylabel("Altruism Score")
    plt.title("Per-Agent Altruism Scores per Day - Active Agents Only")
    plt.tight_layout()
    results_dir = get_results_dir(arglist)
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "per_agent_altruism_boxplot.png"))
    plt.close()

def plot_detour_factor_and_trip_time(detour_distances_by_day, individual_times_by_day, direct_distances_by_day, direct_times_by_day, env, arglist):
    """
    Plot detour factor (distance-based, boxplot per day) and average trip time per day.
    Only includes active agents.
    """
    days = sorted(detour_distances_by_day.keys())
    detour_factors_per_day = []
    avg_trip_times_sharing = []
    avg_trip_times_no_sharing = []

    for day in days:
        detour_factors = []
        trip_times_sharing = []
        trip_times_no_sharing = []
        for agent in detour_distances_by_day[day]:
            # Skip inactive agents
            if env.agent_role.get(agent, 'active') in ['dropout', 'never_joined']:
                continue
                
            if direct_distances_by_day[day][agent] > 0 and detour_distances_by_day[day][agent] > 0:
                detour_factor = detour_distances_by_day[day][agent] / direct_distances_by_day[day][agent]
                detour_factors.append(detour_factor)
            sharing_time = individual_times_by_day[day][agent]
            direct_time = direct_times_by_day[day][agent]
            if sharing_time > 0:
                trip_times_sharing.append(sharing_time)
            if direct_time > 0:
                trip_times_no_sharing.append(direct_time)
        detour_factors_per_day.append(detour_factors)
        avg_trip_times_sharing.append(np.mean(trip_times_sharing) if trip_times_sharing else 0)
        avg_trip_times_no_sharing.append(np.mean(trip_times_no_sharing) if trip_times_no_sharing else 0)

    # Plot Detour Factor (boxplot only)
    plt.figure(figsize=(12, 7))
    b = plt.boxplot(detour_factors_per_day, positions=days, widths=0.5, patch_artist=True,
                    boxprops=dict(facecolor="orange", alpha=0.5), medianprops=dict(color="black"))
    plt.xticks(days)
    plt.xlabel("Day")
    plt.ylabel("Detour Factor (Detour Distance / Direct Distance)")
    plt.title("Per-Driver Detour Factor per Day - Active Agents Only")
    plt.tight_layout()
    results_dir = get_results_dir(arglist)
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "detour_factor_boxplot_per_day.png"))
    plt.close()

    # Plot Average Trip Time (line plot)
    plt.figure(figsize=(10, 6))
    plt.plot(days, avg_trip_times_sharing, marker="o", linestyle="-", color="blue", label="Sharing")
    plt.plot(days, avg_trip_times_no_sharing, marker="o", linestyle="--", color="red", label="No Sharing")
    plt.xlabel("Day")
    plt.ylabel("Average Trip Time")
    plt.title("Average Trip Time per Day - Active Agents Only")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "average_normalized_trip_time_per_day.png"))
    plt.close()

def write_aggregated_results(results_dir, model_label, arglist,
                             distance_total_sharing_by_day, distance_total_no_sharing_by_day,
                             acceptance_rates_by_day=None, vehicle_utilization_by_day=None,
                             multidimensional_stats=None, winner_loser_stats=None,
                             traffic_metrics=None,
                             detour_distances_by_day=None, direct_distances_by_day=None,
                             individual_times_by_day=None, reintegration_metrics=None):
    """Write a compact, uniform aggregated-results summary (results_summary.txt)
    for one run. Designed to be comparable across models/grids in the ablation.
    Any metric whose inputs are unavailable is simply omitted."""
    os.makedirs(results_dir, exist_ok=True)
    tot_share = float(np.sum(list(distance_total_sharing_by_day.values())))
    tot_nosh = float(np.sum(list(distance_total_no_sharing_by_day.values())))
    dist_red_pct = (1.0 - tot_share / tot_nosh) * 100.0 if tot_nosh > 0 else 0.0

    lines = []
    lines.append("=== AGGREGATED RESULTS SUMMARY ===")
    lines.append(f"model: {model_label}")
    lines.append(f"grid_size: {getattr(arglist, 'grid_size', 'NA')}")
    lines.append(f"num_agents: {getattr(arglist, 'num_agents', 'NA')}")
    lines.append(f"altruism_distribution: {getattr(arglist, 'altruism_distribution', getattr(arglist, 'altruism_dist', 'NA'))}")
    lines.append(f"population_dynamics: {'birth_death' if getattr(arglist, 'enable_birth_death', False) else 'fixed'}")
    lines.append(f"days: {getattr(arglist, 'days', 'NA')}")
    lines.append("")
    lines.append(f"total_distance_sharing: {tot_share:.2f}")
    lines.append(f"total_distance_no_sharing: {tot_nosh:.2f}")
    lines.append(f"distance_reduction_percent: {dist_red_pct:.2f}")
    if acceptance_rates_by_day:
        vals = [v for v in acceptance_rates_by_day.values()]
        lines.append(f"mean_acceptance_rate: {float(np.mean(vals)) * (100.0 if max(vals, default=0) <= 1.0 else 1.0):.2f}")
    if vehicle_utilization_by_day:
        uv = [np.mean(v) if hasattr(v, '__len__') else v for v in vehicle_utilization_by_day.values()]
        lines.append(f"mean_vehicle_utilization: {float(np.mean(uv)):.3f}")
    if traffic_metrics:
        lines.append(f"traffic_reduction_percent: {traffic_metrics.get('traffic_reduction_percentage', float('nan')):.2f}")
    if multidimensional_stats:
        lines.append(f"gini_combined_benefits: {multidimensional_stats.get('gini_combined_benefits', float('nan')):.4f}")
        lines.append(f"gini_distance_benefits: {multidimensional_stats.get('gini_distance_benefits', float('nan')):.4f}")
        lines.append(f"mean_distance_benefit: {multidimensional_stats.get('mean_distance_benefit', float('nan')):.2f}")
    if winner_loser_stats:
        lines.append(f"winner_percentage: {winner_loser_stats.get('winner_percentage', float('nan')):.1f}")

    if detour_distances_by_day and direct_distances_by_day:
        factors = []
        for day in detour_distances_by_day:
            dd = detour_distances_by_day[day]; dr = direct_distances_by_day.get(day, {})
            for agent, detour in dd.items():
                direct = dr.get(agent, 0)
                if direct and direct > 0 and detour is not None:
                    factors.append((direct + detour) / direct)
        if factors:
            lines.append(f"avg_detour_factor: {float(np.mean(factors)):.4f}")
    if individual_times_by_day:
        times = []
        for day, td in individual_times_by_day.items():
            for agent, t in (td.items() if hasattr(td, 'items') else []):
                if t and t > 0:
                    times.append(t)
        if times:
            lines.append(f"avg_trip_time: {float(np.mean(times)):.3f}")
    if reintegration_metrics:
        rs = reintegration_metrics.get('final_reintegration_score')
        if rs is not None:
            lines.append(f"reintegration_score: {float(rs):.2f}")

    out_path = os.path.join(results_dir, "results_summary.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Aggregated summary saved to {out_path}")
    return out_path


def save_results(results, save_dir, altruism_points_by_day, roles_by_day, individual_distances_by_day, direct_distances_by_day, detour_distances_by_day, miles_given_by_day, miles_taken_by_day, distance_total_sharing_by_day, distance_total_no_sharing_by_day, arglist):
    """Save the results for all days to a file, including altruism scores, roles, distances, and miles."""
    results_dir = get_results_dir(arglist)
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "results.txt")
    with open(save_path, "w") as f:
        total_distance_sharing = 0
        total_distance_no_sharing = 0
        for day, rewards in results.items():
            f.write(f"Day {day} Rewards:\n")
            for agent_id, reward in rewards.items():
                f.write(f"{agent_id}: Reward = {reward}, Altruism = {altruism_points_by_day[day][agent_id]}, Role = {roles_by_day[day][agent_id]}\n")
                f.write(f"  Individual Distance: {individual_distances_by_day[day].get(agent_id, 0)}\n")
                f.write(f"  Direct Distance: {direct_distances_by_day[day].get(agent_id, 0)}\n")
                f.write(f"  Detour Distance: {detour_distances_by_day[day].get(agent_id, 0)}\n")
                f.write(f"  Miles Given: {miles_given_by_day[day].get(agent_id, 0)}\n")
                f.write(f"  Miles Taken: {miles_taken_by_day[day].get(agent_id, 0)}\n")
            f.write(f"Total Distance (Sharing) by day: {distance_total_sharing_by_day[day]}\n")
            total_distance_sharing += distance_total_sharing_by_day[day]
            f.write(f"Total Distance (No Sharing) by day: {distance_total_no_sharing_by_day[day]}\n")
            total_distance_no_sharing += distance_total_no_sharing_by_day[day]
            f.write("\n")
        f.write(f"Total Distance (Sharing): {total_distance_sharing}\n")
        f.write(f"Total Distance (No Sharing): {total_distance_no_sharing}\n")
    print(f"Results saved to {save_path}")


def plot_sharing_vs_non_sharing(distance_total_sharing_by_day, distance_total_no_sharing_by_day, env, arglist):
    """
    Plot day-wise sharing distance vs. non-sharing distance, with carbon emission annotation.
    """
    days = list(distance_total_sharing_by_day.keys())
    sharing_distances = list(distance_total_sharing_by_day.values())
    non_sharing_distances = list(distance_total_no_sharing_by_day.values())

    # Convert to float to avoid numpy array formatting issues
    total_sharing = float(sum(sharing_distances))
    total_no_sharing = float(sum(non_sharing_distances))

    # Carbon emission calculation (example factor: 0.192 kg CO2 per km)
    emission_factor = 0.192
    sharing_emission = total_sharing * emission_factor
    no_sharing_emission = total_no_sharing * emission_factor

    plt.figure(figsize=(10, 6))
    plt.plot(days, sharing_distances, label="Sharing Distance", marker="o", linestyle="-", color="blue")
    plt.plot(days, non_sharing_distances, label="Non-Sharing Distance", marker="o", linestyle=":", color="red")  # Dotted line

    plt.xlabel("Day")
    plt.ylabel("Total Distance")
    plt.title("Day-wise Sharing Distance vs. Non-Sharing Distance")
    plt.legend()
    plt.grid(True)

    # Annotate total distances and carbon emission at the bottom
    plt.figtext(
        0.5, 0.01,
        f"Total Sharing Distance: {total_sharing:.2f}    |    Total Non-Sharing Distance: {total_no_sharing:.2f}\n"
        f"Total Carbon Emissions: Sharing = {sharing_emission:.2f} kg, No Sharing = {no_sharing_emission:.2f} kg (factor: {emission_factor} kg/km)",
        ha="center", fontsize=12, color="darkgreen"
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    results_dir = get_results_dir(arglist)
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "sharing_vs_non_sharing_distance.png"))
    plt.close()

def plot_vehicle_utilization(vehicle_utilization_by_day, env, arglist):
    """
    Line plot of average vehicle utilization per day.
    """
    days = sorted(vehicle_utilization_by_day.keys())
    avg_utilizations = []
    
    for day in days:
        daily_utilizations = vehicle_utilization_by_day[day]
        avg_utilization = np.mean(daily_utilizations) if daily_utilizations else 0
        avg_utilizations.append(avg_utilization)
    
    plt.figure(figsize=(10, 6))
    plt.plot(days, avg_utilizations, marker="o", linestyle="-", color="purple", linewidth=2, markersize=8)
    plt.xlabel("Day")
    plt.ylabel("Average Vehicle Utilization")
    plt.title("Average Vehicle Utilization per Day")
    plt.grid(True, alpha=0.3)
    
    # Add value annotations on each point
    for day, util in zip(days, avg_utilizations):
        plt.annotate(f'{util:.2f}', (day, util), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    results_dir = get_results_dir(arglist)
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "vehicle_utilization_average.png"))
    plt.close()

def plot_ride_acceptance_rate(acceptance_rates_by_day, env, arglist):
    days = list(acceptance_rates_by_day.keys())
    rates = [acceptance_rates_by_day[d] * 100 for d in days]
    plt.figure(figsize=(10, 6))
    plt.plot(days, rates, marker="o", linestyle="-", color="purple")
    plt.xlabel("Day")
    plt.ylabel("Ride Acceptance Rate (%)")
    plt.title("Ride Acceptance Rate per Day")
    plt.ylim(0, 100)
    plt.grid(True)
    results_dir = get_results_dir(arglist)
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "ride_acceptance_rate.png"))
    plt.close()

def plot_daily_dropouts(env, arglist):
    """
    Plot daily dropout statistics as a simple line plot.
    Shows number of dropouts per day.
    """
    if not env.daily_dropouts:
        print("No dropout data available for plotting.")
        return
    
    days = sorted(env.daily_dropouts.keys())
    dropout_counts = []
    
    for day in days:
        dropped_agents = env.daily_dropouts[day]
        dropout_counts.append(len(dropped_agents))
    
    # Create simple line plot
    plt.figure(figsize=(10, 6))
    
    # Plot dropout counts as line plot
    plt.plot(days, dropout_counts, marker='o', linewidth=2, markersize=8, 
             color='red', markerfacecolor='darkred', markeredgewidth=1, 
             markeredgecolor='black', label='Daily Dropouts')
    
    # Add value labels on points
    for day, count in zip(days, dropout_counts):
        if count > 0:
            plt.annotate(f'{count}', (day, count), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10, fontweight='bold')
    
    plt.xlabel('Day', fontsize=12)
    plt.ylabel('Number of Dropouts', fontsize=12)
    plt.title('Daily Agent Dropouts', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set y-axis to start from 0 and add some padding
    max_dropouts = max(dropout_counts) if dropout_counts else 1
    plt.ylim(0, max_dropouts * 1.1)
    
    # Ensure x-axis shows all days
    plt.xticks(days)
    
    plt.tight_layout()
    results_dir = get_results_dir(arglist)
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "daily_dropouts.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    total_dropouts = sum(dropout_counts)
    print(f"\n=== DROPOUT STATISTICS ===")
    print(f"Total dropouts across all days: {total_dropouts}")
    print(f"Average dropouts per day: {np.mean(dropout_counts):.2f}")
    print(f"Peak dropouts in single day: {max(dropout_counts) if dropout_counts else 0}")
    print(f"Days with dropouts: {sum(1 for c in dropout_counts if c > 0)}/{len(days)}")
    
    return {
        'daily_counts': dropout_counts,
        'total_dropouts': total_dropouts,
        'peak_dropouts': max(dropout_counts) if dropout_counts else 0,
        'days_with_dropouts': sum(1 for c in dropout_counts if c > 0)
    }

def plot_birth_death_analysis(env, arglist):
    """
    Plot birth-death analysis including daily births, deaths, and population dynamics.
    """
    if not hasattr(env, 'daily_births') or not env.daily_births:
        print("No birth-death data available for plotting.")
        return
    
    days = sorted(set(list(env.daily_births.keys()) + list(env.daily_dropouts.keys())))
    birth_counts = []
    death_counts = []
    active_population = []
    
    cumulative_births = 0
    cumulative_deaths = 0
    initial_active = env.initial_active_agents
    
    for day in days:
        daily_births = len(env.daily_births.get(day, []))
        daily_deaths = len(env.daily_dropouts.get(day, []))
        
        birth_counts.append(daily_births)
        death_counts.append(daily_deaths)
        
        cumulative_births += daily_births
        cumulative_deaths += daily_deaths
        current_active = initial_active + cumulative_births - daily_deaths
        active_population.append(current_active)
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Daily births and deaths
    width = 0.35
    x = np.array(days)
    
    bars1 = ax1.bar(x - width/2, birth_counts, width, label='Births', color='green', alpha=0.7)
    bars2 = ax1.bar(x + width/2, death_counts, width, label='Deaths', color='red', alpha=0.7)
    
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Number of Agents')
    ax1.set_title('Daily Births and Deaths')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # 2. Active population over time
    ax2.plot(days, active_population, marker='o', linewidth=2, markersize=6, 
             color='blue', markerfacecolor='lightblue', markeredgewidth=1, 
             markeredgecolor='blue')
    ax2.axhline(y=env.numAgents, color='red', linestyle='--', alpha=0.5, 
                label=f'Max Capacity ({env.numAgents})')
    ax2.axhline(y=initial_active, color='green', linestyle='--', alpha=0.5, 
                label=f'Initial Active ({initial_active})')
    
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Active Agents')
    ax2.set_title('Population Dynamics Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value annotations
    for day, pop in zip(days[::2], active_population[::2]):  # Every other day
        ax2.annotate(f'{pop}', (day, pop), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # 3. Cumulative births and deaths
    cumulative_birth_list = np.cumsum(birth_counts)
    cumulative_death_list = np.cumsum(death_counts)
    
    ax3.plot(days, cumulative_birth_list, marker='s', linewidth=2, 
             color='green', label='Cumulative Births')
    ax3.plot(days, cumulative_death_list, marker='^', linewidth=2, 
             color='red', label='Cumulative Deaths')
    
    ax3.set_xlabel('Day')
    ax3.set_ylabel('Cumulative Count')
    ax3.set_title('Cumulative Births vs Deaths')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Net population change
    net_change = np.array(birth_counts) - np.array(death_counts)
    colors = ['green' if x >= 0 else 'red' for x in net_change]
    
    bars4 = ax4.bar(days, net_change, color=colors, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.set_xlabel('Day')
    ax4.set_ylabel('Net Change (Births - Deaths)')
    ax4.set_title('Daily Net Population Change')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars4, net_change):
        if val != 0:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., 
                    height + (0.1 if height >= 0 else -0.2),
                    f'{int(val)}', ha='center', 
                    va='bottom' if height >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    results_dir = get_results_dir(arglist)
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "birth_death_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    total_births = sum(birth_counts)
    total_deaths = sum(death_counts)
    final_population = active_population[-1] if active_population else initial_active
    
    print(f"\n=== BIRTH-DEATH STATISTICS ===")
    print(f"Initial active agents: {initial_active}")
    print(f"Total births: {total_births}")
    print(f"Total deaths: {total_deaths}")
    print(f"Final active agents: {final_population}")
    print(f"Net population change: {total_births - total_deaths}")
    print(f"Population growth rate: {((final_population - initial_active) / initial_active * 100):.2f}%")
    
    return {
        'total_births': total_births,
        'total_deaths': total_deaths,
        'final_population': final_population,
        'net_change': total_births - total_deaths,
        'growth_rate': (final_population - initial_active) / initial_active * 100
    }

def calculate_reintegration_metrics(env):
    """
    Calculate time-based reintegration metrics and final score.
    """
    if not env.reintegration_events:
        return {
            'basic_rate': 0.0,
            'time_weighted_rate': 0.0,
            'quick_return_rate': 0.0,
            'stability_score': 0.0,
            'final_reintegration_score': 0.0,
            'total_dropouts': 0,
            'total_returns': 0,
            'avg_return_time': 0.0,
            'stable_returners': 0,
            'instability_rate': 0.0,
            'multi_dropout_agents': 0
        }
    
    # Basic calculations
    total_dropouts = sum(len(events) for events in env.agent_dropout_events.values())
    total_returns = len(env.reintegration_events)
    basic_rate = total_returns / total_dropouts if total_dropouts > 0 else 0.0
    
    # Time-based analysis
    return_times = []
    time_weights = []
    quick_returns = 0  # Returns within 2 days
    
    for agent_id, dropout_day, return_day, altruism in env.reintegration_events:
        days_out = return_day - dropout_day
        return_times.append(days_out)
        
        # Time weighting: exponential decay favoring quick returns
        time_weight = np.exp(-0.2 * days_out)  # Decay factor of 0.2
        time_weights.append(time_weight)
        
        if days_out <= 2:
            quick_returns += 1
    
    # Calculate metrics
    avg_return_time = np.mean(return_times) if return_times else 0.0
    quick_return_rate = quick_returns / total_returns if total_returns > 0 else 0.0
    
    # Time-weighted rate
    if time_weights:
        time_weighted_rate = np.mean(time_weights)
    else:
        time_weighted_rate = 0.0
    
    # STABILITY SCORE: Reward single comebacks that lead to stable behavior
    # This is better than rewarding multiple comebacks (which indicate instability)
    unique_returning_agents = len(set(event[0] for event in env.reintegration_events))
    
    # Count agents who returned once and stayed (no subsequent dropouts after their return)
    stable_returners = 0
    current_day = getattr(env, 'current_day', max(env.daily_dropouts.keys()) if env.daily_dropouts else 1)
    
    for agent_id in set(event[0] for event in env.reintegration_events):
        agent_returns = [event[2] for event in env.reintegration_events if event[0] == agent_id]
        agent_dropouts = env.agent_dropout_events.get(agent_id, [])
        
        # Check if agent had a return and then remained stable
        for return_day in agent_returns:
            # Check if there are no dropouts after this return day
            subsequent_dropouts = [d for d in agent_dropouts if d > return_day]
            
            if not subsequent_dropouts:
                # Agent returned and stayed stable
                # Bonus for staying stable for longer periods
                days_stable = current_day - return_day
                if days_stable >= 3:  # Stable for at least 3 days after return
                    stable_returners += 1
                    break  # Count this agent only once
    
    stability_score = stable_returners / unique_returning_agents if unique_returning_agents > 0 else 0.0
    
    # FINAL REINTEGRATION SCORE (0-100 scale)
    # Emphasizes stability over multiple comebacks
    score_components = {
        'basic_rate': basic_rate * 20,           # 20% weight - basic comeback ability
        'time_weighted': time_weighted_rate * 30, # 30% weight - speed of return
        'quick_returns': quick_return_rate * 25,  # 25% weight - immediate comebacks
        'stability': stability_score * 25         # 25% weight - stable reintegration (most important)
    }
    
    final_score = sum(score_components.values())
    
    # Calculate instability metrics for additional insight
    multi_dropout_agents = sum(1 for events in env.agent_dropout_events.values() if len(events) > 1)
    total_agents_with_dropouts = len(env.agent_dropout_events)
    instability_rate = multi_dropout_agents / total_agents_with_dropouts if total_agents_with_dropouts > 0 else 0.0
    
    return {
        'basic_rate': basic_rate,
        'time_weighted_rate': time_weighted_rate,
        'quick_return_rate': quick_return_rate,
        'stability_score': stability_score,
        'final_reintegration_score': final_score,
        'score_breakdown': score_components,
        'total_dropouts': total_dropouts,
        'total_returns': total_returns,
        'avg_return_time': avg_return_time,
        'unique_returning_agents': unique_returning_agents,
        'stable_returners': stable_returners,
        'instability_rate': instability_rate,  # Additional metric for analysis
        'multi_dropout_agents': multi_dropout_agents
    }

def plot_reintegration_analysis(env, arglist):
    """
    Plot reintegration analysis with time-based focus and stability emphasis.
    """
    metrics = calculate_reintegration_metrics(env)
    
    if metrics['total_returns'] == 0:
        print("No reintegration data available for plotting.")
        return metrics
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Score Breakdown - Updated with stability
    score_components = metrics['score_breakdown']
    components = list(score_components.keys())
    values = list(score_components.values())
    colors = ['skyblue', 'lightgreen', 'orange', 'gold']  # Changed last color for stability
    
    bars1 = ax1.bar(components, values, color=colors, alpha=0.8)
    ax1.set_ylabel('Score Points')
    ax1.set_title(f'Reintegration Score Breakdown\nFinal Score: {metrics["final_reintegration_score"]:.1f}/100')
    
    # Fix the warning by setting ticks first, then labels
    ax1.set_xticks(range(len(components)))
    ax1.set_xticklabels(['Basic Rate\n(20%)', 'Time Weighted\n(30%)', 'Quick Returns\n(25%)', 'Stability\n(25%)'])
    
    # Add value labels
    for bar, val in zip(bars1, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Return Time Distribution with Weights (unchanged)
    return_times = []
    weights = []
    for agent_id, dropout_day, return_day, altruism in env.reintegration_events:
        days_out = return_day - dropout_day
        return_times.append(days_out)
        weights.append(np.exp(-0.2 * days_out))
    
    if return_times:
        # Create weighted histogram
        bins = range(1, max(return_times) + 2)
        ax2.hist(return_times, bins=bins, alpha=0.7, color='purple', edgecolor='black', label='Returns')
        
        # Add weight curve
        ax2_twin = ax2.twinx()
        x_curve = np.linspace(1, max(return_times), 100)
        y_curve = np.exp(-0.2 * x_curve)
        ax2_twin.plot(x_curve, y_curve, 'r--', linewidth=2, label='Time Weight')
        
        ax2.axvline(x=metrics['avg_return_time'], color='red', linestyle='-', 
                   label=f'Avg: {metrics["avg_return_time"]:.1f} days')
        
        ax2.set_xlabel('Days Until Return')
        ax2.set_ylabel('Number of Returns')
        ax2_twin.set_ylabel('Time Weight Factor')
        ax2.set_title('Return Time Distribution with Weights')
        ax2.legend(loc='upper right')
        ax2_twin.legend(loc='upper center')
        ax2.grid(True, alpha=0.3)
    
    # 3. Time-based Metrics Comparison - Updated with stability
    metric_names = ['Basic Rate', 'Time Weighted', 'Quick Returns', 'Stability Score']
    metric_values = [
        metrics['basic_rate'],
        metrics['time_weighted_rate'], 
        metrics['quick_return_rate'],
        metrics['stability_score']
    ]
    
    bars3 = ax3.barh(metric_names, metric_values, color=['blue', 'green', 'orange', 'gold'], alpha=0.7)
    ax3.set_xlabel('Rate/Score')
    ax3.set_title('Time-based Reintegration Metrics')
    ax3.set_xlim(0, 1.0)
    
    # Add value labels
    for bar, val in zip(bars3, metric_values):
        width = bar.get_width()
        ax3.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{val:.3f}', ha='left', va='center', fontweight='bold')
    
    # 4. Summary with Final Score - Updated text
    ax4.axis('off')
    summary_text = f"""
    REINTEGRATION ANALYSIS SUMMARY
    
    ╔══════════════════════════════════════╗
    ║    FINAL REINTEGRATION SCORE         ║
    ║         {metrics['final_reintegration_score']:.1f} / 100.0                ║
    ╚══════════════════════════════════════╝
    
    BASIC STATISTICS:
    • Total Dropouts: {metrics['total_dropouts']}
    • Total Returns: {metrics['total_returns']} 
    • Unique Agents Returned: {metrics['unique_returning_agents']}
    • Stable Returners: {metrics['stable_returners']}
    • Multi-Dropout Agents: {metrics['multi_dropout_agents']}
    
    TIME-BASED METRICS:
    • Basic Rate: {metrics['basic_rate']:.3f}
    • Time-Weighted Rate: {metrics['time_weighted_rate']:.3f}
    • Quick Return Rate (≤2 days): {metrics['quick_return_rate']:.3f}
    • Stability Score: {metrics['stability_score']:.3f}
    • Instability Rate: {metrics['instability_rate']:.3f}
    • Average Return Time: {metrics['avg_return_time']:.1f} days
    
    SCORE BREAKDOWN:
    • Basic Rate (20%): {metrics['score_breakdown']['basic_rate']:.1f}
    • Time Weighted (30%): {metrics['score_breakdown']['time_weighted']:.1f}
    • Quick Returns (25%): {metrics['score_breakdown']['quick_returns']:.1f}
    • Stability (25%): {metrics['score_breakdown']['stability']:.1f}
    
    STABILITY INSIGHT:
    Agents with single successful reintegration: {metrics['stable_returners']}
    (Better than multiple comebacks which indicate instability)
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    results_dir = get_results_dir(arglist)
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "reintegration_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics

def track_pickup_interactions(env, actions, pickup_history, day):
    """
    This function is now simplified - the actual tracking happens in env.step()
    We just need to update the global pickup_history from env.daily_pickups
    """
    # Get pickups for current day from environment
    if day in env.daily_pickups:
        for pair, count in env.daily_pickups[day].items():
            if pair not in pickup_history:
                pickup_history[pair] = 0
            pickup_history[pair] = count  # Use the count from environment

def generate_ieee_tikz_data(altruism_points_by_day, individual_distances_by_day, 
                           direct_distances_by_day, detour_distances_by_day,
                           distance_total_sharing_by_day, distance_total_no_sharing_by_day,
                           vehicle_utilization_by_day, acceptance_rates_by_day,
                           individual_times_by_day, direct_times_by_day, env, arglist):
    """Generate IEEE-formatted TikZ data files for import."""
    
    print("Generating IEEE TikZ data files...")
    results_dir = get_results_dir(arglist)
    os.makedirs(results_dir, exist_ok=True)
    latex_dir = os.path.join(results_dir, "latex_data")
    os.makedirs(latex_dir, exist_ok=True)
    
    days = sorted(individual_distances_by_day.keys())
    
    # 1. Distance comparison data
    with open(f"{latex_dir}/distance_sharing.dat", 'w') as f:
        f.write("day distance\n")
        for day in days:
            f.write(f"{day} {distance_total_sharing_by_day[day]:.2f}\n")

    with open(f"{latex_dir}/distance_no_sharing.dat", 'w') as f:
        f.write("day distance\n")
        for day in days:
            f.write(f"{day} {distance_total_no_sharing_by_day[day]:.2f}\n")
    
    # 2. Agent distance statistics
    with open(f"{latex_dir}/agent_distance_sharing.dat", 'w') as f:
        f.write("day mean std\n")
        for day in days:
            # Only include active agents
            vals = [dist for agent, dist in individual_distances_by_day[day].items() 
                   if env.agent_role.get(agent, 'active') not in ['dropout', 'never_joined']]
            mean_val = np.mean(vals) if vals else 0
            std_val = np.std(vals) if vals else 0
            f.write(f"{day} {mean_val:.2f} {std_val:.2f}\n")

    with open(f"{latex_dir}/agent_distance_no_sharing.dat", 'w') as f:
        f.write("day mean std\n")
        for day in days:
            # Only include active agents
            vals = [dist for agent, dist in direct_distances_by_day[day].items() 
                   if env.agent_role.get(agent, 'active') not in ['dropout', 'never_joined']]
            mean_val = np.mean(vals) if vals else 0
            std_val = np.std(vals) if vals else 0
            f.write(f"{day} {mean_val:.2f} {std_val:.2f}\n")
    
    # 3. Vehicle utilization
    with open(f"{latex_dir}/vehicle_utilization.dat", 'w') as f:
        f.write("day utilization\n")
        for day in days:
            util = np.mean(vehicle_utilization_by_day[day]) if vehicle_utilization_by_day[day] else 0
            f.write(f"{day} {util:.3f}\n")
    
    # 4. Acceptance rate
    with open(f"{latex_dir}/acceptance_rate.dat", 'w') as f:
        f.write("day rate\n")
        for day in days:
            f.write(f"{day} {acceptance_rates_by_day[day] * 100:.1f}\n")
    
    # 5. Altruism data
    with open(f"{latex_dir}/altruism.dat", 'w') as f:
        f.write("day mean std\n")
        for day in days:
            # Only include active agents
            vals = [score for agent, score in altruism_points_by_day[day].items() 
                   if env.agent_role.get(agent, 'active') not in ['dropout', 'never_joined']]
            mean_val = np.mean(vals) if vals else 0
            std_val = np.std(vals) if vals else 0
            f.write(f"{day} {mean_val:.3f} {std_val:.3f}\n")
    
    # 6. Trip time data
    with open(f"{latex_dir}/trip_time_sharing.dat", 'w') as f:
        f.write("day time\n")
        for day in days:
            # Only include active agents with positive times
            times = [t for agent, t in individual_times_by_day[day].items() 
                    if t > 0 and env.agent_role.get(agent, 'active') not in ['dropout', 'never_joined']]
            avg_time = np.mean(times) if times else 0
            f.write(f"{day} {avg_time:.1f}\n")

    with open(f"{latex_dir}/trip_time_no_sharing.dat", 'w') as f:
        f.write("day time\n")
        for day in days:
            # Only include active agents with positive times
            times = [t for agent, t in direct_times_by_day[day].items() 
                    if t > 0 and env.agent_role.get(agent, 'active') not in ['dropout', 'never_joined']]
            avg_time = np.mean(times) if times else 0
            f.write(f"{day} {avg_time:.1f}\n")
    
    # 7. Detour factor data
    with open(f"{latex_dir}/detour_factor.dat", 'w') as f:
        f.write("day mean std\n")
        for day in days:
            detour_factors = []
            for agent in detour_distances_by_day[day]:
                # Only include active agents
                if env.agent_role.get(agent, 'active') in ['dropout', 'never_joined']:
                    continue
                if (direct_distances_by_day[day][agent] > 0 and 
                    detour_distances_by_day[day][agent] > 0):
                    factor = detour_distances_by_day[day][agent] / direct_distances_by_day[day][agent]
                    detour_factors.append(factor)
            
            if detour_factors:
                mean_factor = np.mean(detour_factors)
                std_factor = np.std(detour_factors)
            else:
                mean_factor = 0
                std_factor = 0
            f.write(f"{day} {mean_factor:.3f} {std_factor:.3f}\n")
    
    # 9. ENHANCED Summary statistics with ALL metrics and averages
    total_sharing = sum(distance_total_sharing_by_day.values())
    total_no_sharing = sum(distance_total_no_sharing_by_day.values())
    reduction_percent = ((total_no_sharing - total_sharing) / total_no_sharing * 100) if total_no_sharing > 0 else 0
    
    # Calculate averages across all days for daily metrics
    all_sharing_distances = [distance_total_sharing_by_day[day] for day in days]
    all_no_sharing_distances = [distance_total_no_sharing_by_day[day] for day in days]
    
    # Agent-level averages (only active agents)
    all_agent_sharing_distances = []
    all_agent_no_sharing_distances = []
    all_altruism_scores = []
    all_trip_times_sharing = []
    all_trip_times_no_sharing = []
    all_detour_factors = []
    
    for day in days:
        # Collect agent distances for active agents only
        for agent in individual_distances_by_day[day]:
            if env.agent_role.get(agent, 'active') not in ['dropout', 'never_joined']:
                all_agent_sharing_distances.append(individual_distances_by_day[day][agent])
                all_agent_no_sharing_distances.append(direct_distances_by_day[day][agent])
                all_altruism_scores.append(altruism_points_by_day[day].get(agent, 0))
                
                # Trip times
                if individual_times_by_day[day][agent] > 0:
                    all_trip_times_sharing.append(individual_times_by_day[day][agent])
                if direct_times_by_day[day][agent] > 0:
                    all_trip_times_no_sharing.append(direct_times_by_day[day][agent])
                
                # Detour factors
                if (direct_distances_by_day[day][agent] > 0 and 
                    detour_distances_by_day[day][agent] > 0):
                    factor = detour_distances_by_day[day][agent] / direct_distances_by_day[day][agent]
                    all_detour_factors.append(factor)
    
    # Vehicle utilization averages
    all_utilizations = []
    for day in days:
        if vehicle_utilization_by_day[day]:
            all_utilizations.extend(vehicle_utilization_by_day[day])
    
    # Acceptance rates
    all_acceptance_rates = [acceptance_rates_by_day[day] for day in days]

    with open(f"{latex_dir}/summary.dat", 'w') as f:
        # Total metrics
        f.write(f"total_sharing {total_sharing:.2f}\n")
        f.write(f"total_no_sharing {total_no_sharing:.2f}\n")
        f.write(f"reduction_percent {reduction_percent:.1f}\n")
        f.write(f"carbon_sharing {total_sharing * 0.192:.2f}\n")
        f.write(f"carbon_no_sharing {total_no_sharing * 0.192:.2f}\n")
        f.write(f"carbon_reduction_kg {(total_no_sharing - total_sharing) * 0.192:.2f}\n")
        
        # Daily averages
        f.write(f"avg_daily_sharing_distance {np.mean(all_sharing_distances):.2f}\n")
        f.write(f"avg_daily_no_sharing_distance {np.mean(all_no_sharing_distances):.2f}\n")
        
        # Agent-level averages (across all days and active agents)
        f.write(f"avg_agent_sharing_distance {np.mean(all_agent_sharing_distances):.2f}\n")
        f.write(f"avg_agent_no_sharing_distance {np.mean(all_agent_no_sharing_distances):.2f}\n")
        f.write(f"avg_altruism_score {np.mean(all_altruism_scores):.3f}\n")
        f.write(f"std_altruism_score {np.std(all_altruism_scores):.3f}\n")
        
        # Trip time averages
        f.write(f"avg_trip_time_sharing {np.mean(all_trip_times_sharing):.1f}\n")
        f.write(f"avg_trip_time_no_sharing {np.mean(all_trip_times_no_sharing):.1f}\n")
        
        # Detour factor averages
        f.write(f"avg_detour_factor {np.mean(all_detour_factors):.3f}\n")
        f.write(f"std_detour_factor {np.std(all_detour_factors):.3f}\n")
        
        # Vehicle utilization
        f.write(f"avg_vehicle_utilization {np.mean(all_utilizations):.3f}\n")
        f.write(f"std_vehicle_utilization {np.std(all_utilizations):.3f}\n")
        
        # Acceptance rate
        f.write(f"avg_acceptance_rate {np.mean(all_acceptance_rates) * 100:.1f}\n")
        f.write(f"std_acceptance_rate {np.std(all_acceptance_rates) * 100:.1f}\n")
        
        # Birth-death metrics (if available)
        if hasattr(env, 'reintegration_events') and env.reintegration_events:
            reintegration_metrics = calculate_reintegration_metrics(env)
            f.write(f"reintegration_score {reintegration_metrics['final_reintegration_score']:.1f}\n")
            f.write(f"total_dropouts {reintegration_metrics['total_dropouts']}\n")
            f.write(f"total_returns {reintegration_metrics['total_returns']}\n")
            f.write(f"avg_return_time {reintegration_metrics['avg_return_time']:.1f}\n")
            f.write(f"stability_score {reintegration_metrics['stability_score']:.3f}\n")
        
        # Dropout metrics (if available)
        if hasattr(env, 'daily_dropouts') and env.daily_dropouts:
            total_dropouts = sum(len(agents) for agents in env.daily_dropouts.values())
            days_with_dropouts = sum(1 for agents in env.daily_dropouts.values() if len(agents) > 0)
            f.write(f"total_simulation_dropouts {total_dropouts}\n")
            f.write(f"days_with_dropouts {days_with_dropouts}\n")
            f.write(f"avg_dropouts_per_day {total_dropouts / len(days):.2f}\n")
        
        # Simulation parameters for reference
        f.write(f"simulation_days {len(days)}\n")
        f.write(f"initial_agents {env.initial_active_agents}\n")
        f.write(f"max_agents {env.numAgents}\n")

    # NEW: 10. Daily dropout data
    if hasattr(env, 'daily_dropouts') and env.daily_dropouts:
        with open(f"{latex_dir}/daily_dropouts.dat", 'w') as f:
            f.write("day dropouts avg_altruism\n")
            for day in days:
                dropped_agents = env.daily_dropouts.get(day, [])
                dropout_count = len(dropped_agents)
                
                if dropped_agents:
                    altruism_scores = [env.altruism_points_day.get(agent, 0) for agent in dropped_agents]
                    avg_altruism = np.mean(altruism_scores)
                else:
                    avg_altruism = 0
                
                f.write(f"{day} {dropout_count} {avg_altruism:.4f}\n")

    # NEW: 11. Reintegration metrics
    if hasattr(env, 'reintegration_events') and env.reintegration_events:
        reintegration_metrics = calculate_reintegration_metrics(env)

        with open(f"{latex_dir}/reintegration_metrics.dat", 'w') as f:
            f.write("metric value\n")
            f.write(f"final_score {reintegration_metrics['final_reintegration_score']:.2f}\n")
            f.write(f"basic_rate {reintegration_metrics['basic_rate']:.4f}\n")
            f.write(f"time_weighted_rate {reintegration_metrics['time_weighted_rate']:.4f}\n")
            f.write(f"quick_return_rate {reintegration_metrics['quick_return_rate']:.4f}\n")
            f.write(f"stability_score {reintegration_metrics['stability_score']:.4f}\n")
            f.write(f"stable_returners {reintegration_metrics['stable_returners']}\n")
            f.write(f"instability_rate {reintegration_metrics['instability_rate']:.4f}\n")
            f.write(f"avg_return_time {reintegration_metrics['avg_return_time']:.2f}\n")
    
    print("Generated IEEE data files in results/ieee_data/")
    print("Upload these .dat files to your Overleaf project and use the updated LaTeX code.")

def calculate_agent_traffic_reduction_benefits(paths_sharing_by_day, paths_no_sharing_by_day, env, agents):
    """
    Calculate traffic reduction benefits for each agent based on actual ride-sharing participation.
    
    Method: 
    - For drivers: Benefit = total distance of riders they carried (enabling those trips to be shared)
    - For riders: Benefit = their own trip distance (if they were picked up)
    - This represents the actual traffic reduction achieved through ride-sharing
    """
    agent_traffic_benefits = {}
    
    days = sorted(paths_sharing_by_day.keys())
    
    for agent in agents:
        if env.agent_role.get(agent, 'active') in ['dropout', 'never_joined']:
            agent_traffic_benefits[agent] = 0.0
            continue
        
        total_benefit = 0.0
        
        for day in days:
            daily_benefit = calculate_ridesharing_traffic_benefits(agent, env, day)
            total_benefit += daily_benefit
        
        # Average across all days and normalize by number of days
        agent_traffic_benefits[agent] = total_benefit / len(days) if days else 0.0
    
    return agent_traffic_benefits

def calculate_ridesharing_traffic_benefits(agent, env, day):
    """
    Calculate traffic benefits based on actual ride-sharing participation.
    
    For drivers: Benefit = sum of rider trip distances they enabled (avoided individual trips)
    For riders: Benefit = own trip distance (if picked up, representing avoided individual trip)
    """
    if env.agent_role.get(agent) == 'driver':
        # Driver benefit = sum of rider trip distances they enabled to be shared
        total_benefit = 0.0
        
        # Get riders carried by this driver
        riders_carried = env.riders_drivers.get(agent, [])
        
        for rider_obj in riders_carried:
            # Find the rider agent corresponding to this rider object
            rider_agent = None
            for i, agent_name in enumerate(env.agents):
                if env.agent_objects[i] == rider_obj:
                    rider_agent = agent_name
                    break
            
            if rider_agent:
                # Calculate the direct distance this rider would have traveled alone
                rider_direct_path = rider_obj._dijkstra_path(rider_obj.start_position, rider_obj.destination)
                rider_distance = rider_obj._calculate_path_length(rider_direct_path)
                total_benefit += rider_distance
        
        return total_benefit
        
    elif env.agent_role.get(agent) == 'rider':
        # Rider benefit = own trip distance (if they were picked up)
        try:
            agent_idx = env.agents.index(agent)
            agent_obj = env.agent_objects[agent_idx]
            
            # Check if this rider was picked up
            if hasattr(agent_obj, 'being_picked_up') and agent_obj.being_picked_up:
                # Calculate their direct trip distance (which was avoided)
                rider_path = agent_obj._dijkstra_path(agent_obj.start_position, agent_obj.destination)
                return agent_obj._calculate_path_length(rider_path)
        except (ValueError, IndexError):
            pass
    
    return 0.0

def calculate_agent_benefits(individual_distances_by_day, direct_distances_by_day, altruism_points_by_day, 
                           paths_sharing_by_day, paths_no_sharing_by_day, env, arglist):
    """
    Calculate per-agent benefits comparing sharing vs non-sharing across all days.
    Now includes both distance benefits and traffic reduction benefits.
    Only includes active agents.
    Returns individual agent data for scatter plots and multidimensional analysis.
    """
    agent_benefits = {}
    agent_total_altruism = {}
    
    # Get all agents from first day
    first_day = min(individual_distances_by_day.keys())
    agents = list(individual_distances_by_day[first_day].keys())
    
    # Calculate traffic reduction benefits for each agent
    agent_traffic_benefits = calculate_agent_traffic_reduction_benefits(
        paths_sharing_by_day, paths_no_sharing_by_day, env, agents
    )
    
    for agent in agents:
        # Skip inactive agents
        if env.agent_role.get(agent, 'active') in ['dropout', 'never_joined']:
            continue
            
        total_sharing_distance = 0
        total_no_sharing_distance = 0
        total_altruism = 0
        days_count = 0
        
        # Sum across all days (only when agent was active)
        for day in individual_distances_by_day.keys():
            if (agent in individual_distances_by_day[day] and agent in direct_distances_by_day[day] and
                env.agent_role.get(agent, 'active') not in ['dropout', 'never_joined']):
                total_sharing_distance += individual_distances_by_day[day][agent]
                total_no_sharing_distance += direct_distances_by_day[day][agent]
                total_altruism += altruism_points_by_day[day].get(agent, 0)
                days_count += 1
        
        if days_count == 0:  # Agent was never active
            continue
            
        # Calculate distance benefit (positive = saved distance, negative = extra distance)
        distance_benefit = total_no_sharing_distance - total_sharing_distance
        avg_altruism = total_altruism / days_count
        
        # Get traffic reduction benefit for this agent
        traffic_benefit = agent_traffic_benefits.get(agent, 0.0)
        
        agent_benefits[agent] = {
            'distance_benefit': distance_benefit,
            'traffic_benefit': traffic_benefit,  # New traffic reduction benefit
            'total_sharing_distance': total_sharing_distance,
            'total_no_sharing_distance': total_no_sharing_distance,
            'avg_altruism': avg_altruism,
            'benefit_percentage': (distance_benefit / total_no_sharing_distance * 100) if total_no_sharing_distance > 0 else 0,
            'combined_benefit': distance_benefit + traffic_benefit  # Combined multidimensional benefit
        }
        agent_total_altruism[agent] = avg_altruism
    
    return agent_benefits, agent_total_altruism

def plot_benefits_vs_altruism_line_curves(agent_benefits, env, arglist):
    """
    Create line plots showing benefits vs altruism scores.
    Agents are sorted by altruism score to create smooth curves.
    """
    # Extract data
    distance_benefits = []
    traffic_benefits = []
    altruism_scores = []
    agent_names = []
    
    for agent, data in agent_benefits.items():
        distance_benefits.append(data['distance_benefit'])
        traffic_benefits.append(data['traffic_benefit'])
        altruism_scores.append(data['avg_altruism'])
        agent_names.append(agent)
    
    # Convert to numpy arrays for easier manipulation
    distance_benefits = np.array(distance_benefits)
    traffic_benefits = np.array(traffic_benefits)
    altruism_scores = np.array(altruism_scores)
    agent_names = np.array(agent_names)
    
    # Sort all arrays by altruism scores (ascending order)
    sorted_indices = np.argsort(altruism_scores)
    altruism_sorted = altruism_scores[sorted_indices]
    distance_sorted = distance_benefits[sorted_indices]
    traffic_sorted = traffic_benefits[sorted_indices]
    agents_sorted = agent_names[sorted_indices]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Distance Benefits vs Altruism Line Plot
    ax1.plot(altruism_sorted, distance_sorted, 'b-', linewidth=2.5, alpha=0.8, 
             marker='o', markersize=4, markerfacecolor='blue', markeredgecolor='darkblue',
             markeredgewidth=0.5, label='Distance Benefits')
    
    # Add horizontal line at y=0 for reference
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    ax1.set_xlabel('Altruism Score', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Distance Benefit (Distance Saved)', fontsize=12, fontweight='bold')
    ax1.set_title('Distance Benefits vs Altruism Score\n(Agents Ordered by Altruism)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add statistics text box
    mean_distance = np.mean(distance_sorted)
    std_distance = np.std(distance_sorted)
    corr_distance = np.corrcoef(altruism_sorted, distance_sorted)[0, 1]
    
    stats_text_1 = f'Mean: {mean_distance:.2f}\nStd: {std_distance:.2f}\nCorrelation: {corr_distance:.3f}'
    ax1.text(0.02, 0.98, stats_text_1, transform=ax1.transAxes,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    # Highlight extreme values
    min_idx = np.argmin(distance_sorted)
    max_idx = np.argmax(distance_sorted)
    
    ax1.scatter(altruism_sorted[min_idx], distance_sorted[min_idx], 
               color='red', s=100, marker='v', zorder=5, label=f'Min: {agents_sorted[min_idx]}')
    ax1.scatter(altruism_sorted[max_idx], distance_sorted[max_idx], 
               color='green', s=100, marker='^', zorder=5, label=f'Max: {agents_sorted[max_idx]}')
    
    # 2. Traffic Benefits vs Altruism Line Plot
    ax2.plot(altruism_sorted, traffic_sorted, 'g-', linewidth=2.5, alpha=0.8,
             marker='s', markersize=4, markerfacecolor='green', markeredgecolor='darkgreen',
             markeredgewidth=0.5, label='Traffic Benefits')
    
    # Add horizontal line at y=0 for reference
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    ax2.set_xlabel('Altruism Score', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Traffic Benefit (Ride-sharing Participation)', fontsize=12, fontweight='bold')
    ax2.set_title('Traffic Benefits vs Altruism Score\n(Agents Ordered by Altruism)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add statistics text box
    mean_traffic = np.mean(traffic_sorted)
    std_traffic = np.std(traffic_sorted)
    corr_traffic = np.corrcoef(altruism_sorted, traffic_sorted)[0, 1]
    
    stats_text_2 = f'Mean: {mean_traffic:.2f}\nStd: {std_traffic:.2f}\nCorrelation: {corr_traffic:.3f}'
    ax2.text(0.02, 0.98, stats_text_2, transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    # Highlight extreme values for traffic benefits
    min_idx_traffic = np.argmin(traffic_sorted)
    max_idx_traffic = np.argmax(traffic_sorted)
    
    ax2.scatter(altruism_sorted[min_idx_traffic], traffic_sorted[min_idx_traffic], 
               color='red', s=100, marker='v', zorder=5, label=f'Min: {agents_sorted[min_idx_traffic]}')
    ax2.scatter(altruism_sorted[max_idx_traffic], traffic_sorted[max_idx_traffic], 
               color='darkgreen', s=100, marker='^', zorder=5, label=f'Max: {agents_sorted[max_idx_traffic]}')
    
    plt.tight_layout()
    
    # Save the plot
    results_dir = get_results_dir(arglist)
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f"{results_dir}/benefits_vs_altruism_line_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data for LaTeX
    latex_dir = os.path.join(results_dir, "latex_data")
    os.makedirs(latex_dir, exist_ok=True)
    
    # Save the line curve data
    with open(f"{latex_dir}/distance_benefits_vs_altruism_curve.dat", 'w') as f:
        f.write("altruism_score distance_benefit agent_name\n")
        for i in range(len(altruism_sorted)):
            f.write(f"{altruism_sorted[i]:.4f} {distance_sorted[i]:.4f} {agents_sorted[i]}\n")
    
    with open(f"{latex_dir}/traffic_benefits_vs_altruism_curve.dat", 'w') as f:
        f.write("altruism_score traffic_benefit agent_name\n")
        for i in range(len(altruism_sorted)):
            f.write(f"{altruism_sorted[i]:.4f} {traffic_sorted[i]:.4f} {agents_sorted[i]}\n")
    
    # Return statistics for summary
    return {
        'distance_altruism_correlation': corr_distance,
        'traffic_altruism_correlation': corr_traffic,
        'distance_benefit_range': [float(np.min(distance_sorted)), float(np.max(distance_sorted))],
        'traffic_benefit_range': [float(np.min(traffic_sorted)), float(np.max(traffic_sorted))],
        'altruism_range': [float(np.min(altruism_sorted)), float(np.max(altruism_sorted))],
        'agents_analyzed': len(altruism_sorted)
    }

def analyze_multidimensional_benefit_distribution(agent_benefits, env, arglist, save_dir="results"):
    """
    Analyze benefit distribution considering both distance saved and traffic reduction.
    Creates 3D Lorenz surface and calculates separate Gini coefficients.
    """
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
    
    # Calculate Gini coefficients for each dimension
    gini_distance = calculate_gini(positive_distance_benefits) if positive_distance_benefits else 0
    gini_traffic = calculate_gini(positive_traffic_benefits) if positive_traffic_benefits else 0
    
    results_dir = get_results_dir(arglist)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create 3D Lorenz Surface plot using Plotly
    if len(agent_benefits) > 2:  # Need at least 3 agents for meaningful surface
        try:
            # Get all agents and their benefits
            agents = list(agent_benefits.keys())
            n_agents = len(agents)
            
            # Extract benefits for all agents (including zeros)
            agent_distance_benefits = [agent_benefits[agent]['distance_benefit'] for agent in agents]
            agent_traffic_benefits = [agent_benefits[agent]['traffic_benefit'] for agent in agents]
            
            # Sort agents by distance benefits (ascending order)
            distance_sorted_indices = np.argsort(agent_distance_benefits)
            sorted_distance_benefits = [agent_distance_benefits[i] for i in distance_sorted_indices]
            
            # Sort agents by traffic benefits (ascending order)  
            traffic_sorted_indices = np.argsort(agent_traffic_benefits)
            sorted_traffic_benefits = [agent_traffic_benefits[i] for i in traffic_sorted_indices]
            
            # Calculate cumulative distributions
            cumulative_distance = np.cumsum(sorted_distance_benefits)
            cumulative_traffic = np.cumsum(sorted_traffic_benefits)
            
            # Normalize cumulative benefits (handle case where total might be zero or negative)
            total_distance = cumulative_distance[-1] if cumulative_distance[-1] > 0 else 1
            total_traffic = cumulative_traffic[-1] if cumulative_traffic[-1] > 0 else 1
            
            distance_cumulative_norm = cumulative_distance / total_distance
            traffic_cumulative_norm = cumulative_traffic / total_traffic
            
            # Create population shares (uniform spacing from 0 to 1)
            population_shares = np.linspace(0, 1, n_agents)
            
            # Create interpolated values for smooth surface
            n_points = min(50, n_agents)  # Limit resolution for performance
            interp_pop_shares = np.linspace(0, 1, n_points)
            
            # Interpolate cumulative benefits to uniform population grid
            interp_distance_cumulative = np.interp(interp_pop_shares, population_shares, distance_cumulative_norm)
            interp_traffic_cumulative = np.interp(interp_pop_shares, population_shares, traffic_cumulative_norm)
            
            # Create meshgrid
            X, Y = np.meshgrid(interp_distance_cumulative, interp_traffic_cumulative)
            
            # Z represents the cumulative population share for each combination
            Z = np.zeros_like(X)
            for i in range(n_points):
                for j in range(n_points):
                    # Find the maximum population share that achieves both benefit levels
                    distance_threshold = X[i, j]
                    traffic_threshold = Y[i, j]
                    
                    # Find agents that meet both thresholds
                    qualifying_agents_distance = np.sum(distance_cumulative_norm <= distance_threshold)
                    qualifying_agents_traffic = np.sum(traffic_cumulative_norm <= traffic_threshold)
                    
                    # Take the minimum (most restrictive)
                    qualifying_agents = min(qualifying_agents_distance, qualifying_agents_traffic)
                    Z[i, j] = qualifying_agents / n_agents
            
            # Create Plotly 3D surface with corrected axis configuration
            fig = go.Figure()
            
            # Add main Lorenz surface with proper colorbar configuration
            fig.add_trace(go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale='Viridis',
                opacity=0.9,
                name='Lorenz Surface',
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
            
            # Add equality plane (perfect equality reference)
            X_eq, Y_eq = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
            Z_eq = (X_eq + Y_eq) / 2
            Z_eq = np.minimum(Z_eq, 1.0)  # Cap at 1.0
            
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
            
            # Update layout with corrected axis properties
            fig.update_layout(
                title={
                    'text': '3D Lorenz Surface (Multidimensional Benefit Distribution)',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'family': 'Arial, sans-serif'}
                },
                scene=dict(
                    xaxis=dict(
                        title=dict(
                            text='Cumulative Share of Benefits (Distance)',
                            font=dict(size=14)
                        ),
                        tickfont=dict(size=12),
                        range=[0, 1]
                    ),
                    yaxis=dict(
                        title=dict(
                            text='Cumulative Share of Benefits (Traffic)',
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
                        text=f'Analysis of {n_agents} agents',
                        x=0.5, y=0.02,
                        xref='paper', yref='paper',
                        showarrow=False,
                        font=dict(size=12, style='italic')
                    )
                ]
            )
            
            # Configure Plotly for offline use
            pio.renderers.default = "browser"
            
            # Save as interactive HTML with better error handling
            html_path = f"{results_dir}/3D Lorenz Surface (Multidimensional Benefit Distribution).html"
            try:
                # Method 1: Try with plotly.offline.plot
                pyo.plot(fig, filename=html_path, auto_open=False, config={'displayModeBar': True})
                print(f"Interactive 3D Lorenz Surface saved as: {html_path}")
            except Exception as html_error:
                print(f"Error saving HTML with pyo.plot: {html_error}")
                try:
                    # Method 2: Try with fig.write_html
                    fig.write_html(html_path, include_plotlyjs=True)
                    print(f"Interactive 3D Lorenz Surface saved as: {html_path}")
                except Exception as write_error:
                    print(f"Error saving HTML with write_html: {write_error}")
                    # Create a simple fallback HTML
                    html_content = f"""
                    <html>
                    <head><title>3D Lorenz Surface</title></head>
                    <body>
                    <h1>3D Lorenz Surface (Multidimensional Benefit Distribution)</h1>
                    <p>Error generating interactive plot. Analysis of {n_agents} agents.</p>
                    <p>Error details: {write_error}</p>
                    </body>
                    </html>
                    """
                    with open(html_path, 'w') as f:
                        f.write(html_content)
                    print(f"Fallback HTML saved as: {html_path}")
            
            # Save as static PNG (requires kaleido: pip install kaleido)
            png_path = f"{results_dir}/3D Lorenz Surface (Multidimensional Benefit Distribution).png"
            try:
                # Check if kaleido is available
                fig.write_image(png_path, width=1200, height=900, scale=2)
                print(f"3D Lorenz Surface PNG saved as: {png_path}")
            except Exception as png_error:
                print(f"Could not save PNG (install kaleido for static images): {png_error}")
                print("Install with: pip install kaleido")
                
        except Exception as e:
            print(f"Error creating 3D Lorenz surface: {e}")
            import traceback
            traceback.print_exc()
            
            # Create a simple error HTML file
            error_html_path = f"{results_dir}/3D Lorenz Surface (Multidimensional Benefit Distribution).html"
            error_html_content = f"""
            <html>
            <head><title>3D Lorenz Surface - Error</title></head>
            <body>
            <h1>3D Lorenz Surface (Error)</h1>
            <p>Error Creating 3D Surface: {str(e)}</p>
            <p>Please check the console for detailed error information.</p>
            </body>
            </html>
            """
            with open(error_html_path, 'w') as f:
                f.write(error_html_content)
            print(f"Error HTML saved as: {error_html_path}")
    else:
        # Create a simple message HTML for insufficient data
        insufficient_html_path = f"{results_dir}/3D Lorenz Surface (Multidimensional Benefit Distribution).html"
        insufficient_html_content = f"""
        <html>
        <head><title>3D Lorenz Surface - Insufficient Data</title></head>
        <body>
        <h1>3D Lorenz Surface (Insufficient Data)</h1>
        <p>Insufficient Data for 3D Lorenz Surface</p>
        <p>(Need at least 3 agents)</p>
        </body>
        </html>
        """
        with open(insufficient_html_path, 'w') as f:
            f.write(insufficient_html_content)
        print(f"Insufficient data HTML saved as: {insufficient_html_path}")
    
    # Continue with the rest of the matplotlib plots (unchanged)
    # Create comprehensive visualization (original 6-panel plot using matplotlib)
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 3D Lorenz Surface (placeholder - refer to interactive version)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.text(0.5, 0.5, 'Interactive 3D Lorenz Surface\nSaved as HTML file\n(See separate file)', 
             ha='center', va='center', transform=ax1.transAxes, fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax1.set_title('3D Lorenz Surface\n(Interactive Version Available)')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # 2. Distance Benefits Lorenz Curve (2D Projection)
    ax2 = plt.subplot(2, 3, 2)
    if positive_distance_benefits:
        sorted_distance = sorted(positive_distance_benefits)
        cumulative_distance = np.cumsum(sorted_distance)
        cumulative_distance_norm = cumulative_distance / cumulative_distance[-1]
        x_distance = np.linspace(0, 1, len(cumulative_distance_norm))
        
        ax2.plot([0, 1], [0, 1], 'k--', label='Perfect Equality', linewidth=2)
        ax2.plot(x_distance, cumulative_distance_norm, 'b-', linewidth=3, label='Actual Distribution')
        ax2.fill_between(x_distance, cumulative_distance_norm, x_distance, alpha=0.3, color='blue')
        
        ax2.set_xlabel('Cumulative Share of Agents')
        ax2.set_ylabel('Cumulative Share of Distance Benefits')
        ax2.set_title(f'Distance Benefits Lorenz Curve\n(Gini: {gini_distance:.3f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No Positive Distance Benefits', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14)
    
    # 3. Traffic Reduction Lorenz Curve (2D Projection)
    ax3 = plt.subplot(2, 3, 3)
    if positive_traffic_benefits:
        sorted_traffic = sorted(positive_traffic_benefits)
        cumulative_traffic = np.cumsum(sorted_traffic)
        cumulative_traffic_norm = cumulative_traffic / cumulative_traffic[-1]
        x_traffic = np.linspace(0, 1, len(cumulative_traffic_norm))
        
        ax3.plot([0, 1], [0, 1], 'k--', label='Perfect Equality', linewidth=2)
        ax3.plot(x_traffic, cumulative_traffic_norm, 'g-', linewidth=3, label='Actual Distribution')
        ax3.fill_between(x_traffic, cumulative_traffic_norm, x_traffic, alpha=0.3, color='green')
        
        ax3.set_xlabel('Cumulative Share of Agents')
        ax3.set_ylabel('Cumulative Share of Traffic Benefits')
        ax3.set_title(f'Traffic Reduction Lorenz Curve\n(Gini: {gini_traffic:.3f})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No Positive Traffic Benefits', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=14)
    
    # 4. 2D Benefit Scatter Plot
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(distance_benefits, traffic_benefits, alpha=0.7, s=60, 
                         c='purple', edgecolors='black', linewidth=0.5)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Distance Benefit (Distance Saved)')
    ax4.set_ylabel('Traffic Reduction Benefit (Ride-sharing Participation)')
    ax4.set_title('Multidimensional Benefit Distribution\n(Distance vs Traffic Reduction)')
    ax4.grid(True, alpha=0.3)
    
    # Add correlation (handle case where all values might be the same)
    try:
        correlation = np.corrcoef(distance_benefits, traffic_benefits)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    except:
        correlation = 0.0
    
    ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax4.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"), fontsize=10)
    
    # 5. Benefit Distribution Histograms
    ax5 = plt.subplot(2, 3, 5)
    if len(distance_benefits) > 0:
        ax5.hist(distance_benefits, bins=min(20, len(set(distance_benefits))), alpha=0.7, color='blue', label='Distance Benefits', density=True)
        ax5.axvline(x=np.mean(distance_benefits), color='blue', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(distance_benefits):.3f}')
    ax5.set_xlabel('Distance Benefit')
    ax5.set_ylabel('Density')
    ax5.set_title('Distance Benefit Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(2, 3, 6)
    if len(traffic_benefits) > 0:
        ax6.hist(traffic_benefits, bins=min(20, len(set(traffic_benefits))), alpha=0.7, color='green', label='Traffic Benefits', density=True)
        ax6.axvline(x=np.mean(traffic_benefits), color='green', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(traffic_benefits):.3f}')
    ax6.set_xlabel('Traffic Reduction Benefit (Ride-sharing Distance)')
    ax6.set_ylabel('Density')
    ax6.set_title('Traffic Reduction Benefit Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/multidimensional_benefit_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate additional statistics (rest remains the same)
    combined_benefits = [data['combined_benefit'] for data in agent_benefits.values()]
    positive_combined_benefits = [b for b in combined_benefits if b > 0]
    combined_gini = calculate_gini(positive_combined_benefits) if positive_combined_benefits else 0
    
    # Concentration ratios for top 20%
    n_agents = len(distance_benefits)
    top_20_percent = max(1, n_agents // 5)
    
    sorted_distance_benefits = sorted(distance_benefits, reverse=True)
    sorted_traffic_benefits = sorted(traffic_benefits, reverse=True)
    
    top_20_distance = sum(sorted_distance_benefits[:top_20_percent])
    top_20_traffic = sum(sorted_traffic_benefits[:top_20_percent])
    
    total_positive_distance = sum(positive_distance_benefits) if positive_distance_benefits else 1
    total_positive_traffic = sum(positive_traffic_benefits) if positive_traffic_benefits else 1
    
    concentration_distance = (top_20_distance / total_positive_distance * 100) if total_positive_distance > 0 else 0
    concentration_traffic = (top_20_traffic / total_positive_traffic * 100) if total_positive_traffic > 0 else 0
    
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
        'concentration_ratio_distance_top_20': concentration_distance,
        'concentration_ratio_traffic_top_20': concentration_traffic,
        'positive_distance_benefit_agents': len(positive_distance_benefits),
        'positive_traffic_benefit_agents': len(positive_traffic_benefits),
        'total_agents_analyzed': len(agent_benefits)
    }
    
    return stats

def save_multidimensional_metrics_data(agent_benefits, multidimensional_stats, save_dir, env, arglist):
    """
    Save the multidimensional metrics data in IEEE/LaTeX compatible format.
    """
    results_dir = get_results_dir(arglist)
    os.makedirs(results_dir, exist_ok=True)
    latex_dir = os.path.join(results_dir, "latex_data")
    os.makedirs(latex_dir, exist_ok=True)
    
    # 1. Multidimensional benefit scatter data
    with open(f"{latex_dir}/multidimensional_benefits.dat", 'w') as f:
        f.write("distance_benefit traffic_benefit combined_benefit altruism agent_role\n")
        for agent, data in agent_benefits.items():
            agent_role = env.agent_role.get(agent, 'unknown')
            f.write(f"{data['distance_benefit']:.4f} {data['traffic_benefit']:.4f} "
                   f"{data['combined_benefit']:.4f} {data['avg_altruism']:.4f} {agent_role}\n")
    
    # 2. Lorenz curve data for distance benefits
    distance_benefits = [data['distance_benefit'] for data in agent_benefits.values() if data['distance_benefit'] > 0]
    if distance_benefits:
        sorted_distance = sorted(distance_benefits)
        cumulative_distance = np.cumsum(sorted_distance)
        cumulative_distance_norm = cumulative_distance / cumulative_distance[-1]
        x_distance = np.linspace(0, 1, len(cumulative_distance_norm))
        
        with open(f"{latex_dir}/lorenz_distance_benefits.dat", 'w') as f:
            f.write("agent_share cumulative_benefit\n")
            f.write("0.0000 0.0000\n")  # Start point
            for x, y in zip(x_distance, cumulative_distance_norm):
                f.write(f"{x:.4f} {y:.4f}\n")
    
    # 3. Lorenz curve data for traffic benefits
    traffic_benefits = [data['traffic_benefit'] for data in agent_benefits.values() if data['traffic_benefit'] > 0]
    if traffic_benefits:
        sorted_traffic = sorted(traffic_benefits)
        cumulative_traffic = np.cumsum(sorted_traffic)
        cumulative_traffic_norm = cumulative_traffic / cumulative_traffic[-1]
        x_traffic = np.linspace(0, 1, len(cumulative_traffic_norm))
        
        with open(f"{latex_dir}/lorenz_traffic_benefits.dat", 'w') as f:
            f.write("agent_share cumulative_benefit\n")
            f.write("0.0000 0.0000\n")  # Start point
            for x, y in zip(x_traffic, cumulative_traffic_norm):
                f.write(f"{x:.4f} {y:.4f}\n")
    
    # 4. Multidimensional distribution stats
    with open(f"{latex_dir}/multidimensional_stats.dat", 'w') as f:
        f.write("metric value\n")
        for key, value in multidimensional_stats.items():
            f.write(f"{key} {value:.6f}\n")
    
    # 5. Role-based analysis
    driver_distance_benefits = []
    driver_traffic_benefits = []
    rider_distance_benefits = []
    rider_traffic_benefits = []
    
    for agent, data in agent_benefits.items():
        role = env.agent_role.get(agent, 'unknown')
        if role == 'driver':
            driver_distance_benefits.append(data['distance_benefit'])
            driver_traffic_benefits.append(data['traffic_benefit'])
        elif role == 'rider':
            rider_distance_benefits.append(data['distance_benefit'])
            rider_traffic_benefits.append(data['traffic_benefit'])
    
    with open(f"{latex_dir}/role_based_benefits.dat", 'w') as f:
        f.write("role mean_distance_benefit mean_traffic_benefit count\n")
        if driver_distance_benefits:
            f.write(f"driver {np.mean(driver_distance_benefits):.4f} {np.mean(driver_traffic_benefits):.4f} {len(driver_distance_benefits)}\n")
        if rider_distance_benefits:
            f.write(f"rider {np.mean(rider_distance_benefits):.4f} {np.mean(rider_traffic_benefits):.4f} {len(rider_distance_benefits)}\n")
    
    # 6. Updated summary with multidimensional metrics
    with open(f"{latex_dir}/multidimensional_summary.dat", 'w') as f:
        f.write("metric value description\n")
        f.write(f"gini_distance {multidimensional_stats['gini_distance_benefits']:.4f} \"Distance benefit inequality\"\n")
        f.write(f"gini_traffic {multidimensional_stats['gini_traffic_benefits']:.4f} \"Traffic benefit inequality\"\n")
        f.write(f"gini_combined {multidimensional_stats['gini_combined_benefits']:.4f} \"Combined benefit inequality\"\n")
        f.write(f"correlation_benefits {multidimensional_stats['correlation_distance_traffic']:.4f} \"Distance-Traffic benefit correlation\"\n")
        f.write(f"concentration_distance {multidimensional_stats['concentration_ratio_distance_top_20']:.2f} \"Top 20% distance benefit share\"\n")
        f.write(f"concentration_traffic {multidimensional_stats['concentration_ratio_traffic_top_20']:.2f} \"Top 20% traffic benefit share\"\n")
        f.write(f"mean_distance_benefit {multidimensional_stats['mean_distance_benefit']:.4f} \"Average distance benefit\"\n")
        f.write(f"mean_traffic_benefit {multidimensional_stats['mean_traffic_benefit']:.4f} \"Average traffic benefit\"\n")
        
        # Role-specific metrics
        if driver_distance_benefits:
            f.write(f"driver_mean_distance {np.mean(driver_distance_benefits):.4f} \"Driver average distance benefit\"\n")
            f.write(f"driver_mean_traffic {np.mean(driver_traffic_benefits):.4f} \"Driver average traffic benefit\"\n")
        if rider_distance_benefits:
            f.write(f"rider_mean_distance {np.mean(rider_distance_benefits):.4f} \"Rider average distance benefit\"\n")
            f.write(f"rider_mean_traffic {np.mean(rider_traffic_benefits):.4f} \"Rider average traffic benefit\"\n")
    
    print("Multidimensional benefit analysis data saved for LaTeX integration")

def plot_agent_benefit_scatter(agent_benefits, env, arglist):
    """
    Create scatter plot of agent benefits vs altruism scores.
    Now includes both distance and traffic benefits.
    """
    distance_benefits = [data['distance_benefit'] for data in agent_benefits.values()]
    traffic_benefits = [data['traffic_benefit'] for data in agent_benefits.values()]
    combined_benefits = [data['combined_benefit'] for data in agent_benefits.values()]
    altruism_scores = [data['avg_altruism'] for data in agent_benefits.values()]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Distance Benefits vs Altruism
    scatter1 = ax1.scatter(altruism_scores, distance_benefits, alpha=0.6, s=50, 
                          c='blue', edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Average Altruism Score')
    ax1.set_ylabel('Distance Benefit')
    ax1.set_title('Distance Benefits vs Altruism')
    ax1.grid(True, alpha=0.3)
    
    # Calculate and display correlation
    corr_distance = np.corrcoef(altruism_scores, distance_benefits)[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {corr_distance:.3f}', transform=ax1.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    # 2. Traffic Benefits vs Altruism
    scatter2 = ax2.scatter(altruism_scores, traffic_benefits, alpha=0.6, s=50, 
                          c='green', edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Average Altruism Score')
    ax2.set_ylabel('Traffic Benefit')
    ax2.set_title('Traffic Benefits vs Altruism')
    ax2.grid(True, alpha=0.3)
    
    corr_traffic = np.corrcoef(altruism_scores, traffic_benefits)[0, 1]
    ax2.text(0.05, 0.95, f'Correlation: {corr_traffic:.3f}', transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    # 3. Combined Benefits vs Altruism
    scatter3 = ax3.scatter(altruism_scores, combined_benefits, alpha=0.6, s=50, 
                          c='purple', edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Average Altruism Score')
    ax3.set_ylabel('Combined Benefit')
    ax3.set_title('Combined Benefits vs Altruism')
    ax3.grid(True, alpha=0.3)
    
    corr_combined = np.corrcoef(altruism_scores, combined_benefits)[0, 1]
    ax3.text(0.05, 0.95, f'Correlation: {corr_combined:.3f}', transform=ax3.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightpink"))
    
    # 4. Role-based analysis
    driver_benefits = []
    driver_altruism = []
    rider_benefits = []
    rider_altruism = []
    
    for agent, data in agent_benefits.items():
        role = env.agent_role.get(agent, 'unknown')
        if role == 'driver':
            driver_benefits.append(data['combined_benefit'])
            driver_altruism.append(data['avg_altruism'])
        elif role == 'rider':
            rider_benefits.append(data['combined_benefit'])
            rider_altruism.append(data['avg_altruism'])
    
    if driver_benefits:
        ax4.scatter(driver_altruism, driver_benefits, alpha=0.6, s=50, 
                   c='red', label='Drivers', edgecolors='black', linewidth=0.5)
    if rider_benefits:
        ax4.scatter(rider_altruism, rider_benefits, alpha=0.6, s=50, 
                   c='blue', label='Riders', edgecolors='black', linewidth=0.5)
    
    ax4.set_xlabel('Average Altruism Score')
    ax4.set_ylabel('Combined Benefit')
    ax4.set_title('Combined Benefits by Role')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    results_dir = get_results_dir(arglist)
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f"{results_dir}/agent_benefit_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'distance_altruism_correlation': corr_distance,
        'traffic_altruism_correlation': corr_traffic,
        'combined_altruism_correlation': corr_combined
    }

def analyze_winner_loser_distribution(agent_benefits, env, arglist):
    """
    Analyze the distribution of winners and losers based on combined benefits.
    """
    combined_benefits = [data['combined_benefit'] for data in agent_benefits.values()]
    
    winners = sum(1 for benefit in combined_benefits if benefit > 0)
    losers = sum(1 for benefit in combined_benefits if benefit < 0)
    neutral = sum(1 for benefit in combined_benefits if benefit == 0)
    total_agents = len(combined_benefits)
    
    winner_percentage = (winners / total_agents * 100) if total_agents > 0 else 0
    loser_percentage = (losers / total_agents * 100) if total_agents > 0 else 0
    neutral_percentage = (neutral / total_agents * 100) if total_agents > 0 else 0
    
    # Additional statistics
    winner_benefits = [b for b in combined_benefits if b > 0]
    loser_benefits = [b for b in combined_benefits if b < 0]
    
    avg_winner_benefit = np.mean(winner_benefits) if winner_benefits else 0
    avg_loser_benefit = np.mean(loser_benefits) if loser_benefits else 0
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    labels = ['Winners', 'Losers', 'Neutral']
    sizes = [winners, losers, neutral]
    colors = ['green', 'red', 'gray']
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                      startangle=90, textprops={'fontsize': 12})
    ax1.set_title('Winner/Loser Distribution')
    
    # Bar chart with benefit magnitudes
    categories = ['Winners', 'Losers']
    avg_benefits = [avg_winner_benefit, abs(avg_loser_benefit)]
    bar_colors = ['green', 'red']
    
    bars = ax2.bar(categories, avg_benefits, color=bar_colors, alpha=0.7)
    ax2.set_ylabel('Average Benefit Magnitude')
    ax2.set_title('Average Benefit by Category')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_benefits):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    results_dir = get_results_dir(arglist)
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f"{results_dir}/winner_loser_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    stats = {
        'winners': winners,
        'losers': losers,
        'neutral': neutral,
        'total_agents': total_agents,
        'winner_percentage': winner_percentage,
        'loser_percentage': loser_percentage,
        'neutral_percentage': neutral_percentage,
        'avg_winner_benefit': avg_winner_benefit,
        'avg_loser_benefit': avg_loser_benefit
    }
    
    print(f"\n=== WINNER/LOSER ANALYSIS ===")
    print(f"Winners: {winners} ({winner_percentage:.1f}%)")
    print(f"Losers: {losers} ({loser_percentage:.1f}%)")
    print(f"Neutral: {neutral} ({neutral_percentage:.1f}%)")
    print(f"Average winner benefit: {avg_winner_benefit:.2f}")
    print(f"Average loser benefit: {avg_loser_benefit:.2f}")
    
    return stats

def analyze_benefit_distribution(agent_benefits, env, arglist):
    """
    Analyze the distribution of benefits using traditional single-dimension metrics.
    This is the original analysis for comparison with multidimensional approach.
    """
    distance_benefits = [data['distance_benefit'] for data in agent_benefits.values()]
    
    # Separate positive benefits for Gini calculation
    positive_benefits = [b for b in distance_benefits if b > 0]
    
    def calculate_gini(values):
        """Calculate Gini coefficient (0 = perfect equality, 1 = perfect inequality)"""
        if not values or len(values) == 0:
            return 0
        values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(values)
        return (n + 1 - 2 * sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
    
    gini_coefficient = calculate_gini(positive_benefits) if positive_benefits else 0
    
    # Create Lorenz curve
    if positive_benefits:
        sorted_benefits = sorted(positive_benefits)
        cumulative_benefits = np.cumsum(sorted_benefits)
        cumulative_benefits_norm = cumulative_benefits / cumulative_benefits[-1]
        x_values = np.linspace(0, 1, len(cumulative_benefits_norm))
        
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Equality', linewidth=2)
        plt.plot(x_values, cumulative_benefits_norm, 'b-', linewidth=3, label='Actual Distribution')
        plt.fill_between(x_values, cumulative_benefits_norm, x_values, alpha=0.3, color='blue')
        
        plt.xlabel('Cumulative Share of Agents')
        plt.ylabel('Cumulative Share of Benefits')
        plt.title(f'Lorenz Curve for Distance Benefits\n(Gini Coefficient: {gini_coefficient:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        results_dir = get_results_dir(arglist)
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(f"{results_dir}/lorenz_curve_distance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Statistics
    stats = {
        'gini_coefficient': gini_coefficient,
        'mean_benefit': np.mean(distance_benefits),
        'std_benefit': np.std(distance_benefits),
        'median_benefit': np.median(distance_benefits),
        'min_benefit': np.min(distance_benefits),
        'max_benefit': np.max(distance_benefits),
        'positive_benefit_count': len(positive_benefits),
        'total_agents': len(distance_benefits)
    }
    
    return stats

def save_new_metrics_data(agent_benefits, winner_loser_stats, distribution_stats, 
                         benefit_altruism_correlation, env, arglist, line_curve_stats=None):
    """
    Save new metrics data in LaTeX-compatible format.
    """
    results_dir = get_results_dir(arglist)
    os.makedirs(results_dir, exist_ok=True)
    latex_dir = os.path.join(results_dir, "latex_data")
    os.makedirs(latex_dir, exist_ok=True)
    
    # 1. Winner/Loser statistics
    with open(f"{latex_dir}/winner_loser_stats.dat", 'w') as f:
        f.write("metric value\n")
        for key, value in winner_loser_stats.items():
            f.write(f"{key} {value}\n")
    
    # 2. Benefit distribution statistics
    with open(f"{latex_dir}/benefit_distribution_stats.dat", 'w') as f:
        f.write("metric value\n")
        for key, value in distribution_stats.items():
            f.write(f"{key} {value:.6f}\n")
    
    # 3. Benefit-altruism correlations
    with open(f"{latex_dir}/benefit_altruism_correlations.dat", 'w') as f:
        f.write("correlation_type value\n")
        for key, value in benefit_altruism_correlation.items():
            f.write(f"{key} {value:.6f}\n")
    
    # 4. NEW: Line curve statistics
    if line_curve_stats:
        with open(f"{latex_dir}/line_curve_stats.dat", 'w') as f:
            f.write("metric value\n")
            for key, value in line_curve_stats.items():
                if isinstance(value, list):
                    f.write(f"{key}_min {value[0]:.6f}\n")
                    f.write(f"{key}_max {value[1]:.6f}\n")
                else:
                    f.write(f"{key} {value}\n")
    
    # 5. Individual agent data for detailed analysis
    with open(f"{latex_dir}/agent_benefits_detailed.dat", 'w') as f:
        f.write("agent distance_benefit traffic_benefit combined_benefit altruism role\n")
        for agent, data in agent_benefits.items():
            role = env.agent_role.get(agent, 'unknown')
            f.write(f"{agent} {data['distance_benefit']:.4f} {data['traffic_benefit']:.4f} "
                   f"{data['combined_benefit']:.4f} {data['avg_altruism']:.4f} {role}\n")
    
    print("New metrics data saved for LaTeX integration")

def analyze_altruism_transactions(env, arglist, save_dir=None):
    """
    Analyze the actual altruism transactions that occurred during simulation.
    """
    results_dir = get_results_dir(arglist)
    os.makedirs(results_dir, exist_ok=True)
    transactions = env.altruism_transactions
    
    if not transactions:
        print("No altruism transactions found.")
        return
    
    # Statistics
    total_transactions = len(transactions)
    avg_driver_gain = np.mean([t['driver_gain'] for t in transactions])
    avg_rider_loss = np.mean([t['rider_loss'] for t in transactions])
    avg_detour_cost = np.mean([t['detour_cost'] for t in transactions])
    
    # Transactions per day
    daily_counts = {}
    for transaction in transactions:
        day = transaction['day']
        daily_counts[day] = daily_counts.get(day, 0) + 1
    
    print(f"\n=== ALTRUISM TRANSACTION ANALYSIS ===")
    print(f"Total transactions: {total_transactions}")
    print(f"Average driver gain: {avg_driver_gain:.4f}")
    print(f"Average rider loss: {avg_rider_loss:.4f}")
    print(f"Average detour cost: {avg_detour_cost:.4f}")
    print(f"Transactions per day: {daily_counts}")
    
    # Save detailed transaction data
    with open(f"{results_dir}/altruism_transactions.txt", 'w') as f:
        f.write("Day,Driver,Rider,DriverGain,RiderLoss,DetourCost,NormalizedDetourCost\n")
        for t in transactions:
            f.write(f"{t['day']},{t['driver']},{t['rider']},{t['driver_gain']:.4f},"
                   f"{t['rider_loss']:.4f},{t['detour_cost']:.4f},{t['normalized_detour_cost']:.4f}\n")
    
    return transactions

def perform_granger_causality_analysis(daily_altruism_series, daily_benefit_series, 
                                     daily_aggregate_altruism, daily_aggregate_benefits, 
                                     num_days, env, arglist):
    """
    Perform Granger causality analysis between altruism and benefits.
    Tests both individual agent level and aggregate level causality.
    """
    results = {
        'agent_level_results': {},
        'aggregate_results': {},
        'summary_statistics': {}
    }
    
    print("Performing Granger Causality Analysis...")
    
    # 1. AGENT-LEVEL ANALYSIS
    print("\n1. AGENT-LEVEL ANALYSIS")
    print("-" * 30)
    
    valid_agents = 0
    altruism_to_benefit_significant = 0
    benefit_to_altruism_significant = 0
    
    # Only analyze agents with sufficient data points
    min_observations = max(8, num_days // 2)  # Need at least 8 observations or half the days
    
    for agent in daily_altruism_series:
        if (len(daily_altruism_series[agent]) >= min_observations and 
            agent in daily_benefit_series and 
            len(daily_benefit_series[agent]) >= min_observations - 1):  # Benefits start from day 2
            
            try:
                # Align the series (altruism starts from day 1, benefits from day 2)
                altruism_series = daily_altruism_series[agent][1:]  # Skip day 1 altruism
                benefit_series = daily_benefit_series[agent]
                
                if len(altruism_series) != len(benefit_series):
                    min_len = min(len(altruism_series), len(benefit_series))
                    altruism_series = altruism_series[:min_len]
                    benefit_series = benefit_series[:min_len]
                
                if len(altruism_series) >= min_observations - 1:
                    # Create DataFrame for Granger test
                    data = pd.DataFrame({
                        'altruism': altruism_series,
                        'benefits': benefit_series
                    })
                    
                    # Test stationarity and difference if necessary
                    altruism_stationary = make_stationary(altruism_series, f"{agent}_altruism")
                    benefits_stationary = make_stationary(benefit_series, f"{agent}_benefits")
                    
                    if altruism_stationary is not None and benefits_stationary is not None:
                        # Test Altruism -> Benefits
                        causality_alt_to_ben = test_granger_causality(
                            altruism_stationary, benefits_stationary, 
                            f"{agent}: Altruism -> Benefits"
                        )
                        
                        # Test Benefits -> Altruism  
                        causality_ben_to_alt = test_granger_causality(
                            benefits_stationary, altruism_stationary,
                            f"{agent}: Benefits -> Altruism"
                        )
                        
                        results['agent_level_results'][agent] = {
                            'altruism_to_benefits': causality_alt_to_ben,
                            'benefits_to_altruism': causality_ben_to_alt,
                            'data_points': len(altruism_series)
                        }
                        
                        valid_agents += 1
                        if causality_alt_to_ben['significant']:
                            altruism_to_benefit_significant += 1
                        if causality_ben_to_alt['significant']:
                            benefit_to_altruism_significant += 1
                            
            except Exception as e:
                print(f"Error analyzing agent {agent}: {e}")
                continue
    
    print(f"Analyzed {valid_agents} agents with sufficient data")
    print(f"Altruism -> Benefits significant: {altruism_to_benefit_significant}/{valid_agents}")
    print(f"Benefits -> Altruism significant: {benefit_to_altruism_significant}/{valid_agents}")
    
    # 2. AGGREGATE-LEVEL ANALYSIS
    print("\n2. AGGREGATE-LEVEL ANALYSIS")
    print("-" * 30)
    
    if len(daily_aggregate_altruism) >= min_observations and len(daily_aggregate_benefits) >= min_observations - 1:
        # Align aggregate series
        agg_altruism = daily_aggregate_altruism[1:]  # Skip day 1
        agg_benefits = daily_aggregate_benefits
        
        min_len = min(len(agg_altruism), len(agg_benefits))
        agg_altruism = agg_altruism[:min_len]
        agg_benefits = agg_benefits[:min_len]
        
        # Test stationarity
        agg_altruism_stationary = make_stationary(agg_altruism, "Aggregate Altruism")
        agg_benefits_stationary = make_stationary(agg_benefits, "Aggregate Benefits")
        
        if agg_altruism_stationary is not None and agg_benefits_stationary is not None:
            # Test causality in both directions
            agg_causality_alt_to_ben = test_granger_causality(
                agg_altruism_stationary, agg_benefits_stationary,
                "Aggregate: Altruism -> Benefits"
            )
            
            agg_causality_ben_to_alt = test_granger_causality(
                agg_benefits_stationary, agg_altruism_stationary,
                "Aggregate: Benefits -> Altruism" 
            )
            
            results['aggregate_results'] = {
                'altruism_to_benefits': agg_causality_alt_to_ben,
                'benefits_to_altruism': agg_causality_ben_to_alt,
                'data_points': len(agg_altruism)
            }
            
            print(f"Aggregate analysis completed with {len(agg_altruism)} data points")
    else:
        print("Insufficient data for aggregate analysis")
        results['aggregate_results'] = None
    
    # 3. SUMMARY STATISTICS
    results['summary_statistics'] = {
        'total_agents_analyzed': valid_agents,
        'altruism_to_benefits_significant_count': altruism_to_benefit_significant,
        'benefits_to_altruism_significant_count': benefit_to_altruism_significant,
        'altruism_to_benefits_percentage': (altruism_to_benefit_significant / valid_agents * 100) if valid_agents > 0 else 0,
        'benefits_to_altruism_percentage': (benefit_to_altruism_significant / valid_agents * 100) if valid_agents > 0 else 0,
        'min_observations_required': min_observations
    }
    
    return results

def make_stationary(series, series_name):
    """
    Test for stationarity and make series stationary if necessary.
    Returns the stationary series or None if it cannot be made stationary.
    """
    try:
        series = np.array(series)
        
        # Remove any inf or nan values
        series = series[np.isfinite(series)]
        
        if len(series) < 4:
            return None
        
        # Test original series
        adf_result = adfuller(series, autolag='AIC')
        
        if adf_result[1] <= 0.05:  # Series is already stationary
            return series
        else:
            # Take first difference
            diff_series = np.diff(series)
            
            if len(diff_series) < 3:
                return None
                
            # Test differenced series
            adf_result_diff = adfuller(diff_series, autolag='AIC')
            
            if adf_result_diff[1] <= 0.05:
                return diff_series
            else:
                # Take second difference if needed
                diff2_series = np.diff(diff_series)
                
                if len(diff2_series) < 3:
                    return None
                    
                adf_result_diff2 = adfuller(diff2_series, autolag='AIC')
                
                if adf_result_diff2[1] <= 0.05:
                    return diff2_series
                else:
                    print(f"Could not make {series_name} stationary")
                    return None
    except:
        return None

def test_granger_causality(x_series, y_series, test_name):
    """
    Test Granger causality from x to y.
    Returns dictionary with test results.
    """
    try:
        # Ensure series are same length
        min_len = min(len(x_series), len(y_series))
        x_series = x_series[:min_len]
        y_series = y_series[:min_len]
        
        if min_len < 4:
            return {'significant': False, 'p_value': 1.0, 'f_statistic': 0.0, 'error': 'Insufficient data'}
        
        # Create DataFrame
        data = pd.DataFrame({'y': y_series, 'x': x_series})
        
        # Determine maximum lag (use smaller of 4 or 1/3 of data length)
        max_lag = min(4, max(1, len(data) // 3))
        
        # Perform Granger causality test
        gc_result = grangercausalitytests(data[['y', 'x']], maxlag=max_lag, verbose=False)
        
        # Get results for optimal lag (usually lag 1 for daily data)
        optimal_lag = 1 if max_lag >= 1 else max_lag
        
        if optimal_lag in gc_result:
            f_stat = gc_result[optimal_lag][0]['ssr_ftest'][0]
            p_value = gc_result[optimal_lag][0]['ssr_ftest'][1]
            
            is_significant = p_value < 0.05
            
            if is_significant:
                print(f"✓ {test_name}: SIGNIFICANT (p={p_value:.4f}, F={f_stat:.4f})")
            
            return {
                'significant': is_significant,
                'p_value': p_value,
                'f_statistic': f_stat,
                'optimal_lag': optimal_lag,
                'error': None
            }
        else:
            return {'significant': False, 'p_value': 1.0, 'f_statistic': 0.0, 'error': 'Lag not available'}
            
    except Exception as e:
        return {'significant': False, 'p_value': 1.0, 'f_statistic': 0.0, 'error': str(e)}

def plot_time_series_and_causality(daily_altruism_series, daily_benefit_series,
                                  daily_aggregate_altruism, daily_aggregate_benefits,
                                  granger_results, save_dir, env, arglist):
    """
    Create comprehensive time series and causality visualization.
    """
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Aggregate time series
    ax1 = plt.subplot(3, 2, 1)
    days = range(1, len(daily_aggregate_altruism) + 1)
    plt.plot(days, daily_aggregate_altruism, 'b-', linewidth=2, label='Mean Altruism')
    plt.xlabel('Day')
    plt.ylabel('Mean Altruism Score')
    plt.title('Aggregate Altruism Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    ax2 = plt.subplot(3, 2, 2)
    benefit_days = range(2, len(daily_aggregate_benefits) + 2)  # Benefits start from day 2
    plt.plot(benefit_days, daily_aggregate_benefits, 'r-', linewidth=2, label='Mean Benefits')
    plt.xlabel('Day')
    plt.ylabel('Mean Distance Saved')
    plt.title('Aggregate Benefits Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. Sample individual agent time series (top 6 agents with most data)
    ax3 = plt.subplot(3, 2, 3)
    
    # Select top agents by data availability
    agent_data_lengths = {agent: len(series) for agent, series in daily_altruism_series.items()}
    top_agents = sorted(agent_data_lengths.items(), key=lambda x: x[1], reverse=True)[:6]
    
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    for i, (agent, _) in enumerate(top_agents):
        if agent in daily_altruism_series:
            days = range(1, len(daily_altruism_series[agent]) + 1)
            plt.plot(days, daily_altruism_series[agent], color=colors[i], 
                    linewidth=1, alpha=0.7, label=f'{agent}')
    
    plt.xlabel('Day')
    plt.ylabel('Altruism Score')
    plt.title('Individual Agent Altruism Over Time (Top 6 Agents)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    ax4 = plt.subplot(3, 2, 4)
    for i, (agent, _) in enumerate(top_agents):
        if agent in daily_benefit_series and len(daily_benefit_series[agent]) > 0:
            days = range(2, len(daily_benefit_series[agent]) + 2)  # Benefits start from day 2
            plt.plot(days, daily_benefit_series[agent], color=colors[i], 
                    linewidth=1, alpha=0.7, label=f'{agent}')
    
    plt.xlabel('Day')
    plt.ylabel('Distance Saved')
    plt.title('Individual Agent Benefits Over Time (Top 6 Agents)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 3. Granger causality summary
    ax5 = plt.subplot(3, 1, 3)
    ax5.axis('off')
    
    # Prepare summary text
    summary_stats = granger_results['summary_statistics']
    agg_results = granger_results.get('aggregate_results', {})
    
    summary_text = f"""
    GRANGER CAUSALITY ANALYSIS SUMMARY
    
    AGENT-LEVEL RESULTS:
    • Total agents analyzed: {summary_stats['total_agents_analyzed']}
    • Altruism → Benefits significant: {summary_stats['altruism_to_benefits_significant_count']} ({summary_stats['altruism_to_benefits_percentage']:.1f}%)
    • Benefits → Altruism significant: {summary_stats['benefits_to_altruism_significant_count']} ({summary_stats['benefits_to_altruism_percentage']:.1f}%)
    
    AGGREGATE-LEVEL RESULTS:
    """
    
    if agg_results and 'altruism_to_benefits' in agg_results:
        alt_to_ben = agg_results['altruism_to_benefits']
        ben_to_alt = agg_results['benefits_to_altruism']
        
        summary_text += f"""• Altruism → Benefits: {'SIGNIFICANT' if alt_to_ben['significant'] else 'NOT SIGNIFICANT'} (p={alt_to_ben['p_value']:.4f})
    • Benefits → Altruism: {'SIGNIFICANT' if ben_to_alt['significant'] else 'NOT SIGNIFICANT'} (p={ben_to_alt['p_value']:.4f})
    • Data points: {agg_results['data_points']}
    
    INTERPRETATION:
    """
        if alt_to_ben['significant'] and ben_to_alt['significant']:
            summary_text += "• BIDIRECTIONAL CAUSALITY detected at aggregate level"
        elif alt_to_ben['significant']:
            summary_text += "• UNIDIRECTIONAL causality: Altruism drives Benefits"
        elif ben_to_alt['significant']:
            summary_text += "• UNIDIRECTIONAL causality: Benefits drive Altruism"
        else:
            summary_text += "• NO CAUSALITY detected at aggregate level"
            
    else:
        summary_text += "• Insufficient data for aggregate analysis"
    
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/granger_causality_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Granger causality visualization saved to {save_dir}/granger_causality_analysis.png")

def save_granger_causality_results(granger_results, save_dir, env, arglist):
    """
    Save detailed Granger causality results to files.
    """
    import os
    import json
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save JSON results
    json_path = f"{save_dir}/granger_causality_results.json"
    with open(json_path, 'w') as f:
        json.dump(granger_results, f, indent=2, default=str)
    
    # Save CSV for LaTeX import
    latex_dir = f"{save_dir}/latex_data"
    os.makedirs(latex_dir, exist_ok=True)
    
    # Agent-level summary
    with open(f"{latex_dir}/granger_causality_summary.dat", 'w') as f:
        f.write("metric value\n")
        stats = granger_results['summary_statistics']
        f.write(f"agents_analyzed {stats['total_agents_analyzed']}\n")
        f.write(f"altruism_to_benefits_significant {stats['altruism_to_benefits_significant_count']}\n")
        f.write(f"benefits_to_altruism_significant {stats['benefits_to_altruism_significant_count']}\n")
        f.write(f"altruism_to_benefits_percentage {stats['altruism_to_benefits_percentage']:.2f}\n")
        f.write(f"benefits_to_altruism_percentage {stats['benefits_to_altruism_percentage']:.2f}\n")
        
        # Aggregate results
        agg = granger_results.get('aggregate_results', {})
        if agg and 'altruism_to_benefits' in agg:
            f.write(f"aggregate_altruism_to_benefits_significant {1 if agg['altruism_to_benefits']['significant'] else 0}\n")
            f.write(f"aggregate_benefits_to_altruism_significant {1 if agg['benefits_to_altruism']['significant'] else 0}\n")
            f.write(f"aggregate_altruism_to_benefits_pvalue {agg['altruism_to_benefits']['p_value']:.6f}\n")
            f.write(f"aggregate_benefits_to_altruism_pvalue {agg['benefits_to_altruism']['p_value']:.6f}\n")
        else:
            f.write("aggregate_altruism_to_benefits_significant 0\n")
            f.write("aggregate_benefits_to_altruism_significant 0\n")
            f.write("aggregate_altruism_to_benefits_pvalue 1.000000\n")
            f.write("aggregate_benefits_to_altruism_pvalue 1.000000\n")
    
    print(f"Granger causality results saved to {save_dir}")

def calculate_traffic_density_metrics(paths_sharing_by_day, paths_no_sharing_by_day, env_height, env_width, total_days, env, arglist, rho_threshold=2.0):
    """
    Calculate traffic density metrics using the improved method:
    DENSE(d) = sum over grid cells of indicator(density(c) > rho_threshold)
    density(c) = E_t[sum over vehicles of indicator(vehicle location = c)]
    
    Args:
        rho_threshold: Threshold for considering a cell as "dense" (average vehicles per day)
    """
    # FIX (DeepPool traffic): delegate to the corrected metric in ars.metrics, which
    # rasterises each on-road vehicle's DIJKSTRA route (via env.daily_pickups +
    # _driver_route_cells) on the SAME basis as the distance accounting, and uses an
    # adaptive dense threshold. The old body below rasterised the raw per-tick
    # simulation trajectory (inflated, inconsistent with the no-sharing Dijkstra
    # baseline) and produced impossible negatives; it is kept only as unreachable
    # reference code after this early return. Return format is identical
    # (metrics_dict, avg_density_sharing, avg_density_no_sharing).
    from ars.metrics import calculate_traffic_density_metrics as _fixed_traffic
    return _fixed_traffic(paths_sharing_by_day, paths_no_sharing_by_day,
                          env_height, env_width, total_days, env, arglist, rho_threshold)

    print(f"Calculating traffic density metrics with threshold ρ = {rho_threshold}")

    # Initialize density grids
    density_grid_sharing = np.zeros((env_height, env_width))
    density_grid_no_sharing = np.zeros((env_height, env_width))
    
    days = sorted(paths_sharing_by_day.keys())
    
    # For each day, count vehicles passing through each cell
    for day in days:
        paths_sharing = paths_sharing_by_day[day]
        paths_no_sharing = paths_no_sharing_by_day[day]
        
        # Daily grids for this specific day
        daily_sharing_grid = np.zeros((env_height, env_width))
        daily_no_sharing_grid = np.zeros((env_height, env_width))
        
        # Get picked up riders for this day (to avoid double counting)
        picked_up_riders = set()
        for driver_agent, riders_list in env.riders_drivers.items():
            for rider_obj in riders_list:
                for i, agent in enumerate(env.agents):
                    if env.agent_objects[i] == rider_obj:
                        picked_up_riders.add(agent)
                        break
        
        # Count vehicles for sharing scenario
        for agent, path in paths_sharing.items():
            if env.agent_role.get(agent, 'active') in ['dropout', 'never_joined']:
                continue
                
            should_count = False
            if env.agent_role.get(agent) == "driver":
                should_count = True
            elif env.agent_role.get(agent) == "rider" and agent not in picked_up_riders:
                should_count = True
            
            if should_count and path:
                for pos in path:
                    x, y = pos[0], pos[1]
                    if 0 <= x < env_height and 0 <= y < env_width:
                        daily_sharing_grid[x, y] += 1
        
        # Count vehicles for no-sharing scenario
        for agent, path in paths_no_sharing.items():
            if env.agent_role.get(agent, 'active') in ['dropout', 'never_joined']:
                continue
                
            if path:
                for pos in path:
                    x, y = pos[0], pos[1]
                    if 0 <= x < env_height and 0 <= y < env_width:
                        daily_no_sharing_grid[x, y] += 1
        
        # Add to cumulative density
        density_grid_sharing += daily_sharing_grid
        density_grid_no_sharing += daily_no_sharing_grid
    
    # Calculate average density per cell across all days
    avg_density_sharing = density_grid_sharing / len(days)
    avg_density_no_sharing = density_grid_no_sharing / len(days)
    
    # Calculate DENSE metrics
    dense_cells_sharing = np.sum(avg_density_sharing > rho_threshold)
    dense_cells_no_sharing = np.sum(avg_density_no_sharing > rho_threshold)
    
    # Additional metrics
    total_cells = env_height * env_width
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
    
    # Congestion hotspots (cells with very high density)
    high_congestion_threshold = rho_threshold * 2  # Double the threshold for hotspots
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
    
    print(f"\n=== TRAFFIC DENSITY ANALYSIS (ρ = {rho_threshold}) ===")
    print(f"Dense cells - No sharing: {dense_cells_no_sharing} ({dense_percentage_no_sharing:.1f}%)")
    print(f"Dense cells - Sharing: {dense_cells_sharing} ({dense_percentage_sharing:.1f}%)")
    print(f"Dense cell reduction: {dense_cell_reduction} cells ({dense_cell_reduction_percentage:.1f}%)")
    print(f"Total traffic reduction: {traffic_reduction_percentage:.1f}%")
    print(f"Peak density reduction: {peak_density_reduction:.1f}%")
    print(f"Congestion hotspot reduction: {hotspots_no_sharing - hotspots_sharing} hotspots")
    
    return metrics, avg_density_sharing, avg_density_no_sharing

def create_traffic_density_visualization(avg_density_sharing, avg_density_no_sharing, metrics, save_dir, env, arglist):
    """
    Create paper-worthy traffic density visualization with three side-by-side heatmaps.
    """
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
    from matplotlib.colors import LinearSegmentedColormap
    
    # Custom colormap for density: white (0) to red (max)
    density_colors = ['white', 'red']
    density_cmap = LinearSegmentedColormap.from_list('white_to_red', density_colors, N=256)
    
    # Custom colormap for reduction: red (negative) -> white (0) -> green (positive)
    reduction_colors = ['red', 'white', 'green']
    reduction_cmap = LinearSegmentedColormap.from_list('red_white_green', reduction_colors, N=256)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
    
    # Find unified color scale for first two plots
    vmin = 0  # Start from 0 for better visualization
    vmax = max(avg_density_sharing.max(), avg_density_no_sharing.max())
    
    # 1. No-sharing scenario heatmap
    im1 = ax1.imshow(avg_density_no_sharing, cmap=density_cmap, vmin=vmin, vmax=vmax, 
                     interpolation='nearest', aspect='equal')
    ax1.set_title('(a) No Ride-Sharing', fontweight='bold', pad=15)
    ax1.set_xlabel('Grid X', fontweight='bold')
    ax1.set_ylabel('Grid Y', fontweight='bold')
    
    # Add colorbar with proper formatting
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02)
    cbar1.set_label('Vehicle Density', fontweight='bold', rotation=270, labelpad=20)
    cbar1.ax.tick_params(labelsize=11)
    
    # Set ticks to show grid boundaries clearly
    ax1.set_xticks(np.arange(0, avg_density_no_sharing.shape[1], 3))
    ax1.set_yticks(np.arange(0, avg_density_no_sharing.shape[0], 3))
    
    # 2. Sharing scenario heatmap
    im2 = ax2.imshow(avg_density_sharing, cmap=density_cmap, vmin=vmin, vmax=vmax, 
                     interpolation='nearest', aspect='equal')
    ax2.set_title('(b) With Ride-Sharing', fontweight='bold', pad=15)
    ax2.set_xlabel('Grid X', fontweight='bold')
    ax2.set_ylabel('Grid Y', fontweight='bold')
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02)
    cbar2.set_label('Vehicle Density', fontweight='bold', rotation=270, labelpad=20)
    cbar2.ax.tick_params(labelsize=11)
    
    # Set ticks
    ax2.set_xticks(np.arange(0, avg_density_sharing.shape[1], 3))
    ax2.set_yticks(np.arange(0, avg_density_sharing.shape[0], 3))
    
    # 3. Traffic reduction heatmap (diverging colormap)
    density_diff = avg_density_no_sharing - avg_density_sharing
    # Use symmetric scale for diverging data, centered at 0
    vmax_diff = max(abs(density_diff.min()), abs(density_diff.max()))
    vmin_diff = -vmax_diff
    
    im3 = ax3.imshow(density_diff, cmap=reduction_cmap, vmin=vmin_diff, vmax=vmax_diff, 
                     interpolation='nearest', aspect='equal')
    ax3.set_title('(c) Traffic Density Reduction', fontweight='bold', pad=15)
    ax3.set_xlabel('Grid X', fontweight='bold')
    ax3.set_ylabel('Grid Y', fontweight='bold')
    
    # Add colorbar with custom labels
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8, pad=0.02)
    cbar3.set_label('Density Change', fontweight='bold', rotation=270, labelpad=20)
    cbar3.ax.tick_params(labelsize=11)
    
    # Set ticks
    ax3.set_xticks(np.arange(0, density_diff.shape[1], 3))
    ax3.set_yticks(np.arange(0, density_diff.shape[0], 3))
    
    # Adjust layout to prevent overlap with bottom text
    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(bottom=0.15)  # Make room for bottom text
    
    # Add only traffic reduction percentage at bottom
    traffic_reduction_text = f"Overall Traffic Reduction: {metrics['traffic_reduction_percentage']:.1f}%"
    
    fig.text(0.5, 0.02, traffic_reduction_text, ha='center', va='bottom', fontsize=16, 
             style='italic', color='darkblue', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.7))
    
    # Save with high quality for paper inclusion
    results_dir = get_results_dir(arglist)
    os.makedirs(results_dir, exist_ok=True)
    heatmap_dir = os.path.join(results_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # Save in multiple formats for flexibility
    plt.savefig(os.path.join(heatmap_dir, "traffic_density_paper.png"), 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(heatmap_dir, "traffic_density_paper.pdf"), 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(heatmap_dir, "traffic_density_paper.eps"), 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    plt.close()
    
    # Reset matplotlib settings
    plt.rcParams.update(plt.rcParamsDefault)
    
    print(f"Paper-worthy traffic density visualizations saved to {heatmap_dir}/")
    print("Available formats: PNG, PDF, EPS")
    
    return density_diff

def save_traffic_density_data_for_latex(metrics, avg_density_sharing, avg_density_no_sharing, density_diff, save_dir, env, arglist):
    """
    Save traffic density data in LaTeX-compatible formats.
    """
    results_dir = get_results_dir(arglist)
    os.makedirs(results_dir, exist_ok=True)
    latex_dir = os.path.join(results_dir, "latex_data")
    os.makedirs(latex_dir, exist_ok=True)
    
    # 1. Traffic density metrics summary
    with open(f"{latex_dir}/traffic_density_metrics.dat", 'w') as f:
        f.write("metric value\n")
        for key, value in metrics.items():
            f.write(f"{key} {value}\n")
    
    # 2. Density data for TikZ heatmap (if needed)
    height, width = avg_density_sharing.shape
    
    # Sharing density data
    with open(f"{latex_dir}/density_sharing_grid.dat", 'w') as f:
        f.write("x y density\n")
        for i in range(height):
            for j in range(width):
                f.write(f"{j} {height-1-i} {avg_density_sharing[i,j]:.3f}\n")
    
    # No-sharing density data
    with open(f"{latex_dir}/density_no_sharing_grid.dat", 'w') as f:
        f.write("x y density\n")
        for i in range(height):
            for j in range(width):
                f.write(f"{j} {height-1-i} {avg_density_no_sharing[i,j]:.3f}\n")
    
    # Density reduction data
    with open(f"{latex_dir}/density_reduction_grid.dat", 'w') as f:
        f.write("x y reduction\n")
        for i in range(height):
            for j in range(width):
                f.write(f"{j} {height-1-i} {density_diff[i,j]:.3f}\n")
    
    # 3. Key metrics for easy LaTeX access
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
    
    print("Traffic density data saved for LaTeX integration")