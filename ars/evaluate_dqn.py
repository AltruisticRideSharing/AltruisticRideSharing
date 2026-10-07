"""
Parallel to evaluate_oracle.py. Uses the identical environment interface,
metrics pipeline (ars.metrics), and output format so results are directly
comparable to ORACLE, MADDPG, and MAAC in the paper.

Usage (mirrors evaluate_oracle.py CLI):
    python ars/evaluate_dqn.py \
        --num-agents 100 \
        --dataset 100_agents \
        --grid-size 15 \
        --days 10 \
        --num-episodes 100 \
        --weights-file savedagents/dqn/grid15/100agents/dqn_final.pkl

Output:
    results/grid15/dqn/100agents_uniform_fixed/
        ├── results.txt
        ├── results_summary.txt          ← key paper metrics
        ├── latex_data/summary.dat       ← all LaTeX-ready .dat files
        ├── per_agent_distance_boxplot.png
        ├── per_agent_altruism_boxplot.png
        ├── sharing_vs_non_sharing_distance.png
        ├── vehicle_utilization_average.png
        ├── ride_acceptance_rate.png
        ├── agent_benefit_scatter.png
        ├── multidimensional_benefit_analysis.png
        ├── winner_loser_distribution.png
        ├── granger_causality_analysis.png
        └── heatmaps/traffic_density_paper.png  (+ .pdf, .eps)
"""

import argparse
import copy
import os
import warnings
import numpy as np
from ars.env.env import Env
from ars.env.agent import Driver, Rider
from ars.models.deeppool.dqn_baseline import DQNBaseline
from ars.metrics_dqn import *
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CLI — mirrors evaluate_oracle.py exactly so sweep scripts work unchanged
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser("DQN Baseline Evaluation for ARS")

    # Environment — identical to evaluate_oracle.py
    parser.add_argument("--num-agents",          type=int,   default=100)
    parser.add_argument("--initial-active-agents", type=int, default=100)
    parser.add_argument("--grid-size",           type=int,   default=15)
    parser.add_argument("--dataset",             type=str,   default="100_agents")
    parser.add_argument("--perception-field",    type=int,   default=None)
    parser.add_argument("--days",                type=int,   default=10)
    parser.add_argument("--num-episodes",        type=int,   default=1,
                        help="Episodes per day during evaluation (default 1 — greedy rollout)")

    # Altruism — identical defaults to evaluate_oracle.py
    parser.add_argument("--altruism-distribution", type=str, default="uniform",
                        choices=["uniform", "gaussian"])
    parser.add_argument("--altruism-mean",       type=float, default=0.5)
    parser.add_argument("--altruism-std",        type=float, default=0.15)
    parser.add_argument("--alpha-r",             type=float, default=0.4)
    parser.add_argument("--forced-driver-threshold", type=float, default=0.2)
    parser.add_argument("--alpha-s",             type=float, default=0.5)
    parser.add_argument("--beta-s",              type=float, default=0.7)

    # Reward filtering — identical to evaluate_oracle.py
    parser.add_argument("--filter-neg-rew",      action="store_false", default=True)
    parser.add_argument("--reward-threshold",    type=float, default=-0.75)

    # Checkpoint
    parser.add_argument("--save-dir",            type=str,   default="savedagents")
    parser.add_argument("--weights-file",        type=str,   default=None,
                        help="Path to dqn_*.pkl checkpoint. If omitted, uses "
                             "<save-dir>/dqn/grid<N>/<agents>agents/dqn_final.pkl")

    # DQN architecture — must match what was used during training
    parser.add_argument("--hidden1",             type=int,   default=128)
    parser.add_argument("--hidden2",             type=int,   default=256)

    # Birth/death (off by default, matching evaluate_oracle.py)
    parser.add_argument("--enable-birth-death",  action="store_true", default=False)

    # Results dir override (used by ablation sweeps)
    parser.add_argument("--results-dir",         type=str,   default=None)

    args, _ = parser.parse_known_args()
    # model_label drives get_results_dir — "dqn" puts results in results/grid<N>/dqn/
    args.model_label = "dqn"
    return args


def setup_environment(arglist):
    return Env(
        height=arglist.grid_size,
        width=arglist.grid_size,
        numAgents=arglist.num_agents,
        dataset_folder=arglist.dataset,
        initial_active_agents=arglist.initial_active_agents,
        altruism_distribution=arglist.altruism_distribution,
        altruism_mean=arglist.altruism_mean,
        altruism_std=arglist.altruism_std,
        total_days=arglist.days,
        perception_field=arglist.perception_field,
        alpha_r=arglist.alpha_r,
        forced_driver_threshold=arglist.forced_driver_threshold,
        alpha_s=arglist.alpha_s,
        beta_s=arglist.beta_s,
    )


# Checkpoint loading
def load_dqn(arglist, obs_dim, num_agents):
    """Load DQNBaseline from checkpoint."""
    agent = DQNBaseline(
        obs_dim=obs_dim,
        num_agents=num_agents,
        num_agents_grid=arglist.grid_size,
        hidden1=arglist.hidden1,
        hidden2=arglist.hidden2,
    )

    # Resolve checkpoint path
    if arglist.weights_file:
        ckpt_path = arglist.weights_file
        ckpt_dir  = os.path.dirname(ckpt_path)
        ckpt_tag  = os.path.basename(ckpt_path).replace("dqn_", "").replace(".pkl", "")
    else:
        ckpt_dir = os.path.join(
            arglist.save_dir, "dqn",
            f"grid{arglist.grid_size}",
            f"{arglist.num_agents}agents"
        )
        ckpt_tag = "final"

    agent.load(ckpt_dir, tag=ckpt_tag)
    agent.set_epsilon(0.0)   # Fully greedy for evaluation
    return agent



def run_simulation(env, dqn_agent, arglist, day, pickup_history):
    """
    Run one greedy evaluation episode and return all metrics dicts.
    Mirrors evaluate_oracle.py run_simulation() return signature exactly.
    """
    env.current_day = day

    episode_reward      = {agent: 0 for agent in env.agents}
    paths               = {agent: [env.agent_objects[i].start_position]
                           for i, agent in enumerate(env.agents)}
    miles_given         = {agent: 0 for agent in env.agents}
    miles_taken         = {agent: 0 for agent in env.agents}
    ride_offers         = 0
    ride_accepts        = 0

    state, infos = env.reset(
        enable_dropout=False,
        enable_birth_death=arglist.enable_birth_death,
    )

    done = False
    paths_no_sharing = get_no_sharing_paths(env)
    vehicle_utilization_per_driver = {}

    while not done:
        actions = np.full(env.numAgents, env.numAgents, dtype=np.int32)

        for i, agent in enumerate(env.agents):
            if env.agent_role[agent] in ["dropout", "never_joined"]:
                continue

            # Build valid actions (mirrors evaluate_oracle.py exactly)
            nearby_grid      = state[i][3:].astype(np.int32)
            nearby_rider_ids = nearby_grid[nearby_grid != -1]

            if arglist.filter_neg_rew and env.agent_role[agent] == "driver":
                driver_obj = env.agent_objects[i]
                filtered   = []
                for rider_id in nearby_rider_ids:
                    if env.agent_role[env.agents[rider_id]] in ["dropout", "never_joined"]:
                        continue
                    rider_obj  = env.agent_objects[rider_id]
                    reward_val = env.get_reward(driver_obj, rider_obj, 0)
                    if reward_val > arglist.reward_threshold:
                        filtered.append(rider_id)
                nearby_rider_ids = np.array(filtered, dtype=np.int32)

            valid_actions = np.concatenate((nearby_rider_ids, [env.numAgents]))

            if env.agent_role[agent] == "driver":
                ride_offers += len(valid_actions) - 1   # exclude no-op

            # DQN action selection (greedy, epsilon=0)
            if infos[i] == 1:   # driver
                action = dqn_agent.select_action(state[i], is_driver=True, evaluation=True)
                # Enforce valid_actions constraint (mask may not cover filter above)
                if action not in valid_actions:
                    action = int(env.numAgents)
                actions[i] = action

                if action != env.numAgents:
                    ride_accepts += 1

        track_pickup_interactions(env, actions, pickup_history, day)

        next_state, reward, done, next_infos = env.step(actions)

        # Track paths
        for i, agent in enumerate(env.agents):
            if env.agent_role[agent] in ["dropout", "never_joined"]:
                continue
            agent_obj = env.agent_objects[i]
            if agent_obj.is_at_destination() and np.array_equal(
                paths[agent][-1], agent_obj.destination
            ):
                continue
            paths[agent].append(tuple(agent_obj.position))

        state = next_state
        infos = next_infos
        for i, agent in enumerate(env.agents):
            episode_reward[agent] += reward[i]

        if done:
            # ---- Distance / time calculations (identical to evaluate_oracle.py) ----
            direct_distances     = {}
            individual_distances = {}
            individual_times     = {}
            direct_times         = {}
            detour_distances     = {agent: 0 for agent in env.agents}

            for agent in env.agents:
                if env.agent_role[agent] in ["dropout", "never_joined"]:
                    individual_distances[agent] = 0
                    direct_distances[agent]     = 0
                    individual_times[agent]     = 0
                    direct_times[agent]         = 0
                    continue

                idx       = env.agents.index(agent)
                agent_obj = env.agent_objects[idx]

                if isinstance(agent_obj, Driver):
                    riders_picked = len(env.riders_drivers[agent])
                    vehicle_utilization_per_driver[agent] = 1.0 + riders_picked

                    total_time = calculate_time(paths[agent], env.time_weight_matrix)
                    individual_times[agent] = total_time
                    total_dist = calculate_distance(paths[agent], env.weight_matrix)
                    individual_distances[agent] = total_dist

                    direct_dist = agent_obj._calculate_path_length(
                        agent_obj._dijkstra_path(
                            agent_obj.start_position, agent_obj.destination
                        )
                    )
                    direct_distances[agent]  = direct_dist
                    detour_distances[agent]  = total_dist - direct_dist

                    for rider in env.riders_drivers[agent]:
                        rider_path = agent_obj._dijkstra_path(
                            rider.start_position, rider.destination
                        )
                        miles_given[agent] += agent_obj._calculate_path_length(rider_path)

                    direct_path = agent_obj._dijkstra_path(
                        agent_obj.start_position, agent_obj.destination
                    )
                    direct_times[agent] = calculate_time(direct_path, env.time_weight_matrix)

                else:  # Rider
                    rider_path = agent_obj._dijkstra_path(
                        agent_obj.start_position, agent_obj.destination
                    )
                    if agent_obj.being_picked_up:
                        miles_taken[agent]         = agent_obj._calculate_path_length(rider_path)
                        individual_distances[agent] = 0
                        direct_distances[agent]     = miles_taken[agent]
                        individual_times[agent]     = 0
                        direct_times[agent]         = calculate_time(rider_path, env.time_weight_matrix)
                    else:
                        individual_distances[agent] = agent_obj._calculate_path_length(rider_path)
                        direct_distances[agent]     = individual_distances[agent]
                        individual_times[agent]     = calculate_time(rider_path, env.time_weight_matrix)
                        direct_times[agent]         = individual_times[agent]
            break

    # Dropout stats (mirrors evaluate_oracle.py)
    dropout_stats = env.get_dropout_statistics(day)
    if dropout_stats["agents_dropped"] > 0:
        print(f"Day {day} Dropouts: {dropout_stats['agents_dropped']} agents")

    acceptance_rate = ride_accepts / ride_offers if ride_offers > 0 else 0

    return (
        episode_reward, individual_distances, direct_distances, detour_distances,
        miles_given, miles_taken, paths, paths_no_sharing,
        ride_offers, ride_accepts, acceptance_rate,
        individual_times, direct_times, vehicle_utilization_per_driver,
    )

# Evaluation loop
def evaluate(arglist):
    print(f"\n=== DQN BASELINE EVALUATION ===")
    print(f"Agents: {arglist.num_agents} | Grid: {arglist.grid_size} | Days: {arglist.days}")

    # Set up env FIRST to get correct obs_dim for this grid size
    env = setup_environment(arglist)
    obs_dim = env.observation_space().nvec.shape[0]   # grid-aware: 28 for 15x15, 228 for 45x45

    stats = env.get_altruism_distribution_stats()
    print(f"\n=== INITIAL ALTRUISM ===")
    print(f"Mean={stats['mean']:.3f}  Std={stats['std']:.3f}  "
          f"Range=[{stats['min']:.3f},{stats['max']:.3f}]")

    dqn_agent = load_dqn(arglist, obs_dim, env.numAgents)
    print("DQN checkpoint loaded. Running greedy evaluation (ε=0).\n")

    # ---- Per-day accumulators (mirrors evaluate_oracle.py) ----
    results                         = {}
    individual_distances_by_day     = {}
    direct_distances_by_day         = {}
    detour_distances_by_day         = {}
    miles_given_by_day              = {}
    miles_taken_by_day              = {}
    altruism_points_by_day          = {}
    roles_by_day                    = {}
    distance_total_sharing_by_day   = {}
    distance_total_no_sharing_by_day = {}
    acceptance_rates_by_day         = {}
    ride_offers_by_day              = {}
    ride_accepts_by_day             = {}
    individual_times_by_day         = {}
    direct_times_by_day             = {}
    vehicle_utilization_by_day      = {}
    paths_sharing_by_day            = {}
    paths_no_sharing_by_day         = {}

    daily_altruism_series   = {}
    daily_benefit_series    = {}
    daily_aggregate_altruism  = []
    daily_aggregate_benefits  = []

    pickup_history = {}

    for day in range(1, arglist.days + 1):
        if arglist.enable_birth_death:
            env.reset_day(day, enable_birth_death=True)
        else:
            env.reset_day(day)
        env.current_day = day
        print(f"Day {day}/{arglist.days} ...")

        (episode_reward, individual_distances, direct_distances, detour_distances,
         miles_given, miles_taken, paths_sharing, paths_no_sharing,
         ride_offers, ride_accepts, acceptance_rate,
         individual_times, direct_times, vehicle_utilization_per_driver) = run_simulation(
            env, dqn_agent, arglist, day, pickup_history
        )

        track_pickup_interactions(env, None, pickup_history, day)

        paths_sharing_by_day[day]    = paths_sharing
        paths_no_sharing_by_day[day] = paths_no_sharing

        individual_times_by_day[day]  = copy.deepcopy(individual_times)
        direct_times_by_day[day]      = copy.deepcopy(direct_times)
        acceptance_rates_by_day[day]  = acceptance_rate
        ride_offers_by_day[day]       = ride_offers
        ride_accepts_by_day[day]      = ride_accepts

        results[day]                       = episode_reward
        individual_distances_by_day[day]   = copy.deepcopy(individual_distances)
        direct_distances_by_day[day]       = copy.deepcopy(direct_distances)
        detour_distances_by_day[day]       = copy.deepcopy(detour_distances)
        miles_given_by_day[day]            = copy.deepcopy(miles_given)
        miles_taken_by_day[day]            = copy.deepcopy(miles_taken)
        vehicle_utilization_by_day[day]    = list(vehicle_utilization_per_driver.values())
        altruism_points_by_day[day]        = copy.deepcopy(env.altruism_points_day)
        roles_by_day[day]                  = copy.deepcopy(env.agent_role)

        distance_total_sharing_by_day[day] = sum(
            dist for agent, dist in individual_distances_by_day[day].items()
            if env.agent_role.get(agent, "active") not in ["dropout", "never_joined"]
        )
        distance_total_no_sharing_by_day[day] = sum(
            dist for agent, dist in direct_distances_by_day[day].items()
            if env.agent_role.get(agent, "active") not in ["dropout", "never_joined"]
        )

        # Altruism time-series (for Granger analysis)
        for agent in env.agents:
            if env.agent_role.get(agent, "active") not in ["dropout", "never_joined"]:
                daily_altruism_series.setdefault(agent, []).append(
                    altruism_points_by_day[day][agent]
                )

        active_altruism = [
            altruism_points_by_day[day][a]
            for a in env.agents
            if env.agent_role.get(a, "active") not in ["dropout", "never_joined"]
               and a in altruism_points_by_day[day]
        ]
        daily_aggregate_altruism.append(np.mean(active_altruism) if active_altruism else 0.0)
        print(f"  Day {day}: mean altruism={daily_aggregate_altruism[-1]:.4f}  "
              f"total_reward={sum(episode_reward.values()):.2f}  "
              f"acceptance={acceptance_rate*100:.1f}%")

        if day > 1:
            for agent in env.agents:
                if (env.agent_role.get(agent, "active") not in ["dropout", "never_joined"]
                        and agent in individual_distances_by_day[day]
                        and agent in direct_distances_by_day[day]):
                    benefit = (direct_distances_by_day[day][agent]
                               - individual_distances_by_day[day][agent])
                    daily_benefit_series.setdefault(agent, []).append(benefit)

            active_benefits = [
                direct_distances_by_day[day][a] - individual_distances_by_day[day][a]
                for a in env.agents
                if (env.agent_role.get(a, "active") not in ["dropout", "never_joined"]
                    and a in individual_distances_by_day[day])
            ]
            daily_aggregate_benefits.append(
                np.mean(active_benefits) if active_benefits else 0.0
            )

    # ---- Metrics & plots (identical call sequence to evaluate_oracle.py) ----
    print("\nGenerating metrics and plots ...")
    results_dir = get_results_dir(arglist)

    analyze_altruism_transactions(env, arglist)

    plot_sharing_vs_non_sharing(
        distance_total_sharing_by_day, distance_total_no_sharing_by_day, env, arglist
    )
    plot_vehicle_utilization(vehicle_utilization_by_day, env, arglist)
    plot_ride_acceptance_rate(acceptance_rates_by_day, env, arglist)
    plot_agent_distance_boxplots(
        individual_distances_by_day, direct_distances_by_day, env, arglist
    )
    plot_altruism_boxplots(altruism_points_by_day, env, arglist)
    plot_detour_factor_and_trip_time(
        detour_distances_by_day, individual_times_by_day,
        direct_distances_by_day, direct_times_by_day, env, arglist
    )
    save_results(
        results,
        save_dir=f"results/{arglist.dataset}",
        altruism_points_by_day=altruism_points_by_day,
        roles_by_day=roles_by_day,
        individual_distances_by_day=individual_distances_by_day,
        direct_distances_by_day=direct_distances_by_day,
        detour_distances_by_day=detour_distances_by_day,
        miles_given_by_day=miles_given_by_day,
        miles_taken_by_day=miles_taken_by_day,
        distance_total_sharing_by_day=distance_total_sharing_by_day,
        distance_total_no_sharing_by_day=distance_total_no_sharing_by_day,
        arglist=arglist,
    )
    generate_ieee_tikz_data(
        altruism_points_by_day, individual_distances_by_day,
        direct_distances_by_day, detour_distances_by_day,
        distance_total_sharing_by_day, distance_total_no_sharing_by_day,
        vehicle_utilization_by_day, acceptance_rates_by_day,
        individual_times_by_day, direct_times_by_day, env, arglist
    )

    agent_benefits, agent_total_altruism = calculate_agent_benefits(
        individual_distances_by_day, direct_distances_by_day,
        altruism_points_by_day, paths_sharing_by_day, paths_no_sharing_by_day,
        env, arglist
    )
    benefit_altruism_correlation  = plot_agent_benefit_scatter(agent_benefits, env, arglist)
    line_curve_stats              = plot_benefits_vs_altruism_line_curves(agent_benefits, env, arglist)
    winner_loser_stats            = analyze_winner_loser_distribution(agent_benefits, env, arglist)
    multidimensional_stats        = analyze_multidimensional_benefit_distribution(agent_benefits, env, arglist)
    distribution_stats            = analyze_benefit_distribution(agent_benefits, env, arglist)

    save_new_metrics_data(
        agent_benefits, winner_loser_stats, distribution_stats,
        benefit_altruism_correlation, env, arglist, line_curve_stats
    )
    save_multidimensional_metrics_data(
        agent_benefits, multidimensional_stats, results_dir, env, arglist
    )

    print(f"\n=== BENEFIT ANALYSIS SUMMARY ===")
    print(f"Distance Benefit Gini:   {multidimensional_stats['gini_distance_benefits']:.4f}")
    print(f"Traffic Benefit Gini:    {multidimensional_stats['gini_traffic_benefits']:.4f}")
    print(f"Combined Benefit Gini:   {multidimensional_stats['gini_combined_benefits']:.4f}")
    print(f"Mean Distance Benefit:   {multidimensional_stats['mean_distance_benefit']:.2f}")
    print(f"Mean Traffic Benefit:    {multidimensional_stats['mean_traffic_benefit']:.2f}")
    print(f"Winners: {winner_loser_stats['winners']} ({winner_loser_stats['winner_percentage']:.1f}%)")
    print(f"Losers:  {winner_loser_stats['losers']} ({winner_loser_stats['loser_percentage']:.1f}%)")

    # Granger causality
    print("\n" + "=" * 60)
    print("GRANGER CAUSALITY ANALYSIS")
    print("=" * 60)
    granger_results = perform_granger_causality_analysis(
        daily_altruism_series, daily_benefit_series,
        daily_aggregate_altruism, daily_aggregate_benefits,
        arglist.days, env, arglist
    )
    save_granger_causality_results(granger_results, results_dir, env, arglist)
    plot_time_series_and_causality(
        daily_altruism_series, daily_benefit_series,
        daily_aggregate_altruism, daily_aggregate_benefits,
        granger_results, results_dir, env, arglist
    )

    # Birth/death or dropout
    if arglist.enable_birth_death:
        bd_stats = plot_birth_death_analysis(env, arglist)
        reintegration_metrics = plot_reintegration_analysis(env, arglist)
        print(f"\n=== REINTEGRATION SCORE: {reintegration_metrics['final_reintegration_score']:.1f}/100 ===")
    else:
        plot_daily_dropouts(env, arglist)

    # Traffic density
    rho_threshold = 2.0
    traffic_metrics, avg_density_sharing, avg_density_no_sharing = \
        calculate_traffic_density_metrics(
            paths_sharing_by_day, paths_no_sharing_by_day,
            env.height, env.width, arglist.days, env, arglist, rho_threshold
        )
    density_diff = create_traffic_density_visualization(
        avg_density_sharing, avg_density_no_sharing, traffic_metrics, results_dir, env, arglist
    )
    save_traffic_density_data_for_latex(
        traffic_metrics, avg_density_sharing, avg_density_no_sharing,
        density_diff, results_dir, env, arglist
    )

    # Append traffic metrics to summary.dat
    latex_dir = os.path.join(results_dir, "latex_data")
    with open(f"{latex_dir}/summary.dat", "a") as f:
        f.write(f"dense_cell_reduction {traffic_metrics['dense_cell_reduction']}\n")
        f.write(f"dense_cell_reduction_percentage {traffic_metrics['dense_cell_reduction_percentage']:.2f}\n")
        f.write(f"traffic_reduction_percentage {traffic_metrics['traffic_reduction_percentage']:.2f}\n")
        f.write(f"peak_density_reduction_percentage {traffic_metrics['peak_density_reduction_percentage']:.2f}\n")
        f.write(f"hotspot_reduction {traffic_metrics['hotspot_reduction']}\n")
        f.write(f"congestion_threshold {traffic_metrics['rho_threshold']:.1f}\n")

    # Aggregated summary (for cross-model comparison)
    write_aggregated_results(
        results_dir, arglist.model_label, arglist,
        distance_total_sharing_by_day, distance_total_no_sharing_by_day,
        acceptance_rates_by_day=acceptance_rates_by_day,
        vehicle_utilization_by_day=vehicle_utilization_by_day,
        multidimensional_stats=multidimensional_stats,
        winner_loser_stats=winner_loser_stats,
        traffic_metrics=traffic_metrics,
        detour_distances_by_day=detour_distances_by_day,
        direct_distances_by_day=direct_distances_by_day,
        individual_times_by_day=individual_times_by_day,
        reintegration_metrics=(reintegration_metrics if arglist.enable_birth_death else None),
    )

    print(f"\n=== EVALUATION COMPLETE ===")
    print(f"All outputs written to: {results_dir}")


if __name__ == "__main__":
    arglist = parse_args()
    evaluate(arglist)
