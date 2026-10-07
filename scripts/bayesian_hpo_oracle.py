#!/usr/bin/env python
"""Bayesian hyperparameter optimisation for the ORACLE policy using Optuna.

This drives ``python -m ars.train_oracle`` as a black-box objective and searches
the key ORACLE hyperparameters with Optuna's Tree-structured Parzen Estimator
(TPE) sampler -- a standard Bayesian optimisation method [Bergstra et al., 2011].

Each trial runs a *short* ORACLE training job (a few days x a few episodes) and
reads back the held-out evaluation reward written via ``--tune-mode``. Keeping
each trial cheap is deliberate: the goal is a convincing, reproducible
justification for the chosen hyperparameters, not a full-length run per trial.
The recommended configuration is then retrained at full length for the paper.

The study is persisted to a SQLite database, so the search is resumable and
inspectable after the fact (e.g. with ``optuna-dashboard``).

Outputs (under ``--out-dir``):
  * ``study.db``          -- Optuna study (SQLite; resume with the same path)
  * ``trials.csv``        -- every trial: params, value, state, duration
  * ``best.json``         -- best config and its objective value
  * ``convergence.dat``   -- running best objective vs. trial (pgfplots-ready)
  * ``importance.dat``    -- Optuna parameter importances (pgfplots-ready)

Example
-------
    python scripts/bayesian_hpo_oracle.py \
        --dataset 100_agents --grid-size 15 --num-agents 100 \
        --trials 25 --startup-trials 6 \
        --days 3 --num-episodes 30 --num-envs 10 \
        --out-dir runs/hpo/oracle_grid15

Note on per-trial budget: ``num_episodes // num_envs`` must be >= 2 so the
exploration-temperature decay period is non-zero in ``ars.train_oracle``.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile

import optuna


def parse_args():
    p = argparse.ArgumentParser(description="Optuna (TPE) Bayesian HPO for ORACLE")
    # search control
    p.add_argument("--trials", type=int, default=25,
                   help="total number of Optuna trials")
    p.add_argument("--startup-trials", type=int, default=6,
                   help="random trials before TPE modelling begins")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--objective", type=str, default="mean_eval_reward",
                   help="key in the tune-metrics JSON to maximise")
    p.add_argument("--out-dir", type=str, default="runs/hpo/oracle")
    p.add_argument("--study-name", type=str, default="oracle_hpo")
    # per-trial training budget (kept small on purpose)
    p.add_argument("--dataset", type=str, default="100_agents")
    p.add_argument("--grid-size", type=int, default=15)
    p.add_argument("--num-agents", type=int, default=100)
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--num-episodes", type=int, default=30)
    p.add_argument("--num-envs", type=int, default=10)
    p.add_argument("--exploration", type=str, default="tags")
    p.add_argument("--alpha-r", type=float, default=0.4,
                   help="reward trade-off weight held fixed across trials; must "
                        "match the value used for the final full-length training")
    p.add_argument("--eval-days", type=int, default=20,
                   help="simulation days used to score a trial when --objective "
                        "is an evaluation metric (e.g. distance_reduction_percent)")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--gpu-id", type=str, default="0")
    p.add_argument("--python", type=str, default=sys.executable,
                   help="python interpreter used to launch train_oracle")
    return p.parse_args()


def suggest_config(trial):
    """The ORACLE search space (reported in the paper). Each entry maps directly
    to a ``train_oracle`` CLI flag. Learning rates / regularisation are searched
    in log space."""
    return {
        "--lr":              trial.suggest_float("critic_lr", 1e-5, 5e-3, log=True),
        "--actor-lr":        trial.suggest_float("actor_lr_mult", 1e-2, 1.0, log=True),
        "--gamma":           trial.suggest_float("gamma", 0.90, 0.99),
        "--tau":             trial.suggest_float("tau", 5e-3, 1e-1, log=True),
        "--reg-coef":        trial.suggest_float("reg_coef", 1e-6, 1e-3, log=True),
        "--gumbel-temp-end": trial.suggest_float("gumbel_temp_end", 1e-2, 0.3, log=True),
        "--attn_temp":       trial.suggest_float("attn_temp", 0.5, 2.0),
        "--num-heads":       trial.suggest_categorical("num_heads", [1, 2, 4, 8]),
    }


# Objectives that only the evaluation pipeline can produce. Selecting one of
# these makes each trial run train -> evaluate, and score the trial on the
# evaluator's own reported metric.
EVAL_OBJECTIVES = {
    "distance_reduction_percent",
    "mean_acceptance_rate",
    "mean_vehicle_utilization",
}


def run_eval_for_trial(args, ckpt_root, trial_number, num_heads):
    """Evaluate a trial's checkpoint with ars.evaluate_oracle and return the
    metrics parsed from its results_summary.txt (or None if it failed).

    ``num_heads`` must be the value this trial trained with: the attention
    critic's shape depends on it, so evaluating with a different head count
    would restore a mismatched network."""
    ckpt_dir = os.path.join(ckpt_root, "oracle", f"grid{args.grid_size}",
                            f"{args.num_agents}agents")
    ckpt_file = f"final_weights_attention_{args.num_agents}_tags.pkl"
    if not os.path.isfile(os.path.join(ckpt_dir, ckpt_file)):
        sys.stderr.write(f"[trial {trial_number}] no checkpoint to evaluate\n")
        return None

    results_dir = os.path.join(args.out_dir, "eval", f"trial_{trial_number}")
    cmd = [
        args.python, "-m", "ars.evaluate_oracle",
        "--model-type", "attention",
        "--dataset", args.dataset,
        "--grid-size", str(args.grid_size),
        "--num-agents", str(args.num_agents),
        "--initial-active-agents", str(args.num_agents),
        "--days", str(args.eval_days),
        "--altruism-distribution", "uniform", "--altruism-mean", "0.5",
        "--alpha-r", str(args.alpha_r),
        "--num-heads", str(num_heads),
        "--save-dir", ckpt_dir, "--restore", "--weights-file", ckpt_file,
        "--results-dir", results_dir,
        "--device", args.device, "--gpu-id", args.gpu_id,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    summary = os.path.join(results_dir, "results_summary.txt")
    if proc.returncode != 0 or not os.path.isfile(summary):
        tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
        sys.stderr.write(f"[trial {trial_number}] evaluation failed: {tail}\n")
        return None

    out = {}
    with open(summary) as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            try:
                out[k.strip()] = float(v.strip())
            except ValueError:
                pass
    if args.objective not in out:
        sys.stderr.write(f"[trial {trial_number}] '{args.objective}' not in summary\n")
        return None
    return out


def make_objective(args):
    def objective(trial):
        cfg = suggest_config(trial)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            metrics_path = tf.name
        save_dir = os.path.join(args.out_dir, "checkpoints", f"trial_{trial.number}")

        cmd = [
            args.python, "-m", "ars.train_oracle",
            "--model", "attention",
            "--dataset", args.dataset,
            "--grid-size", str(args.grid_size),
            "--num-agents", str(args.num_agents),
            "--days", str(args.days),
            "--num-episodes", str(args.num_episodes),
            "--num-envs", str(args.num_envs),
            "--exploration", args.exploration,
            "--alpha-r", str(args.alpha_r),
            "--device", args.device, "--gpu-id", args.gpu_id,
            "--wandb-mode", "disabled",
            "--save-dir", save_dir,
            "--tune-mode", metrics_path,
        ]
        for flag, val in cfg.items():
            cmd += [flag, str(val)]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
            sys.stderr.write(f"[trial {trial.number}] training failed "
                             f"(rc={proc.returncode}): {tail}\n")
            try:
                os.remove(metrics_path)
            except OSError:
                pass
            raise optuna.TrialPruned()

        # The reward metrics always come from --tune-mode. The objective itself
        # is only read from here when it is a *training* metric; evaluation
        # objectives are scored further below, from the evaluator's own output.
        needs_eval = args.objective in EVAL_OBJECTIVES
        score = None
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
            if not needs_eval:
                score = float(metrics[args.objective])
        except (OSError, KeyError, ValueError) as e:
            sys.stderr.write(f"[trial {trial.number}] could not read objective: {e}\n")
            raise optuna.TrialPruned()
        finally:
            try:
                os.remove(metrics_path)
            except OSError:
                pass

        # Objectives that are only defined by the *evaluation* pipeline (e.g.
        # distance_reduction_percent) are obtained by running the real evaluator
        # on the trained checkpoint. This guarantees the tuned objective is
        # literally the number reported in the paper, rather than a
        # reimplementation that could silently drift from it.
        if needs_eval:
            eval_metrics = run_eval_for_trial(
                args, save_dir, trial.number, cfg["--num-heads"])
            if eval_metrics is None:
                raise optuna.TrialPruned()
            for k, v in eval_metrics.items():
                trial.set_user_attr(k, v)
            for k, v in metrics.items():
                trial.set_user_attr(k, v)   # keep the reward metrics too
            score = float(eval_metrics[args.objective])
            return score

        # expose secondary metrics for later inspection
        for k, v in metrics.items():
            if k != args.objective:
                trial.set_user_attr(k, v)
        return score

    return objective


def export_results(study, args):
    os.makedirs(args.out_dir, exist_ok=True)
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]

    # trials.csv
    df = study.trials_dataframe(attrs=("number", "value", "state",
                                       "duration", "params"))
    df.to_csv(os.path.join(args.out_dir, "trials.csv"), index=False)

    # best.json
    with open(os.path.join(args.out_dir, "best.json"), "w") as f:
        json.dump({"objective": args.objective,
                   "best_value": study.best_value,
                   "best_trial": study.best_trial.number,
                   "config": study.best_params}, f, indent=2)

    # convergence.dat (running best over trial index, in trial order)
    with open(os.path.join(args.out_dir, "convergence.dat"), "w") as f:
        f.write("trial best_objective\n")
        running = None
        for t in sorted(study.trials, key=lambda x: x.number):
            if t.value is not None:
                running = t.value if running is None else max(running, t.value)
            if running is not None:
                f.write(f"{t.number} {running:.6f}\n")

    # importance.dat (Optuna's default fANOVA-based importance)
    with open(os.path.join(args.out_dir, "importance.dat"), "w") as f:
        f.write("param importance\n")
        if len(completed) >= 2:
            try:
                imp = optuna.importance.get_param_importances(study)
                for name, val in imp.items():
                    f.write(f"{name} {val:.6f}\n")
            except Exception as e:  # importance needs >=2 distinct values
                sys.stderr.write(f"importance unavailable: {e}\n")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    storage = f"sqlite:///{os.path.join(args.out_dir, 'study.db')}"

    sampler = optuna.samplers.TPESampler(
        seed=args.seed, n_startup_trials=args.startup_trials)
    study = optuna.create_study(
        study_name=args.study_name, storage=storage,
        sampler=sampler, direction="maximize", load_if_exists=True)

    print(f"Optuna TPE HPO for ORACLE | objective={args.objective} (maximise)")
    print(f"Trials: {args.trials} (startup random: {args.startup_trials})")
    print(f"Per-trial budget: days={args.days} episodes={args.num_episodes} "
          f"envs={args.num_envs} grid={args.grid_size}")
    print(f"Storage: {storage}\n")

    already = len([t for t in study.trials
                   if t.state in (optuna.trial.TrialState.COMPLETE,
                                  optuna.trial.TrialState.PRUNED)])
    remaining = max(0, args.trials - already)
    if already:
        print(f"Resuming study: {already} trials already recorded, "
              f"running {remaining} more.\n")

    study.optimize(make_objective(args), n_trials=remaining,
                   show_progress_bar=False)

    export_results(study, args)

    print("\n" + "=" * 60)
    print(f"Best objective ({args.objective}): {study.best_value:.4f} "
          f"at trial {study.best_trial.number}")
    print("Best config:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"\nResults written to {args.out_dir}/")
    print("  study.db  trials.csv  best.json  convergence.dat  importance.dat")


if __name__ == "__main__":
    main()
