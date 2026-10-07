#!/usr/bin/env python
"""Bayesian hyperparameter optimisation for the DQN baseline using Optuna.

Mirrors ``scripts/bayesian_hpo_oracle.py`` exactly in structure (Optuna TPE
sampler, resumable SQLite study, same output files), but targets
``ars/train_dqn.py`` / ``ars/evaluate_dqn.py`` and searches DQN-specific
hyperparameters instead of ORACLE's actor/critic/attention ones.

IMPORTANT PREREQUISITE: this script assumes ``ars/train_dqn.py`` has been
patched with a ``--tune-mode <path>`` flag that writes a JSON metrics dict
(mean_eval_reward, last_eval_reward, mean_train_reward, final_epsilon,
final_buffer_size) at the end of training -- see the accompanying patch.
Without it, trials have no reliable per-trial signal to optimise against.

Each trial runs a *short* DQN training job (a few days x a few episodes/day)
and reads back the held-out evaluation reward written via ``--tune-mode``.
The recommended configuration is then retrained at full length for the paper.

Outputs (under ``--out-dir``), identical in format to the ORACLE HPO script:
  * ``study.db``          -- Optuna study (SQLite; resume with the same path)
  * ``trials.csv``        -- every trial: params, value, state, duration
  * ``best.json``         -- best config and its objective value
  * ``convergence.dat``   -- running best objective vs. trial (pgfplots-ready)
  * ``importance.dat``    -- Optuna parameter importances (pgfplots-ready)

Example
-------
    python scripts/bayesian_hpo_dqn.py \
        --dataset 100_agents --grid-size 15 --num-agents 100 \
        --trials 25 --startup-trials 6 \
        --days 3 --num-episodes 30 \
        --out-dir runs/hpo/dqn_grid15

Note: unlike train_oracle.py, train_dqn.py has no --num-envs flag (it runs a
single environment sequentially), so no such argument appears below.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import optuna


def parse_args():
    p = argparse.ArgumentParser(description="Optuna (TPE) Bayesian HPO for the DQN baseline")
    # search control
    p.add_argument("--trials", type=int, default=25,
                   help="total number of Optuna trials")
    p.add_argument("--startup-trials", type=int, default=6,
                   help="random trials before TPE modelling begins")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--objective", type=str, default="mean_eval_reward",
                   help="key in the tune-metrics JSON (or evaluator summary) to maximise")
    p.add_argument("--out-dir", type=str, default="runs/hpo/dqn")
    p.add_argument("--study-name", type=str, default="dqn_hpo")
    p.add_argument("--fresh", action="store_true", default=False,
                   help="wipe any existing study.db and trial checkpoints under "
                        "--out-dir before starting, instead of resuming. Use this "
                        "after a batch of trials failed for a reason unrelated to "
                        "the hyperparameters themselves (e.g. a code bug) -- "
                        "otherwise those failed/pruned trials count toward "
                        "--trials and get skipped on the next run.")
    # per-trial training budget (kept small on purpose)
    p.add_argument("--dataset", type=str, default="100_agents")
    p.add_argument("--grid-size", type=int, default=15)
    p.add_argument("--num-agents", type=int, default=100)
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--num-episodes", type=int, default=30,
                   help="episodes per day (train_dqn.py's --num-episodes semantics)")
    p.add_argument("--eval-days", type=int, default=10,
                   help="simulation days used to score a trial when --objective "
                        "is an evaluation-only metric (e.g. distance_reduction_percent)")
    # fixed (not searched) evaluation-side altruism config, must match what
    # train_dqn.py's Env() defaults to, since train_dqn.py does not expose
    # these as CLI flags at all
    p.add_argument("--altruism-distribution", type=str, default="uniform")
    p.add_argument("--altruism-mean", type=float, default=0.5)
    p.add_argument("--alpha-r", type=float, default=0.4)
    p.add_argument("--python", type=str, default=sys.executable,
                   help="python interpreter used to launch train_dqn / evaluate_dqn")
    return p.parse_args()


def suggest_config(trial):
    """The DQN search space. Each entry maps directly to a ``train_dqn.py``
    CLI flag. Learning rate / tau / buffer-related quantities are searched in
    log space, matching the ORACLE script's treatment of similarly-scaled
    hyperparameters."""
    return {
        "--lr":                      trial.suggest_float("lr", 1e-5, 5e-3, log=True),
        "--gamma":                   trial.suggest_float("gamma", 0.90, 0.999),
        "--tau":                     trial.suggest_float("tau", 1e-3, 1e-1, log=True),
        "--buffer-size":             trial.suggest_categorical("buffer_size",
                                                                [10_000, 50_000, 100_000, 200_000]),
        "--batch-size":              trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "--hidden1":                 trial.suggest_categorical("hidden1", [64, 128, 256]),
        "--hidden2":                 trial.suggest_categorical("hidden2", [128, 256, 512]),
        "--epsilon-min":             trial.suggest_float("epsilon_min", 1e-3, 1e-1, log=True),
        "--epsilon-decay-episodes":  trial.suggest_float("epsilon_decay_episodes", 0.5, 0.95),
    }


# Objectives that only the evaluation pipeline can produce. Selecting one of
# these makes each trial run train -> evaluate, and score the trial on the
# evaluator's own reported metric (same rationale as the ORACLE script: the
# tuned objective should be literally the number reported in the paper).
EVAL_OBJECTIVES = {
    "distance_reduction_percent",
    "mean_acceptance_rate",
    "mean_vehicle_utilization",
}


def run_eval_for_trial(args, ckpt_root, trial_number, hidden1, hidden2):
    """Evaluate a trial's checkpoint with ars.evaluate_dqn and return the
    metrics parsed from its results_summary.txt (or None if it failed).

    ``hidden1``/``hidden2`` must be the values this trial trained with: the
    DQN network's shape depends on them, so evaluating with mismatched sizes
    would fail to restore the checkpoint. ``ckpt_root`` is passed as
    --save-dir to both train and evaluate so the checkpoint path resolution
    (<save-dir>/dqn/grid<N>/<agents>agents/dqn_final.pkl) matches on both
    sides without needing an explicit --weights-file."""
    ckpt_dir = os.path.join(ckpt_root, "dqn", f"grid{args.grid_size}",
                            f"{args.num_agents}agents")
    ckpt_file = "dqn_final.pkl"
    if not os.path.isfile(os.path.join(ckpt_dir, ckpt_file)):
        sys.stderr.write(f"[trial {trial_number}] no checkpoint to evaluate\n")
        return None

    results_dir = os.path.join(args.out_dir, "eval", f"trial_{trial_number}")
    cmd = [
        args.python, "-m", "ars.evaluate_dqn",
        "--dataset", args.dataset,
        "--grid-size", str(args.grid_size),
        "--num-agents", str(args.num_agents),
        "--initial-active-agents", str(args.num_agents),
        "--days", str(args.eval_days),
        "--altruism-distribution", args.altruism_distribution,
        "--altruism-mean", str(args.altruism_mean),
        "--alpha-r", str(args.alpha_r),
        "--hidden1", str(hidden1),
        "--hidden2", str(hidden2),
        "--save-dir", ckpt_root,
        "--results-dir", results_dir,
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
            args.python, "-m", "ars.train_dqn",
            "--dataset", args.dataset,
            "--grid-size", str(args.grid_size),
            "--num-agents", str(args.num_agents),
            "--initial-active-agents", str(args.num_agents),
            "--days", str(args.days),
            "--num-episodes", str(args.num_episodes),
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

        # The reward metrics always come from --tune-mode. The objective
        # itself is only read from here when it is a *training* metric;
        # evaluation objectives are scored further below, from the
        # evaluator's own output.
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
        # distance_reduction_percent) are obtained by running the real
        # evaluator on the trained checkpoint, so the tuned objective is
        # literally the number reported in the paper.
        if needs_eval:
            eval_metrics = run_eval_for_trial(
                args, save_dir, trial.number, cfg["--hidden1"], cfg["--hidden2"])
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

    if not completed:
        sys.stderr.write(
            "\nNo COMPLETE trials in this study -- every trial failed or was "
            "pruned, so there is no best config to report. Check trials.csv "
            f"(written to {args.out_dir}/trials.csv) for the failure reason "
            "of individual trials, fix the underlying issue, then rerun with "
            "--fresh (or delete --out-dir) to discard this failed study "
            "rather than resuming it.\n"
        )
        return False

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

    return True


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.fresh:
        import shutil
        study_db = os.path.join(args.out_dir, "study.db")
        ckpt_dir = os.path.join(args.out_dir, "checkpoints")
        eval_dir = os.path.join(args.out_dir, "eval")
        removed = []
        if os.path.isfile(study_db):
            os.remove(study_db)
            removed.append(study_db)
        for d in (ckpt_dir, eval_dir):
            if os.path.isdir(d):
                shutil.rmtree(d)
                removed.append(d)
        if removed:
            print(f"--fresh: removed {', '.join(removed)}\n")
        else:
            print("--fresh: nothing to remove (no existing study found)\n")

    storage = f"sqlite:///{os.path.join(args.out_dir, 'study.db')}"

    sampler = optuna.samplers.TPESampler(
        seed=args.seed, n_startup_trials=args.startup_trials)
    study = optuna.create_study(
        study_name=args.study_name, storage=storage,
        sampler=sampler, direction="maximize", load_if_exists=True)

    print(f"Optuna TPE HPO for DQN baseline | objective={args.objective} (maximise)")
    print(f"Trials: {args.trials} (startup random: {args.startup_trials})")
    print(f"Per-trial budget: days={args.days} episodes/day={args.num_episodes} "
          f"grid={args.grid_size}")
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

    ok = export_results(study, args)
    if not ok:
        sys.exit(1)

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