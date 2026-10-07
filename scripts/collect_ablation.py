#!/usr/bin/env python
"""Collect the sensitivity/ablation sweep into a table + pgfplots data.

Reads each run produced by ``run_ablation_sensitivity.sh`` under
``results/ablation/<knob>/<value>/`` and pulls the headline efficiency and
fairness metrics out of:
    latex_data/summary.dat                  (reduction_percent, detour, util, ...)
    latex_data/benefit_distribution_stats.dat (gini_coefficient)

Outputs:
    <out>                                   one CSV with every (knob, value) row
    results/ablation/<knob>.dat             pgfplots-ready, one file per knob

Example:
    python scripts/collect_ablation.py --root results/ablation \
        --out results/ablation/sensitivity_summary.csv
"""

import argparse
import csv
import os
import glob


# metric key in summary.dat -> short column name in the output
SUMMARY_KEYS = {
    "reduction_percent": "dist_reduction_pct",
    "avg_detour_factor": "detour",
    "avg_vehicle_utilization": "utilization",
    "avg_acceptance_rate": "acceptance",
    "avg_altruism_score": "mean_altruism",
    "traffic_reduction_percentage": "traffic_reduction_pct",
}


def read_kv(path):
    """Read a whitespace-separated 'key value' .dat file into a dict."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    out[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return out


def collect_run(run_dir):
    latex = os.path.join(run_dir, "latex_data")
    summary = read_kv(os.path.join(latex, "summary.dat"))
    bench = read_kv(os.path.join(latex, "benefit_distribution_stats.dat"))
    row = {col: summary.get(key) for key, col in SUMMARY_KEYS.items()}
    row["gini"] = bench.get("gini_coefficient")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/ablation")
    ap.add_argument("--out", default="results/ablation/sensitivity_summary.csv")
    args = ap.parse_args()

    rows = []
    # results/ablation/<knob>/<value>/
    for knob_dir in sorted(glob.glob(os.path.join(args.root, "*"))):
        if not os.path.isdir(knob_dir):
            continue
        knob = os.path.basename(knob_dir)
        for val_dir in sorted(glob.glob(os.path.join(knob_dir, "*")),
                              key=lambda p: _as_float(os.path.basename(p))):
            if not os.path.isdir(val_dir):
                continue
            value = os.path.basename(val_dir)
            metrics = collect_run(val_dir)
            if all(v is None for v in metrics.values()):
                print(f"  (skip {knob}={value}: no metrics found)")
                continue
            rows.append({"knob": knob, "value": _as_float(value), **metrics})

    if not rows:
        print(f"No ablation runs found under {args.root}.")
        return

    cols = ["knob", "value", "dist_reduction_pct", "traffic_reduction_pct",
            "detour", "utilization", "acceptance", "mean_altruism", "gini"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"Wrote {len(rows)} rows -> {args.out}")

    # per-knob pgfplots .dat
    by_knob = {}
    for r in rows:
        by_knob.setdefault(r["knob"], []).append(r)
    for knob, knob_rows in by_knob.items():
        knob_rows.sort(key=lambda r: r["value"])
        dat = os.path.join(args.root, f"{knob}.dat")
        plot_cols = [c for c in cols if c not in ("knob",)]
        with open(dat, "w") as f:
            f.write(" ".join(plot_cols) + "\n")
            for r in knob_rows:
                f.write(" ".join(_fmt(r.get(c)) for c in plot_cols) + "\n")
        print(f"  pgfplots: {dat}")


def _as_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return s


def _fmt(v):
    if v is None or v == "":
        return "nan"
    return f"{v:.6g}" if isinstance(v, float) else str(v)


if __name__ == "__main__":
    main()
