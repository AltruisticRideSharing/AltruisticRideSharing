#!/usr/bin/env python3
"""Plot three heatmaps from latex_data .dat files:
- density_no_sharing_grid.dat
- density_sharing_grid.dat
- density_reduction_grid.dat

Creates a figure with 3 side-by-side subplots and saves/shows it.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_dat(path):
    # Files include a header line: "x y density". Read with a regex separator and header=0.
    df = pd.read_csv(path, sep=r"\s+", header=0)
    # Ensure numeric columns
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['density'] = pd.to_numeric(df['density'], errors='coerce')
    return df


def df_to_grid(df):
    pivot = df.pivot(index="y", columns="x", values="density")
    # Convert index/columns to numeric sorts (they may be floats after coercion)
    pivot.index = pd.to_numeric(pivot.index, errors='coerce')
    pivot.columns = pd.to_numeric(pivot.columns, errors='coerce')
    pivot = pivot.sort_index(axis=0, ascending=True).sort_index(axis=1, ascending=True)
    return pivot


def main():
    parser = argparse.ArgumentParser(description="Plot density heatmaps from latex_data .dat files")
    parser.add_argument("--latex_dir", default="./latex_data", help="Path to folder containing the .dat files")
    parser.add_argument("--out", default=None, help="Output image file (png). If omitted the plot will be shown")
    parser.add_argument("--dpi", type=int, default=150, help="Output image DPI")
    args = parser.parse_args()

    no_path = os.path.join(args.latex_dir, "density_no_sharing_grid.dat")
    sh_path = os.path.join(args.latex_dir, "density_sharing_grid.dat")

    for p in (no_path, sh_path):
        if not os.path.isfile(p):
            print(f"Required file not found: {p}")
            sys.exit(1)

    df_no = load_dat(no_path)
    df_sh = load_dat(sh_path)

    grid_no = df_to_grid(df_no)
    grid_sh = df_to_grid(df_sh)

    if not grid_no.index.equals(grid_sh.index) or not grid_no.columns.equals(grid_sh.columns):
        grid_sh = grid_sh.reindex(index=grid_no.index, columns=grid_no.columns, fill_value=0)

    grid_diff = grid_no - grid_sh

    total_no = np.nansum(grid_no.values)
    total_sh = np.nansum(grid_sh.values)
    overall_reduction_pct = 0.0
    if total_no > 0:
        overall_reduction_pct = 100.0 * (total_no - total_sh) / total_no

    sns.set(style="white")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    vmin = float(np.nanmin([grid_no.values.min(), grid_sh.values.min()]))
    vmax = float(np.nanmax([grid_no.values.max(), grid_sh.values.max()]))

    ax = axs[0]
    sns.heatmap(grid_no, ax=ax, cmap="Reds", cbar_kws={"label": "Density"}, vmin=vmin, vmax=vmax)
    ax.set_title("(a) No Ride-Sharing")
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax = axs[1]
    sns.heatmap(grid_sh, ax=ax, cmap="Reds", cbar_kws={"label": "Density"}, vmin=vmin, vmax=vmax)
    ax.set_title("(b) With OR-Tools Ride-Sharing")
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax = axs[2]
    sns.heatmap(grid_diff, ax=ax, cmap="RdYlGn", center=0, cbar_kws={"label": "Density Change"})
    ax.set_title("(c) Traffic Density Reduction")
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.text(0.5, 0.02, f"Overall Traffic Reduction: {overall_reduction_pct:.1f}%", ha="center", fontsize=12,
             bbox=dict(facecolor="white", alpha=0.8, boxstyle="round"))

    if args.out:
        fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
