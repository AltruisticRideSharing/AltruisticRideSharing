#!/usr/bin/env python
"""Generate the ARS grid dataset (weight matrices + sampled agent trips) for an
arbitrary square grid size, derived from the NYC TLC notebook
(Ride_Sharing_Datatset.ipynb).

For each grid size it writes a self-contained dataset folder containing the four
files the environment loads (env/env.py:34-48):

    weight_matrix.npy        (rows, cols, 4) directional road distances (km)
    time_weight_matrix.npy   (rows, cols, 4) directional travel times (min)
    fixed_positions.npy      (num_agents, 2) int (row, col) pickup cells
    fixed_destinations.npy   (num_agents, 2) int (row, col) dropoff cells

plus diagnostics (grid coords, rides CSV, coverage PNG).

Differences from the notebook, all behaviour-preserving:
  * Grid size is a parameter; no hardcoded 15 / 225.
  * Only the 4 directional neighbours of each cell are routed (4*rows*cols
    Dijkstra calls) instead of the full (rows*cols)^2 dense tensor the notebook
    built and then discarded. This produces an identical weight_matrix while
    making large grids (e.g. 45x45) fast.
  * The OSM graph and the polygon-filtered ride table are cached to disk so a
    multi-grid sweep downloads / filters once.
  * Trip-distance sampling regime is held fixed across grid sizes (default the
    notebook's "medium" 1.5-5.0 km range) so the ablation varies only
    discretisation granularity, not the trip distribution.

Examples
--------
    # the three ablation grids, 100 agents each, into dataset/100_agents_grid{N}/
    python generate_grid_dataset.py --grid-sizes 15 30 45 --num-agents 100

    # regenerate the canonical 15x15 baseline into dataset/100_agents/
    python generate_grid_dataset.py --grid-sizes 15 --num-agents 100 \
        --out-template 100_agents
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from geopy.distance import geodesic

# Heavy / optional imports are done lazily inside the functions that need them
# (osmnx, networkx, datasets, matplotlib, seaborn) so that --help and cache
# reuse do not require the full stack.

# ---------------------------------------------------------------------------
# Study areas (lon, lat), one quadrilateral per city. NYC is identical to the
# notebook. Beijing covers a dense inner-ring slice matching ingest_tdrive's
# default bbox. A city's polygon may also be derived from its rides table
# (--polygon auto) so a different bbox does not need editing here.
# ---------------------------------------------------------------------------
CITY_POLYGONS = {
    "nyc": np.array([
        (-73.965, 40.815),   # top-left
        (-73.930, 40.800),   # top-right
        (-73.975, 40.740),   # bottom-right
        (-74.008, 40.754),   # bottom-left
    ]),
    # Compact ~5x5 km commercial core, matching ingest_tdrive.DEFAULT_BBOX
    # (116.37,39.89,116.43,39.94). Sized so a 15x15 grid gives ~0.3-0.4 km
    # cells, comparable to the NYC study area (avoids the km-scale reward
    # mismatch the broad 15x15 km bbox caused).
    "beijing": np.array([
        (116.37, 39.94),   # top-left
        (116.43, 39.94),   # top-right
        (116.43, 39.89),   # bottom-right
        (116.37, 39.89),   # bottom-left
    ]),
    # Chicago downtown core, sized to match NYC's physical footprint:
    # 6.61 x 8.31 km (NYC is 6.58 x 8.33 km), giving ~0.44 x 0.55 km cells on a
    # 15x15 grid -- the same physical AND cell size as NYC/Beijing for direct
    # comparability. Centred on the densest Loop centroid cluster (~-87.632,
    # 41.887). The open taxi data masks pickup/dropoff to census-tract centroids,
    # so only ~44 distinct downtown locations exist -> ~17% cell coverage (vs
    # NYC/Beijing 48-64%): a documented data limitation, accepted in exchange for
    # matched grid geometry. Grids 15/30/45. See ars/data/ingest_chicago.py.
    "chicago": np.array([
        (-87.6718, 41.9244),   # top-left
        (-87.5922, 41.9244),   # top-right
        (-87.5922, 41.8496),   # bottom-right
        (-87.6718, 41.8496),   # bottom-left
    ]),
}

# Backwards-compat alias: existing callers / docs referencing POLYGON get NYC.
POLYGON = CITY_POLYGONS["nyc"]


def polygon_from_rides(rides_df, pad=0.002):
    """Axis-aligned quadrilateral bounding the rides table (lon/lat), padded.
    Used when --polygon auto, e.g. for a Beijing bbox other than the default."""
    lon = pd.concat([rides_df["pickup_longitude"], rides_df["dropoff_longitude"]])
    lat = pd.concat([rides_df["pickup_latitude"], rides_df["dropoff_latitude"]])
    lo_lon, hi_lon = lon.min() - pad, lon.max() + pad
    lo_lat, hi_lat = lat.min() - pad, lat.max() + pad
    return np.array([
        (lo_lon, hi_lat),  # top-left
        (hi_lon, hi_lat),  # top-right
        (hi_lon, lo_lat),  # bottom-right
        (lo_lon, lo_lat),  # bottom-left
    ])

DEFAULT_SPEED_KPH = 25.0
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

# Repo root = two levels up from this file (dataset/dataset_creation/..)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CACHE_DIR = os.path.join(SCRIPT_DIR, "cache")


# ---------------------------------------------------------------------------
# Grid + geometry helpers (verbatim from the notebook)
# ---------------------------------------------------------------------------
def create_rotated_grid(polygon, rows, cols):
    """rows x cols grid of lat/lon centres inside the quadrilateral."""
    top_edge = np.linspace(polygon[0], polygon[1], cols)
    bottom_edge = np.linspace(polygon[3], polygon[2], cols)
    grid = np.zeros((rows, cols, 2))
    for i in range(cols):
        grid[:, i] = np.linspace(top_edge[i], bottom_edge[i], rows)
    return grid


def point_in_polygon(point, polygon):
    """Ray-casting point-in-polygon test."""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


# ---------------------------------------------------------------------------
# Cached, grid-independent stages: ride filtering + OSM graph
# ---------------------------------------------------------------------------
def load_filtered_rides(cache_dir, polygon, city="nyc", rides_source=None):
    """Load the polygon-filtered ride table, building (and caching) it on first
    use. Grid-independent, so shared across all grid sizes for one city.

    nyc      : pulls the NYC TLC slice from the HF dataset (notebook behaviour).
    other    : reads a pre-built rides parquet/CSV (e.g. Beijing T-Drive output
               from ingest_tdrive) given by ``rides_source`` and filters it to
               ``polygon``. The file must carry pickup/dropoff lon/lat columns.
    """
    cache_path = os.path.join(cache_dir, f"{city}_filtered_rides.parquet")
    if os.path.exists(cache_path):
        print(f"[cache] {city} filtered rides <- {cache_path}")
        return pd.read_parquet(cache_path)

    if rides_source is not None:
        print(f"[build] loading {city} rides from {rides_source} ...")
        if rides_source.endswith(".parquet"):
            df = pd.read_parquet(rides_source)
        else:
            df = pd.read_csv(rides_source)
        if "tpep_pickup_datetime" in df.columns:
            df["time"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
        df_t = df
    else:
        print("[build] loading NYC TLC dataset and filtering to study window ...")
        from datasets import load_dataset
        ds = load_dataset("Divyanshudiv/NYC_TLC_Data_2016")
        df = ds["train"].to_pandas()
        df["time"] = pd.to_datetime(df["tpep_pickup_datetime"])
        df_t = df[(df["time"].dt.date == pd.to_datetime("2016-01-02").date())
                  & (df["time"].dt.hour == 9)]

    rows = []
    for _, row in df_t.iterrows():
        pu = (row["pickup_longitude"], row["pickup_latitude"])
        do = (row["dropoff_longitude"], row["dropoff_latitude"])
        if point_in_polygon(pu, polygon) and point_in_polygon(do, polygon):
            rows.append(row)
    out = pd.DataFrame(rows).reset_index(drop=True)
    print(f"[build] {city} rides within polygon: {len(out)}")
    if out.empty:
        raise ValueError(
            f"No {city} rides fell inside the polygon. Check --polygon / bbox.")

    os.makedirs(cache_dir, exist_ok=True)
    out.to_parquet(cache_path)
    return out


def load_graph(cache_dir, polygon, city="nyc"):
    """Load the drive-network graph for the study area, caching it. Grid-independent."""
    cache_path = os.path.join(cache_dir, f"{city}_osm_graph.pkl")
    if os.path.exists(cache_path):
        print(f"[cache] {city} OSM graph <- {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"[build] downloading {city} OSM drive network ...")
    import osmnx as ox
    pad = 0.005
    min_lat, max_lat = polygon[:, 1].min(), polygon[:, 1].max()
    min_lon, max_lon = polygon[:, 0].min(), polygon[:, 0].max()
    graph = ox.graph_from_bbox(
        bbox=(min_lon - pad, min_lat - pad, max_lon + pad, max_lat + pad),
        network_type="drive",
    )
    print(f"[build] graph nodes={len(graph.nodes)} edges={len(graph.edges)}")
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(graph, f)
    return graph


# ---------------------------------------------------------------------------
# Grid-dependent stages
# ---------------------------------------------------------------------------
def assign_rides_to_grid(rides_df, grid):
    """Snap each ride's pickup/dropoff to the nearest grid cell and record the
    inter-cell geodesic distance. Returns a DataFrame of grid-cell rides."""
    rows, cols = grid.shape[0], grid.shape[1]
    flat = grid.reshape(-1, 2)  # (rows*cols, 2) as (lon, lat)

    def nearest_cell(lon, lat):
        # geodesic over all cells; rows*cols is small even for 45x45 (2025)
        dmin, best = float("inf"), 0
        for k in range(flat.shape[0]):
            d = geodesic((lat, lon), (flat[k, 1], flat[k, 0])).meters
            if d < dmin:
                dmin, best = d, k
        return divmod(best, cols)

    out = []
    for idx, ride in rides_df.iterrows():
        pr, pc = nearest_cell(ride["pickup_longitude"], ride["pickup_latitude"])
        dr, dc = nearest_cell(ride["dropoff_longitude"], ride["dropoff_latitude"])
        if (pr, pc) == (dr, dc):
            continue
        dist_km = geodesic((grid[pr, pc][1], grid[pr, pc][0]),
                           (grid[dr, dc][1], grid[dr, dc][0])).kilometers
        out.append({
            "pickup_cell_row": pr, "pickup_cell_col": pc,
            "dropoff_cell_row": dr, "dropoff_cell_col": dc,
            "distance_km": dist_km, "original_index": idx,
        })
    df = pd.DataFrame(out)
    print(f"[grid {rows}x{cols}] rides snapped to distinct cells: {len(df)}")
    return df


def build_directional_matrices(grid, graph, speed_kph=DEFAULT_SPEED_KPH):
    """Compute the (rows, cols, 4) directional distance and time matrices.

    Only the 4 axis neighbours of each cell are routed on the road network,
    which is all the environment's Dijkstra ever consumes. This matches the
    notebook's directional matrices exactly while avoiding the full dense
    (rows*cols)^2 routing the notebook performed and threw away.
    """
    import networkx as nx
    import osmnx as ox

    rows, cols = grid.shape[0], grid.shape[1]

    # Nearest road node per cell (one query per cell)
    cell_node = {}
    for i in range(rows):
        for j in range(cols):
            lon, lat = grid[i, j]
            try:
                cell_node[(i, j)] = ox.distance.nearest_nodes(graph, X=lon, Y=lat)
            except Exception:
                print(f"  [warn] no road node for cell ({i},{j})")
                cell_node[(i, j)] = None

    def road_km(a, b):
        """Road distance (km) between cells a,b, geodesic fallback."""
        na, nb = cell_node.get(a), cell_node.get(b)
        if na is not None and nb is not None:
            try:
                return nx.shortest_path_length(graph, na, nb, weight="length") / 1000.0
            except Exception:
                pass
        ca, cb = grid[a[0], a[1]], grid[b[0], b[1]]
        return geodesic((ca[1], ca[0]), (cb[1], cb[0])).kilometers

    w = np.zeros((rows, cols, 4))
    t = np.zeros((rows, cols, 4))
    neighbours = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

    for i in range(rows):
        for j in range(cols):
            for d, (di, dj) in neighbours.items():
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    km = road_km((i, j), (ni, nj))
                    w[i, j, d] = km
                    t[i, j, d] = (km / speed_kph) * 60.0
                else:
                    w[i, j, d] = float("inf")
                    t[i, j, d] = float("inf")
    return w, t


def sample_uniform_positions(rides_df, grid, num_agents, distance_range, seed=42):
    """Sample agent trips uniformly across distance bins (notebook logic),
    optionally filtered to a distance range. Returns (starts, ends, samples)."""
    np.random.seed(seed)
    df = rides_df.copy()
    if distance_range is not None:
        lo, hi = distance_range
        df = df[(df["distance_km"] >= lo) & (df["distance_km"] <= hi)]
        print(f"  filtered to {len(df)} rides in {lo}-{hi} km")
    if len(df) == 0:
        raise ValueError("No rides available after distance filtering")

    distances = df["distance_km"].values
    n_bins = max(1, min(10, len(df) // 5))
    bins = np.linspace(distances.min(), distances.max(), n_bins + 1)
    bin_idx = np.digitize(distances, bins) - 1

    per_bin, remaining = divmod(num_agents, n_bins)
    selected = []
    for b in range(n_bins):
        bucket = df[bin_idx == b]
        if len(bucket) == 0:
            continue
        k = min(per_bin + (1 if b < remaining else 0), len(bucket))
        selected.extend(bucket.sample(k, replace=True).index.tolist())
    while len(selected) < num_agents:
        selected.extend(df.sample(1).index.tolist())

    samples = df.loc[selected[:num_agents]]
    starts = [(int(r["pickup_cell_row"]), int(r["pickup_cell_col"]))
              for _, r in samples.iterrows()]
    ends = [(int(r["dropoff_cell_row"]), int(r["dropoff_cell_col"]))
            for _, r in samples.iterrows()]
    return starts, ends, samples


def save_coverage_plot(starts, ends, grid, path):
    rows, cols = grid.shape[0], grid.shape[1]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as e:
        print(f"  [skip] coverage plot ({e})")
        return
    counts = np.zeros((rows, cols))
    for pos in set(starts + ends):
        counts[pos[0], pos[1]] = 1
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(counts, cmap="YlOrRd", cbar=True, ax=ax)
    cov = len(set(starts + ends)) / (rows * cols) * 100
    ax.set_title(f"Unique trip cells ({rows}x{cols}) — {cov:.1f}% coverage")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-grid driver
# ---------------------------------------------------------------------------
def _grid_cache(cache_dir, city, grid_size):
    """Paths for the agent-INDEPENDENT per-(city,grid) products: the directional
    road matrices and the snapped ride table. These depend only on the city's
    road graph and grid resolution, never on num_agents, so they are computed
    once and reused across every agent count / sampling seed for that grid."""
    base = os.path.join(cache_dir, f"{city}_grid{grid_size}")
    return {
        "w": base + "_weight_matrix.npy",
        "t": base + "_time_weight_matrix.npy",
        "rides": base + "_grid_rides.parquet",
    }


def build_or_load_grid_products(grid, graph, city, grid_size, rides_df, cache_dir):
    """Return (w, t, grid_rides) for one (city, grid), using the cache if present.

    Routing the road network (build_directional_matrices) and snapping every
    ride to its nearest cell (assign_rides_to_grid) are the two expensive,
    agent-independent stages. Caching them by (city, grid) means a second agent
    count for the same grid is generated in seconds instead of re-routing.
    """
    cp = _grid_cache(cache_dir, city, grid_size)
    if all(os.path.exists(p) for p in cp.values()):
        print(f"[cache] {city} grid{grid_size} matrices + snapped rides <- {cache_dir}")
        w = np.load(cp["w"])
        t = np.load(cp["t"])
        grid_rides = pd.read_parquet(cp["rides"])
        return w, t, grid_rides

    grid_rides = assign_rides_to_grid(rides_df, grid)
    w, t = build_directional_matrices(grid, graph)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cp["w"], w)
    np.save(cp["t"], t)
    grid_rides.to_parquet(cp["rides"])
    print(f"[cache] {city} grid{grid_size} matrices + snapped rides -> {cache_dir}")
    return w, t, grid_rides


def generate_for_grid(grid_size, num_agents, out_dir, rides_df, graph,
                      distance_range, seed, polygon, city, cache_dir):
    rows = cols = grid_size
    print(f"\n=== grid {rows}x{cols} -> {out_dir} ===")
    os.makedirs(out_dir, exist_ok=True)

    grid = create_rotated_grid(polygon, rows, cols)
    w, t, grid_rides = build_or_load_grid_products(
        grid, graph, city, grid_size, rides_df, cache_dir)
    starts, ends, samples = sample_uniform_positions(
        grid_rides, grid, num_agents, distance_range, seed)

    # Env-loaded files (shapes/dtypes match the existing 15x15 artifacts)
    np.save(os.path.join(out_dir, "weight_matrix.npy"), w)
    np.save(os.path.join(out_dir, "time_weight_matrix.npy"), t)
    np.save(os.path.join(out_dir, "fixed_positions.npy"),
            np.array(starts, dtype=np.int64))
    np.save(os.path.join(out_dir, "fixed_destinations.npy"),
            np.array(ends, dtype=np.int64))

    # Diagnostics
    np.save(os.path.join(out_dir, "rotated_grid_coordinates.npy"), grid)
    grid_rides.to_csv(os.path.join(out_dir, "grid_rides.csv"), index=False)
    save_coverage_plot(starts, ends, grid,
                       os.path.join(out_dir, "sampling_coverage.png"))

    cov = len(set(starts + ends)) / (rows * cols) * 100
    print(f"  saved: weight {w.shape}, positions {len(starts)} agents, "
          f"mean trip {samples['distance_km'].mean():.2f} km, "
          f"grid coverage {cov:.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-sizes", type=int, nargs="+", default=[15, 30, 45],
                    help="square grid side lengths to generate")
    ap.add_argument("--num-agents", type=int, default=100,
                    help="number of agent trips to sample per grid")
    ap.add_argument("--out-template", type=str, default="{num_agents}_agents_grid{grid}",
                    help="dataset folder name template (under dataset/). "
                         "Fields: {num_agents}, {grid}. "
                         "Use e.g. '{num_agents}_agents' to write the canonical folder.")
    ap.add_argument("--distance-min", type=float, default=1.5,
                    help="min trip distance km for sampling (notebook 'medium')")
    ap.add_argument("--distance-max", type=float, default=5.0,
                    help="max trip distance km for sampling")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", type=str, default=CACHE_DIR)
    ap.add_argument("--city", type=str, default="nyc",
                    choices=sorted(CITY_POLYGONS.keys()) + ["other"],
                    help="study city. nyc pulls NYC TLC; others read --rides-source.")
    ap.add_argument("--rides-source", type=str, default=None,
                    help="parquet/CSV of pre-built rides (pickup/dropoff lon-lat). "
                         "Required for any --city other than nyc, e.g. the "
                         "Beijing output of `python -m ars.data.ingest_tdrive`.")
    ap.add_argument("--polygon", type=str, default="city",
                    help="'city' (preset for --city), 'auto' (bound the rides "
                         "table), or 8 numbers 'lon1 lat1 ... lon4 lat4'.")
    args = ap.parse_args()

    dist_range = (args.distance_min, args.distance_max)

    if args.city != "nyc" and args.rides_source is None:
        raise SystemExit(f"--city {args.city} requires --rides-source "
                         "(build it with ingest_tdrive for Beijing).")

    # Resolve the study polygon.
    if args.polygon == "city":
        if args.city not in CITY_POLYGONS:
            raise SystemExit(f"No preset polygon for city '{args.city}'; "
                             "pass --polygon auto or explicit coordinates.")
        polygon = CITY_POLYGONS[args.city]
    elif args.polygon == "auto":
        polygon = None  # derived after the rides table loads
    else:
        nums = [float(x) for x in args.polygon.split()]
        if len(nums) != 8:
            raise SystemExit("--polygon needs 8 numbers: lon1 lat1 ... lon4 lat4")
        polygon = np.array(list(zip(nums[0::2], nums[1::2])))

    # Grid-independent stages, computed once.
    if polygon is None:
        # 'auto': peek at the rides source to bound the area, then filter.
        src = args.rides_source
        raw = (pd.read_parquet(src) if src.endswith(".parquet")
               else pd.read_csv(src))
        polygon = polygon_from_rides(raw)
        print(f"[auto-polygon] {polygon.tolist()}")
    rides_df = load_filtered_rides(args.cache_dir, polygon,
                                   city=args.city, rides_source=args.rides_source)
    graph = load_graph(args.cache_dir, polygon, city=args.city)

    for g in args.grid_sizes:
        folder = args.out_template.format(num_agents=args.num_agents, grid=g)
        out_dir = os.path.join(REPO_ROOT, "dataset", folder)
        generate_for_grid(g, args.num_agents, out_dir, rides_df, graph,
                          dist_range, args.seed, polygon, args.city, args.cache_dir)

    print("\nDone. Use the folder name(s) as --dataset when launching runs, e.g.:")
    for g in args.grid_sizes:
        folder = args.out_template.format(num_agents=args.num_agents, grid=g)
        print(f"  --dataset {folder} --grid-size {g}")


if __name__ == "__main__":
    main()
