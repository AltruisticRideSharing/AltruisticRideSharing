#!/usr/bin/env python
"""Ingest the Beijing T-Drive taxi GPS dataset into the ride table that
``generate_grid_dataset.py`` consumes for an arbitrary city.

T-Drive (Microsoft Research Asia; Yuan et al., 2010/2011) is one week of GPS
tracks for 10,357 Beijing taxis. Unlike NYC TLC it is *raw point streams*, not
pre-segmented pickup -> dropoff trips, so this script segments each taxi's track
into trips and emits a rides table with the SAME columns the NYC loader produces:

    pickup_longitude, pickup_latitude,
    dropoff_longitude, dropoff_latitude,
    tpep_pickup_datetime         (kept for column-compat with the NYC path)

Raw input
---------
A directory of per-taxi CSV files (the canonical T-Drive `release/taxi_log_2008_by_id/`
layout), each row:

    taxi_id, datetime, longitude, latitude
    e.g.  1, 2008-02-02 13:30:00, 116.51172, 39.92123

T-Drive carries no occupied/vacant flag, so "trips" are derived geometrically:
a track is split wherever the taxi is idle (does not move beyond `--stop-radius-m`)
for at least `--stop-gap-min` minutes, or where consecutive fixes are more than
`--max-fix-gap-min` apart (GPS dropout). Each resulting moving segment with a
plausible length becomes one pickup -> dropoff trip. This is the standard way
T-Drive is turned into OD trips in the literature and is sufficient for sampling
agent origin/destination cells; it does not claim to recover fare meter states.

Output
------
A parquet (default ``ars/data/cache/beijing_filtered_rides.parquet``) that the
generator reads via ``--rides-source`` with ``--city beijing``.

Example
-------
    python -m ars.data.ingest_tdrive \
        --raw-dir ~/tdrive/release/taxi_log_2008_by_id \
        --out ars/data/cache/beijing_filtered_rides.parquet
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from geopy.distance import geodesic

# Default Beijing study window: a compact ~5x5 km commercial core (Tiananmen /
# Forbidden City -> Wangfujing -> toward the CBD) chosen so a 15x15 grid yields
# cells of ~0.3-0.4 km, matching the NYC study area's physical cell size (the
# broad 3rd-ring bbox gave ~1.1 km cells, which broke the km-scaled reward).
# Overridable with --bbox. (min_lon, min_lat, max_lon, max_lat)
DEFAULT_BBOX = (116.37, 39.89, 116.43, 39.94)


# Plausible China bounding box; raw T-Drive carries occasional corrupt fixes
# (zeros, swapped fields, out-of-range values) that must be dropped before any
# geodesic call, which rejects |lat| > 90.
CHINA_LON = (73.0, 135.0)
CHINA_LAT = (18.0, 54.0)


def read_one_taxi(path):
    """Read a single T-Drive per-taxi CSV. Tolerant of header/no-header and of
    corrupt GPS fixes (out-of-range or zero coordinates)."""
    try:
        df = pd.read_csv(path, header=None,
                         names=["taxi_id", "time", "lon", "lat"])
        # If the first row was actually a header, the parse makes lon/lat NaN.
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df = df.dropna(subset=["lon", "lat"])
        # Drop physically-impossible / corrupt fixes before any geodesic call.
        df = df[
            df["lon"].between(*CHINA_LON) & df["lat"].between(*CHINA_LAT)
        ]
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"])
        return df
    except Exception:
        return None


def segment_trips(df, stop_radius_m, stop_gap_min, max_fix_gap_min,
                  min_trip_km, max_trip_km):
    """Split one taxi's ordered track into OD trips.

    A trip ends when the taxi stays within `stop_radius_m` for >= `stop_gap_min`
    minutes (a dwell), or a fix-to-fix time gap exceeds `max_fix_gap_min`
    (dropout). Each moving segment whose straight-line OD length is within
    [min_trip_km, max_trip_km] becomes one (pickup, dropoff) record.
    """
    df = df.sort_values("time").reset_index(drop=True)
    if len(df) < 2:
        return []

    trips = []
    seg_start = 0
    dwell_anchor = 0  # index where the current dwell began

    def emit(s, e):
        if e <= s:
            return
        p = df.iloc[s]
        d = df.iloc[e]
        km = geodesic((p["lat"], p["lon"]), (d["lat"], d["lon"])).kilometers
        if min_trip_km <= km <= max_trip_km:
            trips.append({
                "pickup_longitude": p["lon"], "pickup_latitude": p["lat"],
                "dropoff_longitude": d["lon"], "dropoff_latitude": d["lat"],
                "tpep_pickup_datetime": p["time"],
                "distance_km_raw": km,
            })

    for i in range(1, len(df)):
        dt_min = (df["time"].iloc[i] - df["time"].iloc[i - 1]).total_seconds() / 60.0
        move_m = geodesic(
            (df["lat"].iloc[i], df["lon"].iloc[i]),
            (df["lat"].iloc[dwell_anchor], df["lon"].iloc[dwell_anchor]),
        ).meters

        if dt_min > max_fix_gap_min:
            # GPS dropout: close the current segment, restart after the gap.
            emit(seg_start, i - 1)
            seg_start = i
            dwell_anchor = i
            continue

        if move_m > stop_radius_m:
            # Still moving; advance the dwell anchor to "now".
            dwell_anchor = i
        else:
            # Within stop radius of the dwell anchor; check dwell duration.
            dwell_min = (df["time"].iloc[i] - df["time"].iloc[dwell_anchor]).total_seconds() / 60.0
            if dwell_min >= stop_gap_min:
                # A real stop: the segment up to the dwell anchor is a trip.
                emit(seg_start, dwell_anchor)
                seg_start = i
                dwell_anchor = i

    emit(seg_start, len(df) - 1)
    return trips


def in_bbox(df, bbox):
    lo_lon, lo_lat, hi_lon, hi_lat = bbox
    return df[
        (df["pickup_longitude"].between(lo_lon, hi_lon))
        & (df["pickup_latitude"].between(lo_lat, hi_lat))
        & (df["dropoff_longitude"].between(lo_lon, hi_lon))
        & (df["dropoff_latitude"].between(lo_lat, hi_lat))
    ].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True,
                    help="dir of T-Drive per-taxi CSVs (taxi_log_2008_by_id/)")
    ap.add_argument("--out", default="ars/data/cache/beijing_filtered_rides.parquet",
                    help="output parquet path (feed to generator --rides-source)")
    ap.add_argument("--bbox", type=float, nargs=4, default=list(DEFAULT_BBOX),
                    metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                    help="study window; trips kept only if pickup+dropoff inside")
    ap.add_argument("--max-taxis", type=int, default=0,
                    help="cap number of taxi files processed (0 = all)")
    ap.add_argument("--stop-radius-m", type=float, default=50.0)
    ap.add_argument("--stop-gap-min", type=float, default=5.0)
    ap.add_argument("--max-fix-gap-min", type=float, default=10.0)
    ap.add_argument("--min-trip-km", type=float, default=0.5)
    ap.add_argument("--max-trip-km", type=float, default=30.0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.raw_dir, "*.txt"))
                   + glob.glob(os.path.join(args.raw_dir, "*.csv")))
    if not files:
        raise SystemExit(
            f"No T-Drive files (*.txt/*.csv) found in {args.raw_dir}. "
            "Point --raw-dir at the unzipped taxi_log_2008_by_id/ folder.")
    if args.max_taxis:
        files = files[:args.max_taxis]
    print(f"[t-drive] {len(files)} taxi files from {args.raw_dir}")

    all_trips = []
    for n, path in enumerate(files):
        df = read_one_taxi(path)
        if df is None or df.empty:
            continue
        all_trips.extend(segment_trips(
            df, args.stop_radius_m, args.stop_gap_min, args.max_fix_gap_min,
            args.min_trip_km, args.max_trip_km))
        if (n + 1) % 500 == 0:
            print(f"  processed {n+1}/{len(files)} taxis, "
                  f"{len(all_trips)} trips so far")

    trips = pd.DataFrame(all_trips)
    print(f"[t-drive] segmented trips (pre-bbox): {len(trips)}")
    if trips.empty:
        raise SystemExit("No trips segmented; loosen --min-trip-km/--stop-gap-min.")

    trips = in_bbox(trips, tuple(args.bbox))
    print(f"[t-drive] trips inside bbox {tuple(args.bbox)}: {len(trips)}")
    if trips.empty:
        raise SystemExit("No trips inside bbox; widen --bbox.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    trips.to_parquet(args.out)
    lon = pd.concat([trips["pickup_longitude"], trips["dropoff_longitude"]])
    lat = pd.concat([trips["pickup_latitude"], trips["dropoff_latitude"]])
    print(f"[t-drive] wrote {len(trips)} trips -> {args.out}")
    print(f"[t-drive] lon [{lon.min():.4f},{lon.max():.4f}] "
          f"lat [{lat.min():.4f},{lat.max():.4f}] "
          f"mean trip {trips['distance_km_raw'].mean():.2f} km")
    print("\nNext: generate the grid dataset with")
    print(f"  python -m ars.data.generate_grid_dataset --city beijing \\")
    print(f"      --rides-source {args.out} --grid-sizes 15 --num-agents 100 \\")
    print(f"      --out-template 'beijing_{{num_agents}}_agents'")


if __name__ == "__main__":
    main()
