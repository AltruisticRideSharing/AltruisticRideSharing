#!/usr/bin/env python
"""Ingest the City of Chicago Taxi Trips dataset into the ride table that
``generate_grid_dataset.py`` consumes for an arbitrary city.

Unlike NYC TLC (raw GPS) and Beijing T-Drive (raw GPS tracks), Chicago's open
taxi data **masks** pickup/dropoff coordinates to the centroid of the trip's
census tract for rider privacy (community-area centroid when a tract has fewer
than three trips in the 15-minute reporting bin). We therefore *embrace* the
tract structure rather than pretend to a precision the data does not have:
each trip's origin/destination is the real census-tract centroid it was assigned
to. Those centroids become the pickup/dropoff coordinates in the emitted table,
and the downstream generator snaps them onto the study grid exactly as it does
for NYC/Beijing -- so Chicago stays directly comparable to the other two cities
(same lattice env, same state/action space) while using Chicago's genuine
administrative geography instead of an arbitrary square.

This script emits a rides table with the SAME columns the NYC loader produces:

    pickup_longitude, pickup_latitude,
    dropoff_longitude, dropoff_latitude,
    tpep_pickup_datetime         (kept for column-compat with the NYC path)

Source
------
Socrata SODA API for the Taxi Trips dataset (resource id ``ajtu-isnz``). We pull
only trips whose start falls in the requested day/hour window and whose pickup
and dropoff centroids are both present, so the download stays small.

Relevant source fields:
    trip_start_timestamp,
    pickup_centroid_latitude, pickup_centroid_longitude,
    dropoff_centroid_latitude, dropoff_centroid_longitude,
    trip_miles

Output
------
A parquet (default ``ars/data/cache/chicago_filtered_rides.parquet``) that the
generator reads via ``--rides-source`` with ``--city chicago``.

Example
-------
    python -m ars.data.ingest_chicago \
        --date 2016-01-04 --hour 9 \
        --out ars/data/cache/chicago_filtered_rides.parquet
"""
import argparse
import os
import sys
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

# Socrata SODA v2 endpoint for the Chicago Taxi Trips resource.
SODA_BASE = "https://data.cityofchicago.org/resource/ajtu-isnz.json"

# Source (masked, census-tract-centroid) coordinate columns -> the NYC-style
# names the generator expects. The centroid IS the pickup/dropoff location here.
COL_MAP = {
    "pickup_centroid_longitude": "pickup_longitude",
    "pickup_centroid_latitude": "pickup_latitude",
    "dropoff_centroid_longitude": "dropoff_longitude",
    "dropoff_centroid_latitude": "dropoff_latitude",
}


def _fetch_page(where, limit, offset, app_token=None, retries=4):
    """Fetch one SODA page as a list of dicts, with basic backoff."""
    params = {
        "$select": (
            "trip_start_timestamp,trip_miles,"
            "pickup_centroid_latitude,pickup_centroid_longitude,"
            "dropoff_centroid_latitude,dropoff_centroid_longitude"
        ),
        "$where": where,
        "$limit": limit,
        "$offset": offset,
        "$order": "trip_start_timestamp",
    }
    url = f"{SODA_BASE}?{urlencode(params)}"
    headers = {"Accept": "application/json"}
    if app_token:
        headers["X-App-Token"] = app_token
    last_err = None
    for attempt in range(retries):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=120) as resp:
                import json
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001 - network is best-effort here
            last_err = e
            wait = 2 ** attempt
            print(f"[warn] page offset={offset} failed ({e}); retry in {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"SODA fetch failed after {retries} tries: {last_err}")


def fetch_rides(date, hour, app_token=None, page=50000, max_rows=None):
    """Pull all taxi trips starting on `date` (optionally within a single `hour`)
    that carry both pickup and dropoff census-tract centroids.

    `hour < 0` pulls the whole calendar day (more volume; recommended since the
    tract-centroid masking caps distinct locations regardless of window, so a
    full day maximises OD pairs without adding spatial resolution)."""
    # trip_start_timestamp is a floating timestamp; bound it to the requested
    # window on the calendar day.
    next_day = (pd.to_datetime(date) + pd.Timedelta(days=1)).date().isoformat()
    if hour < 0:
        start = f"{date}T00:00:00"
        end = f"{next_day}T00:00:00"
    else:
        start = f"{date}T{hour:02d}:00:00"
        end_hour = hour + 1
        if end_hour == 24:
            end = f"{next_day}T00:00:00"
        else:
            end = f"{date}T{end_hour:02d}:00:00"

    where = (
        f"trip_start_timestamp >= '{start}' "
        f"AND trip_start_timestamp < '{end}' "
        "AND pickup_centroid_latitude IS NOT NULL "
        "AND dropoff_centroid_latitude IS NOT NULL"
    )
    print(f"[build] Chicago taxi trips where: {where}")

    all_rows = []
    offset = 0
    while True:
        rows = _fetch_page(where, page, offset, app_token=app_token)
        if not rows:
            break
        all_rows.extend(rows)
        print(f"[build]   fetched {len(all_rows)} rows (offset {offset})")
        if len(rows) < page:
            break
        if max_rows is not None and len(all_rows) >= max_rows:
            all_rows = all_rows[:max_rows]
            break
        offset += page
    return all_rows


def to_ride_table(rows):
    """Convert raw SODA dicts into the NYC-style ride table (typed, renamed)."""
    if not rows:
        raise ValueError(
            "No Chicago trips returned for the requested window. Try another "
            "date/hour (weekday mornings have the most volume) or widen the hour."
        )
    df = pd.DataFrame(rows)
    # Coerce coordinate columns to float; drop any row missing one of the four.
    for src in COL_MAP:
        df[src] = pd.to_numeric(df.get(src), errors="coerce")
    df = df.dropna(subset=list(COL_MAP.keys())).reset_index(drop=True)
    df = df.rename(columns=COL_MAP)

    # Column-compat with the NYC path (generate_grid_dataset looks for this to
    # derive a `time` column; harmless if unused).
    if "trip_start_timestamp" in df.columns:
        df["tpep_pickup_datetime"] = pd.to_datetime(
            df["trip_start_timestamp"], errors="coerce"
        )
    if "trip_miles" in df.columns:
        df["trip_miles"] = pd.to_numeric(df["trip_miles"], errors="coerce")

    keep = [
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
        "tpep_pickup_datetime", "trip_miles",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].reset_index(drop=True)
    # Drop degenerate trips whose masked origin == destination (same tract): the
    # generator would discard these anyway (no inter-cell distance), but doing it
    # here makes the retained-trip count honest in the log.
    same = (
        (out["pickup_longitude"] == out["dropoff_longitude"])
        & (out["pickup_latitude"] == out["dropoff_latitude"])
    )
    n_same = int(same.sum())
    out = out[~same].reset_index(drop=True)
    print(f"[build] retained {len(out)} inter-tract trips "
          f"({n_same} same-tract trips dropped)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="2016-01-04",
                    help="calendar day of trip start, YYYY-MM-DD "
                         "(default a January weekday, ~matching the NYC window)")
    ap.add_argument("--hour", type=int, default=-1,
                    help="hour of day [0-23] to slice; <0 pulls the whole day "
                         "(default -1, recommended for volume)")
    ap.add_argument("--out", default="ars/data/cache/chicago_filtered_rides.parquet",
                    help="output parquet path")
    ap.add_argument("--app-token", default=os.environ.get("CHICAGO_APP_TOKEN"),
                    help="optional Socrata app token (raises rate limits)")
    ap.add_argument("--page", type=int, default=50000,
                    help="SODA page size")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap total rows fetched (debug)")
    args = ap.parse_args()

    rows = fetch_rides(args.date, args.hour, app_token=args.app_token,
                       page=args.page, max_rows=args.max_rows)
    out = to_ride_table(rows)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_parquet(args.out)
    print(f"[done] wrote {len(out)} Chicago rides -> {args.out}")
    # Quick spatial summary to help pick/verify the study polygon.
    print("[info] pickup lon range: "
          f"[{out['pickup_longitude'].min():.4f}, {out['pickup_longitude'].max():.4f}]")
    print("[info] pickup lat range: "
          f"[{out['pickup_latitude'].min():.4f}, {out['pickup_latitude'].max():.4f}]")
    print("[info] distinct pickup centroids: "
          f"{out[['pickup_longitude','pickup_latitude']].drop_duplicates().shape[0]}")


if __name__ == "__main__":
    main()
