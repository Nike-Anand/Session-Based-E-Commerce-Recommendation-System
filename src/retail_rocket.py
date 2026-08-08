"""
RetailRocket specific data-loading utilities.

RetailRocket's events.csv has columns:
    timestamp (unix epoch in MILLISECONDS), visitorid, event, itemid, transactionid

It has NO explicit session_id, user_id, product_id, or event_type columns.
This module:
  1. Maps columns to the schema the rest of the pipeline expects
     (session_id, user_id, product_id, event_type, timestamp-as-ISO-string).
  2. Derives sessions per visitor via an inactivity timeout (default 30 min).
  3. Builds a real popularity ranking from event counts (replacing the fake
     `range(1,k+1)` fallbacks used previously).

Usage:
    from src.retail_rocket import load_retail_rocket, build_popularity_ranking
"""
import os

import numpy as np
import pandas as pd

# Default inactivity timeout (seconds) between successive events from the same
# visitor before we start a new session. 30 minutes is the standard choice.
SESSION_TIMEOUT_SECONDS = 1800


def load_retail_rocket(events_path, session_timeout_seconds=SESSION_TIMEOUT_SECONDS):
    """
    Load RetailRocket events.csv and return a DataFrame with the project's
    canonical schema:
        session_id, user_id, product_id, event_type, timestamp (ISO string)

    Sessions are derived by grouping each visitor's events (sorted by time)
    and breaking whenever a gap exceeds `session_timeout_seconds`.
    """
    raw = pd.read_csv(
        events_path,
        usecols=["timestamp", "visitorid", "event", "itemid"],
        dtype={"timestamp": "int64", "visitorid": "int64",
               "event": "str", "itemid": "int64"},
    )

    # Convert epoch-ms -> seconds, then sort by (visitor, time).
    raw["ts_sec"] = raw["timestamp"] / 1000.0

    # Sort by visitor then time. This scalar sort is fast enough for ~2.7M rows.
    raw = raw.sort_values(["visitorid", "ts_sec"]).reset_index(drop=True)

    # Detect a new session whenever the gap from the previous event (same
    # visitor) exceeds the timeout. First event of each visitor is always a
    # new session.
    same_visitor = raw["visitorid"] == raw["visitorid"].shift(1)
    gap = raw["ts_sec"] - raw["ts_sec"].shift(1)
    new_session_flag = (~same_visitor) | (gap > session_timeout_seconds)

    # Assign a session id like: {visitorid}_{running_index_within_visitor}
    session_seq = new_session_flag.groupby(raw["visitorid"]).cumsum()
    raw["session_id"] = raw["visitorid"].astype(str) + "_" + session_seq.astype(str)

    # Now build the canonical schema.
    out = pd.DataFrame({
        "session_id": raw["session_id"].values,
        "user_id": raw["visitorid"].values,
        "product_id": raw["itemid"].values,
        "event_type": raw["event"].values,
        # Keep a stable ISO-ish string for sorting/dedup downstream.
        "timestamp": pd.to_datetime(raw["ts_sec"], unit="s").astype(str).values,
    })
    return out


def build_popularity_ranking(events: pd.DataFrame, top_k=1000):
    """
    Compute a real popularity ranking from event counts (most-viewed first).

    events: DataFrame with at least a 'product_id' column (raw ids).
    Returns: list[int] of raw product_ids ordered most-popular first.
    """
    counts = events["product_id"].value_counts()
    return counts.index.tolist()[:top_k]


def prepare_retail_rocket(events_csv_path, out_csv_path,
                          session_timeout_seconds=SESSION_TIMEOUT_SECONDS):
    """
    One-shot helper: load RetailRocket events, derive sessions, and write the
    canonical schema to `out_csv_path` so the rest of the pipeline (which
    reads a plain events.csv) can consume it without a --retail flag.
    """
    events = load_retail_rocket(events_csv_path, session_timeout_seconds)
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    events.to_csv(out_csv_path, index=False)
    print(f"RetailRocket -> canonical schema written to: {out_csv_path}")
    print(f"  {len(events)} events | {events['session_id'].nunique()} sessions | "
          f"{events['user_id'].nunique()} users | {events['product_id'].nunique()} products")
    return events


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True,
                        help="Path to RetailRocket events.csv")
    parser.add_argument("--out", default="data/retail_rocket_events.csv",
                        help="Where to write the canonical-schema events.csv")
    parser.add_argument("--timeout", type=int, default=SESSION_TIMEOUT_SECONDS,
                        help="Session inactivity timeout in seconds")
    args = parser.parse_args()

    prepare_retail_rocket(args.src, args.out, args.timeout)
    # Also emit popularity ranking for reference.
    events = pd.read_csv(args.out)
    pop = build_popularity_ranking(events)
    print(f"Top-10 most popular products: {pop[:10]}")
