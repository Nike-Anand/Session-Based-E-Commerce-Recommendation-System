"""
Exploratory data analysis on the events log.
Run as a plain script (or paste into a Jupyter cell).

Usage:
    python notebooks/eda.py --data data/events.csv
"""
import argparse

import pandas as pd


def run_eda(events_path):
    events = pd.read_csv(events_path)

    print(f"Total events: {len(events)}")
    print(f"Unique users: {events['user_id'].nunique()}")
    print(f"Unique products: {events['product_id'].nunique()}")
    print(f"Unique sessions: {events['session_id'].nunique()}")

    session_lengths = events.groupby("session_id").size()
    print(f"\nSession length — mean: {session_lengths.mean():.2f}, "
          f"median: {session_lengths.median():.1f}, max: {session_lengths.max()}")

    product_freq = events["product_id"].value_counts()
    top10_share = product_freq.head(10).sum() / len(events) * 100
    print(f"\nTop 10 products account for {top10_share:.1f}% of events "
          f"(power-law check — high % implies popularity bias risk)")

    events["hour"] = pd.to_datetime(events["timestamp"]).dt.hour
    print("\nEvents by hour of day:")
    print(events.groupby("hour").size())

    if "event_type" in events.columns:
        purchases = (events["event_type"] == "purchase").sum()
        print(f"\nPurchase rate: {purchases / len(events) * 100:.2f}%")
        print("\nEvent type breakdown:")
        print(events["event_type"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/events.csv")
    args = parser.parse_args()
    run_eda(args.data)
