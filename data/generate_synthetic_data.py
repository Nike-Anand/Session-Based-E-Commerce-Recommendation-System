"""
Generate synthetic but *structured* e-commerce session data.

Unlike pure random product assignment, this generator gives products a
category, and biases each session so that consecutive views are likely to
come from the same or a related category — mimicking real "building a
gaming setup" style browsing behavior described in the spec doc.

Usage:
    python data/generate_synthetic_data.py --num_users 2000 --num_products 3000 \
        --out data/events.csv
"""
import argparse
import json
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

CATEGORIES = [
    "laptops", "gaming_peripherals", "monitors", "audio",
    "smart_home", "kitchen", "fitness", "books", "apparel", "furniture",
]

# Categories that are commonly browsed together (adjacency list)
RELATED = {
    "laptops": ["gaming_peripherals", "monitors", "audio"],
    "gaming_peripherals": ["laptops", "monitors"],
    "monitors": ["laptops", "gaming_peripherals"],
    "audio": ["laptops", "fitness"],
    "smart_home": ["kitchen", "furniture"],
    "kitchen": ["smart_home", "furniture"],
    "fitness": ["apparel", "audio"],
    "books": ["furniture", "apparel"],
    "apparel": ["fitness", "books"],
    "furniture": ["smart_home", "kitchen"],
}


def build_product_catalog(num_products, seed=42):
    rng = np.random.default_rng(seed)
    brands = ["ASUS", "Logitech", "Sony", "Samsung", "IKEA", "Nike", "Generic", "Anker"]
    products = []
    for pid in range(num_products):
        category = CATEGORIES[pid % len(CATEGORIES)]
        products.append({
            "product_id": pid,
            "category": category,
            "brand": rng.choice(brands),
            "price": round(float(rng.lognormal(mean=3.5, sigma=0.8)), 2),
            "rating": round(float(rng.uniform(3.0, 5.0)), 1),
        })
    return pd.DataFrame(products)


def sample_session(catalog_by_category, session_len, rng):
    """Sample a session that has realistic category coherence."""
    category = rng.choice(CATEGORIES)
    session = []
    for _ in range(session_len):
        # 70% chance: stay in current/related category, 30%: jump randomly
        if rng.random() < 0.7:
            candidates = [category] + RELATED[category]
            category = rng.choice(candidates)
        else:
            category = rng.choice(CATEGORIES)

        pool = catalog_by_category[category]
        # Power-law-ish popularity bias within category
        idx = min(int(rng.exponential(scale=len(pool) / 5)), len(pool) - 1)
        product_id = pool[idx]
        session.append(product_id)
    return session


def generate(num_users, num_products, sessions_per_user_mean=3, seed=42, out_path="data/events.csv"):
    rng = np.random.default_rng(seed)
    random.seed(seed)

    catalog = build_product_catalog(num_products, seed=seed)
    catalog_by_category = {
        cat: catalog[catalog.category == cat].product_id.tolist() for cat in CATEGORIES
    }

    events = []
    start_time = datetime(2024, 1, 1)
    event_types = ["view", "view", "view", "click", "add_to_cart", "purchase"]

    for user_id in range(num_users):
        num_sessions = max(1, int(rng.poisson(sessions_per_user_mean)))
        cursor = start_time + timedelta(days=int(rng.integers(0, 150)))
        for s in range(num_sessions):
            session_len = max(2, min(15, int(rng.poisson(6))))
            session_items = sample_session(catalog_by_category, session_len, rng)
            session_id = f"u{user_id}_s{s}"
            t = cursor
            for item in session_items:
                event_type = event_types[rng.integers(0, len(event_types))]
                events.append({
                    "user_id": f"user_{user_id}",
                    "session_id": session_id,
                    "product_id": item,
                    "event_type": event_type,
                    "timestamp": t.isoformat(),
                })
                t += timedelta(seconds=int(rng.integers(5, 120)))
            cursor = t + timedelta(hours=int(rng.integers(1, 72)))

    events_df = pd.DataFrame(events)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    events_df.to_csv(out_path, index=False)

    catalog_path = os.path.join(os.path.dirname(out_path) or ".", "product_catalog.csv")
    catalog.to_csv(catalog_path, index=False)

    print(f"Generated {len(events_df)} events across {events_df.session_id.nunique()} sessions "
          f"for {events_df.user_id.nunique()} users and {num_products} products.")
    print(f"Events written to: {out_path}")
    print(f"Product catalog written to: {catalog_path}")
    return events_df, catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_users", type=int, default=2000)
    parser.add_argument("--num_products", type=int, default=3000)
    parser.add_argument("--sessions_per_user_mean", type=float, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/events.csv")
    args = parser.parse_args()

    generate(
        num_users=args.num_users,
        num_products=args.num_products,
        sessions_per_user_mean=args.sessions_per_user_mean,
        seed=args.seed,
        out_path=args.out,
    )
