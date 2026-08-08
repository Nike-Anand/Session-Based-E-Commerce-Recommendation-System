"""
Ranking evaluation metrics for the recommender (not plain accuracy).

Usage:
    python -m src.evaluate --checkpoint models/gru4rec.pt --data data/events.csv
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from src.dataset import PadCollate, load_and_split
from src.models import GRU4REC


def dcg_at_k(relevance, k):
    relevance = np.asarray(relevance)[:k]
    if relevance.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevance.size + 2))
    return float(np.sum(relevance / discounts))


@torch.no_grad()
def compute_metrics(model, loader, device, k=10):
    """
    Compute Recall@K, MRR@K, NDCG@K over a held-out set of (session, next_item)
    examples. Each example has exactly one relevant item (the true next click),
    so NDCG@K here simplifies to 1/log2(rank+1) if the item is in the top-k, else 0.
    """
    model.eval()
    recall_hits, mrr_scores, ndcg_scores = [], [], []

    for input_seqs, targets in loader:
        input_seqs, targets = input_seqs.to(device), targets.to(device)
        scores = model(input_seqs)  # [B, num_products]
        _, top_k_indices = torch.topk(scores, k, dim=1)

        top_k_indices = top_k_indices.cpu().numpy()
        targets_np = targets.cpu().numpy()

        for i, target in enumerate(targets_np):
            top_k = top_k_indices[i]
            hits = np.where(top_k == target)[0]
            if hits.size > 0:
                rank = int(hits[0]) + 1  # 1-indexed
                recall_hits.append(1)
                mrr_scores.append(1.0 / rank)
                ndcg_scores.append(1.0 / np.log2(rank + 1))
            else:
                recall_hits.append(0)
                mrr_scores.append(0.0)
                ndcg_scores.append(0.0)

    return {
        f"Recall@{k}": float(np.mean(recall_hits)) if recall_hits else 0.0,
        f"MRR@{k}": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        f"NDCG@{k}": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "num_examples": len(recall_hits),
    }


def popularity_baseline_metrics(train_targets, test_targets, num_products, k=10):
    """A simple 'always recommend the top-k most popular items' baseline for comparison."""
    counts = np.bincount(train_targets, minlength=num_products)
    top_k = np.argsort(-counts)[:k]

    recall_hits, mrr_scores, ndcg_scores = [], [], []
    for target in test_targets:
        hits = np.where(top_k == target)[0]
        if hits.size > 0:
            rank = int(hits[0]) + 1
            recall_hits.append(1)
            mrr_scores.append(1.0 / rank)
            ndcg_scores.append(1.0 / np.log2(rank + 1))
        else:
            recall_hits.append(0)
            mrr_scores.append(0.0)
            ndcg_scores.append(0.0)

    return {
        f"Recall@{k}": float(np.mean(recall_hits)) if recall_hits else 0.0,
        f"MRR@{k}": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        f"NDCG@{k}": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=CONFIG.checkpoint_path)
    parser.add_argument("--data", type=str, default=CONFIG.events_path)
    parser.add_argument("--retail", action="store_true",
                        help="Use the RetailRocket canonical-schema events.csv "
                             "(data/retail_rocket_events.csv).")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=CONFIG.batch_size)
    args = parser.parse_args()

    if args.retail:
        args.data = os.path.join(os.path.dirname(CONFIG.events_path), "retail_rocket_events.csv")
        print("Using RetailRocket dataset:", args.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds, product_to_idx, idx_to_product = load_and_split(
        args.data, max_session_length=CONFIG.max_session_length,
        val_split=CONFIG.val_split, test_split=CONFIG.test_split, seed=CONFIG.seed,
    )
    num_products = len(product_to_idx) + 1

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = GRU4REC(
        num_products=ckpt["num_products"],
        embedding_dim=ckpt["embedding_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    collate = PadCollate(pad_value=CONFIG.pad_value)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    metrics = compute_metrics(model, test_loader, device, k=args.k)
    print("Model metrics:")
    for name, val in metrics.items():
        print(f"  {name}: {val:.4f}" if isinstance(val, float) else f"  {name}: {val}")

    # Popularity baseline for comparison
    train_targets = np.array([t.item() for _, t in train_ds])
    test_targets = np.array([t.item() for _, t in test_ds])
    if len(test_targets) > 0:
        baseline = popularity_baseline_metrics(train_targets, test_targets, num_products, k=args.k)
        print("\nPopularity baseline:")
        for name, val in baseline.items():
            print(f"  {name}: {val:.4f}")

        improvement = (
            (metrics[f"Recall@{args.k}"] - baseline[f"Recall@{args.k}"])
            / max(baseline[f"Recall@{args.k}"], 1e-9) * 100
        )
        print(f"\nGRU4REC improvement over popularity baseline: {improvement:.1f}%")


if __name__ == "__main__":
    main()
