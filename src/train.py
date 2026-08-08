"""
Training pipeline for GRU4REC.

Usage:
    python -m src.train --data data/events.csv --epochs 5
"""
import argparse
import json
import os
import sys

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from src.dataset import PadCollate, load_and_split
from src.models import GRU4REC


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for input_seqs, targets in loader:
        input_seqs, targets = input_seqs.to(device), targets.to(device)

        scores = model(input_seqs)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for input_seqs, targets in loader:
        input_seqs, targets = input_seqs.to(device), targets.to(device)
        scores = model(input_seqs)
        loss = criterion(scores, targets)
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=CONFIG.events_path)
    parser.add_argument("--retail", action="store_true",
                        help="Use the RetailRocket canonical-schema events.csv "
                             "(data/retail_rocket_events.csv) and larger default dims.")
    parser.add_argument("--epochs", type=int, default=CONFIG.epochs)
    parser.add_argument("--batch_size", type=int, default=CONFIG.batch_size)
    parser.add_argument("--lr", type=float, default=CONFIG.lr)
    parser.add_argument("--embedding_dim", type=int, default=CONFIG.embedding_dim)
    parser.add_argument("--hidden_dim", type=int, default=CONFIG.hidden_dim)
    parser.add_argument("--checkpoint", type=str, default=CONFIG.checkpoint_path)
    args = parser.parse_args()

    if args.retail:
        args.data = os.path.join(os.path.dirname(CONFIG.events_path), "retail_rocket_events.csv")
        # Larger model for the ~235K-item RetailRocket catalog.
        if args.embedding_dim == CONFIG.embedding_dim:
            args.embedding_dim = 128
        if args.hidden_dim == CONFIG.hidden_dim:
            args.hidden_dim = 128
        print("Using RetailRocket dataset:", args.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds, test_ds, product_to_idx, idx_to_product = load_and_split(
        args.data, max_session_length=CONFIG.max_session_length,
        val_split=CONFIG.val_split, test_split=CONFIG.test_split, seed=CONFIG.seed,
    )
    num_products = len(product_to_idx) + 1  # +1 for padding index 0
    print(f"Vocab size (incl. padding): {num_products}")
    print(f"Train examples: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    collate = PadCollate(pad_value=CONFIG.pad_value)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = GRU4REC(
        num_products=num_products,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=CONFIG.num_layers,
        dropout=CONFIG.dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = CrossEntropyLoss()

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_products": num_products,
                "embedding_dim": args.embedding_dim,
                "hidden_dim": args.hidden_dim,
                "num_layers": CONFIG.num_layers,
                "dropout": CONFIG.dropout,
            }, args.checkpoint)

    # Persist vocab so inference/serving can map raw product_ids <-> indices
    with open(CONFIG.vocab_path, "w") as f:
        json.dump({str(k): v for k, v in product_to_idx.items()}, f)

    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {args.checkpoint}")
    print(f"Vocab saved to: {CONFIG.vocab_path}")


if __name__ == "__main__":
    main()
