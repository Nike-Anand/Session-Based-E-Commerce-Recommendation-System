"""
Ablation experiments: compare model architectures on a subsampled dataset.

Trains each variant (GRU4REC, AttentionRNN, ContextAwareRNN, TwoTower) from
scratch on a percentage of the RetailRocket data (or synthetic) and evaluates
on a shared held-out split so the comparison is fair.

Usage:
    python experiments/ablation.py --retail --subsample 0.2 --epochs 2 --embedding_dim 64 --hidden_dim 64

Output:
    experiments/ablation_report.md
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from src.dataset import PadCollate, SessionDataset, build_sessions, build_vocab
from src.models import GRU4REC, AttentionRNN, ContextAwareRNN, TwoTowerModel


def subsample_events(events_df, frac, seed=0):
    """Keep a fraction of sessions (by session_id) to make training tractable."""
    rng = np.random.default_rng(seed)
    sessions = events_df["session_id"].unique()
    keep = set(rng.choice(sessions, size=int(len(sessions) * frac), replace=False))
    return events_df[events_df["session_id"].isin(keep)]


@torch.no_grad()
def evaluate_split(model, sessions, collate, device, k=10, batch_size=256):
    model.eval()
    ds = SessionDataset(sessions, max_session_length=CONFIG.max_session_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    hits, mrr, ndcg, n = 0, 0.0, 0.0, 0
    for input_seqs, targets in loader:
        input_seqs, targets = input_seqs.to(device), targets.to(device)
        scores = model(input_seqs)
        _, top_k = torch.topk(scores, k, dim=1)
        top_k = top_k.cpu().numpy()
        targets_np = targets.cpu().numpy()
        for i, t in enumerate(targets_np):
            w = np.where(top_k[i] == t)[0]
            n += 1
            if w.size:
                rank = int(w[0]) + 1
                hits += 1
                mrr += 1.0 / rank
                ndcg += 1.0 / np.log2(rank + 1)
    return {
        "Recall@10": hits / max(n, 1),
        "MRR@10": mrr / max(n, 1),
        "NDCG@10": ndcg / max(n, 1),
        "num_examples": n,
    }


def train_model(model, train_loader, val_sessions, collate, device, epochs,
                lr=1e-3, batch_size=256):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()
    best_val = -1.0
    best_metrics = None
    for epoch in range(epochs):
        model.train()
        tot = 0.0
        for input_seqs, targets in train_loader:
            input_seqs, targets = input_seqs.to(device), targets.to(device)
            scores = model(input_seqs)
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot += loss.item()
        val = evaluate_split(model, val_sessions, collate, device)
        print(f"  epoch {epoch+1}/{epochs} loss={tot/len(train_loader):.4f} "
              f"val_Recall@10={val['Recall@10']:.4f}")
        if val["Recall@10"] > best_val:
            best_val = val["Recall@10"]
            best_metrics = val
    return best_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retail", action="store_true")
    ap.add_argument("--subsample", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--embedding_dim", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ablation_report.md"))
    args = ap.parse_args()

    data_path = (os.path.join(os.path.dirname(CONFIG.events_path), "retail_rocket_events.csv")
                 if args.retail else CONFIG.events_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | data: {data_path} | subsample={args.subsample}")

    events = pd.read_csv(data_path)
    rng = np.random.default_rng(CONFIG.seed)
    sessions_all = events["session_id"].unique()
    perm = rng.permutation(len(sessions_all))
    n_val = int(len(sessions_all) * 0.1)
    n_test = int(len(sessions_all) * 0.1)
    val_sess = set(perm[:n_val].tolist())
    test_sess = set(perm[n_val:n_val + n_test].tolist())
    train_sess = set(perm[n_val + n_test:].tolist())

    # Subsample train sessions for tractability
    train_sess = set(rng.choice(list(train_sess),
                                size=int(len(train_sess) * args.subsample), replace=False))

    def to_ds(sess_set):
        df = events[events["session_id"].isin(sess_set)]
        p2i, _ = build_vocab(df)
        return build_sessions(df, p2i)

    train_sessions = to_ds(train_sess)
    val_sessions = to_ds(val_sess)
    test_sessions = to_ds(test_sess)

    # Build vocab over ALL events (so all splits share the same index space)
    ps_all, _ = build_vocab(events)
    num_products = len(ps_all) + 1
    print(f"num_products (shared vocab): {num_products} | "
          f"train_sessions={len(train_sessions)} val={len(val_sessions)} "
          f"test={len(test_sessions)}")

    collate = PadCollate(pad_value=CONFIG.pad_value)
    train_ds = SessionDataset(train_sessions, CONFIG.max_session_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate)

    report = ["# Ablation Report", "",
              f"- Dataset: {'RetailRocket' if args.retail else 'Synthetic'}",
              f"- Subsample (train): {args.subsample}",
              f"- Epochs: {args.epochs} | embedding_dim={args.embedding_dim} "
              f"hidden_dim={args.hidden_dim}",
              f"- Device: {device}", "",
              "| Model | Recall@10 | MRR@10 | NDCG@10 | N |",
              "|---|---|---|---|---|"]

    def build_and_run(name, model):
        print(f"\nTraining {name}...")
        best = train_model(model, train_loader, val_sessions, collate, device,
                           args.epochs, lr=args.lr, batch_size=args.batch_size)
        test = evaluate_split(model, test_sessions, collate, device)
        report.append(f"| {name} | {test['Recall@10']:.4f} | {test['MRR@10']:.4f} | "
                      f"{test['NDCG@10']:.4f} | {test['num_examples']} |")
        print(f"  {name} TEST Recall@10={test['Recall@10']:.4f}")
        return test

    # GRU4REC
    build_and_run("GRU4REC", GRU4REC(
        num_products=num_products, embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim, num_layers=CONFIG.num_layers,
        dropout=CONFIG.dropout).to(device))

    # AttentionRNN
    build_and_run("AttentionRNN", AttentionRNN(
        num_products=num_products, embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim).to(device))

# TwoTowerModel wrapped to accept a single seq input (dummy features).
    # Its forward(session_items, all_item_ids, item_features) needs all item ids;
    # we wrap it to compute scores over the full vocab with zero item features.
    class TwoTowerWrapper(torch.nn.Module):
        def __init__(self, num_products, user_dim, item_dim, item_feature_dim):
            super().__init__()
            self.inner = TwoTowerModel(
                num_products=num_products, user_dim=user_dim, item_dim=item_dim,
                item_feature_dim=item_feature_dim)
            self._all_ids = torch.arange(1, num_products, dtype=torch.long)

        def forward(self, input_seqs):
            all_ids = self._all_ids.to(input_seqs.device)
            return self.inner(input_seqs, all_ids)

    build_and_run("TwoTowerModel", TwoTowerWrapper(
        num_products=num_products, user_dim=args.hidden_dim,
        item_dim=args.embedding_dim, item_feature_dim=4).to(device))

    # ContextAwareRNN wrapped to accept a single seq input (dummy user features).
    class ContextWrapper(torch.nn.Module):
        def __init__(self, num_products, embedding_dim, hidden_dim, user_feature_dim):
            super().__init__()
            self.inner = ContextAwareRNN(
                num_products=num_products, embedding_dim=embedding_dim,
                hidden_dim=hidden_dim, user_feature_dim=user_feature_dim)

        def forward(self, input_seqs):
            zeros = torch.zeros(input_seqs.size(0), 4, device=input_seqs.device)
            return self.inner(input_seqs, zeros)

    build_and_run("ContextAwareRNN", ContextWrapper(
        num_products=num_products, embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim, user_feature_dim=4).to(device))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\nAblation report written to {args.out}")


if __name__ == "__main__":
    main()
