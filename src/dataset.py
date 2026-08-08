"""
Session dataset construction: turns raw event logs into
(input_sequence -> next_item) training examples, with padding for batching.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def build_vocab(events: pd.DataFrame):
    """Map raw product_id values to contiguous 1..N indices (0 reserved for padding)."""
    unique_products = sorted(events["product_id"].unique())
    product_to_idx = {pid: idx + 1 for idx, pid in enumerate(unique_products)}  # 0 = pad
    idx_to_product = {idx: pid for pid, idx in product_to_idx.items()}
    return product_to_idx, idx_to_product


def build_sessions(events: pd.DataFrame, product_to_idx: dict):
    """Group events into ordered sessions of vocab indices."""
    events_sorted = events.sort_values(["session_id", "timestamp"])
    sessions = (
        events_sorted.groupby("session_id")["product_id"]
        .apply(lambda s: [product_to_idx[p] for p in s.tolist()])
        .tolist()
    )
    # Drop sessions that are too short to form a training pair
    return [s for s in sessions if len(s) >= 2]


class SessionDataset(Dataset):
    """
    Builds (input_seq, target) pairs from sessions using the standard
    "next item prediction" framing: for a session [p1, p2, p3],
    examples are ([p1] -> p2) and ([p1, p2] -> p3).
    """

    def __init__(self, sessions, max_session_length=20):
        self.examples = []
        for session in sessions:
            session = session[-max_session_length:]
            for i in range(1, len(session)):
                input_seq = session[:i]
                target = session[i]
                self.examples.append((input_seq, target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_seq, target = self.examples[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class PadCollate:
    """Left-pads-free, right-pads sequences to the max length in a batch."""

    def __init__(self, pad_value=0):
        self.pad_value = pad_value

    def __call__(self, batch):
        inputs, targets = zip(*batch)
        max_len = max(len(seq) for seq in inputs)
        padded_inputs = torch.full((len(inputs), max_len), self.pad_value, dtype=torch.long)
        for i, seq in enumerate(inputs):
            padded_inputs[i, : len(seq)] = seq
        return padded_inputs, torch.stack(targets)


def load_and_split(events_path, max_session_length=20, val_split=0.1, test_split=0.1, seed=42):
    """
    Load raw events CSV, build vocab + sessions, and split sessions (not
    individual examples) into train/val/test to avoid leakage across splits.
    """
    events = pd.read_csv(events_path)
    product_to_idx, idx_to_product = build_vocab(events)
    sessions = build_sessions(events, product_to_idx)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(sessions))
    n_val = int(len(sessions) * val_split)
    n_test = int(len(sessions) * test_split)

    val_idx = set(indices[:n_val])
    test_idx = set(indices[n_val : n_val + n_test])

    train_sessions, val_sessions, test_sessions = [], [], []
    for i, s in enumerate(sessions):
        if i in val_idx:
            val_sessions.append(s)
        elif i in test_idx:
            test_sessions.append(s)
        else:
            train_sessions.append(s)

    train_ds = SessionDataset(train_sessions, max_session_length)
    val_ds = SessionDataset(val_sessions, max_session_length)
    test_ds = SessionDataset(test_sessions, max_session_length)

    return train_ds, val_ds, test_ds, product_to_idx, idx_to_product
