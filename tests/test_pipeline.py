"""
Lightweight sanity tests (no pytest dependency required — just run this file).

Usage:
    python tests/test_pipeline.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import PadCollate, SessionDataset, build_sessions, build_vocab
from src.models import GRU4REC, AttentionRNN


def test_vocab_and_sessions():
    import pandas as pd
    events = pd.DataFrame({
        "session_id": ["s1", "s1", "s1", "s2", "s2"],
        "product_id": [10, 20, 30, 20, 10],
        "timestamp": ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00",
                      "2024-01-01T01:00:00", "2024-01-01T01:01:00"],
    })
    product_to_idx, idx_to_product = build_vocab(events)
    assert set(product_to_idx.keys()) == {10, 20, 30}
    assert 0 not in product_to_idx.values()  # 0 reserved for padding

    sessions = build_sessions(events, product_to_idx)
    assert len(sessions) == 2
    assert all(len(s) >= 2 for s in sessions)
    print("test_vocab_and_sessions: PASS")


def test_dataset_examples():
    sessions = [[1, 2, 3], [4, 5]]
    ds = SessionDataset(sessions, max_session_length=20)
    # session [1,2,3] -> 2 examples, session [4,5] -> 1 example
    assert len(ds) == 3
    seq, target = ds[0]
    assert seq.tolist() == [1]
    assert target.item() == 2
    print("test_dataset_examples: PASS")


def test_pad_collate():
    batch = [
        (torch.tensor([1, 2]), torch.tensor(3)),
        (torch.tensor([4]), torch.tensor(5)),
    ]
    collate = PadCollate(pad_value=0)
    inputs, targets = collate(batch)
    assert inputs.shape == (2, 2)
    assert inputs[1].tolist() == [4, 0]
    assert targets.tolist() == [3, 5]
    print("test_pad_collate: PASS")


def test_gru4rec_forward_shapes():
    model = GRU4REC(num_products=50, embedding_dim=8, hidden_dim=8)
    batch = torch.randint(1, 50, (4, 6))  # [batch=4, seq_len=6]
    scores = model(batch)
    assert scores.shape == (4, 50)
    print("test_gru4rec_forward_shapes: PASS")


def test_attention_rnn_forward_shapes():
    model = AttentionRNN(num_products=50, embedding_dim=8, hidden_dim=8)
    batch = torch.randint(1, 50, (4, 6))
    scores = model(batch)
    assert scores.shape == (4, 50)
    print("test_attention_rnn_forward_shapes: PASS")


def test_get_recommendations():
    model = GRU4REC(num_products=50, embedding_dim=8, hidden_dim=8)
    session = torch.tensor([[1, 2, 3]])
    top_indices, top_scores = model.get_recommendations(session, k=5)
    assert len(top_indices) == 5
    assert len(top_scores) == 5
    print("test_get_recommendations: PASS")


if __name__ == "__main__":
    test_vocab_and_sessions()
    test_dataset_examples()
    test_pad_collate()
    test_gru4rec_forward_shapes()
    test_attention_rnn_forward_shapes()
    test_get_recommendations()
    print("\nAll tests passed.")
