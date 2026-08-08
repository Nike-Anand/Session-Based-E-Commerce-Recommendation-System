"""
Production-style inference utilities:
  - FAISS approximate nearest-neighbor index over product embeddings
  - Dynamic int8 quantization for faster CPU inference
  - A simple TTL cache for user embeddings

Usage (build index from a trained checkpoint):
    python -m src.inference --checkpoint models/gru4rec.pt --data data/events.csv --build_index
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from src.dataset import load_and_split
from src.models import GRU4REC

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def load_model(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = GRU4REC(
        num_products=ckpt["num_products"],
        embedding_dim=ckpt["embedding_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def build_faiss_index(model, nlist=100):
    """
    Build an approximate nearest-neighbor index over the model's *output*
    projection weights (self.fc.weight), which act as learned product
    embeddings compatible with the GRU hidden state via dot product.
    """
    if not FAISS_AVAILABLE:
        raise ImportError("faiss is not installed. `pip install faiss-cpu`.")

    product_embeddings = model.fc.weight.detach().cpu().numpy().astype("float32")
    num_products, dim = product_embeddings.shape

    # Small vocabularies don't have enough points to train IVF clusters well;
    # fall back to an exact flat index in that case.
    if num_products < nlist * 40:
        index = faiss.IndexFlatIP(dim)
        index.add(product_embeddings)
    else:
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.train(product_embeddings)
        index.add(product_embeddings)

    return index, product_embeddings


def quantize_model(model):
    """Dynamic int8 quantization of Linear/Embedding layers for faster CPU inference."""
    return torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )


class EmbeddingCache:
    """TTL cache mapping user_id -> (embedding, timestamp), with simple LRU eviction."""

    def __init__(self, ttl_seconds=3600, max_size=10000):
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._store = {}

    def get(self, user_id):
        entry = self._store.get(user_id)
        if entry is None:
            return None
        embedding, ts = entry
        if time.time() - ts > self.ttl:
            del self._store[user_id]
            return None
        return embedding

    def set(self, user_id, embedding):
        if len(self._store) >= self.max_size and user_id not in self._store:
            oldest = min(self._store, key=lambda u: self._store[u][1])
            del self._store[oldest]
        self._store[user_id] = (embedding, time.time())


class FastRecommender:
    """End-to-end fast recommender: cache -> GRU -> FAISS."""

    def __init__(self, model, faiss_index, cache_ttl=3600, cache_size=10000, device="cpu"):
        self.model = model
        self.index = faiss_index
        self.cache = EmbeddingCache(cache_ttl, cache_size)
        self.device = device

    def recommend(self, user_id, session_items, k=10, use_cache=True):
        start = time.time()

        embedding = self.cache.get(user_id) if use_cache else None
        if embedding is None:
            input_tensor = torch.tensor([session_items], dtype=torch.long, device=self.device)
            embedding = self.model.get_user_embedding(input_tensor).cpu().numpy().astype("float32")
            if use_cache:
                self.cache.set(user_id, embedding)

        # The FAISS index holds fc.weight rows (one per product) in the same
        # hidden_dim space as the GRU's final hidden state, so we can search
        # directly with the user embedding as the query vector.
        distances, indices = self.index.search(embedding.reshape(1, -1).astype("float32"), k)
        elapsed_ms = (time.time() - start) * 1000
        return indices[0].tolist(), elapsed_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=CONFIG.checkpoint_path)
    parser.add_argument("--data", type=str, default=CONFIG.events_path)
    parser.add_argument("--retail", action="store_true",
                        help="Use the RetailRocket canonical-schema events.csv "
                             "(data/retail_rocket_events.csv).")
    parser.add_argument("--build_index", action="store_true")
    parser.add_argument("--nlist", type=int, default=CONFIG.faiss_nlist)
    args = parser.parse_args()

    if args.retail:
        args.data = os.path.join(os.path.dirname(CONFIG.events_path), "retail_rocket_events.csv")
        print("Using RetailRocket dataset:", args.data)

    device = torch.device("cpu")
    model = load_model(args.checkpoint, device)

    if args.build_index:
        if not FAISS_AVAILABLE:
            print("faiss not installed; skipping index build. `pip install faiss-cpu`")
            return
        index, embeddings = build_faiss_index(model, nlist=args.nlist)
        faiss.write_index(index, CONFIG.faiss_index_path)
        np.save(CONFIG.embeddings_path, embeddings)
        print(f"FAISS index written to {CONFIG.faiss_index_path}")
        print(f"Product embeddings written to {CONFIG.embeddings_path}")

        # Quick smoke test: recommend for a random session
        train_ds, val_ds, test_ds, product_to_idx, idx_to_product = load_and_split(
            args.data, max_session_length=CONFIG.max_session_length,
        )
        if len(test_ds) > 0:
            sample_seq, _ = test_ds[0]
            recommender = FastRecommender(model, index, device=device)
            recs, latency_ms = recommender.recommend("demo_user", sample_seq.tolist(), k=10)
            print(f"Sample recommendations (vocab idx): {recs}")
            print(f"Latency: {latency_ms:.2f} ms")


if __name__ == "__main__":
    main()
