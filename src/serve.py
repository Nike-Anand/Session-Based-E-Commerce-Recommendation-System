"""
Real-time recommendation REST API.

Usage:
    python -m src.serve --checkpoint models/gru4rec.pt --index models/products.index

Endpoints:
    POST /recommend
        body: {"user_id": "user_1", "session_items": [3, 17, 42], "k": 10}
        Note: session_items must already be vocab indices (see models/vocab.json
        for the raw product_id -> vocab index mapping produced during training).
    GET /health
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from src.inference import EmbeddingCache, load_model
from src.retail_rocket import build_popularity_ranking

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

app = Flask(__name__)

STATE = {"model": None, "index": None, "cache": None, "vocab": None, "device": "cpu",
         "popularity_rank": None}


def get_popular_products(k=10):
    """Fallback when the model/index path fails.

    Uses a real popularity ranking loaded from the events data (list of raw
    product_ids, most-popular first) when available; otherwise returns the
    first-k raw vocab ids.
    """
    rank = STATE.get("popularity_rank")
    if rank is not None:
        return rank[:k]
    return list(range(1, k + 1))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": STATE["model"] is not None,
        "index_loaded": STATE["index"] is not None,
    })


@app.route("/recommend", methods=["POST"])
def recommend():
    start_time = time.time()
    data = request.get_json(force=True)
    user_id = data.get("user_id", "anonymous")
    session_items = data.get("session_items", [])
    k = int(data.get("k", CONFIG.top_k))

    model, index, cache, vocab = STATE["model"], STATE["index"], STATE["cache"], STATE["vocab"]

    try:
        if model is None:
            raise RuntimeError("Model not loaded")
        if not session_items:
            raise ValueError("session_items is empty")

        # Map raw product_ids -> vocab indices (0 reserved for padding).
        # If the caller already sends vocab indices, they pass through unchanged.
        if vocab is not None:
            mapped = []
            for pid in session_items:
                pid = int(pid)
                idx = vocab.get(str(pid), vocab.get(pid, 0))
                if idx == 0:
                    # Unknown product -> skip (or map to pad). We drop it.
                    continue
                mapped.append(idx)
            if not mapped:
                raise ValueError("No known products in session_items")
            session_items = mapped
        else:
            session_items = [int(x) for x in session_items]

        embedding = cache.get(user_id)
        if embedding is None:
            input_tensor = torch.tensor([session_items], dtype=torch.long, device=STATE["device"])
            embedding = model.get_user_embedding(input_tensor).cpu().numpy().astype("float32")
            cache.set(user_id, embedding)

        if index is not None:
            _, indices = index.search(embedding.reshape(1, -1), k)
            top_vocab = indices[0].tolist()
            method = "faiss"
        else:
            with torch.no_grad():
                scores = model.fc(torch.tensor(embedding, device=STATE["device"]))
                top_vocab = torch.topk(scores, k, dim=1).indices[0].tolist()
            method = "dense"

        # Map vocab indices back to raw product_ids for the caller.
        if vocab is not None:
            idx_to_product = {int(v): kk for kk, v in vocab.items()}
            top_products = [idx_to_product.get(int(v), int(v)) for v in top_vocab]
        else:
            top_products = top_vocab

        elapsed_ms = (time.time() - start_time) * 1000
        return jsonify({
            "user_id": user_id,
            "recommendations": top_products,
            "method": method,
            "latency_ms": round(elapsed_ms, 2),
        })

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        return jsonify({
            "user_id": user_id,
            "recommendations": get_popular_products(k),
            "method": "fallback_popularity",
            "error": str(e),
            "latency_ms": round(elapsed_ms, 2),
        }), 200


def load_state(checkpoint_path, index_path, cache_ttl, cache_size, vocab_path=None,
               popularity_csv=None):
    device = torch.device("cpu")
    STATE["device"] = device
    STATE["model"] = load_model(checkpoint_path, device)
    STATE["cache"] = EmbeddingCache(ttl_seconds=cache_ttl, max_size=cache_size)

    if vocab_path and os.path.exists(vocab_path):
        with open(vocab_path, "r") as f:
            STATE["vocab"] = json.load(f)  # keys are str(raw_product_id) -> vocab index
        print(f"Vocab loaded from {vocab_path} ({len(STATE['vocab'])} products)")
    else:
        STATE["vocab"] = None
        print("Warning: vocab not found — raw product_id mapping disabled.")

    if popularity_csv and os.path.exists(popularity_csv):
        events = pd.read_csv(popularity_csv, usecols=["product_id"])
        STATE["popularity_rank"] = build_popularity_ranking(events)
        print(f"Popularity ranking built from {len(events)} events.")
    else:
        STATE["popularity_rank"] = None
        print("Warning: popularity data not provided — using placeholder fallback.")

    if FAISS_AVAILABLE and index_path and os.path.exists(index_path):
        STATE["index"] = faiss.read_index(index_path)
    else:
        STATE["index"] = None
        print("Warning: FAISS index not found/available — using dense fallback scoring.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=CONFIG.checkpoint_path)
    parser.add_argument("--index", type=str, default=CONFIG.faiss_index_path)
    parser.add_argument("--vocab", type=str, default=CONFIG.vocab_path)
    parser.add_argument("--popularity", type=str, default=None,
                        help="Path to a canonical-schema events.csv used to build the "
                             "real popularity fallback ranking (e.g. data/retail_rocket_events.csv)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    load_state(args.checkpoint, args.index, CONFIG.cache_ttl_seconds, CONFIG.cache_size,
               vocab_path=args.vocab, popularity_csv=args.popularity)
    app.run(host=args.host, port=args.port, debug=False)
