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
import torch
from flask import Flask, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from src.inference import EmbeddingCache, load_model

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

app = Flask(__name__)

STATE = {"model": None, "index": None, "cache": None, "vocab": None, "device": "cpu"}


def get_popular_products(k=10):
    """Fallback when the model/index path fails: return first-k vocab items.
    In production, replace with a precomputed real popularity ranking."""
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

    model, index, cache = STATE["model"], STATE["index"], STATE["cache"]

    try:
        if model is None:
            raise RuntimeError("Model not loaded")
        if not session_items:
            raise ValueError("session_items is empty")

        embedding = cache.get(user_id)
        if embedding is None:
            input_tensor = torch.tensor([session_items], dtype=torch.long, device=STATE["device"])
            embedding = model.get_user_embedding(input_tensor).cpu().numpy().astype("float32")
            cache.set(user_id, embedding)

        if index is not None:
            _, indices = index.search(embedding.reshape(1, -1), k)
            top_products = indices[0].tolist()
            method = "faiss"
        else:
            with torch.no_grad():
                scores = model.fc(torch.tensor(embedding, device=STATE["device"]))
                top_products = torch.topk(scores, k, dim=1).indices[0].tolist()
            method = "dense"

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


def load_state(checkpoint_path, index_path, cache_ttl, cache_size):
    device = torch.device("cpu")
    STATE["device"] = device
    STATE["model"] = load_model(checkpoint_path, device)
    STATE["cache"] = EmbeddingCache(ttl_seconds=cache_ttl, max_size=cache_size)

    if FAISS_AVAILABLE and index_path and os.path.exists(index_path):
        STATE["index"] = faiss.read_index(index_path)
    else:
        STATE["index"] = None
        print("Warning: FAISS index not found/available — using dense fallback scoring.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=CONFIG.checkpoint_path)
    parser.add_argument("--index", type=str, default=CONFIG.faiss_index_path)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    load_state(args.checkpoint, args.index, CONFIG.cache_ttl_seconds, CONFIG.cache_size)
    app.run(host=args.host, port=args.port, debug=False)
