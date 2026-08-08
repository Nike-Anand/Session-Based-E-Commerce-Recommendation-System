# E-Commerce Session-Based Recommendation System (GRU4REC)

A complete, runnable implementation of a session-based recommendation system
using GRU-based RNNs, matching the architecture described in
`rank1_recommendation_system_detailed.md`.

## Project layout

```
ecommerce-rec-system/
├── README.md
├── requirements.txt
├── config.py                  # Central config (hyperparams, paths)
├── data/
│   └── generate_synthetic_data.py   # Creates realistic synthetic session data
├── src/
│   ├── __init__.py
│   ├── dataset.py              # SessionDataset + padding collate fn
│   ├── models.py                # GRU4REC, AttentionRNN, TwoTowerModel, ContextAwareRNN
│   ├── train.py                 # Training loop + checkpointing
│   ├── evaluate.py              # Recall@K, MRR@K, NDCG@K, business metrics
│   ├── cold_start.py            # HybridRecommender (RNN + content-based fallback)
│   ├── inference.py             # FAISS ANN index, quantization, embedding cache
│   └── serve.py                 # Flask REST API for real-time recommendations
├── tests/
│   └── test_pipeline.py         # Sanity tests for dataset/model/eval
└── notebooks/
    └── eda.py                   # Exploratory data analysis script
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate synthetic session data (swap for real events.csv later)
python data/generate_synthetic_data.py --num_users 2000 --num_products 3000 --out data/events.csv

# 2. Train the model
python -m src.train --data data/events.csv --epochs 5 --embedding_dim 64 --hidden_dim 64

# 3. Evaluate
python -m src.evaluate --checkpoint models/gru4rec.pt --data data/events.csv

# 4. Build a FAISS index + serve recommendations
python -m src.inference --checkpoint models/gru4rec.pt --data data/events.csv --build_index
python -m src.serve --checkpoint models/gru4rec.pt --index models/products.index
```

Then query the API:

```bash
curl -X POST http://localhost:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_1", "session_items": [12, 45, 8], "k": 10}'
```

## What's implemented

- **GRU4REC** — the core session-based recommender (embeddings → GRU → scores)
- **AttentionRNN** — attention-weighted variant that upweights recent items
- **TwoTowerModel** — separate user/item towers for better generalization
- **ContextAwareRNN** — injects user features (age, tier, etc.) for partial cold-start help
- **HybridRecommender** — blends RNN scores with content-based similarity for true cold-start users/items
- **FAISS-backed inference** — sub-millisecond approximate nearest-neighbor product search
- **Dynamic quantization** — int8 inference for lower latency
- **LRU-style embedding cache** — avoids recomputing embeddings for returning users within a TTL
- **Flask serving layer** — `/recommend` endpoint with graceful fallback to popularity-based recs
- **Evaluation suite** — Recall@K, MRR@K, NDCG@K, plus a business-metrics helper

## Notes on scale

The defaults here (embedding_dim=64, small synthetic data) are tuned to train
in seconds/minutes on CPU so you can verify the whole pipeline end-to-end.
For a "real" run, bump `--embedding_dim 128 --hidden_dim 128`, use a real
session dataset (e.g. RetailRocket), and train on GPU.
