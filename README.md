<<<<<<< HEAD

# 🛒 Session-Based E-Commerce Recommendation System

<p align="center">
  <b>GRU4Rec • PyTorch • FAISS • FastAPI • RetailRocket</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch">
  <img src="https://img.shields.io/badge/FAISS-ANN%20Search-green">
  <img src="https://img.shields.io/badge/FastAPI-Production%20API-009688?logo=fastapi">
  <img src="https://img.shields.io/badge/Dataset-RetailRocket-orange">
  <img src="https://img.shields.io/badge/Status-Working-success">
</p>

<p align="center">
  <i>End-to-end session-based recommendation system trained on real e-commerce interaction data.</i>
</p>

---

## ⚡ At a Glance

| | Result |
|---|---:|
| 🧠 Model | GRU4Rec |
| 📦 Products | 235,062 |
| 📊 Sessions | 382,780 |
| 🧪 Test Examples | 91,289 |
| 🎯 Recall@10 | **31.02%** |
| 📈 MRR@10 | **21.33%** |
| 📐 NDCG@10 | **23.65%** |
| ⚡ p50 Latency | **1.27 ms** |
| 🚀 Retrieval | FAISS |
| 🌐 Serving | FastAPI |
| 🖥️ Training GPU | RTX 4060 |

---

## 🎯 What This Project Does

This project predicts the **next product a user is likely to interact with** based on the sequence of products viewed during their current session.

The system uses **GRU4Rec** to learn sequential behavior patterns from real e-commerce interactions and **FAISS** to efficiently retrieve recommendations from learned product embeddings.

```text
User Session
     │
     ▼
[Product A → Product B → Product C]
     │
     ▼
   GRU4Rec
     │
     ▼
Session Representation
     │
     ▼
Product Embeddings
     │
     ▼
    FAISS
     │
     ▼
Top-K Recommendations
````

---

# 🧠 Architecture

```text
                    RetailRocket
                         │
                         ▼
                Data Preprocessing
                         │
                         ▼
                  Session Builder
                         │
                         ▼
                 Sequence Dataset
                         │
                         ▼
                    GRU4Rec
                  ┌──────┴──────┐
                  │             │
             Embeddings      Prediction
                  │
                  ▼
             FAISS Index
                  │
                  ▼
             FastAPI API
                  │
                  ▼
          Top-K Recommendations
```

---

# 📊 Model Performance

### Benchmark

| Model       |  Recall@10 |     MRR@10 |    NDCG@10 |
| ----------- | ---------: | ---------: | ---------: |
| Random      |     0.0000 |     0.0000 |     0.0000 |
| Popularity  |     0.0089 |     0.0031 |     0.0045 |
| ItemKNN     |     0.0094 |     0.0032 |     0.0047 |
| **GRU4Rec** | **0.3102** | **0.2133** | **0.2365** |

GRU4Rec achieves approximately **34× the Recall@10 of the popularity baseline** on the evaluated held-out examples.

### Evaluation

* **382,780 sessions** analyzed
* **91,289 held-out test examples**
* Train / Validation / Test: **306K / 38K / 38K sessions**
* **0 session overlap** across train, validation and test

---

# 🔬 Key Finding: Popularity Bias

The model performs strongly overall, but deeper analysis revealed a significant **popularity bias**.

| Product Frequency |  Recall@10 |
| ----------------- | ---------: |
| 🔥 Popular        |  **0.412** |
| Mid-frequency     |  **0.092** |
| Rare              | **0.0016** |

The model performs substantially better on frequently interacted products while struggling with rare products.

This leads to the project's next research question:

> **Can recommendation quality be maintained while reducing popularity bias and improving long-tail recommendation performance?**

---

# 📈 Session-Length Analysis

| Session Length | Recall@10 |
| -------------- | --------: |
| 1              |     0.334 |
| 2              |     0.351 |
| 19             |     0.130 |

Short sessions perform better on this dataset, while performance decreases as session length increases.

The analysis indicates that the high short-session performance is partly influenced by repeated interactions with popular products rather than representing a general cold-start advantage.

---

# ⚡ Inference Performance

The complete recommendation pipeline is running through a FastAPI service.

### Uncached latency

```text
p50     1.27 ms
p95    31.90 ms
p99    45.10 ms
mean    5.88 ms
```

### Live Example

```text
Input Session

[100, 200, 300, 400, 500]

            ↓

       GRU4Rec + FAISS

            ↓

Recommendations

[1608, 405, 15, 720, 780]
```

---

# 🗂️ Dataset

The project uses the **RetailRocket Recommender System Dataset**.

Raw interaction events are transformed into sequential shopping sessions.

```text
Raw Events
    ↓
Cleaning
    ↓
Timestamp Ordering
    ↓
Session Identification
    ↓
Product Encoding
    ↓
Sequence Generation
    ↓
Train / Validation / Test
```

### Dataset Split

```text
Training       ~306K sessions
Validation      ~38K sessions
Testing         ~38K sessions
```

Data leakage verification found **zero session overlap** between the three splits.

---

# 🛠️ Technology Stack

### Machine Learning

`Python` · `PyTorch` · `GRU4Rec` · `NumPy` · `Pandas` · `Scikit-learn`

### Retrieval

`FAISS`

### Serving

`FastAPI` · `Uvicorn`

### Infrastructure

`Docker` · `Docker Compose`

### Experimentation

`TensorBoard` · `GPU Profiling` · `Custom Evaluation Pipeline`

---

# 📁 Repository Structure

```text
recommendation-system/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── gru4rec_retail.pt
│
├── experiments/
│   ├── analyze.py
│   ├── ablation.py
│   └── report.md
│
├── src/
│   ├── dataset/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── api/
│
├── tests/
├── configs/
├── scripts/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# 🚀 Quick Start

## 1. Clone

```bash
git clone <repository-url>
cd recommendation-system
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Prepare Dataset

Place the RetailRocket event data inside:

```text
data/raw/
```

## 4. Train

```bash
python scripts/train.py
```

## 5. Evaluate

```bash
python experiments/analyze.py
```

## 6. Start API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 5000
```

## 7. Request Recommendations
=======
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
>>>>>>> 15947c771cff675a28a40292660babaacb739448

```bash
curl -X POST http://localhost:5000/recommend \
  -H "Content-Type: application/json" \
<<<<<<< HEAD
  -d '{"session":[100,200,300,400,500]}'
```

---

# 📊 Current Status

| Component                  | Status |
| -------------------------- | ------ |
| RetailRocket Pipeline      | ✅      |
| Session Generation         | ✅      |
| GRU4Rec Training           | ✅      |
| GPU Training               | ✅      |
| Model Evaluation           | ✅      |
| Baseline Comparison        | ✅      |
| Leakage Verification       | ✅      |
| Popularity Analysis        | ✅      |
| FAISS Retrieval            | ✅      |
| FastAPI Serving            | ✅      |
| Latency Benchmarking       | ✅      |
| Live Inference             | ✅      |
| Ablation Study             | 🔄     |
| Popularity-Bias Mitigation | 🔜     |

---

# 🔬 Next Experiments

The next stage focuses on **scientific improvement rather than adding UI features**.

### Planned

* [ ] GRU4Rec ablation experiments
* [ ] AttentionRNN comparison
* [ ] ContextAwareRNN comparison
* [ ] Two-Tower comparison
* [ ] Popularity-bias mitigation
* [ ] Long-tail recommendation analysis
* [ ] Re-evaluation after mitigation
* [ ] Production optimization

---

# 🔍 Evaluation Philosophy

The project does not rely on a single accuracy number.

The recommendation system is evaluated through:

* Recall@K
* MRR@K
* NDCG@K
* Popularity baseline
* ItemKNN baseline
* Session-length stratification
* Popularity stratification
* Cold-start analysis
* Data-leakage verification
* Inference latency
* Recommendation inspection

This makes it possible to understand not only **how well the model performs**, but also **where and why it fails**.

---

# 🧪 End-to-End ML Lifecycle

```text
Problem Definition
        ↓
Data Collection
        ↓
Data Processing
        ↓
Session Construction
        ↓
Model Training
        ↓
Offline Evaluation
        ↓
Baseline Comparison
        ↓
Failure Analysis
        ↓
Model Improvement
        ↓
Vector Retrieval
        ↓
API Serving
        ↓
Latency & Production Evaluation
```

---

# 📚 References

* RetailRocket Recommender System Dataset
* GRU4Rec: Session-based Recommendations with Recurrent Neural Networks
* FAISS: Efficient Similarity Search and Clustering of Dense Vectors
* PyTorch Documentation
* FastAPI Documentation

---

## ⚠️ Disclaimer

This is an independent machine-learning project inspired by large-scale e-commerce recommendation problems.

It is **not affiliated with or officially developed by Amazon**.

---

<p align="center">
  <b>Model → Evaluation → Failure Analysis → Retrieval → Serving</b>
  <br>
  <sub>Built as an end-to-end ML research and engineering project.</sub>
</p>
```
=======
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
>>>>>>> 15947c771cff675a28a40292660babaacb739448
