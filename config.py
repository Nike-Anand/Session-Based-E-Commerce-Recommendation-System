"""
Central configuration for the recommendation system.
Import this instead of hardcoding paths/hyperparams throughout the project.
"""
import os
from dataclasses import dataclass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(MODELS_DIR, exist_ok=True)


@dataclass
class Config:
    # Data
    events_path: str = os.path.join(DATA_DIR, "events.csv")
    max_session_length: int = 20
    pad_value: int = 0

    # Model
    embedding_dim: int = 64
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.1

    # Training
    batch_size: int = 64
    epochs: int = 5
    lr: float = 1e-3
    device: str = "cpu"  # overridden to 'cuda' automatically if available
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42

    # Cold-start blending
    rnn_weight: float = 0.7
    content_weight: float = 0.3

    # Serving / inference
    cache_ttl_seconds: int = 3600
    cache_size: int = 10000
    faiss_nlist: int = 100
    top_k: int = 10

    # Paths for saved artifacts
    checkpoint_path: str = os.path.join(MODELS_DIR, "gru4rec.pt")
    faiss_index_path: str = os.path.join(MODELS_DIR, "products.index")
    embeddings_path: str = os.path.join(MODELS_DIR, "product_embeddings.npy")
    vocab_path: str = os.path.join(MODELS_DIR, "vocab.json")


CONFIG = Config()
