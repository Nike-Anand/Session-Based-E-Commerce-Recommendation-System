"""
Model architectures for session-based recommendation.

- GRU4REC: the baseline session RNN (recommended starting point).
- AttentionRNN: adds attention over session steps, upweighting recent items.
- TwoTowerModel: separate user/item towers, for better generalization at scale.
- ContextAwareRNN: injects user features to partially address cold-start.
"""
import torch
import torch.nn as nn


class GRU4REC(nn.Module):
    """
    Session-based RNN for product recommendations.

    Args:
        num_products: Vocab size (including the padding index 0).
        embedding_dim: Dimension of product embeddings.
        hidden_dim: Dimension of GRU hidden state.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout between GRU layers (only active if num_layers > 1).
    """

    def __init__(self, num_products, embedding_dim=128, hidden_dim=128,
                 num_layers=1, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(num_products, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, num_products)

    def forward(self, product_ids):
        """
        Args:
            product_ids: [batch, seq_len]
        Returns:
            scores: [batch, num_products] — prediction for the *next* item
                     after the given sequence (uses final hidden state).
        """
        embeddings = self.embedding(product_ids)          # [B, T, E]
        _, hidden_state = self.gru(embeddings)             # hidden_state: [num_layers, B, H]
        final_hidden = hidden_state[-1]                    # [B, H]
        scores = self.fc(final_hidden)                     # [B, num_products]
        return scores

    def get_user_embedding(self, product_ids):
        """Return the GRU's final hidden state as a 'user intent' vector."""
        with torch.no_grad():
            embeddings = self.embedding(product_ids)
            _, hidden_state = self.gru(embeddings)
            return hidden_state[-1]

    def get_recommendations(self, product_ids, k=10):
        """product_ids: [1, seq_len] -> top-k product vocab indices."""
        with torch.no_grad():
            scores = self.forward(product_ids)
            top_scores, top_indices = torch.topk(scores, k, dim=1)
            return top_indices[0].cpu().numpy(), top_scores[0].cpu().numpy()


class AttentionRNN(nn.Module):
    """RNN with attention over session items (recent items weighted more)."""

    def __init__(self, num_products, embedding_dim=128, hidden_dim=128, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(num_products, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, num_products)

    def forward(self, product_ids):
        embeddings = self.embedding(product_ids)           # [B, T, E]
        gru_output, _ = self.gru(embeddings)                # [B, T, H]

        attention_scores = self.attention_w(gru_output)     # [B, T, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)

        context = torch.sum(gru_output * attention_weights, dim=1)  # [B, H]
        scores = self.fc(context)
        return scores

    def get_user_embedding(self, product_ids):
        with torch.no_grad():
            embeddings = self.embedding(product_ids)
            gru_output, _ = self.gru(embeddings)
            attention_scores = self.attention_w(gru_output)
            attention_weights = torch.softmax(attention_scores, dim=1)
            return torch.sum(gru_output * attention_weights, dim=1)


class TwoTowerModel(nn.Module):
    """Separate user tower (GRU over session) and item tower (embedding + features)."""

    def __init__(self, num_products, user_dim=128, item_dim=128, item_feature_dim=10, pad_idx=0):
        super().__init__()
        self.item_embedding_user_side = nn.Embedding(num_products, 64, padding_idx=pad_idx)
        self.user_gru = nn.GRU(64, user_dim, batch_first=True)

        self.item_embedding = nn.Embedding(num_products, item_dim, padding_idx=pad_idx)
        self.item_feature_processor = nn.Linear(item_feature_dim, item_dim)

    def user_tower(self, session_items):
        embeddings = self.item_embedding_user_side(session_items)  # [B, T, 64]
        _, hidden = self.user_gru(embeddings)                       # [1, B, user_dim]
        return hidden.squeeze(0)                                    # [B, user_dim]

    def item_tower(self, item_ids, item_features=None):
        embeddings = self.item_embedding(item_ids)  # [N, item_dim]
        if item_features is not None:
            embeddings = embeddings + self.item_feature_processor(item_features)
        return embeddings

    def forward(self, session_items, all_item_ids, item_features=None):
        user_emb = self.user_tower(session_items)                   # [B, user_dim]
        item_embs = self.item_tower(all_item_ids, item_features)    # [N, item_dim]
        scores = torch.matmul(user_emb, item_embs.T)                # [B, N]
        return scores


class ContextAwareRNN(nn.Module):
    """GRU4REC + user-level features, to partially mitigate new-user cold-start."""

    def __init__(self, num_products, embedding_dim=128, hidden_dim=128,
                 user_feature_dim=10, pad_idx=0):
        super().__init__()
        self.item_embedding = nn.Embedding(num_products, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        self.user_feature_processor = nn.Sequential(
            nn.Linear(user_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.fc = nn.Linear(hidden_dim, num_products)

    def forward(self, product_ids, user_features):
        embeddings = self.item_embedding(product_ids)
        _, hidden = self.gru(embeddings)               # [1, B, H]
        hidden = hidden.squeeze(0)                      # [B, H]

        user_embedding = self.user_feature_processor(user_features)  # [B, H]
        combined = torch.cat([hidden, user_embedding], dim=1)         # [B, 2H]
        fused = self.fusion(combined)                                 # [B, H]
        return self.fc(fused)


MODEL_REGISTRY = {
    "gru4rec": GRU4REC,
    "attention": AttentionRNN,
    "two_tower": TwoTowerModel,
    "context_aware": ContextAwareRNN,
}
