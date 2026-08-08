"""
Cold-start handling: blend RNN predictions with content-based similarity
for new users (little/no session history) and provide feature-based
bootstrapping for brand-new products.
"""
import numpy as np
import torch


class HybridRecommender:
    """
    Combines the trained GRU4REC model (for warm users, i.e. those with
    enough session history) with a content-based fallback (for cold-start
    users/products).
    """

    def __init__(self, model, product_features: dict, device="cpu",
                 rnn_weight=0.7, content_weight=0.3, warm_threshold=2,
                 popularity_rank=None):
        """
        Args:
            model: trained GRU4REC (or compatible) model.
            product_features: dict[product_idx] -> {category, brand, price, rating}
            warm_threshold: min session length to be treated as a "warm" user.
            popularity_rank: optional list[int] of vocab indices ordered
                most-popular-first, used as the empty-session fallback.
        """
        self.model = model
        self.product_features = product_features
        self.device = device
        self.rnn_weight = rnn_weight
        self.content_weight = content_weight
        self.warm_threshold = warm_threshold
        self.num_products = model.fc.out_features
        self.popularity_rank = popularity_rank

    def recommend(self, session_items, k=10):
        """session_items: list[int] of vocab indices in current session."""
        if len(session_items) < self.warm_threshold:
            return self.content_based_recommend(session_items, k)

        input_tensor = torch.tensor([session_items], dtype=torch.long, device=self.device)
        with torch.no_grad():
            rnn_scores = torch.softmax(self.model(input_tensor), dim=1)[0]  # [num_products]

        content_scores = self._content_scores_all(session_items[-1])
        content_scores = torch.tensor(content_scores, dtype=torch.float32, device=self.device)
        # Normalize content scores to a comparable range
        if content_scores.max() > 0:
            content_scores = content_scores / content_scores.max()

        blended = self.rnn_weight * rnn_scores + self.content_weight * content_scores
        top_k = torch.topk(blended, min(k, blended.numel())).indices
        return top_k.cpu().numpy()

    def content_based_recommend(self, session_items, k=10):
        if not session_items:
            return self.get_popular_products(k)
        last_item = session_items[-1]
        scores = self._content_scores_all(last_item)
        top_k = np.argsort(-scores)[:k]
        return top_k

    def _content_scores_all(self, reference_item):
        ref = self.product_features.get(reference_item)
        scores = np.zeros(self.num_products, dtype=np.float32)
        if ref is None:
            return scores
        for pid, feat in self.product_features.items():
            scores[pid] = self.feature_similarity(ref, feat)
        return scores

    @staticmethod
    def feature_similarity(feat1, feat2):
        similarity = 0.0
        if feat1.get("category") == feat2.get("category"):
            similarity += 0.5
        price1, price2 = feat1.get("price", 0), feat2.get("price", 0)
        price_diff = abs(price1 - price2)
        if price_diff < 50:
            similarity += 0.3
        elif price_diff < 200:
            similarity += 0.1
        if feat1.get("brand") == feat2.get("brand"):
            similarity += 0.2
        return similarity

    def get_popular_products(self, k=10):
        """Fallback for truly empty sessions.

        Uses a real popularity ranking (list of vocab indices, most-popular
        first) when provided; otherwise falls back to the first-k indices.
        """
        if self.popularity_rank is not None:
            return np.array(self.popularity_rank[:k], dtype=np.int64)
        return np.arange(min(k, self.num_products))


class ProductEmbeddingBootstrap:
    """Initialize embeddings for newly launched products via similarity averaging."""

    def __init__(self, embedding_layer):
        self.embedding_layer = embedding_layer  # nn.Embedding

    def initialize_new_product(self, new_product_idx, similar_product_indices):
        with torch.no_grad():
            similar_embeddings = self.embedding_layer.weight[similar_product_indices]
            new_embedding = similar_embeddings.mean(dim=0)
            self.embedding_layer.weight[new_product_idx] = new_embedding
        return new_embedding
