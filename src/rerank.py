import numpy as np
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, corpus, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=192):
        self.text_of = dict(zip(corpus["product_id"], corpus["search_text"]))
        self.model = CrossEncoder(model_name, max_length=max_length, device="cpu")

    def rerank(self, query, candidates, top_k=10):
        """candidates: list of product_ids. Returns [(product_id, score)]"""
        pairs = [(query, self.text_of[p]) for p in candidates]
        scores = self.model.predict(pairs, batch_size=32)
        order = np.argsort(scores)[::-1][:top_k]
        return [(candidates[i], float(scores[i])) for i in order]