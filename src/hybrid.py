from collections import defaultdict


class HybridSearch:
    def __init__(self, bm25_search, semantic_search, k=60, w_bm=1.0, w_sem=1.0, depth=100):
        self.bm25 = bm25_search
        self.sem = semantic_search
        self.k, self.w_bm, self.w_sem, self.depth = k, w_bm, w_sem, depth

    def search(self, query, top_k=10):
        scores = defaultdict(float)
        for rank, (pid, _) in enumerate(self.bm25.search(query, self.depth), start=1):
            scores[pid] += self.w_bm / (self.k + rank)
        for rank, (pid, _) in enumerate(self.sem.search(query, self.depth), start=1):
            scores[pid] += self.w_sem / (self.k + rank)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked