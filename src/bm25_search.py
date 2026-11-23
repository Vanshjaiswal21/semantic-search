import re
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


class BM25Search:
    def __init__(self, corpus: pd.DataFrame):
        self.product_ids = corpus["product_id"].values
        self.bm25 = BM25Okapi([tokenize(t) for t in corpus["search_text"]])

    def search(self, query, top_k=10):
        scores = self.bm25.get_scores(tokenize(query))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.product_ids[i], float(scores[i])) for i in top_idx]