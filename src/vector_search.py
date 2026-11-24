import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self, corpus, embeddings,
                 model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"):
        self.product_ids = corpus["product_id"].values
        self.model = SentenceTransformer(model_name, device="cpu")
        self.model.max_seq_length = 128
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype("float32"))

    def search(self, query, top_k=10):
        qvec = self.model.encode(query, normalize_embeddings=True)
        qvec = np.asarray(qvec, dtype="float32").reshape(1, -1)
        scores, idx = self.index.search(qvec, top_k)
        return [(self.product_ids[i], float(s)) for i, s in zip(idx[0], scores[0])]