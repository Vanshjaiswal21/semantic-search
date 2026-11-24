import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from src.bm25_search import BM25Search
from src.vector_search import SemanticSearch
from src.hybrid import HybridSearch
from src.rerank import Reranker

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# ---- Settings (use the best values from your Step 9 grid search) ----
HYBRID_K = 60
HYBRID_W_SEM = 1.0
RERANK_DEPTH = 20      # how many hybrid candidates the re-ranker scores

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts: load data and models."""
    print("Loading data and models... (about 1 minute)")
    t = time.time()

    corpus = pd.read_parquet(DATA / "corpus.parquet")
    emb = np.load(DATA / "corpus_embeddings.npy")

    bm25 = BM25Search(corpus)
    sem = SemanticSearch(corpus, emb)

    state["bm25"] = bm25
    state["sem"] = sem
    state["hybrid"] = HybridSearch(bm25, sem, k=HYBRID_K, w_sem=HYBRID_W_SEM)
    state["reranker"] = Reranker(corpus, max_length=128)
    state["products"] = corpus.set_index("product_id")[
        ["product_title", "product_brand", "product_color", "product_description"]
    ].to_dict("index")
    state["n_products"] = len(corpus)

    print(f"Ready in {time.time() - t:.0f} seconds. {len(corpus)} products loaded.")
    yield
    state.clear()


app = FastAPI(title="Semantic Product Search", version="1.0", lifespan=lifespan)

Method = Literal["bm25", "semantic", "hybrid", "hybrid_rerank"]


@lru_cache(maxsize=512)
def run_search(q: str, method: str, top_k: int):
    """Cached: the same query + method + top_k is answered instantly the second time."""
    if method == "bm25":
        hits = state["bm25"].search(q, top_k)
    elif method == "semantic":
        hits = state["sem"].search(q, top_k)
    elif method == "hybrid":
        hits = state["hybrid"].search(q, top_k)
    else:  # hybrid_rerank
        candidates = [p for p, _ in state["hybrid"].search(q, max(RERANK_DEPTH, top_k))]
        hits = state["reranker"].rerank(q, candidates, top_k)
    return tuple((str(p), float(s)) for p, s in hits)


@app.get("/health")
def health():
    return {"status": "ok", "products": state.get("n_products", 0)}


@app.get("/search")
def search(
    q: str = Query(..., description="Search text"),
    method: Method = "hybrid",
    top_k: int = Query(10, ge=1, le=50),
):
    q = q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    start = time.perf_counter()
    hits = run_search(q, method, top_k)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    results = []
    for rank, (pid, score) in enumerate(hits, start=1):
        info = state["products"].get(pid, {})
        results.append({
            "rank": rank,
            "product_id": pid,
            "score": round(score, 4),
            "title": info.get("product_title", ""),
            "brand": info.get("product_brand", ""),
            "color": info.get("product_color", ""),
            "description": (info.get("product_description", "") or "")[:200],
        })

    return {"query": q, "method": method, "latency_ms": latency_ms, "results": results}