import numpy as np
import pandas as pd


def build_qrels_dict(qrels: pd.DataFrame) -> dict:
    """{query_id: {product_id: gain}}"""
    d = {}
    for qid, pid, gain in zip(qrels["query_id"], qrels["product_id"], qrels["gain"]):
        d.setdefault(qid, {})[pid] = gain
    return d


def build_run_dict(run: pd.DataFrame) -> dict:
    """{query_id: [product_ids ordered by rank]}"""
    run = run.sort_values(["query_id", "rank"])
    return run.groupby("query_id")["product_id"].apply(list).to_dict()


def dcg(gains):
    return sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(gains))


def evaluate_run(run: pd.DataFrame, qrels: pd.DataFrame, k=10, rel_threshold=2):
    qrels_d = build_qrels_dict(qrels)
    run_d = build_run_dict(run)

    recalls, mrrs, ndcgs = [], [], []

    for qid, judged in qrels_d.items():
        relevant = {p for p, g in judged.items() if g >= rel_threshold}
        if not relevant:
            continue  # skip queries with no relevant product

        retrieved = run_d.get(qid, [])[:k]

        # Recall@k
        hits = sum(1 for p in retrieved if p in relevant)
        recalls.append(hits / len(relevant))

        # MRR@k
        rr = 0.0
        for rank, p in enumerate(retrieved, start=1):
            if p in relevant:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

        # NDCG@k
        gains = [judged.get(p, 0) for p in retrieved]
        ideal = sorted(judged.values(), reverse=True)[:k]
        ideal_dcg = dcg(ideal)
        ndcgs.append(dcg(gains) / ideal_dcg if ideal_dcg > 0 else 0.0)

    return {
        f"Recall@{k}": round(float(np.mean(recalls)), 4),
        f"MRR@{k}": round(float(np.mean(mrrs)), 4),
        f"NDCG@{k}": round(float(np.mean(ndcgs)), 4),
        "queries_evaluated": len(ndcgs),
    }