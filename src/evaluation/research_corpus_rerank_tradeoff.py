import json
import statistics
import time
from pathlib import Path

from src.evaluation.retrieval_metrics import (
    hit_at_k,
    reciprocal_rank,
)
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import VectorStore


QUERY_PATH = Path(
    "data/evaluation/research_corpus_chunk_queries.json"
)
CHUNK_PATH = Path(
    "data/processed/research_corpus_chunks.jsonl"
)


def load_queries():
    return json.loads(
        QUERY_PATH.read_text(encoding="utf-8")
    )["queries"]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def percentile(values, p):
    values = sorted(values)

    if not values:
        return 0.0

    index = (len(values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    fraction = index - lower

    return (
        values[lower]
        + (values[upper] - values[lower]) * fraction
    )


def evaluate_depth(hybrid, reranker, queries, depth):
    metric_rows = []
    latency_ms = []

    for item in queries:
        query = item["query"]
        relevant = set(item["relevant_chunks"])

        start = time.perf_counter()

        candidates = hybrid.search(
            query,
            top_k=depth,
            candidate_k=depth,
        )

        reranked = reranker.rerank(
            query,
            candidates,
            top_k=min(5, depth),
        )

        elapsed = time.perf_counter() - start
        latency_ms.append(elapsed * 1000)

        ids = [r["chunk_id"] for r in reranked]

        metric_rows.append(
            {
                "hit@1": hit_at_k(ids, relevant, 1),
                "hit@3": hit_at_k(
                    ids,
                    relevant,
                    min(3, depth),
                ),
                "hit@5": hit_at_k(
                    ids,
                    relevant,
                    min(5, depth),
                ),
                "mrr": reciprocal_rank(
                    ids,
                    relevant,
                ),
            }
        )

    return {
        "depth": depth,
        "hit@1": mean(
            [r["hit@1"] for r in metric_rows]
        ),
        "hit@3": mean(
            [r["hit@3"] for r in metric_rows]
        ),
        "hit@5": mean(
            [r["hit@5"] for r in metric_rows]
        ),
        "mrr": mean(
            [r["mrr"] for r in metric_rows]
        ),
        "mean_ms": mean(latency_ms),
        "median_ms": statistics.median(latency_ms),
        "p95_ms": percentile(latency_ms, 0.95),
    }


def main():
    queries = load_queries()

    dense = VectorStore(
        persist_directory="indexes/chroma",
        collection_name="research_chunks",
    )

    bm25 = BM25Store(CHUNK_PATH)

    hybrid = HybridRetriever(
        dense_store=dense,
        bm25_store=bm25,
    )

    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="mps",
    )

    print("Dataset: research_corpus_chunk_dev_v1")
    print(f"Queries: {len(queries)}")
    print()

    warmup_query = queries[0]["query"]

    warmup_candidates = hybrid.search(
        warmup_query,
        top_k=20,
        candidate_k=20,
    )

    reranker.rerank(
        warmup_query,
        warmup_candidates,
        top_k=5,
    )

    print("Warm-up complete")
    print()

    for depth in [5, 10, 20]:
        result = evaluate_depth(
            hybrid,
            reranker,
            queries,
            depth,
        )

        print(
            f"Depth={result['depth']}: "
            f"Hit@1={result['hit@1']:.3f}, "
            f"Hit@3={result['hit@3']:.3f}, "
            f"Hit@5={result['hit@5']:.3f}, "
            f"MRR={result['mrr']:.3f}, "
            f"mean={result['mean_ms']:.2f} ms, "
            f"median={result['median_ms']:.2f} ms, "
            f"p95={result['p95_ms']:.2f} ms"
        )


if __name__ == "__main__":
    main()
