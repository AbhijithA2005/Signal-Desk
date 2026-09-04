import json
import statistics
import time
from pathlib import Path

from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import VectorStore


QUERY_PATH = Path("data/evaluation/research_corpus_chunk_queries.json")
CHUNK_PATH = Path("data/processed/research_corpus_chunks.jsonl")


def load_queries():
    return json.loads(
        QUERY_PATH.read_text(encoding="utf-8")
    )["queries"]


def summarize(times):
    return {
        "mean_ms": statistics.mean(times) * 1000,
        "median_ms": statistics.median(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
    }


def benchmark(name, search_fn, queries):
    times = []

    for item in queries:
        start = time.perf_counter()
        search_fn(item["query"])
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    summary = summarize(times)

    print(
        f"{name}: "
        f"mean={summary['mean_ms']:.2f} ms, "
        f"median={summary['median_ms']:.2f} ms, "
        f"min={summary['min_ms']:.2f} ms, "
        f"max={summary['max_ms']:.2f} ms"
    )


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

    # Warm-up each component so model/index initialization does not
    # dominate the measured query latency.
    warmup_query = queries[0]["query"]

    dense.search(warmup_query, top_k=5)
    bm25.search(warmup_query, top_k=5)
    hybrid.search(
        warmup_query,
        top_k=5,
        candidate_k=20,
    )

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

    benchmark(
        "Dense",
        lambda query: dense.search(
            query,
            top_k=5,
        ),
        queries,
    )

    benchmark(
        "BM25",
        lambda query: bm25.search(
            query,
            top_k=5,
        ),
        queries,
    )

    benchmark(
        "Hybrid/RRF",
        lambda query: hybrid.search(
            query,
            top_k=5,
            candidate_k=20,
        ),
        queries,
    )

    def hybrid_reranked(query):
        candidates = hybrid.search(
            query,
            top_k=20,
            candidate_k=20,
        )
        return reranker.rerank(
            query,
            candidates,
            top_k=5,
        )

    benchmark(
        "Hybrid+Reranker",
        hybrid_reranked,
        queries,
    )


if __name__ == "__main__":
    main()
