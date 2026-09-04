import json
from pathlib import Path

from src.evaluation.retrieval_metrics import hit_at_k, reciprocal_rank
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import VectorStore


QUERY_PATH = Path("data/evaluation/research_corpus_chunk_queries.json")
CHUNK_PATH = Path("data/processed/research_corpus_chunks.jsonl")


def load_queries():
    data = json.loads(QUERY_PATH.read_text(encoding="utf-8"))
    return data["queries"]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def evaluate_method(name, search_fn, queries):
    results = []

    for item in queries:
        retrieved = search_fn(item["query"])
        retrieved_ids = [result["chunk_id"] for result in retrieved]
        relevant = set(item["relevant_chunks"])

        results.append(
            {
                "query_id": item["query_id"],
                "hit@1": hit_at_k(retrieved_ids, relevant, 1),
                "hit@3": hit_at_k(retrieved_ids, relevant, 3),
                "hit@5": hit_at_k(retrieved_ids, relevant, 5),
                "mrr": reciprocal_rank(retrieved_ids, relevant),
            }
        )

    return {
        "method": name,
        "hit@1": mean([r["hit@1"] for r in results]),
        "hit@3": mean([r["hit@3"] for r in results]),
        "hit@5": mean([r["hit@5"] for r in results]),
        "mrr": mean([r["mrr"] for r in results]),
    }


def main():
    queries = load_queries()

    print(f"Dataset: research_corpus_chunk_dev_v1")
    print(f"Queries: {len(queries)}")
    print()

    dense_store = VectorStore(
        persist_directory="indexes/chroma",
        collection_name="research_chunks",
    )

    bm25_store = BM25Store(CHUNK_PATH)

    hybrid = HybridRetriever(
        dense_store=dense_store,
        bm25_store=bm25_store,
    )

    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="mps",
    )

    dense_metrics = evaluate_method(
        "Dense",
        lambda query: dense_store.search(query, top_k=5),
        queries,
    )

    bm25_metrics = evaluate_method(
        "BM25",
        lambda query: bm25_store.search(query, top_k=5),
        queries,
    )

    hybrid_metrics = evaluate_method(
        "Hybrid/RRF",
        lambda query: hybrid.search(
            query,
            top_k=5,
            candidate_k=10,
        ),
        queries,
    )

    def hybrid_reranked(query):
        candidates = hybrid.search(
            query,
            top_k=10,
            candidate_k=20,
        )
        return reranker.rerank(
            query,
            candidates,
            top_k=5,
        )

    reranked_metrics = evaluate_method(
        "Hybrid+Reranker",
        hybrid_reranked,
        queries,
    )

    for metrics in [
        dense_metrics,
        bm25_metrics,
        hybrid_metrics,
        reranked_metrics,
    ]:
        print(
            f"{metrics['method']}: "
            f"Hit@1={metrics['hit@1']:.3f}, "
            f"Hit@3={metrics['hit@3']:.3f}, "
            f"Hit@5={metrics['hit@5']:.3f}, "
            f"MRR={metrics['mrr']:.3f}"
        )


if __name__ == "__main__":
    main()
