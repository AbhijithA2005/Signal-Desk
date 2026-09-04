import json
from pathlib import Path

from src.evaluation.retrieval_metrics import hit_at_k, reciprocal_rank
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import VectorStore


QUERY_PATH = Path("data/evaluation/research_corpus_chunk_queries.json")
CHUNK_PATH = Path("data/processed/research_corpus_chunks.jsonl")


def mean(values):
    return sum(values) / len(values) if values else 0.0


def load_queries():
    return json.loads(
        QUERY_PATH.read_text(encoding="utf-8")
    )["queries"]


def evaluate_depth(hybrid, reranker, queries, depth):
    rows = []

    for item in queries:
        query = item["query"]
        relevant = set(item["relevant_chunks"])

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

        ids = [r["chunk_id"] for r in reranked]

        rows.append(
            {
                "hit@1": hit_at_k(ids, relevant, 1),
                "hit@3": hit_at_k(ids, relevant, min(3, depth)),
                "hit@5": hit_at_k(ids, relevant, min(5, depth)),
                "mrr": reciprocal_rank(ids, relevant),
            }
        )

    return {
        "depth": depth,
        "hit@1": mean([r["hit@1"] for r in rows]),
        "hit@3": mean([r["hit@3"] for r in rows]),
        "hit@5": mean([r["hit@5"] for r in rows]),
        "mrr": mean([r["mrr"] for r in rows]),
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

    for depth in [5, 10, 20]:
        metrics = evaluate_depth(
            hybrid,
            reranker,
            queries,
            depth,
        )

        print(
            f"Candidate depth={metrics['depth']}: "
            f"Hit@1={metrics['hit@1']:.3f}, "
            f"Hit@3={metrics['hit@3']:.3f}, "
            f"Hit@5={metrics['hit@5']:.3f}, "
            f"MRR={metrics['mrr']:.3f}"
        )


if __name__ == "__main__":
    main()
