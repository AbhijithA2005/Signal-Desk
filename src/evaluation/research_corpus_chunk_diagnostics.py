import json
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


def find_rank(results, relevant):
    for rank, result in enumerate(results, start=1):
        if result["chunk_id"] in relevant:
            return rank
    return None


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

    for item in queries:
        query = item["query"]
        relevant = set(item["relevant_chunks"])

        dense_results = dense.search(query, top_k=5)
        bm25_results = bm25.search(query, top_k=5)
        hybrid_results = hybrid.search(
            query,
            top_k=5,
            candidate_k=10,
        )

        reranked_candidates = hybrid.search(
            query,
            top_k=10,
            candidate_k=20,
        )
        reranked_results = reranker.rerank(
            query,
            reranked_candidates,
            top_k=5,
        )

        print("=" * 100)
        print(f"{item['query_id']} | {query}")
        print(f"Gold: {', '.join(sorted(relevant))}")

        for name, results in [
            ("Dense", dense_results),
            ("BM25", bm25_results),
            ("Hybrid", hybrid_results),
            ("Reranked", reranked_results),
        ]:
            rank = find_rank(results, relevant)
            ids = [r["chunk_id"] for r in results]
            print(
                f"{name:<10} rank={rank if rank is not None else '-'} "
                f"| {ids}"
            )


if __name__ == "__main__":
    main()
