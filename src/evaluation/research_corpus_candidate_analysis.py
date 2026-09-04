import json
from pathlib import Path

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.bm25_store import BM25Store
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import VectorStore


QUERY_PATH = Path("data/evaluation/research_corpus_chunk_queries.json")
CHUNK_PATH = Path("data/processed/research_corpus_chunks.jsonl")


def load_queries():
    return json.loads(
        QUERY_PATH.read_text(encoding="utf-8")
    )["queries"]


def hit(results, relevant):
    return any(
        result["chunk_id"] in relevant
        for result in results
    )


def rank_of(results, relevant):
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

    candidate_hits = {5: [], 10: [], 20: []}
    reranked_hits = []

    print(f"Dataset: research_corpus_chunk_dev_v1")
    print(f"Queries: {len(queries)}")
    print()

    for item in queries:
        query = item["query"]
        relevant = set(item["relevant_chunks"])

        candidates = hybrid.search(
            query,
            top_k=20,
            candidate_k=20,
        )

        reranked = reranker.rerank(
            query,
            candidates,
            top_k=5,
        )

        ranks = {
            k: rank_of(candidates[:k], relevant)
            for k in [5, 10, 20]
        }

        reranked_rank = rank_of(reranked, relevant)

        print("=" * 100)
        print(f"{item['query_id']} | {query}")
        print(f"Gold: {', '.join(sorted(relevant))}")
        print(
            "Hybrid candidate rank: "
            f"@5={ranks[5] if ranks[5] is not None else '-'} "
            f"@10={ranks[10] if ranks[10] is not None else '-'} "
            f"@20={ranks[20] if ranks[20] is not None else '-'}"
        )
        print(
            f"Reranked top-5 rank="
            f"{reranked_rank if reranked_rank is not None else '-'}"
        )

        for k in candidate_hits:
            candidate_hits[k].append(
                hit(candidates[:k], relevant)
            )

        reranked_hits.append(
            hit(reranked, relevant)
        )

    print()
    print("=" * 100)
    print("CANDIDATE RECALL SUMMARY")
    print("=" * 100)

    for k in [5, 10, 20]:
        recall = sum(candidate_hits[k]) / len(candidate_hits[k])
        print(f"Hybrid candidate Recall@{k}: {recall:.3f}")

    reranked_recall = sum(reranked_hits) / len(reranked_hits)
    print(f"Hybrid+Reranker top-5 recall: {reranked_recall:.3f}")


if __name__ == "__main__":
    main()
