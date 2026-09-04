from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from src.evaluation.document_metrics import (
    hit_at_k_documents,
    mean,
    reciprocal_rank_documents,
)
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import VectorStore


QUERY_FILE = Path(
    "data/evaluation/research_corpus_queries.json"
)


def load_queries() -> list[dict]:
    """Load the four-paper document-level benchmark."""
    data = json.loads(
        QUERY_FILE.read_text(encoding="utf-8")
    )
    return data["queries"]


def unique_documents(results: list[dict]) -> list[str]:
    """Convert ranked chunk results into ranked unique documents."""
    documents: list[str] = []
    seen: set[str] = set()

    for result in results:
        document_id = result["metadata"]["document_id"]

        if document_id not in seen:
            seen.add(document_id)
            documents.append(document_id)

    return documents


def evaluate_method(
    name: str,
    queries: list[dict],
    retrieve: Callable[[str], list[dict]],
) -> dict:
    """Evaluate one retrieval strategy at document level."""
    hit1: list[float] = []
    hit3: list[float] = []
    hit5: list[float] = []
    mrr: list[float] = []

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    for item in queries:
        results = retrieve(item["query"])
        retrieved_documents = unique_documents(results)

        relevant_documents = item["relevant_documents"]

        q_hit1 = hit_at_k_documents(
            retrieved_documents,
            relevant_documents,
            1,
        )

        q_hit3 = hit_at_k_documents(
            retrieved_documents,
            relevant_documents,
            3,
        )

        q_hit5 = hit_at_k_documents(
            retrieved_documents,
            relevant_documents,
            5,
        )

        q_mrr = reciprocal_rank_documents(
            retrieved_documents,
            relevant_documents,
        )

        hit1.append(q_hit1)
        hit3.append(q_hit3)
        hit5.append(q_hit5)
        mrr.append(q_mrr)

        print(
            f"{item['id']} | "
            f"Hit@1={q_hit1:.0f} | "
            f"Hit@3={q_hit3:.0f} | "
            f"Hit@5={q_hit5:.0f} | "
            f"MRR={q_mrr:.3f} | "
            f"Retrieved={retrieved_documents}"
        )

    summary = {
        "method": name,
        "Hit@1": mean(hit1),
        "Hit@3": mean(hit3),
        "Hit@5": mean(hit5),
        "MRR": mean(mrr),
    }

    print("\nSummary:")
    print(
        f"Hit@1={summary['Hit@1']:.3f} | "
        f"Hit@3={summary['Hit@3']:.3f} | "
        f"Hit@5={summary['Hit@5']:.3f} | "
        f"MRR={summary['MRR']:.3f}"
    )

    return summary


def main() -> None:
    """Run the complete four-paper document evaluation."""
    queries = load_queries()

    dense = VectorStore(
        persist_directory="indexes/chroma",
        collection_name="research_chunks",
        embedding_model_name="BAAI/bge-small-en-v1.5",
        device="mps",
    )

    bm25 = BM25Store(
        "data/processed/research_corpus_chunks.jsonl"
    )

    hybrid = HybridRetriever(
        dense_store=dense,
        bm25_store=bm25,
        rrf_k=60,
    )

    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="mps",
    )

    results: list[dict] = []

    results.append(
        evaluate_method(
            "DENSE",
            queries,
            lambda query: dense.search(
                query,
                top_k=10,
            ),
        )
    )

    results.append(
        evaluate_method(
            "BM25",
            queries,
            lambda query: bm25.search(
                query,
                top_k=10,
            ),
        )
    )

    results.append(
        evaluate_method(
            "HYBRID / RRF",
            queries,
            lambda query: hybrid.search(
                query,
                top_k=10,
                candidate_k=10,
            ),
        )
    )

    results.append(
        evaluate_method(
            "HYBRID + RERANKER",
            queries,
            lambda query: reranker.rerank(
                query,
                hybrid.search(
                    query,
                    top_k=10,
                    candidate_k=10,
                ),
                top_k=10,
            ),
        )
    )

    print("\n" + "=" * 80)
    print("FINAL DOCUMENT-LEVEL COMPARISON")
    print("=" * 80)

    print(
        f"{'Method':<24}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Hit@5':>10}"
        f"{'MRR':>10}"
    )

    for result in results:
        print(
            f"{result['method']:<24}"
            f"{result['Hit@1']:>10.3f}"
            f"{result['Hit@3']:>10.3f}"
            f"{result['Hit@5']:>10.3f}"
            f"{result['MRR']:>10.3f}"
        )


if __name__ == "__main__":
    main()