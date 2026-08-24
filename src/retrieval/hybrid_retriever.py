from __future__ import annotations

from typing import Any


class HybridRetriever:
    """Combine dense and BM25 rankings using Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense_store,
        bm25_store,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative.")

        self.dense_store = dense_store
        self.bm25_store = bm25_store
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Return a fused ranking from dense and BM25 retrieval."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than zero."
            )

        dense_results = self.dense_store.search(
            query,
            top_k=candidate_k,
        )

        bm25_results = self.bm25_store.search(
            query,
            top_k=candidate_k,
        )

        fused: dict[str, dict[str, Any]] = {}

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            chunk_id = result["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                    "dense_distance": None,
                    "bm25_score": None,
                }

            fused[chunk_id]["rrf_score"] += (
                1.0 / (self.rrf_k + rank)
            )
            fused[chunk_id]["dense_rank"] = rank
            fused[chunk_id]["dense_distance"] = result[
                "distance"
            ]

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):
            chunk_id = result["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                    "dense_distance": None,
                    "bm25_score": None,
                }

            fused[chunk_id]["rrf_score"] += (
                1.0 / (self.rrf_k + rank)
            )
            fused[chunk_id]["bm25_rank"] = rank
            fused[chunk_id]["bm25_score"] = result[
                "score"
            ]

        ranked = sorted(
            fused.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        return ranked[:top_k]