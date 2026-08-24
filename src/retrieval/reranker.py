from __future__ import annotations

from typing import Any, Sequence

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Rerank retrieved candidates using a cross-encoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "mps",
    ) -> None:
        self.model_name = model_name
        self.device = device

        self.model = CrossEncoder(
            self.model_name,
            device=self.device,
        )

    def rerank(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Score and rerank candidate chunks."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not candidates:
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        pairs = [
            [query, candidate["text"]]
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        ranked = []

        for candidate, score in zip(
            candidates,
            scores,
        ):
            item = dict(candidate)
            item["reranker_score"] = float(score)
            ranked.append(item)

        ranked.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        return ranked[:top_k]