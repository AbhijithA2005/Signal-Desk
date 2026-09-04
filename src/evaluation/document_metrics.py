from __future__ import annotations

from typing import Iterable, Sequence


def hit_at_k_documents(
    retrieved_document_ids: Sequence[str],
    relevant_document_ids: Iterable[str],
    k: int,
) -> float:
    """Return 1.0 if a relevant document appears in top-k."""
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    relevant = set(relevant_document_ids)

    return float(
        any(
            document_id in relevant
            for document_id in retrieved_document_ids[:k]
        )
    )


def reciprocal_rank_documents(
    retrieved_document_ids: Sequence[str],
    relevant_document_ids: Iterable[str],
) -> float:
    """Return reciprocal rank of the first relevant document."""
    relevant = set(relevant_document_ids)

    for rank, document_id in enumerate(
        retrieved_document_ids,
        start=1,
    ):
        if document_id in relevant:
            return 1.0 / rank

    return 0.0


def mean(values: Sequence[float]) -> float:
    """Return arithmetic mean."""
    if not values:
        return 0.0

    return sum(values) / len(values)