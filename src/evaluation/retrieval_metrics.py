from __future__ import annotations

from typing import Iterable, Sequence


def hit_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    """Return 1.0 if any relevant chunk appears in top-k."""
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    relevant = set(relevant_ids)

    return float(
        any(chunk_id in relevant for chunk_id in retrieved_ids[:k])
    )


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
) -> float:
    """Return reciprocal rank of the first relevant result."""
    relevant = set(relevant_ids)

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def mean(values: Sequence[float]) -> float:
    """Return arithmetic mean."""
    if not values:
        return 0.0

    return sum(values) / len(values)