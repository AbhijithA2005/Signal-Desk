from __future__ import annotations


DEFAULT_MIN_RERANKER_SCORE = -2.0


def is_evidence_sufficient(
    reranked_results: list[dict],
    *,
    min_reranker_score: float = DEFAULT_MIN_RERANKER_SCORE,
) -> bool:
    """
    Decide whether retrieved evidence is sufficiently relevant
    to proceed to grounded generation.

    The threshold is calibrated on the current development set.
    It is not intended as a universal threshold for all models/corpora.
    """
    if not reranked_results:
        return False

    top_score = float(
        reranked_results[0]["reranker_score"]
    )

    return top_score >= min_reranker_score
