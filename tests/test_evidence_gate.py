from src.evaluation.evidence_gate import (
    DEFAULT_MIN_RERANKER_SCORE,
    is_evidence_sufficient,
)


def test_empty_results_are_insufficient():
    assert not is_evidence_sufficient([])


def test_score_above_threshold_is_sufficient():
    results = [
        {"reranker_score": -0.479},
    ]

    assert is_evidence_sufficient(results)


def test_score_below_threshold_is_insufficient():
    results = [
        {"reranker_score": -3.095},
    ]

    assert not is_evidence_sufficient(results)


def test_default_threshold_is_documented():
    assert DEFAULT_MIN_RERANKER_SCORE == -2.0
