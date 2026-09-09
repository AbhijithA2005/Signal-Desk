from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.citations.citation import citation_from_metadata
from src.evaluation.evidence_gate import (
    DEFAULT_MIN_RERANKER_SCORE,
    is_evidence_sufficient,
)
from src.generation.ollama_client import OllamaClient
from src.generation.prompt_builder import build_grounded_prompt
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import VectorStore


ANSWERABLE_PATH = Path(
    "data/evaluation/research_corpus_generation_queries.json"
)

UNANSWERABLE_PATH = Path(
    "data/evaluation/research_corpus_abstention_queries.json"
)

CHUNK_PATH = Path(
    "data/processed/research_corpus_chunks.jsonl"
)

RESULTS_PATH = Path(
    "data/evaluation/research_corpus_gated_abstention_results.jsonl"
)

SUMMARY_PATH = Path(
    "data/evaluation/research_corpus_gated_abstention_summary.json"
)


def load_queries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
    return payload["queries"]


def prepare_evidence(
    reranked: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []

    for result in reranked:
        item = dict(result)

        citation = citation_from_metadata(
            result["chunk_id"],
            result["metadata"],
        )

        item["citation"] = citation

        evidence.append(item)

    return evidence


def evaluate_query(
    item: dict[str, Any],
    expected_answerable: bool,
    client: OllamaClient,
    hybrid: HybridRetriever,
    reranker: CrossEncoderReranker,
) -> dict[str, Any]:

    candidates = hybrid.search(
        item["query"],
        top_k=20,
        candidate_k=20,
    )

    reranked = reranker.rerank(
        item["query"],
        candidates,
        top_k=5,
    )

    top_score = float(
        reranked[0]["reranker_score"]
    )

    gate_pass = is_evidence_sufficient(
        reranked
    )

    result: dict[str, Any] = {
        "query_id": item["query_id"],
        "query": item["query"],
        "expected_answerable": expected_answerable,
        "gate_threshold": DEFAULT_MIN_RERANKER_SCORE,
        "top_reranker_score": top_score,
        "gate_pass": gate_pass,
        "decision": "generate" if gate_pass else "abstain",
        "retrieved_chunks": [
            result["chunk_id"]
            for result in reranked
        ],
    }

    if not gate_pass:
        result["answer"] = ""
        return result

    evidence = prepare_evidence(reranked)

    prompt = build_grounded_prompt(
        query=item["query"],
        evidence=evidence,
    )

    answer = client.generate(prompt)

    result["answer"] = answer.strip()

    return result


def main() -> None:
    answerable = load_queries(
        ANSWERABLE_PATH
    )

    unanswerable = load_queries(
        UNANSWERABLE_PATH
    )

    dense = VectorStore(
        persist_directory="indexes/chroma",
        collection_name="research_chunks",
    )

    bm25 = BM25Store(
        CHUNK_PATH
    )

    hybrid = HybridRetriever(
        dense_store=dense,
        bm25_store=bm25,
    )

    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="mps",
    )

    client = OllamaClient(
        model="qwen3:8b",
    )

    results: list[dict[str, Any]] = []

    print("=" * 100)
    print("ANSWERABLE QUERIES")
    print("=" * 100)

    for item in answerable:
        result = evaluate_query(
            item=item,
            expected_answerable=True,
            client=client,
            hybrid=hybrid,
            reranker=reranker,
        )

        results.append(result)

        print(
            f"{result['query_id']} | "
            f"score={result['top_reranker_score']:.3f} | "
            f"gate={result['gate_pass']} | "
            f"decision={result['decision']}"
        )

    print("\n" + "=" * 100)
    print("UNANSWERABLE QUERIES")
    print("=" * 100)

    for item in unanswerable:
        result = evaluate_query(
            item=item,
            expected_answerable=False,
            client=client,
            hybrid=hybrid,
            reranker=reranker,
        )

        results.append(result)

        print(
            f"{result['query_id']} | "
            f"score={result['top_reranker_score']:.3f} | "
            f"gate={result['gate_pass']} | "
            f"decision={result['decision']}"
        )

    answerable_results = [
        result
        for result in results
        if result["expected_answerable"]
    ]

    unanswerable_results = [
        result
        for result in results
        if not result["expected_answerable"]
    ]

    answerable_accepted = sum(
        result["gate_pass"]
        for result in answerable_results
    )

    unanswerable_rejected = sum(
        not result["gate_pass"]
        for result in unanswerable_results
    )

    false_abstentions = (
        len(answerable_results)
        - answerable_accepted
    )

    false_acceptances = (
        len(unanswerable_results)
        - unanswerable_rejected
    )

    answerable_acceptance_rate = (
        answerable_accepted
        / len(answerable_results)
        if answerable_results
        else 0.0
    )

    unanswerable_abstention_rate = (
        unanswerable_rejected
        / len(unanswerable_results)
        if unanswerable_results
        else 0.0
    )

    summary = {
        "experiment": "gated_abstention",
        "gate_type": "reranker_score_threshold",
        "gate_threshold": DEFAULT_MIN_RERANKER_SCORE,
        "num_answerable": len(answerable_results),
        "num_unanswerable": len(unanswerable_results),
        "answerable_accepted": answerable_accepted,
        "unanswerable_rejected": unanswerable_rejected,
        "false_abstentions": false_abstentions,
        "false_acceptances": false_acceptances,
        "answerable_acceptance_rate": answerable_acceptance_rate,
        "unanswerable_abstention_rate": unanswerable_abstention_rate,
    }

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for result in results:
            handle.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("GATED ABSTENTION SUMMARY")
    print("=" * 100)
    print(
        f"Gate threshold: "
        f"{DEFAULT_MIN_RERANKER_SCORE:.3f}"
    )
    print(
        f"Answerable accepted: "
        f"{answerable_accepted}/"
        f"{len(answerable_results)}"
    )
    print(
        f"Unanswerable rejected: "
        f"{unanswerable_rejected}/"
        f"{len(unanswerable_results)}"
    )
    print(
        f"False abstentions: "
        f"{false_abstentions}"
    )
    print(
        f"False acceptances: "
        f"{false_acceptances}"
    )
    print(
        f"Answerable acceptance rate: "
        f"{answerable_acceptance_rate:.3f}"
    )
    print(
        f"Unanswerable abstention rate: "
        f"{unanswerable_abstention_rate:.3f}"
    )

    print("\nSaved:")
    print(RESULTS_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
