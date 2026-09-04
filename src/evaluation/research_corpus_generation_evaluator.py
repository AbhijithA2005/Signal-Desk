import json
from pathlib import Path

from src.citations.citation import citation_from_metadata
from src.generation.ollama_client import OllamaClient
from src.generation.prompt_builder import build_grounded_prompt
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import VectorStore


QUERY_PATH = Path(
    "data/evaluation/research_corpus_generation_queries.json"
)

CHUNK_PATH = Path(
    "data/processed/research_corpus_chunks.jsonl"
)

OUTPUT_PATH = Path(
    "data/evaluation/research_corpus_generation_results.jsonl"
)


def load_queries():
    return json.loads(
        QUERY_PATH.read_text(encoding="utf-8")
    )["queries"]


def build_evidence(results):
    evidence = []

    for result in results:
        metadata = result["metadata"]

        evidence.append(
            {
                "text": result["text"],
                "citation": citation_from_metadata(
                    result["chunk_id"],
                    metadata,
                ),
            }
        )

    return evidence


def serialize_evidence(evidence):
    serialized = []

    for item in evidence:
        citation = item["citation"]

        serialized.append(
            {
                "text": item["text"],
                "citation": {
                    "document_id": citation.document_id,
                    "page_label": citation.page_label,
                    "chunk_id": citation.chunk_id,
                },
            }
        )

    return serialized


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

    ollama = OllamaClient(
        model="qwen3:8b",
        base_url="http://localhost:11434",
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output:

        for item in queries:
            query = item["query"]

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

            evidence = build_evidence(reranked)

            prompt = build_grounded_prompt(
                query=query,
                evidence=evidence,
            )

            answer = ollama.generate(
                prompt,
                temperature=0.0,
            )

            record = {
                "query_id": item["query_id"],
                "query": query,
                "relevant_chunks": item["relevant_chunks"],
                "expected_claims": item["expected_claims"],
                "retrieved_chunks": [
                    result["chunk_id"]
                    for result in reranked
                ],
                "evidence": serialize_evidence(
                    evidence
                ),
                "answer": answer,
            }

            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            print("=" * 100)
            print(
                f"{item['query_id']} | {query}"
            )
            print(
                "Retrieved:",
                ", ".join(record["retrieved_chunks"]),
            )
            print()
            print(answer)
            print()

    print(
        f"Saved generation results to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
