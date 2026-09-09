from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    AskRequest,
    AskResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    Source,
)
from src.citations.citation import citation_from_metadata
from src.evaluation.evidence_gate import is_evidence_sufficient
from src.generation.ollama_client import OllamaClient
from src.generation.prompt_builder import build_grounded_prompt
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import VectorStore


CHUNK_PATH = Path(
    "data/processed/research_corpus_chunks.jsonl"
)

CHROMA_PATH = "indexes/chroma"
COLLECTION_NAME = "research_chunks"


@lru_cache(maxsize=1)
def get_pipeline() -> tuple[
    HybridRetriever,
    CrossEncoderReranker,
    OllamaClient,
]:
    dense = VectorStore(
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
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

    client = OllamaClient(
        model="qwen3:8b",
    )

    return hybrid, reranker, client


app = FastAPI(
    title="Research RAG Platform",
    description=(
        "Local RAG API exposing hybrid retrieval, "
        "reranking, evidence gating, and grounded generation."
    ),
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "research-rag",
    }


@app.get("/corpus")
def corpus() -> dict[str, object]:
    documents: dict[str, int] = {}

    if not CHUNK_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Corpus file not found: {CHUNK_PATH}",
        )

    import json

    with CHUNK_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            record = json.loads(line)
            document_id = record["document_id"]
            documents[document_id] = (
                documents.get(document_id, 0) + 1
            )

    return {
        "documents": len(documents),
        "chunks": sum(documents.values()),
        "chunks_by_document": documents,
    }


@app.post(
    "/search",
    response_model=SearchResponse,
)
def search(request: SearchRequest) -> SearchResponse:
    hybrid, reranker, _ = get_pipeline()

    candidates = hybrid.search(
        request.query,
        top_k=20,
        candidate_k=20,
    )

    reranked = reranker.rerank(
        request.query,
        candidates,
        top_k=request.top_k,
    )

    results: list[SearchResult] = []

    for result in reranked:
        citation = citation_from_metadata(
            result["chunk_id"],
            result["metadata"],
        )

        results.append(
            SearchResult(
                chunk_id=result["chunk_id"],
                document_id=result["metadata"]["document_id"],
                text=result["text"],
                score=result.get("score"),
                reranker_score=result.get(
                    "reranker_score"
                ),
                citation=citation.label,
            )
        )

    return SearchResponse(
        query=request.query,
        results=results,
    )


@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask(request: AskRequest) -> AskResponse:
    hybrid, reranker, client = get_pipeline()

    candidates = hybrid.search(
        request.query,
        top_k=20,
        candidate_k=20,
    )

    reranked = reranker.rerank(
        request.query,
        candidates,
        top_k=5,
    )

    if not reranked:
        return AskResponse(
            query=request.query,
            answer=(
                "The provided evidence is insufficient "
                "to answer the question."
            ),
            decision="abstain",
            gate_score=None,
            sources=[],
        )

    gate_score = float(
        reranked[0]["reranker_score"]
    )

    if not is_evidence_sufficient(reranked):
        return AskResponse(
            query=request.query,
            answer=(
                "The provided evidence is insufficient "
                "to answer the question."
            ),
            decision="abstain",
            gate_score=gate_score,
            sources=[],
        )

    evidence = []

    sources: list[Source] = []

    for result in reranked:
        citation = citation_from_metadata(
            result["chunk_id"],
            result["metadata"],
        )

        item = dict(result)
        item["citation"] = citation
        evidence.append(item)

        sources.append(
            Source(
                document_id=citation.document_id,
                chunk_id=citation.chunk_id,
                page_label=citation.page_label,
                chunk_index=citation.chunk_index,
                score=float(
                    result["reranker_score"]
                ),
            )
        )

    prompt = build_grounded_prompt(
        query=request.query,
        evidence=evidence,
    )

    answer = client.generate(prompt)

    return AskResponse(
        query=request.query,
        answer=answer.strip(),
        decision="generate",
        gate_score=gate_score,
        sources=sources,
    )
