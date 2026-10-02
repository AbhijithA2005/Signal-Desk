from __future__ import annotations
import os
import hashlib
from collections.abc import Iterator
import requests
import json
import re
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from tokenizers import Tokenizer

from src.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentListResponse,
    ImageAnalysisResponse,
    NotesResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    Source,
    UploadDocumentResponse,
)
from src.chunking.chunk_store import save_chunks_jsonl
from src.chunking.chunker import chunk_pages
from src.citations.citation import citation_from_metadata
from src.evaluation.evidence_gate import is_evidence_sufficient
from src.generation.ollama_client import OllamaClient
from src.generation.prompt_builder import (
    build_general_chat_prompt,
    build_grounded_prompt,
    build_research_notes_prompt,
)
from src.ingestion.document_registry import DocumentRegistry
from src.ingestion.pdf_loader import load_pdf
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import VectorStore


UPLOAD_ROOT = Path("data/uploads")
UPLOAD_FILE_PATH = UPLOAD_ROOT / "files"
UPLOAD_CHUNK_PATH = Path("data/processed/uploads")
DOCUMENT_DATABASE_PATH = UPLOAD_ROOT / "documents.sqlite3"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_DOCUMENT_OVERVIEW_CHUNKS = 12
MAX_GENERATION_EVIDENCE_CHUNKS = 3

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

    bm25 = BM25Store(
        None,
        additional_chunk_directory=UPLOAD_CHUNK_PATH,
        include_primary_corpus=False,
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
        vision_model=os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:4b"),
    )

    return hybrid, reranker, client


@lru_cache(maxsize=1)
def get_document_registry() -> DocumentRegistry:
    return DocumentRegistry(DOCUMENT_DATABASE_PATH)


@lru_cache(maxsize=1)
def get_upload_tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained("bert-base-uncased")


def is_document_overview_request(query: str) -> bool:
    return re.search(
        r"\b(?:tell(?:\s+me)?\s+about|summari[sz]e|summary|overview|"
        r"describe|notes?|presentations?|ppt|slides|study\s+guide|"
        r"key\s+points|main\s+points)\b",
        query,
        flags=re.IGNORECASE,
    ) is not None


def select_document_overview_chunks(
    records: list[dict],
) -> tuple[list[dict], bool]:
    if len(records) <= MAX_DOCUMENT_OVERVIEW_CHUNKS:
        return records, False

    last_index = len(records) - 1
    indices = sorted(
        {
            round(
                index * last_index / (MAX_DOCUMENT_OVERVIEW_CHUNKS - 1)
            )
            for index in range(MAX_DOCUMENT_OVERVIEW_CHUNKS)
        }
    )
    return [records[index] for index in indices], True


app = FastAPI(
    title="Research RAG Platform",
    description=(
        "Local RAG API exposing hybrid retrieval, "
        "reranking, evidence gating, and grounded generation."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def warm_model() -> None:
    """
    Preload the local Qwen model so the first real
    user request does not pay the model-load penalty.
    """
    try:
        _, _, client = get_pipeline()

        requests.post(
            client.generate_url,
            json={
                "model": client.model,
                "prompt": "",
                "stream": False,
                "keep_alive": -1,
            },
            timeout=120,
        ).raise_for_status()

        print("Ollama model warm-up: PASSED")

    except Exception as exc:
        print(
            f"Ollama model warm-up skipped: {exc}"
        )

def build_retrieval_results(
    reranked: list[dict],
) -> list[SearchResult]:
    results: list[SearchResult] = []

    for result in reranked:
        citation = citation_from_metadata(
            result["chunk_id"],
            result["metadata"],
        )

        results.append(
            SearchResult(
                chunk_id=result["chunk_id"],
                document_id=citation.document_id,
                text=result["text"],
                score=result.get("score"),
                reranker_score=result.get("reranker_score"),
                citation=citation.label,
            )
        )

    return results


def retrieve_query_evidence(
    query: str,
    document_ids: list[str],
) -> tuple[
    list[dict],
    list[SearchResult],
    list[Source],
    float | None,
    bool,
    dict | None,
    bool,
]:
    validate_document_scope(document_ids)
    hybrid, reranker, _ = get_pipeline()
    overview_document = None
    overview_sampled = False

    if (
        document_ids is not None
        and len(document_ids) == 1
        and document_ids[0].startswith("UPL_")
        and is_document_overview_request(query)
    ):
        overview_document = get_document_registry().get_document(document_ids[0])

    if overview_document is not None:
        document_records = hybrid.bm25_store.records_for_documents(document_ids)
        reranked, overview_sampled = select_document_overview_chunks(document_records)
    else:
        candidates = hybrid.search(
            query,
            top_k=20,
            candidate_k=20,
            document_ids=document_ids,
        )
        reranked = (
            reranker.rerank(
                query,
                candidates,
                top_k=MAX_GENERATION_EVIDENCE_CHUNKS,
            )
            if candidates
            else []
        )

    retrieval = build_retrieval_results(reranked)
    gate_score = (
        None
        if not reranked or overview_document is not None
        else float(reranked[0]["reranker_score"])
    )

    sufficient = bool(reranked) and (
        overview_document is not None or is_evidence_sufficient(reranked)
    )

    evidence = []
    sources: list[Source] = []
    for result in reranked:
        citation = citation_from_metadata(result["chunk_id"], result["metadata"])
        item = dict(result)
        item["citation"] = citation
        evidence.append(item)
        sources.append(
            Source(
                document_id=citation.document_id,
                chunk_id=citation.chunk_id,
                page_label=citation.page_label,
                chunk_index=citation.chunk_index,
                score=(
                    float(result["reranker_score"])
                    if result.get("reranker_score") is not None
                    else None
                ),
            )
        )

    return (
        evidence,
        retrieval,
        sources,
        gate_score,
        sufficient,
        overview_document,
        overview_sampled,
    )


def validate_document_scope(document_ids: list[str]) -> None:
    if not document_ids:
        return

    available_ids = {
        document["document_id"]
        for document in get_document_registry().list_documents()
    }
    unknown_ids = set(document_ids) - available_ids
    if unknown_ids:
        raise HTTPException(
            status_code=404,
            detail="One or more selected documents are not uploaded documents.",
        )


def is_supported_image(content: bytes) -> bool:
    return (
        content.startswith(b"\x89PNG\r\n\x1a\n")
        or content.startswith(b"\xff\xd8\xff")
        or content.startswith((b"GIF87a", b"GIF89a"))
        or (content.startswith(b"RIFF") and content[8:12] == b"WEBP")
    )


def validate_image_upload(file: UploadFile, query: str) -> tuple[bytes, str]:
    content = file.file.read(MAX_IMAGE_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image analysis uploads are limited to 10 MB.",
        )
    if not is_supported_image(content):
        raise HTTPException(
            status_code=415,
            detail="Use a PNG, JPEG, GIF, or WebP image.",
        )
    clean_query = query.strip()
    if not clean_query:
        raise HTTPException(status_code=422, detail="An image question is required.")
    if len(clean_query) > 2000:
        raise HTTPException(status_code=422, detail="Image questions are limited to 2000 characters.")
    return content, clean_query


def build_answer_prompt(
    query: str,
    evidence: list[dict],
    overview_document: dict | None,
    overview_sampled: bool,
) -> str:
    prompt_query = query
    if overview_document is not None:
        prompt_query = (
            f'Give an overview of "{overview_document["filename"]}". '
            "Summarize its purpose, main topics, key points, and conclusions. "
            "Describe claims as claims made by the document, not as independently verified facts."
        )
        if overview_sampled:
            prompt_query += (
                " The supplied passages are sampled across the document, so "
                "note that the overview may not cover every section."
            )
    return build_grounded_prompt(query=prompt_query, evidence=evidence)


def ndjson_stream(
    metadata: dict[str, object],
    chunks: Iterator[str],
) -> StreamingResponse:
    def events() -> Iterator[str]:
        yield json.dumps({"type": "metadata", "payload": metadata}) + "\n"
        try:
            for chunk in chunks:
                yield json.dumps({"type": "token", "text": chunk}) + "\n"
        except requests.RequestException as exc:
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                if exc.response.status_code == 404:
                    yield json.dumps(
                        {
                            "type": "error",
                            "message": "The configured Ollama model is unavailable. Check the model name with `ollama list`.",
                        }
                    ) + "\n"
                else:
                    yield json.dumps(
                        {"type": "error", "message": "Ollama could not complete the request."}
                    ) + "\n"
            else:
                yield json.dumps(
                    {"type": "error", "message": "Ollama is unreachable. Start the local service and retry."}
                ) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "research-rag",
    }


@app.get(
    "/documents",
    response_model=DocumentListResponse,
)
def list_documents() -> DocumentListResponse:
    return DocumentListResponse(
        documents=get_document_registry().list_documents(),
    )


@app.post(
    "/documents",
    response_model=UploadDocumentResponse,
)
def upload_document(
    file: UploadFile = File(...),
) -> UploadDocumentResponse:
    filename = Path(
        (file.filename or "upload.pdf").replace("\\", "/")
    ).name
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=415,
            detail="Only PDF uploads are supported in this phase.",
        )

    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="PDF uploads are limited to 25 MB.",
        )
    if b"%PDF-" not in content[:1024]:
        raise HTTPException(
            status_code=415,
            detail="The uploaded file is not a valid PDF.",
        )

    checksum = hashlib.sha256(content).hexdigest()
    registry = get_document_registry()
    existing = registry.get_by_sha256(checksum)
    if existing is not None:
        return UploadDocumentResponse(**existing, duplicate=True)

    document_id = f"UPL_{uuid.uuid4().hex[:16].upper()}"
    pdf_path = UPLOAD_FILE_PATH / f"{document_id}.pdf"
    chunk_path = UPLOAD_CHUNK_PATH / f"{document_id}.jsonl"
    pipeline = None
    vector_index_attempted = False
    bm25_indexed = False

    try:
        UPLOAD_FILE_PATH.mkdir(parents=True, exist_ok=True)
        UPLOAD_CHUNK_PATH.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(content)

        try:
            pages = load_pdf(pdf_path)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail="The uploaded PDF could not be read.",
            ) from exc
        if not any(page.text.strip() for page in pages):
            raise HTTPException(
                status_code=422,
                detail=(
                    "No selectable text was found. Scanned PDFs require OCR, "
                    "which is not included in this phase."
                ),
            )

        for page in pages:
            page.document_id = document_id
            page.source_file = filename

        chunks = chunk_pages(pages, get_upload_tokenizer())
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No searchable text could be extracted from this PDF.",
            )

        records = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source_file": chunk.source_file,
                "source_pages": chunk.source_pages,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "token_count": chunk.token_count,
            }
            for chunk in chunks
        ]
        save_chunks_jsonl(chunks, chunk_path)

        pipeline = get_pipeline()
        hybrid, _, _ = pipeline
        vector_index_attempted = True
        hybrid.dense_store.add_chunks(
            chunk_ids=[record["chunk_id"] for record in records],
            texts=[record["text"] for record in records],
            metadatas=[
                {
                    "document_id": record["document_id"],
                    "source_file": record["source_file"],
                    "source_pages": ",".join(
                        map(str, record["source_pages"])
                    ),
                    "chunk_index": record["chunk_index"],
                    "token_count": record["token_count"],
                }
                for record in records
            ],
        )
        hybrid.bm25_store.add_records(records)
        bm25_indexed = True

        metadata = {
            "document_id": document_id,
            "filename": filename,
            "sha256": checksum,
            "size_bytes": len(content),
            "page_count": len(pages),
            "chunk_count": len(chunks),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        registry.add_document(metadata)
        return UploadDocumentResponse(**metadata, duplicate=False)
    except HTTPException:
        if pipeline is not None:
            hybrid, _, _ = pipeline
            if bm25_indexed:
                hybrid.bm25_store.remove_document(document_id)
            if vector_index_attempted:
                hybrid.dense_store.delete_document(document_id)
        pdf_path.unlink(missing_ok=True)
        chunk_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        if pipeline is not None:
            hybrid, _, _ = pipeline
            try:
                if bm25_indexed:
                    hybrid.bm25_store.remove_document(document_id)
                if vector_index_attempted:
                    hybrid.dense_store.delete_document(document_id)
            except Exception:
                pass
        pdf_path.unlink(missing_ok=True)
        chunk_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="The PDF could not be indexed.",
        ) from exc


@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, object]:
    registry = get_document_registry()
    document = registry.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    hybrid, _, _ = get_pipeline()
    hybrid.dense_store.delete_document(document_id)
    hybrid.bm25_store.remove_document(document_id)
    registry.delete_document(document_id)
    (UPLOAD_FILE_PATH / f"{document_id}.pdf").unlink(missing_ok=True)
    (UPLOAD_CHUNK_PATH / f"{document_id}.jsonl").unlink(missing_ok=True)
    return {"document_id": document_id, "deleted": True}


@app.post(
    "/search",
    response_model=SearchResponse,
)
def search(request: SearchRequest) -> SearchResponse:
    validate_document_scope(request.document_ids)
    hybrid, reranker, _ = get_pipeline()

    candidates = hybrid.search(
        request.query,
        top_k=20,
        candidate_k=20,
        document_ids=request.document_ids,
    )

    reranked = reranker.rerank(
        request.query,
        candidates,
        top_k=request.top_k,
    )

    return SearchResponse(
        query=request.query,
        results=build_retrieval_results(reranked),
    )


@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask(request: AskRequest) -> AskResponse:
    _, _, client = get_pipeline()
    (
        evidence,
        retrieval,
        sources,
        gate_score,
        sufficient,
        overview_document,
        overview_sampled,
    ) = retrieve_query_evidence(request.query, request.document_ids)

    if not sufficient:
        return AskResponse(
            query=request.query,
            answer=(
                "The provided evidence is insufficient "
                "to answer the question."
            ),
            decision="abstain",
            gate_score=gate_score,
            sources=[],
            retrieval=retrieval,
        )

    prompt = build_answer_prompt(
        request.query,
        evidence,
        overview_document,
        overview_sampled,
    )

    answer = client.generate(prompt)

    return AskResponse(
        query=request.query,
        answer=answer.strip(),
        decision="generate",
        gate_score=gate_score,
        sources=sources,
        retrieval=retrieval,
    )


@app.post("/chat/stream")
def general_chat_stream(request: AskRequest) -> StreamingResponse:
    _, _, client = get_pipeline()
    response = AskResponse(
        query=request.query,
        answer="",
        decision="generate",
        gate_score=None,
        sources=[],
        retrieval=[],
    )
    return ndjson_stream(
        response.model_dump(mode="json"),
        client.stream(build_general_chat_prompt(request.query)),
    )


@app.post("/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    _, _, client = get_pipeline()
    (
        evidence,
        retrieval,
        sources,
        gate_score,
        sufficient,
        overview_document,
        overview_sampled,
    ) = retrieve_query_evidence(request.query, request.document_ids)

    if not sufficient:
        response = AskResponse(
            query=request.query,
            answer="The provided evidence is insufficient to answer the question.",
            decision="abstain",
            gate_score=gate_score,
            sources=[],
            retrieval=retrieval,
        )
        return ndjson_stream(response.model_dump(mode="json"), iter(()))

    prompt = build_answer_prompt(
        request.query,
        evidence,
        overview_document,
        overview_sampled,
    )
    response = AskResponse(
        query=request.query,
        answer="",
        decision="generate",
        gate_score=gate_score,
        sources=sources,
        retrieval=retrieval,
    )
    return ndjson_stream(
        response.model_dump(mode="json"),
        client.stream(prompt),
    )


@app.post("/notes", response_model=NotesResponse)
def create_notes(request: AskRequest) -> NotesResponse:
    _, _, client = get_pipeline()
    (
        evidence,
        retrieval,
        sources,
        gate_score,
        sufficient,
        _,
        _,
    ) = retrieve_query_evidence(request.query, request.document_ids)

    if not sufficient:
        return NotesResponse(
            query=request.query,
            notes="The provided evidence is insufficient to create reliable notes.",
            decision="abstain",
            gate_score=gate_score,
            sources=[],
            retrieval=retrieval,
        )

    notes = client.generate(
        build_research_notes_prompt(request.query, evidence)
    )
    return NotesResponse(
        query=request.query,
        notes=notes,
        decision="generate",
        gate_score=gate_score,
        sources=sources,
        retrieval=retrieval,
    )


@app.post("/notes/stream")
def create_notes_stream(request: AskRequest) -> StreamingResponse:
    _, _, client = get_pipeline()
    evidence, retrieval, sources, gate_score, sufficient, _, _ = (
        retrieve_query_evidence(request.query, request.document_ids)
    )
    if not sufficient:
        response = NotesResponse(
            query=request.query,
            notes="The provided evidence is insufficient to create reliable notes.",
            decision="abstain",
            gate_score=gate_score,
            sources=[],
            retrieval=retrieval,
        )
        return ndjson_stream(response.model_dump(mode="json"), iter(()))

    response = NotesResponse(
        query=request.query,
        notes="",
        decision="generate",
        gate_score=gate_score,
        sources=sources,
        retrieval=retrieval,
    )
    return ndjson_stream(
        response.model_dump(mode="json"),
        client.stream(build_research_notes_prompt(request.query, evidence)),
    )


@app.post("/analyze-image", response_model=ImageAnalysisResponse)
def analyze_image(
    file: UploadFile = File(...),
    query: str = Form(
        default="Describe the image, visible text, and any uncertainty."
    ),
) -> ImageAnalysisResponse:
    content, clean_query = validate_image_upload(file, query)

    client = get_pipeline()[2]
    try:
        analysis = client.analyze_image(clean_query, content)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            model_name = getattr(client, "vision_model", "configured vision model")
            detail = (
                f"Ollama vision model '{model_name}' is not installed. "
                f"Run `ollama pull {model_name}` or set OLLAMA_VISION_MODEL."
            )
        else:
            detail = "Ollama could not complete the image-analysis request."
        raise HTTPException(status_code=503, detail=detail) from exc
    except requests.ConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail="Ollama is unreachable. Start the local Ollama service and retry.",
        ) from exc
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail="Ollama could not complete the image-analysis request.",
        ) from exc

    return ImageAnalysisResponse(
        query=clean_query,
        analysis=analysis,
        notice="Visual interpretation can be mistaken; verify important details against the original image.",
    )


@app.post("/analyze-image/stream")
def analyze_image_stream(
    file: UploadFile = File(...),
    query: str = Form(
        default="Describe the image, visible text, and any uncertainty."
    ),
) -> StreamingResponse:
    content, clean_query = validate_image_upload(file, query)
    client = get_pipeline()[2]
    response = ImageAnalysisResponse(
        query=clean_query,
        analysis="",
        notice="Visual interpretation can be mistaken; verify important details against the original image.",
    )
    return ndjson_stream(
        response.model_dump(mode="json"),
        client.stream(clean_query, image=content),
    )
