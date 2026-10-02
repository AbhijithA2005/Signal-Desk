from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    document_ids: list[str] = Field(default_factory=list)


class UploadedDocument(BaseModel):
    document_id: str
    filename: str
    sha256: str
    size_bytes: int
    page_count: int
    chunk_count: int
    created_at: str


class DocumentListResponse(BaseModel):
    documents: list[UploadedDocument]


class UploadDocumentResponse(UploadedDocument):
    duplicate: bool = False


class Source(BaseModel):
    document_id: str
    chunk_id: str
    page_label: str
    chunk_index: int
    score: float | None = None


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float | None = None
    reranker_score: float | None = None
    citation: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    document_ids: list[str] = Field(default_factory=list)


class AskResponse(BaseModel):
    query: str
    answer: str
    decision: str
    gate_score: float | None = None
    sources: list[Source]
    retrieval: list[SearchResult]


class NotesResponse(BaseModel):
    query: str
    notes: str
    decision: str
    gate_score: float | None = None
    sources: list[Source]
    retrieval: list[SearchResult]


class ImageAnalysisResponse(BaseModel):
    query: str
    analysis: str
    notice: str
