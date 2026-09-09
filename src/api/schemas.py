from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


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


class AskResponse(BaseModel):
    query: str
    answer: str
    decision: str
    gate_score: float | None = None
    sources: list[Source]
