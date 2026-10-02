import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import VectorStore


def _record(chunk_id: str, document_id: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "source_file": f"{document_id}.pdf",
        "source_pages": [1],
        "chunk_index": 0,
        "text": text,
        "token_count": len(text.split()),
    }


def test_bm25_supports_uploaded_documents_and_scope_filters(tmp_path: Path):
    base_record = _record("BASE_C000", "BASE", "cranberry fruit harvest")
    chunk_file = tmp_path / "base.jsonl"
    chunk_file.write_text(json.dumps(base_record) + "\n", encoding="utf-8")
    store = BM25Store(chunk_file)

    uploaded = _record("DOC_TEST_C000", "DOC_TEST", "orbital mechanics satellite")
    store.add_records([uploaded])

    matches = store.search("orbital satellite", document_ids=["DOC_TEST"])
    assert [match["chunk_id"] for match in matches] == ["DOC_TEST_C000"]
    assert store.search("orbital satellite", document_ids=[]) == []

    store.remove_document("DOC_TEST")
    assert all(
        match["metadata"]["document_id"] != "DOC_TEST"
        for match in store.search("orbital satellite")
    )


def test_bm25_reloads_persisted_uploaded_chunks(tmp_path: Path):
    base_record = _record("BASE_C000", "BASE", "cranberry fruit harvest")
    uploaded_record = _record(
        "DOC_TEST_C000",
        "DOC_TEST",
        "orbital mechanics satellite",
    )
    base_path = tmp_path / "base.jsonl"
    upload_directory = tmp_path / "uploads"
    upload_directory.mkdir()
    base_path.write_text(json.dumps(base_record) + "\n", encoding="utf-8")
    (upload_directory / "DOC_TEST.jsonl").write_text(
        json.dumps(uploaded_record) + "\n",
        encoding="utf-8",
    )

    restored = BM25Store(base_path, upload_directory)

    assert [
        match["chunk_id"]
        for match in restored.search(
            "orbital satellite",
            document_ids=["DOC_TEST"],
        )
    ] == ["DOC_TEST_C000"]


def test_bm25_can_start_without_the_bundled_corpus(tmp_path: Path):
    upload_directory = tmp_path / "uploads"
    upload_directory.mkdir()
    store = BM25Store(None, upload_directory, include_primary_corpus=False)

    assert store.count() == 0
    assert store.search("orbital satellite") == []

    store.add_records(
        [_record("DOC_TEST_C000", "DOC_TEST", "orbital mechanics satellite")]
    )
    assert [
        result["chunk_id"] for result in store.search("orbital satellite")
    ] == ["DOC_TEST_C000"]

    store.remove_document("DOC_TEST")
    assert store.count() == 0


def test_vector_store_passes_document_filter_to_chroma():
    store = VectorStore.__new__(VectorStore)
    store.model = Mock()
    store.model.encode.return_value = np.array([0.1, 0.2])
    store.collection = Mock()
    store.collection.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    store.search("satellite", document_ids=["DOC_TEST"])

    assert store.collection.query.call_args.kwargs["where"] == {
        "document_id": {"$in": ["DOC_TEST"]}
    }


def test_hybrid_retriever_passes_document_filter_to_both_stores():
    dense_store = Mock()
    bm25_store = Mock()
    dense_store.search.return_value = []
    bm25_store.search.return_value = []
    retriever = HybridRetriever(dense_store, bm25_store)

    retriever.search("satellite", document_ids=["DOC_TEST"])

    assert dense_store.search.call_args.kwargs["document_ids"] == ["DOC_TEST"]
    assert bm25_store.search.call_args.kwargs["document_ids"] == ["DOC_TEST"]