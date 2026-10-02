from src.ingestion.document_registry import DocumentRegistry


def test_document_registry_persists_and_deletes_documents(tmp_path):
    registry = DocumentRegistry(tmp_path / "documents.sqlite3")
    document = {
        "document_id": "DOC_TEST",
        "filename": "research.pdf",
        "sha256": "abc123",
        "size_bytes": 1024,
        "page_count": 2,
        "chunk_count": 1,
        "created_at": "2026-10-02T00:00:00+00:00",
    }

    registry.add_document(document)

    restored_registry = DocumentRegistry(tmp_path / "documents.sqlite3")
    assert restored_registry.get_document("DOC_TEST") == document
    assert restored_registry.list_documents() == [document]
    assert restored_registry.delete_document("DOC_TEST") == document
    assert restored_registry.get_document("DOC_TEST") is None