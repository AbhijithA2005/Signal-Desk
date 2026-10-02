import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, models, pre_tokenizers

from src.api import app as api
from src.generation.ollama_client import OllamaClient
from src.ingestion.document_registry import DocumentRegistry


class FakeVectorStore:
    def __init__(self):
        self.document_ids = set()

    def add_chunks(self, chunk_ids, texts, metadatas):
        self.document_ids.update(item["document_id"] for item in metadatas)

    def delete_document(self, document_id):
        self.document_ids.discard(document_id)


class FakeBM25Store:
    def __init__(self):
        self.records = []

    def add_records(self, records):
        self.records.extend(records)

    def remove_document(self, document_id):
        self.records = [
            record
            for record in self.records
            if record["document_id"] != document_id
        ]

    def records_for_documents(self, document_ids):
        selected = set(document_ids)
        return [
            {
                "chunk_id": record["chunk_id"],
                "text": record["text"],
                "metadata": {
                    "document_id": record["document_id"],
                    "source_file": record["source_file"],
                    "source_pages": ",".join(
                        str(page) for page in record["source_pages"]
                    ),
                    "chunk_index": record["chunk_index"],
                    "token_count": record["token_count"],
                },
                "score": None,
                "reranker_score": None,
            }
            for record in self.records
            if record["document_id"] in selected
        ]


class FakeHybridRetriever:
    def __init__(self):
        self.dense_store = FakeVectorStore()
        self.bm25_store = FakeBM25Store()
        self.last_document_ids = None

    def search(self, query, top_k=5, candidate_k=10, document_ids=None):
        self.last_document_ids = document_ids
        records = self.bm25_store.records
        if document_ids is not None:
            records = [
                record
                for record in records
                if record["document_id"] in document_ids
            ]
        return [
            {
                "chunk_id": record["chunk_id"],
                "text": record["text"],
                "metadata": {
                    "document_id": record["document_id"],
                    "source_file": record["source_file"],
                    "source_pages": ",".join(
                        str(page) for page in record["source_pages"]
                    ),
                    "chunk_index": record["chunk_index"],
                    "token_count": record["token_count"],
                },
                "score": 1.0,
            }
            for record in records[:candidate_k]
        ][:top_k]


class FakeReranker:
    def __init__(self):
        self.score = -1.0
        self.calls = 0

    def rerank(self, query, candidates, top_k=5):
        self.calls += 1
        return [
            {**candidate, "reranker_score": self.score}
            for candidate in candidates[:top_k]
        ]


class FakeOllamaClient:
    model = "test-model"
    generate_url = "http://ollama.invalid/api/generate"

    def generate(self, prompt):
        self.last_prompt = prompt
        assert "Orbital mechanics describes" in prompt
        if "Create concise research notes" in prompt:
            assert "Critical follow-up questions" in prompt
        return "The document describes orbital mechanics. [SOURCE 1]"

    def analyze_image(self, query, image):
        assert query == "What is shown?"
        assert image.startswith(b"\x89PNG\r\n\x1a\n")
        return "A diagram with labeled orbital paths."

    def stream(self, prompt, image=None):
        if image is not None:
            assert prompt == "What is shown?"
            assert image.startswith(b"\x89PNG\r\n\x1a\n")
            yield "A diagram "
            yield "with labeled orbital paths."
            return

        if "friendly general-purpose assistant" in prompt:
            assert "Hi" in prompt
            yield "Hi! "
            yield "How can I help?"
            return

        assert "Orbital mechanics describes" in prompt
        if "Create concise research notes" in prompt:
            assert "Critical follow-up questions" in prompt
        yield "The document "
        yield "describes orbital mechanics. [SOURCE 1]"


def make_text_pdf() -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Orbital mechanics describes the motion of satellites. "
        "A satellite follows an orbit around a larger body.",
    )
    contents = document.tobytes()
    document.close()
    return contents


def make_blank_pdf() -> bytes:
    document = pymupdf.open()
    document.new_page()
    contents = document.tobytes()
    document.close()
    return contents


def make_test_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


@pytest.fixture
def api_client(monkeypatch, tmp_path):
    hybrid = FakeHybridRetriever()
    reranker = FakeReranker()
    client = FakeOllamaClient()
    monkeypatch.setattr(api, "get_pipeline", lambda: (hybrid, reranker, client))
    monkeypatch.setattr(
        api,
        "get_document_registry",
        lambda: DocumentRegistry(tmp_path / "documents.sqlite3"),
    )
    monkeypatch.setattr(api, "UPLOAD_FILE_PATH", tmp_path / "files")
    monkeypatch.setattr(api, "UPLOAD_CHUNK_PATH", tmp_path / "chunks")
    monkeypatch.setattr(api, "get_upload_tokenizer", make_test_tokenizer)
    monkeypatch.setattr(
        api.requests,
        "post",
        lambda *args, **kwargs: SimpleNamespace(raise_for_status=lambda: None),
    )

    with TestClient(api.app) as test_client:
        yield test_client, hybrid, reranker


def test_upload_scope_search_ask_and_delete(api_client):
    test_client, hybrid, _ = api_client
    pdf_bytes = make_text_pdf()

    response = test_client.post(
        "/documents",
        files={"file": ("orbital-notes.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200, response.text
    uploaded = response.json()
    document_id = uploaded["document_id"]
    assert uploaded["filename"] == "orbital-notes.pdf"
    assert uploaded["page_count"] == 1
    assert uploaded["chunk_count"] > 0
    assert uploaded["duplicate"] is False

    duplicate = test_client.post(
        "/documents",
        files={"file": ("renamed-copy.pdf", pdf_bytes, "application/pdf")},
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["document_id"] == document_id
    assert duplicate.json()["duplicate"] is True

    listed = test_client.get("/documents").json()["documents"]
    assert [item["document_id"] for item in listed] == [document_id]

    search = test_client.post(
        "/search",
        json={"query": "orbital mechanics", "document_ids": [document_id]},
    )
    assert search.status_code == 200, search.text
    assert hybrid.last_document_ids == [document_id]
    assert search.json()["results"][0]["document_id"] == document_id

    ask = test_client.post(
        "/ask",
        json={"query": "What does this document describe?", "document_ids": [document_id]},
    )
    assert ask.status_code == 200, ask.text
    assert ask.json()["decision"] == "generate"
    assert ask.json()["sources"][0]["document_id"] == document_id

    deleted = test_client.delete(f"/documents/{document_id}")
    assert deleted.status_code == 200
    assert test_client.get("/documents").json()["documents"] == []
    assert not list((Path(api.UPLOAD_FILE_PATH)).glob("*.pdf"))
    assert not list((Path(api.UPLOAD_CHUNK_PATH)).glob("*.jsonl"))
    assert document_id not in hybrid.dense_store.document_ids
    assert all(
        record["document_id"] != document_id
        for record in hybrid.bm25_store.records
    )


def test_document_overview_uses_all_selected_pdf_chunks(api_client):
    test_client, _, reranker = api_client
    response = test_client.post(
        "/documents",
        files={"file": ("orbital-notes.pdf", make_text_pdf(), "application/pdf")},
    )
    document_id = response.json()["document_id"]
    reranker.score = -10.0

    answer = test_client.post(
        "/ask",
        json={
            "query": "Tell about the uploaded PDF.",
            "document_ids": [document_id],
        },
    )

    assert answer.status_code == 200, answer.text
    assert answer.json()["decision"] == "generate"
    assert answer.json()["gate_score"] is None
    assert len(answer.json()["sources"]) == response.json()["chunk_count"]
    assert reranker.calls == 0


def test_notes_include_critical_followups_and_cited_evidence(api_client):
    test_client, _, _ = api_client
    uploaded = test_client.post(
        "/documents",
        files={"file": ("orbital-notes.pdf", make_text_pdf(), "application/pdf")},
    ).json()

    response = test_client.post(
        "/notes",
        json={
            "query": "What should I verify about satellite motion?",
            "document_ids": [uploaded["document_id"]],
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["decision"] == "generate"
    assert response.json()["sources"]


def test_presentation_notes_summarize_single_uploaded_document(api_client):
    test_client, _, _ = api_client
    uploaded = test_client.post(
        "/documents",
        files={"file": ("orbital-notes.pdf", make_text_pdf(), "application/pdf")},
    ).json()

    response = test_client.post(
        "/notes/stream",
        json={
            "query": "Can you make me notes for PPT presentation?",
            "document_ids": [uploaded["document_id"]],
        },
    )
    events = [json.loads(line) for line in response.text.splitlines()]

    assert response.status_code == 200
    assert events[0]["payload"]["decision"] == "generate"
    assert len(events[0]["payload"]["sources"]) == uploaded["chunk_count"]
    assert any(event["type"] == "token" for event in events)


def test_notes_abstain_when_retrieved_evidence_is_weak(api_client):
    test_client, _, reranker = api_client
    uploaded = test_client.post(
        "/documents",
        files={("file"): ("orbital-notes.pdf", make_text_pdf(), "application/pdf")},
    ).json()
    reranker.score = -10.0

    response = test_client.post(
        "/notes",
        json={
            "query": "What should I verify?",
            "document_ids": [uploaded["document_id"]],
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["decision"] == "abstain"
    assert response.json()["sources"] == []


def test_empty_document_scope_does_not_search_bundled_corpus(api_client):
    test_client, hybrid, reranker = api_client
    response = test_client.post(
        "/ask/stream",
        json={"query": "What does this source say?"},
    )
    events = [json.loads(line) for line in response.text.splitlines()]

    assert response.status_code == 200
    assert events[0]["payload"]["decision"] == "abstain"
    assert events[0]["payload"]["sources"] == []
    assert hybrid.last_document_ids == []
    assert reranker.calls == 0


@pytest.mark.parametrize("endpoint", ["/ask/stream", "/search"])
def test_live_api_rejects_legacy_benchmark_document_ids(api_client, endpoint):
    test_client, _, _ = api_client
    response = test_client.post(
        endpoint,
        json={"query": "What does this document say?", "document_ids": ["RAG_001"]},
    )

    assert response.status_code == 404


def test_image_analysis_validates_and_analyzes_image(api_client):
    test_client, _, _ = api_client
    response = test_client.post(
        "/analyze-image",
        data={"query": "What is shown?"},
        files={"file": ("figure.png", b"\x89PNG\r\n\x1a\nimage", "image/png")},
    )

    assert response.status_code == 200, response.text
    assert response.json()["analysis"] == "A diagram with labeled orbital paths."
    assert "verify important details" in response.json()["notice"]

    invalid = test_client.post(
        "/analyze-image",
        files={"file": ("not-an-image.png", b"plain text", "image/png")},
    )
    assert invalid.status_code == 415


def test_ollama_vision_client_sends_encoded_image(monkeypatch):
    request = {}

    def fake_post(url, json, timeout):
        request.update(url=url, payload=json, timeout=timeout)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"response": "A chart."},
        )

    monkeypatch.setattr("src.generation.ollama_client.requests.post", fake_post)
    image = b"\x89PNG\r\n\x1a\nexample"
    client = OllamaClient(vision_model="vision-test")

    assert client.analyze_image("Describe this image", image) == "A chart."
    assert request["payload"]["model"] == "vision-test"
    assert request["payload"]["images"] == [base64.b64encode(image).decode("ascii")]
    assert request["payload"]["prompt"] == "Describe this image"


def test_streamed_ask_and_notes_include_metadata_and_tokens(api_client):
    test_client, hybrid, _ = api_client
    uploaded = test_client.post(
        "/documents",
        files={"file": ("orbital-notes.pdf", make_text_pdf(), "application/pdf")},
    ).json()
    seed_record = hybrid.bm25_store.records[0]
    for index in range(4):
        record = seed_record.copy()
        record["chunk_id"] = f'{uploaded["document_id"]}_C{index + 2:03d}'
        record["chunk_index"] = index + 2
        hybrid.bm25_store.records.append(record)
    request = {
        "query": "How are satellites moved?",
        "document_ids": [uploaded["document_id"]],
    }

    answer = test_client.post("/ask/stream", json=request)
    answer_events = [json.loads(line) for line in answer.text.splitlines()]
    assert answer.status_code == 200
    assert answer_events[0]["type"] == "metadata"
    assert answer_events[0]["payload"]["sources"]
    assert len(answer_events[0]["payload"]["sources"]) == 3
    assert "The document describes orbital mechanics. [SOURCE 1]" == "".join(
        event["text"] for event in answer_events if event["type"] == "token"
    )
    assert answer_events[-1]["type"] == "done"

    notes = test_client.post("/notes/stream", json=request)
    notes_events = [json.loads(line) for line in notes.text.splitlines()]
    assert notes_events[0]["payload"]["sources"]
    assert any(event["type"] == "token" for event in notes_events)


def test_streamed_image_analysis_emits_incremental_tokens(api_client):
    test_client, _, _ = api_client
    response = test_client.post(
        "/analyze-image/stream",
        data={"query": "What is shown?"},
        files={"file": ("figure.png", b"\x89PNG\r\n\x1a\nimage", "image/png")},
    )
    events = [json.loads(line) for line in response.text.splitlines()]

    assert response.status_code == 200
    assert events[0]["payload"]["notice"]
    assert "".join(event["text"] for event in events if event["type"] == "token") == (
        "A diagram with labeled orbital paths."
    )
    assert events[-1]["type"] == "done"


def test_general_chat_handles_greeting_without_uploaded_documents(api_client):
    test_client, _, _ = api_client
    response = test_client.post("/chat/stream", json={"query": "Hi"})
    events = [json.loads(line) for line in response.text.splitlines()]

    assert response.status_code == 200
    assert events[0]["payload"]["decision"] == "generate"
    assert events[0]["payload"]["sources"] == []
    assert "".join(event["text"] for event in events if event["type"] == "token") == (
        "Hi! How can I help?"
    )


def test_upload_rejects_non_pdf(api_client):
    test_client, _, _ = api_client
    response = test_client.post(
        "/documents",
        files={"file": ("notes.txt", b"not a PDF", "text/plain")},
    )
    assert response.status_code == 415


def test_upload_rejects_oversized_pdf(api_client, monkeypatch):
    test_client, _, _ = api_client
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 8)
    response = test_client.post(
        "/documents",
        files={"file": ("large.pdf", b"%PDF-" + b"x" * 16, "application/pdf")},
    )
    assert response.status_code == 413


def test_upload_rejects_corrupt_and_textless_pdfs(api_client):
    test_client, _, _ = api_client
    corrupt = test_client.post(
        "/documents",
        files={"file": ("corrupt.pdf", b"%PDF-not-valid", "application/pdf")},
    )
    assert corrupt.status_code == 422

    textless = test_client.post(
        "/documents",
        files={"file": ("scanned.pdf", make_blank_pdf(), "application/pdf")},
    )
    assert textless.status_code == 422
    assert test_client.get("/documents").json()["documents"] == []
