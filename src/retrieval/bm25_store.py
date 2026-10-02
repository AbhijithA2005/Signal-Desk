from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 lexical matching."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Store:
    """In-memory BM25 index over retrieval chunks."""

    def __init__(
        self,
        chunk_file: str | Path | None,
        additional_chunk_directory: str | Path | None = None,
        include_primary_corpus: bool = True,
    ) -> None:
        self.chunk_file = Path(chunk_file) if chunk_file is not None else None
        self.records: list[dict[str, Any]] = []

        if include_primary_corpus:
            if self.chunk_file is None:
                raise ValueError("A primary chunk file is required for corpus indexing.")
            self.records = [
                json.loads(line)
                for line in self.chunk_file.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

        if additional_chunk_directory is not None:
            directory = Path(additional_chunk_directory)
            if directory.exists():
                for path in sorted(directory.glob("*.jsonl")):
                    self.records.extend(
                        json.loads(line)
                        for line in path.read_text(
                            encoding="utf-8"
                        ).splitlines()
                        if line.strip()
                    )

        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self.corpus_tokens = [
            tokenize(record["text"])
            for record in self.records
        ]
        self.index = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def add_records(self, records: Iterable[dict[str, Any]]) -> None:
        additions = list(records)
        if not additions:
            return
        self.records.extend(additions)
        self._rebuild_index()

    def remove_document(self, document_id: str) -> None:
        remaining = [
            record
            for record in self.records
            if record["document_id"] != document_id
        ]
        if len(remaining) != len(self.records):
            self.records = remaining
            self._rebuild_index()

    def records_for_documents(
        self,
        document_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Return all stored chunks for selected documents in corpus order."""
        selected = set(document_ids)
        return [
            {
                "chunk_id": record["chunk_id"],
                "text": record["text"],
                "metadata": {
                    "document_id": record["document_id"],
                    "source_file": record["source_file"],
                    "source_pages": ",".join(
                        map(str, record["source_pages"])
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

    def count(self) -> int:
        """Return the number of indexed chunks."""
        return len(self.records)

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-k BM25 results."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.index is None:
            return []

        query_tokens = tokenize(query)

        scores = self.index.get_scores(query_tokens)
        allowed_documents = (
            set(document_ids)
            if document_ids is not None
            else None
        )

        ranked_indices = sorted(
            (
                index
                for index, record in enumerate(self.records)
                if allowed_documents is None
                or record["document_id"] in allowed_documents
            ),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results: list[dict[str, Any]] = []

        for index in ranked_indices:
            record = self.records[index]

            results.append(
                {
                    "chunk_id": record["chunk_id"],
                    "text": record["text"],
                    "metadata": {
                        "document_id": record["document_id"],
                        "source_file": record["source_file"],
                        "source_pages": ",".join(
                            map(
                                str,
                                record["source_pages"],
                            )
                        ),
                        "chunk_index": record["chunk_index"],
                        "token_count": record["token_count"],
                    },
                    "score": float(scores[index]),
                }
            )

        return results