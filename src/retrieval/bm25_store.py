from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 lexical matching."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Store:
    """In-memory BM25 index over retrieval chunks."""

    def __init__(
        self,
        chunk_file: str | Path,
    ) -> None:
        self.chunk_file = Path(chunk_file)

        records = [
            json.loads(line)
            for line in self.chunk_file.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if not records:
            raise ValueError("Chunk file contains no records.")

        self.records = records
        self.corpus_tokens = [
            tokenize(record["text"])
            for record in records
        ]

        self.index = BM25Okapi(self.corpus_tokens)

    def count(self) -> int:
        """Return the number of indexed chunks."""
        return len(self.records)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the top-k BM25 results."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_tokens = tokenize(query)

        scores = self.index.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
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