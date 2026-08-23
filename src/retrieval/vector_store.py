from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:
    """Persistent Chroma vector store for retrieval chunks."""

    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "mps",
    ) -> None:
        self.persist_directory = str(persist_directory)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.device = device

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        self.model = SentenceTransformer(
            self.embedding_model_name,
            device=self.device,
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "embedding_model": self.embedding_model_name,
                "distance": "cosine",
            },
        )

    def count(self) -> int:
        """Return the number of stored vectors."""
        return self.collection.count()

    def metadata(self) -> dict[str, Any]:
        """Return basic collection metadata."""
        return dict(self.collection.metadata or {})