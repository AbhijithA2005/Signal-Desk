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

    def add_chunks(
        self,
        chunk_ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]],
        batch_size: int = 16,
    ) -> None:
        """Embed and store retrieval chunks in Chroma."""
        if not (
            len(chunk_ids)
            == len(texts)
            == len(metadatas)
        ):
            raise ValueError(
                "chunk_ids, texts, and metadatas must have equal lengths."
            )

        for start in range(0, len(texts), batch_size):
            end = start + batch_size

            batch_ids = chunk_ids[start:end]
            batch_texts = texts[start:end]
            batch_metadatas = metadatas[start:end]

            embeddings = self.model.encode(
                batch_texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas,
                embeddings=embeddings.tolist(),
            )
    def reset(self) -> None:
        """Delete and recreate the current collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "embedding_model": self.embedding_model_name,
                "distance": "cosine",
            },
        )    
def build_collection_from_jsonl(
    chunk_file: str | Path,
    persist_directory: str | Path,
    collection_name: str,
    embedding_model_name: str = "BAAI/bge-small-en-v1.5",
    device: str = "mps",
    reset: bool = False,
) -> VectorStore:
    """Build a Chroma collection from a chunk JSONL file."""
    import json

    store = VectorStore(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
        device=device,
    )

    if reset:
        store.reset()

    path = Path(chunk_file)

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    chunk_ids = [record["chunk_id"] for record in records]
    texts = [record["text"] for record in records]

    metadatas = [
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
    ]

    store.add_chunks(
        chunk_ids=chunk_ids,
        texts=texts,
        metadatas=metadatas,
        batch_size=16,
    )

    return store