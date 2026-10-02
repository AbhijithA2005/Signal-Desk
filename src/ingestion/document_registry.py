from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class DocumentRegistry:
    """Persist metadata for locally uploaded documents."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS uploaded_documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    page_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def list_documents(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT document_id, filename, sha256, size_bytes,
                       page_count, chunk_count, created_at
                FROM uploaded_documents
                ORDER BY created_at DESC, document_id
                """
            ).fetchall()

        return [dict(row) for row in rows]

    def get_document(
        self,
        document_id: str,
    ) -> dict[str, Any] | None:
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT document_id, filename, sha256, size_bytes,
                       page_count, chunk_count, created_at
                FROM uploaded_documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

        return dict(row) if row is not None else None

    def get_by_sha256(self, sha256: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT document_id, filename, sha256, size_bytes,
                       page_count, chunk_count, created_at
                FROM uploaded_documents
                WHERE sha256 = ?
                ORDER BY created_at
                LIMIT 1
                """,
                (sha256,),
            ).fetchone()

        return dict(row) if row is not None else None

    def add_document(self, document: dict[str, Any]) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO uploaded_documents (
                    document_id, filename, sha256, size_bytes,
                    page_count, chunk_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document["document_id"],
                    document["filename"],
                    document["sha256"],
                    document["size_bytes"],
                    document["page_count"],
                    document["chunk_count"],
                    document["created_at"],
                ),
            )

    def delete_document(
        self,
        document_id: str,
    ) -> dict[str, Any] | None:
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT document_id, filename, sha256, size_bytes,
                       page_count, chunk_count, created_at
                FROM uploaded_documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

            if row is None:
                return None

            connection.execute(
                "DELETE FROM uploaded_documents WHERE document_id = ?",
                (document_id,),
            )

        return dict(row)