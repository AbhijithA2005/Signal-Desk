from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Citation:
    """Citation metadata for a retrieved evidence chunk."""

    chunk_id: str
    document_id: str
    source_file: str
    source_pages: tuple[int, ...]
    chunk_index: int

    @property
    def page_label(self) -> str:
        """Return a human-readable page label."""
        if not self.source_pages:
            return "p. ?"

        if len(self.source_pages) == 1:
            return f"p. {self.source_pages[0]}"

        return "pp. " + ", ".join(
            str(page) for page in self.source_pages
        )

    @property
    def label(self) -> str:
        """Return a compact source citation label."""
        return (
            f"{self.document_id}, "
            f"{self.page_label}, "
            f"chunk {self.chunk_id}"
        )


def citation_from_metadata(
    chunk_id: str,
    metadata: dict[str, Any],
) -> Citation:
    """Create a Citation from Chroma metadata."""
    raw_pages = metadata.get("source_pages", "")

    if isinstance(raw_pages, str):
        source_pages = tuple(
            int(page.strip())
            for page in raw_pages.split(",")
            if page.strip()
        )
    else:
        source_pages = tuple(int(page) for page in raw_pages)

    return Citation(
        chunk_id=chunk_id,
        document_id=str(metadata["document_id"]),
        source_file=str(metadata["source_file"]),
        source_pages=source_pages,
        chunk_index=int(metadata["chunk_index"]),
    )