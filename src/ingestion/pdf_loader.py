from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pymupdf


@dataclass
class PageDocument:
    """Represents one extracted PDF page and its metadata."""

    document_id: str
    source_file: str
    page_number: int
    text: str


def load_pdf(pdf_path: str | Path) -> List[PageDocument]:
    """
    Extract text from a PDF page by page.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list containing one PageDocument per page.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the provided path is not a PDF.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, received: {path.name}")

    document_id = path.stem
    pages: List[PageDocument] = []

    with pymupdf.open(path) as pdf:
        for page_index, page in enumerate(pdf):
            text = page.get_text("text").strip()

            pages.append(
                PageDocument(
                    document_id=document_id,
                    source_file=path.name,
                    page_number=page_index + 1,
                    text=text,
                )
            )

    return pages