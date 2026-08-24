from __future__ import annotations

from pathlib import Path

from src.ingestion.pdf_loader import PageDocument, load_pdf


def load_corpus(
    raw_directory: str | Path,
) -> list[PageDocument]:
    """
    Load every PDF in a raw corpus directory.

    PDFs are processed in filename order so that the result
    is deterministic.
    """
    directory = Path(raw_directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Raw corpus directory not found: {directory}"
        )

    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            f"No PDF files found in {directory}"
        )

    pages: list[PageDocument] = []

    for pdf_path in pdf_files:
        pages.extend(load_pdf(pdf_path))

    return pages