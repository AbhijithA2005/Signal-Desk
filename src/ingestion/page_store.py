from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.ingestion.pdf_loader import PageDocument


def save_pages_jsonl(
    pages: Iterable[PageDocument],
    output_path: str | Path,
) -> None:
    """
    Save page-level documents as JSON Lines.

    Each output line contains:
    document_id, source_file, page_number, and text.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as output_file:
        for page in pages:
            record = {
                "document_id": page.document_id,
                "source_file": page.source_file,
                "page_number": page.page_number,
                "text": page.text,
            }

            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )