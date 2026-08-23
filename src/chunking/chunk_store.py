from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.chunking.chunker import Chunk


def save_chunks_jsonl(
    chunks: Iterable[Chunk],
    output_path: str | Path,
) -> None:
    """
    Save retrieval chunks as JSON Lines while preserving
    all citation-relevant metadata.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as output_file:
        for chunk in chunks:
            record = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source_file": chunk.source_file,
                "source_pages": chunk.source_pages,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "token_count": chunk.token_count,
            }

            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )