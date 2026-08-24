from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from nltk.tokenize import sent_tokenize
from tokenizers import Tokenizer

from src.ingestion.pdf_loader import PageDocument


@dataclass
class Chunk:
    """A retrieval chunk with complete source provenance."""

    chunk_id: str
    document_id: str
    source_file: str
    source_pages: List[int]
    chunk_index: int
    text: str
    token_count: int


@dataclass
class SentenceRecord:
    """A sentence plus its source page and token count."""

    text: str
    page_number: int
    token_count: int


def _token_count(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text).ids)


def _split_long_sentence(
    sentence: SentenceRecord,
    tokenizer: Tokenizer,
    max_tokens: int,
) -> List[SentenceRecord]:
    """
    Fallback for a single sentence longer than max_tokens.

    Such a sentence must be split because retaining it intact would
    violate the maximum chunk size.
    """
    encoding = tokenizer.encode(sentence.text)
    tokens = encoding.tokens

    pieces: List[SentenceRecord] = []

    for start in range(0, len(tokens), max_tokens):
        piece_tokens = tokens[start : start + max_tokens]
        piece_text = " ".join(piece_tokens)

        pieces.append(
            SentenceRecord(
                text=piece_text,
                page_number=sentence.page_number,
                token_count=len(piece_tokens),
            )
        )

    return pieces


def build_sentence_stream(
    pages: Sequence[PageDocument],
    tokenizer: Tokenizer,
    max_tokens: int = 800,
) -> List[SentenceRecord]:
    """Convert ordered PDF pages into one ordered sentence stream."""
    sentences: List[SentenceRecord] = []

    for page in pages:
        for sentence_text in sent_tokenize(page.text):
            sentence_text = sentence_text.strip()

            if not sentence_text:
                continue

            record = SentenceRecord(
                text=sentence_text,
                page_number=page.page_number,
                token_count=_token_count(
                    tokenizer,
                    sentence_text,
                ),
            )

            if record.token_count <= max_tokens:
                sentences.append(record)
            else:
                sentences.extend(
                    _split_long_sentence(
                        record,
                        tokenizer,
                        max_tokens,
                    )
                )

    return sentences


def _choose_overlap_start(
    sentences: Sequence[SentenceRecord],
    chunk_start: int,
    chunk_end: int,
    target_overlap: int,
) -> int:
    """
    Select a sentence-aligned suffix whose token count is
    closest to the requested overlap.

    The selected overlap may be slightly below or above the target.
    Sentence boundaries are never broken.
    """
    if chunk_end <= chunk_start:
        return chunk_end

    best_start = chunk_end - 1
    best_difference = float("inf")
    accumulated_tokens = 0

    for index in range(chunk_end - 1, chunk_start - 1, -1):
        accumulated_tokens += sentences[index].token_count

        difference = abs(accumulated_tokens - target_overlap)

        if difference < best_difference:
            best_difference = difference
            best_start = index

        # Once the overlap has passed the target substantially,
        # continuing farther backward cannot improve the local
        # sentence-boundary candidate.
        if accumulated_tokens > target_overlap:
            break

    return best_start
    """
    Select a sentence inside the current chunk so that the tail
    of the current chunk becomes the beginning of the next chunk.

    The chosen overlap is the largest sentence-aligned suffix whose
    token count does not exceed target_overlap. If no whole sentence
    fits, the final sentence is used as the minimum overlap.
    """
    if chunk_end <= chunk_start:
        return chunk_end

    overlap_tokens = 0
    overlap_start = chunk_end - 1

    for index in range(chunk_end - 1, chunk_start - 1, -1):
        sentence_tokens = sentences[index].token_count

        if (
            overlap_tokens + sentence_tokens > target_overlap
            and overlap_tokens > 0
        ):
            break

        overlap_tokens += sentence_tokens
        overlap_start = index

        if overlap_tokens >= target_overlap:
            break

    return overlap_start


def chunk_pages(
    pages: Sequence[PageDocument],
    tokenizer: Tokenizer,
    *,
    target_tokens: int = 700,
    min_tokens: int = 500,
    max_tokens: int = 800,
    overlap_tokens: int = 100,
) -> List[Chunk]:
    """
    Create sentence-aware chunks independently for each document.

    Documents never share chunk boundaries.

    Target:
        ~700 tokens

    Normal range:
        500–800 tokens

    Overlap:
        approximately 100 tokens, constrained to sentence boundaries

    Provenance:
        all source pages touched by each chunk
    """
    if not pages:
        return []

    if not (0 < min_tokens <= target_tokens <= max_tokens):
        raise ValueError(
            "Require 0 < min_tokens <= target_tokens <= max_tokens."
        )

    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError(
            "overlap_tokens must be >= 0 and < max_tokens."
        )

    # Group pages by document while preserving the original
    # deterministic ordering.
    documents: dict[str, List[PageDocument]] = {}

    for page in pages:
        documents.setdefault(page.document_id, []).append(page)

    all_chunks: List[Chunk] = []

    for document_id, document_pages in documents.items():
        sentences = build_sentence_stream(
            document_pages,
            tokenizer,
            max_tokens=max_tokens,
        )

        if not sentences:
            continue

        document_chunks: List[Chunk] = []
        start = 0

        while start < len(sentences):
            current_tokens = 0
            end = start

            while end < len(sentences):
                sentence_tokens = sentences[end].token_count

                if (
                    end > start
                    and current_tokens + sentence_tokens > max_tokens
                ):
                    break

                current_tokens += sentence_tokens
                end += 1

                if current_tokens >= target_tokens:
                    break

            # If the chunk is still too short, add sentences up to
            # min_tokens without exceeding max_tokens.
            while (
                current_tokens < min_tokens
                and end < len(sentences)
                and current_tokens + sentences[end].token_count <= max_tokens
            ):
                current_tokens += sentences[end].token_count
                end += 1

            # Safety fallback for an oversized individual sentence.
            if end == start:
                end += 1
                current_tokens = sentences[start].token_count

            chunk_sentences = sentences[start:end]

            chunk_text = " ".join(
                sentence.text
                for sentence in chunk_sentences
            )

            source_pages = sorted(
                {
                    sentence.page_number
                    for sentence in chunk_sentences
                }
            )

            chunk_index = len(document_chunks)

            document_chunks.append(
                Chunk(
                    chunk_id=(
                        f"{document_id}"
                        f"_C{chunk_index:03d}"
                    ),
                    document_id=document_id,
                    source_file=document_pages[0].source_file,
                    source_pages=source_pages,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    token_count=_token_count(
                        tokenizer,
                        chunk_text,
                    ),
                )
            )

            if end >= len(sentences):
                break

            next_start = _choose_overlap_start(
                sentences,
                start,
                end,
                overlap_tokens,
            )

            if next_start <= start:
                next_start = start + 1

            start = next_start

        all_chunks.extend(document_chunks)

    return all_chunks