from __future__ import annotations

from typing import Sequence


def build_grounded_prompt(
    query: str,
    evidence: Sequence[dict],
) -> str:
    """
    Build a prompt that restricts the model to retrieved evidence.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not evidence:
        raise ValueError("Evidence cannot be empty.")

    context_blocks = []

    for index, item in enumerate(evidence, start=1):
        citation = item["citation"]

        context_blocks.append(
            "\n".join(
                [
                    f"[SOURCE {index}]",
                    f"Document: {citation.document_id}",
                    f"Pages: {citation.page_label}",
                    f"Chunk: {citation.chunk_id}",
                    f"Text:\n{item['text']}",
                ]
            )
        )

    context = "\n\n".join(context_blocks)

    return f"""You are a grounded question-answering system.

Answer the user's question using ONLY the evidence provided below.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. Every factual claim must be supported by the evidence.
4. Cite the supporting source using [SOURCE N].
5. If the evidence is insufficient, say that the provided evidence is insufficient.
6. Give a direct, concise answer.

USER QUESTION:
{query}

RETRIEVED EVIDENCE:
{context}

ANSWER:
"""