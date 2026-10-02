from __future__ import annotations

from typing import Sequence


def build_grounded_prompt(
    query: str,
    evidence: Sequence[dict],
    insufficient_instruction: str = (
        "If the evidence is insufficient, say that the provided evidence is insufficient."
    ),
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
5. {insufficient_instruction}
6. Answer in 1-2 sentences first. Do not restate that answer in another section.
7. For a simple question, stop after the direct answer. For a complex question, add at most 3 short Markdown sections with up to 5 useful bullets total.
8. Use numbered lists only for ordered steps, bold sparingly for key terms, and avoid tables unless they materially improve a comparison.
9. Keep the default response under 180 words, use short paragraphs, and make qualifications or uncertainty explicit.

USER QUESTION:
{query}

RETRIEVED EVIDENCE:
{context}

ANSWER:
"""


def build_general_chat_prompt(query: str) -> str:
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    return f"""You are a friendly general-purpose assistant. Use a concise, professional tone and do not use emojis.

Respond naturally to greetings and everyday conversation. Answer general questions clearly and concisely using your learned knowledge. Do not claim to have searched the web, checked current facts, or read the user's documents. If a detail may be uncertain or out of date, say so. If the user asks about a specific document, explain that you can answer from uploaded documents when they switch to the source-grounded Ask mode.

USER MESSAGE:
{query}

RESPONSE:
"""


def build_research_notes_prompt(
    query: str,
    evidence: Sequence[dict],
) -> str:
    return build_grounded_prompt(
        query=(
            f"Create presentation-ready research notes tailored to this need: {query}\n\n"
            "Format the notes with a concise title, Key points, a short presentation "
            "outline, and Critical follow-up questions. Summarize every useful point "
            "supported by the supplied passages and cite each factual claim using "
            "[SOURCE N]. Do not invent missing details. If evidence is sparse, still "
            "provide the supported notes and turn uncovered details into specific "
            "follow-up questions rather than refusing the task."
        ),
        evidence=evidence,
        insufficient_instruction=(
            "For notes, do not refuse solely because evidence is limited. Summarize "
            "supported points and identify uncovered aspects as unknown follow-up questions."
        ),
    )