from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])\s+"
)


class DeterministicCitationAttacher:
    """
    Attach retrieved-source citations to generated sentences using
    embedding similarity.

    This is a provenance heuristic, not a factuality proof.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "mps",
        threshold: float = 0.55,
    ):
        self.model = SentenceTransformer(
            model_name,
            device=device,
        )
        self.threshold = threshold

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        return [
            sentence.strip()
            for sentence in _SENTENCE_SPLIT.split(text.strip())
            if sentence.strip()
        ]

    def attach(
        self,
        answer: str,
        evidence: Sequence[dict],
    ) -> list[dict]:
        if not answer.strip():
            return []

        if not evidence:
            raise ValueError("Evidence cannot be empty.")

        sentences = self.split_sentences(answer)

        if not sentences:
            return []

        sentence_embeddings = self.model.encode(
            sentences,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        evidence_texts = [
            item["text"]
            for item in evidence
        ]

        evidence_embeddings = self.model.encode(
            evidence_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        similarity_matrix = (
            sentence_embeddings
            @ evidence_embeddings.T
        )

        attached = []

        for sentence, similarities in zip(
            sentences,
            similarity_matrix,
        ):
            best_index = int(
                np.argmax(similarities)
            )
            best_score = float(
                similarities[best_index]
            )

            source_number = (
                best_index + 1
                if best_score >= self.threshold
                else None
            )

            attached.append(
                {
                    "sentence": sentence,
                    "source_number": source_number,
                    "similarity": best_score,
                    "chunk_id": (
                        evidence[best_index]["citation"]["chunk_id"]
                        if source_number is not None
                        else None
                    ),
                }
            )

        return attached


def format_cited_answer(
    sentence_records: Sequence[dict],
) -> str:
    lines = []

    for record in sentence_records:
        sentence = record["sentence"]
        source_number = record["source_number"]

        if source_number is None:
            lines.append(sentence)
        else:
            lines.append(
                f"{sentence} [SOURCE {source_number}]"
            )

    return " ".join(lines)
