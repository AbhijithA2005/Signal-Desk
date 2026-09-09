from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


INPUT_PATH = Path(
    "data/evaluation/research_corpus_generation_results.jsonl"
)

OUTPUT_PATH = Path(
    "data/evaluation/research_corpus_deterministic_citation_analysis.jsonl"
)

SOURCE_PATTERN = re.compile(
    r"\[SOURCE\s+\d+\]",
)

SENTENCE_PATTERN = re.compile(
    r"(?<=[.!?])\s+"
)


def load_records():
    return [
        json.loads(line)
        for line in INPUT_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def clean_answer(text: str) -> str:
    text = SOURCE_PATTERN.sub("", text)

    # Remove markdown headings.
    text = re.sub(
        r"(?m)^\s*#{1,6}\s*.*$",
        "",
        text,
    )

    # Remove markdown horizontal rules.
    text = re.sub(
        r"(?m)^\s*[-*_]{3,}\s*$",
        "",
        text,
    )

    return text.strip()


def extract_prose_sentences(text: str) -> list[str]:
    cleaned = clean_answer(text)

    sentences = []

    for paragraph in cleaned.splitlines():
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # Skip obvious labels, headings, and table rows.
        if paragraph.startswith("|"):
            continue

        paragraph = re.sub(
            r"^\s*[-*+]\s+",
            "",
            paragraph,
        )

        paragraph = re.sub(
            r"^\s*\d+[.)]\s+",
            "",
            paragraph,
        )

        parts = SENTENCE_PATTERN.split(paragraph)

        for part in parts:
            sentence = part.strip()

            if not sentence:
                continue

            if len(sentence.split()) < 5:
                continue

            sentences.append(sentence)

    return sentences


def analyze_record(
    record,
    model,
    threshold: float,
):
    sentences = extract_prose_sentences(
        record["answer"]
    )

    evidence = record["evidence"]

    if not sentences or not evidence:
        return {
            "query_id": record["query_id"],
            "sentence_count": len(sentences),
            "sentence_analysis": [],
        }

    sentence_embeddings = model.encode(
        sentences,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    evidence_texts = [
        item["text"]
        for item in evidence
    ]

    evidence_embeddings = model.encode(
        evidence_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    similarities = (
        sentence_embeddings
        @ evidence_embeddings.T
    )

    sentence_analysis = []

    for sentence, row in zip(
        sentences,
        similarities,
    ):
        best_index = int(np.argmax(row))
        best_score = float(row[best_index])

        sentence_analysis.append(
            {
                "sentence": sentence,
                "best_source_number": best_index + 1,
                "best_chunk_id": evidence[
                    best_index
                ]["citation"]["chunk_id"],
                "similarity": best_score,
                "supported_candidate": (
                    best_score >= threshold
                ),
            }
        )

    return {
        "query_id": record["query_id"],
        "query": record["query"],
        "sentence_count": len(sentences),
        "sentence_analysis": sentence_analysis,
    }


def main():
    records = load_records()

    model = SentenceTransformer(
        "BAAI/bge-small-en-v1.5",
        device="mps",
    )

    threshold = 0.55

    analyses = [
        analyze_record(
            record,
            model,
            threshold,
        )
        for record in records
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output:

        for analysis in analyses:
            output.write(
                json.dumps(
                    analysis,
                    ensure_ascii=False,
                )
                + "\n"
            )

    total_sentences = sum(
        item["sentence_count"]
        for item in analyses
    )

    candidate_sentences = sum(
        sum(
            1
            for sentence in item["sentence_analysis"]
            if sentence["supported_candidate"]
        )
        for item in analyses
    )

    coverage = (
        candidate_sentences / total_sentences
        if total_sentences
        else 0.0
    )

    print(
        "Dataset: research_corpus_generation_dev_v1"
    )
    print(
        f"Queries: {len(records)}"
    )
    print(
        f"Prose sentences analyzed: "
        f"{total_sentences}"
    )
    print(
        f"Similarity-supported candidates: "
        f"{candidate_sentences}"
    )
    print(
        f"Candidate provenance coverage: "
        f"{coverage:.3f}"
    )
    print(
        f"Threshold: {threshold:.2f}"
    )
    print(
        f"Saved analysis to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
