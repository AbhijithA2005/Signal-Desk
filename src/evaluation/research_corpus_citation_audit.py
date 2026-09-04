import json
import re
from pathlib import Path


INPUT_PATH = Path(
    "data/evaluation/research_corpus_generation_results.jsonl"
)

OUTPUT_PATH = Path(
    "data/evaluation/research_corpus_citation_audit.jsonl"
)

SOURCE_PATTERN = re.compile(
    r"\[SOURCE\s+(\d+)\]"
)


def load_records():
    return [
        json.loads(line)
        for line in INPUT_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def audit_record(record):
    answer = record["answer"]
    evidence = record["evidence"]

    matches = SOURCE_PATTERN.findall(answer)
    source_numbers = [int(value) for value in matches]

    valid = [
        number
        for number in source_numbers
        if 1 <= number <= len(evidence)
    ]

    invalid = [
        number
        for number in source_numbers
        if number < 1 or number > len(evidence)
    ]

    unique_valid = sorted(set(valid))

    cited_chunks = [
        evidence[number - 1]["citation"]["chunk_id"]
        for number in unique_valid
    ]

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(
            r"\n\s*\n",
            answer,
        )
        if paragraph.strip()
    ]

    cited_paragraphs = [
        paragraph
        for paragraph in paragraphs
        if SOURCE_PATTERN.search(paragraph)
    ]

    return {
        "query_id": record["query_id"],
        "query": record["query"],
        "citation_count": len(source_numbers),
        "unique_citation_count": len(unique_valid),
        "valid_citations": valid,
        "invalid_citations": invalid,
        "cited_chunks": cited_chunks,
        "has_citation": bool(source_numbers),
        "all_citations_valid": not invalid,
        "cited_paragraph_count": len(cited_paragraphs),
        "paragraph_count": len(paragraphs),
        "citation_coverage": (
            len(cited_paragraphs) / len(paragraphs)
            if paragraphs
            else 0.0
        ),
    }


def main():
    records = load_records()

    audits = [
        audit_record(record)
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

        for audit in audits:
            output.write(
                json.dumps(
                    audit,
                    ensure_ascii=False,
                )
                + "\n"
            )

    total = len(audits)
    with_citations = sum(
        audit["has_citation"]
        for audit in audits
    )
    fully_valid = sum(
        audit["all_citations_valid"]
        for audit in audits
    )

    mean_coverage = (
        sum(
            audit["citation_coverage"]
            for audit in audits
        ) / total
        if total
        else 0.0
    )

    total_citations = sum(
        audit["citation_count"]
        for audit in audits
    )

    invalid_citations = sum(
        len(audit["invalid_citations"])
        for audit in audits
    )

    print("Dataset: research_corpus_generation_dev_v1")
    print(f"Queries: {total}")
    print()
    print(
        f"Answers with >=1 citation: "
        f"{with_citations}/{total} "
        f"({with_citations / total:.3f})"
    )
    print(
        f"Answers with all citations valid: "
        f"{fully_valid}/{total} "
        f"({fully_valid / total:.3f})"
    )
    print(
        f"Total citations: {total_citations}"
    )
    print(
        f"Invalid citations: {invalid_citations}"
    )
    print(
        f"Mean paragraph citation coverage: "
        f"{mean_coverage:.3f}"
    )
    print()
    print("Per-query results:")

    for audit in audits:
        print(
            f"{audit['query_id']}: "
            f"citations={audit['citation_count']}, "
            f"unique={audit['unique_citation_count']}, "
            f"valid={audit['all_citations_valid']}, "
            f"coverage={audit['citation_coverage']:.3f}"
        )

    print()
    print(
        f"Saved audit to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
