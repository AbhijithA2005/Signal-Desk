from tokenizers import Tokenizer

from src.chunking.chunker import chunk_pages
from src.chunking.chunk_store import save_chunks_jsonl
from src.ingestion.corpus_loader import load_corpus


RAW_DIRECTORY = "data/raw"
OUTPUT_PATH = "data/processed/research_corpus_chunks.jsonl"


tokenizer = Tokenizer.from_pretrained(
    "bert-base-uncased"
)

pages = load_corpus(RAW_DIRECTORY)

chunks = chunk_pages(
    pages,
    tokenizer,
    target_tokens=700,
    min_tokens=500,
    max_tokens=800,
    overlap_tokens=100,
)

save_chunks_jsonl(
    chunks,
    OUTPUT_PATH,
)

print("Research corpus chunking: PASSED")
print("Pages:", len(pages))
print("Chunks:", len(chunks))
print("Output:", OUTPUT_PATH)
print(
    "Total chunk tokens:",
    sum(chunk.token_count for chunk in chunks),
)

from collections import Counter

document_counts = Counter(
    chunk.document_id
    for chunk in chunks
)

print("Chunks by document:")
for document_id in sorted(document_counts):
    print(
        f"  {document_id}: "
        f"{document_counts[document_id]}"
    )
