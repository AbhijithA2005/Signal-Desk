# Research RAG

A reproducible local Retrieval-Augmented Generation (RAG) research project evaluating dense retrieval, BM25, hybrid RRF retrieval, cross-encoder reranking, grounded generation, citation behavior, provenance alignment, and evidence-aware abstention.

## Pipeline

Research PDFs
-> PDF extraction
-> sentence-aware chunking
-> dense retrieval + BM25
-> Reciprocal Rank Fusion
-> cross-encoder reranking
-> evidence sufficiency gate
-> Qwen3:8b generation
-> grounded answer with citations

Uploaded PDFs are extracted, chunked, and added to both dense and BM25 retrieval. The chat also supports query-focused research notes with critical follow-up questions, plus direct image analysis using a local Ollama vision model.

## Corpus

Four research papers are currently benchmarked:

- RAG_001: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- RAG_002: Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
- RAG_003: ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT
- RAG_004: Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods

Processed corpus: 42 pages, 85 chunks, 55,497 chunk tokens.

## Components

- Dense retrieval: BAAI/bge-small-en-v1.5
- BM25: rank-bm25
- Hybrid retrieval: Reciprocal Rank Fusion
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- Generator: local qwen3:8b through Ollama
- Vector store: Chroma
- Python: 3.12

## Retrieval Results

20-query four-paper chunk-level development benchmark:

| Method | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| Dense | 0.600 | 0.750 | 0.850 | 0.692 |
| BM25 | 0.250 | 0.600 | 0.800 | 0.464 |
| Hybrid RRF | 0.600 | 0.850 | 0.850 | 0.717 |
| Hybrid + Reranker | 0.750 | 0.950 | 0.950 | 0.850 |

## Reranking Depth

| Depth | Hit@1 | Hit@3 | Hit@5 | MRR | Mean latency |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.700 | 0.850 | 0.850 | 0.775 | 35.89 ms |
| 10 | 0.750 | 0.900 | 0.900 | 0.825 | 51.13 ms |
| 20 | 0.800 | 1.000 | 1.000 | 0.900 | 90.42 ms |

Depth 20 is the strongest tested configuration on this development benchmark, with increased latency.

## Citation Experiment

| Prompt | Answers with citations | Citation adherence | Invalid citations |
|---|---:|---:|---:|
| Baseline | 3 / 12 | 0.250 | 0 |
| Strict citation | 1 / 12 | 0.083 | 0 |

The strict citation prompt was not adopted.

## Provenance Analysis

- 167 prose sentences analyzed
- 165 similarity-supported candidate matches
- candidate provenance coverage: 0.988

This is a semantic alignment signal, not a factuality or entailment metric.

## Evidence Sufficiency Gate

A reranker-score gate is applied before generation.

Development threshold: -2.0

| Measure | Result |
|---|---:|
| Answerable accepted | 12 / 12 |
| Unanswerable rejected | 8 / 8 |
| False abstentions | 0 |
| False acceptances | 0 |

The threshold was calibrated on development data and is not claimed to generalize universally.

## Reproduction

Install dependencies:

    python -m pip install -r requirements.txt

Run tests:

    python -m pytest -q

Build the corpus:

    python build_research_corpus.py

Run gated abstention evaluation:

    python -m src.evaluation.research_corpus_abstention_evaluator

Generation experiments require Ollama with qwen3:8b available locally.

The API's image-analysis route uses `qwen3-vl:4b` by default, a smaller local vision model for lower response latency. Pull it with `ollama pull qwen3-vl:4b`; set `OLLAMA_VISION_MODEL` to use another Ollama vision model. Image questions are analyzed directly and are not added to the searchable document corpus.

## Interactive Features

- Add PDF documents from the chat; new uploads are indexed for both dense and lexical search and can be selected, removed, or queried independently.
- The interactive assistant searches uploaded PDFs only. The bundled RAG papers and their indexes remain available to the evaluation scripts, not the live source picker or default API scope.
- General chat handles greetings and everyday questions without document citations; Ask, Notes, and Search use selected uploads and preserve source-grounded behavior.
- Use Notes for concise findings tailored to a question or decision need, with source citations and critical follow-up questions. Notes abstain when retrieved evidence fails the existing sufficiency gate.
- Grounded generation uses the top three reranked passages; on the current benchmark, Hit@3 matches Hit@5 while keeping the generation prompt smaller.
- Attach PNG, JPEG, GIF, or WebP images for direct visual analysis. Image uploads are limited to 10 MB and are not persisted.
- Expand answer sources to inspect the retrieved passage used to support each citation.

## Evaluation Scripts

Located under src/evaluation/:

- research_corpus_chunk_evaluator.py
- research_corpus_candidate_analysis.py
- research_corpus_rerank_depth.py
- research_corpus_rerank_tradeoff.py
- research_corpus_latency.py
- research_corpus_generation_evaluator.py
- research_corpus_citation_audit.py
- deterministic_citation_analysis.py
- research_corpus_abstention_evaluator.py

## Limitations

This is a controlled local research benchmark rather than a production-scale RAG service.

The main limitations are the small corpus, development-set evaluation, manually defined chunk-level gold evidence, environment-dependent latency, limited citation adherence, the distinction between semantic provenance and factual entailment, and the absence of independent held-out validation for the evidence-gate threshold.

The evidence gate and citations reduce risk but do not guarantee factual correctness. Notes and image analysis remain model-generated; verify important conclusions against the cited original passages or image, and validate the evidence threshold on held-out data before relying on it in high-stakes settings. Scanned PDFs without selectable text still require OCR, which is not currently included.

## Conclusion

The experiments support a modular retrieval-first RAG architecture. Hybrid retrieval improves evidence selection over BM25 alone, cross-encoder reranking further improves ranking quality, and deeper reranking trades additional latency for higher measured retrieval quality.

The evidence sufficiency gate addresses a distinct failure mode in which generation may proceed despite poor retrieved evidence. It performed perfectly on the current development abstention set, but independent held-out evaluation is required before generalizing the threshold or the observed quality levels.
