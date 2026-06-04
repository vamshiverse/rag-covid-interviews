"""Conversational RAG pipeline over COVID-19 doctor interview transcripts.

Modules:
    config            - central configuration & provider switch
    ingestion         - load PDF/DOCX, Q&A-aware chunking, metadata + topics
    embeddings        - pluggable embedder (sentence-transformers / TF-IDF / OpenAI)
    vectorstore       - persistent hybrid index (dense + BM25 + chunks + metadata)
    retriever         - hybrid retrieval with reciprocal-rank fusion
    llm               - thin provider-agnostic LLM client (optional)
    response_engine   - grounded answers with citations + fallback system
    evaluation        - RAG Triad + retrieval/citation/latency metrics
"""

__version__ = "1.0.0"
