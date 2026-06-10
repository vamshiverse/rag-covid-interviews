"""ACCESS POINT 1 - Ingestion & Processing Pipeline (CLI).

Ingests every transcript in the data folder, chunks it (Q&A kept together),
embeds + indexes it, and persists the hybrid index to disk so the Response
Engine and Evaluation Pipeline can load it instantly.

Usage:
    python build_index.py                # build with the configured backend
    python build_index.py --backend tfidf
"""
from __future__ import annotations

import argparse
import json
import time

from rag_app import config
from rag_app.ingestion import ingest_corpus_with_docs, corpus_stats
from rag_app.vectorstore import HybridIndex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None,
                    help="embedding backend: auto|sentence-transformers|tfidf|openai")
    args = ap.parse_args()

    print(f"[1/3] Ingesting transcripts from: {config.DATA_DIR}")
    t0 = time.perf_counter()
    chunks, documents = ingest_corpus_with_docs()
    stats = corpus_stats(chunks)
    print(json.dumps(stats, indent=2))

    print(f"\n[2/3] Building hybrid index (embeddings + BM25) ...")
    index = HybridIndex.build(chunks, backend=args.backend, documents=documents)
    print(f"      embedder: {index.embedder.name}  ({index.embedder_note})")

    print(f"\n[3/3] Persisting index to: {config.INDEX_DIR}")
    index.save()
    dt = time.perf_counter() - t0
    print(f"\nDone in {dt:.1f}s. Indexed {len(chunks)} chunks from "
          f"{stats['n_documents']} documents.")


if __name__ == "__main__":
    main()
