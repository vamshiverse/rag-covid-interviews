"""ACCESS POINT 2 - Response Engine (CLI).

Answer a single question against the knowledge base, showing the grounded
answer, citations, and the fallback decision.

Usage:
    python ask.py "What is MIS-C and how was it treated in children?"
"""
from __future__ import annotations

import sys

from rag_app.vectorstore import HybridIndex
from rag_app.retriever import HybridRetriever
from rag_app.response_engine import ResponseEngine


def main():
    if len(sys.argv) < 2:
        print('Usage: python ask.py "your question"')
        sys.exit(1)
    query = " ".join(sys.argv[1:])

    if not HybridIndex.exists():
        print("No index found. Run:  python build_index.py")
        sys.exit(1)

    engine = ResponseEngine(HybridRetriever(HybridIndex.load()))
    r = engine.answer(query)

    print("\n" + "=" * 78)
    print(f"Q: {query}")
    print("=" * 78)
    if r.is_fallback:
        print(f"\n[FALLBACK]  {r.answer}")
    else:
        print(f"\n{r.answer}\n")
        print("-" * 78)
        print("CITATIONS:")
        for c in r.citations:
            print(f"  • {c.doctor} ({c.country}) [{c.source_file}]  rel={c.relevance:.2f}")
            print(f"      “{c.quote}”")
    print("-" * 78)
    print(f"confidence={r.confidence:.3f}  provider={r.provider}  "
          f"latency={r.timings_ms.get('total')} ms")


if __name__ == "__main__":
    main()
