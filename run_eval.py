"""ACCESS POINT 3 - Evaluation Pipeline (CLI).

Runs the Response Engine over the golden benchmark and reports the RAG Triad
plus all supporting KPIs. Saves full results to storage/eval_results.json.

Usage:
    python run_eval.py
"""
from __future__ import annotations

import datetime
import json

from rag_app import config
from rag_app.vectorstore import HybridIndex
from rag_app.retriever import HybridRetriever
from rag_app.response_engine import ResponseEngine
from rag_app.evaluation import load_golden, build_judge, run_evaluation


def main():
    if not HybridIndex.exists():
        print("No index found. Run:  python build_index.py")
        return

    index = HybridIndex.load()
    retriever = HybridRetriever(index)
    engine = ResponseEngine(retriever)
    judge = build_judge(retriever)
    golden = load_golden()

    print(f"Running evaluation on {len(golden)} golden questions ...")
    print(f"  embedder={index.embedder.name}  generator={engine.provider}  "
          f"judge={'LLM' if judge.use_llm else 'math'}\n")

    def prog(i, n, qid):
        print(f"  [{i:2d}/{n}] {qid}")

    out = run_evaluation(engine, judge, golden, progress=prog)
    out["meta"] = {"embedder": index.embedder.name, "provider": engine.provider,
                   "judge_llm": judge.use_llm,
                   "generated_at": datetime.datetime.now().isoformat(timespec="seconds")}

    agg = out["aggregate"]
    m = agg["all_metrics"]
    print("\n" + "=" * 64)
    print("RAG TRIAD")
    for k, v in agg["rag_triad"].items():
        print(f"  {k:22s}: {v:.3f}")
    print(f"  {'RAG TRIAD (mean)':22s}: {agg['rag_triad_score']:.3f}")
    print("-" * 64)
    print("RETRIEVAL QUALITY")
    for k in ("hit_rate", "mrr", "context_recall", "context_precision", "context_relevance"):
        print(f"  {k:22s}: {m[k]:.3f}")
    print("-" * 64)
    print("ANSWER & GROUNDING")
    for k in ("answer_relevance", "answer_correctness", "citation_grounding",
              "citation_source_precision", "citation_source_recall"):
        print(f"  {k:22s}: {m[k]:.3f}")
    print("-" * 64)
    print("RELIABILITY / OPS")
    print(f"  {'fallback_correct':22s}: {m['fallback_correct']:.3f}")
    print(f"  {'latency_ms_total':22s}: {m['latency_ms_total']:.1f} ms")
    print("=" * 64)

    path = config.PROJECT_ROOT / "storage" / "eval_results.json"
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nFull results saved to: {path}")


if __name__ == "__main__":
    main()
