"""Evaluation Pipeline.

Scores the Response Engine against the Golden Benchmark across the THREE stages
the spec calls out, using the RAG Triad plus supporting KPIs:

  (4a) RETRIEVING chunks  -> Quality of Retrieval
        * Context Relevance (RAG Triad)   - are retrieved chunks on-topic?
        * Hit Rate @k, MRR                - did we fetch the gold chunks, how high?
        * Context Precision / Recall @k   - signal vs noise in the context

  (4b) WRITING answers    -> Answer Relevance & Quality
        * Answer Relevance (RAG Triad)    - does the answer address the question?
        * Answer Correctness              - does it match the ideal answer?

  (4c) GROUNDING answers  -> Citation Accuracy
        * Groundedness / Faithfulness (RAG Triad) - is every claim supported?
        * Citation Grounding              - do quoted lines really appear in sources?
        * Citation Source P/R             - are cited sources the expected ones?

Plus operational KPIs: Fallback Correctness (did it abstain exactly when it
should) and Latency (ms per stage).

Two judge backends:
  * "math"  (default, no key) - embedding cosine + lexical overlap proxies.
  * "llm"   (if a provider key is set) - LLM-as-judge feedback functions,
            mirroring the TruLens RAG Triad from the Advanced RAG course.
"""
from __future__ import annotations

import json
import re
import time
from statistics import mean

import numpy as np

from . import config
from .response_engine import ResponseEngine, AnswerResult, split_sentences


# ----------------------------------------------------------- helpers --------
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def _token_overlap(a: str, b: str) -> float:
    sa, sb = set(_norm(a).split()), set(_norm(b).split())
    if not sa:
        return 0.0
    return len(sa & sb) / len(sa)


def load_golden(path=None) -> list[dict]:
    path = path or config.GOLDEN_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)["items"]


# =====================================================================
#  JUDGE  (math by default, LLM if available)
# =====================================================================
class Judge:
    """Computes the quality scores. `embedder` is the index's embedder so the
    judge speaks the same vector space as retrieval."""

    def __init__(self, embedder, use_llm: bool = False, llm=None):
        self.embedder = embedder
        self.use_llm = use_llm and llm is not None
        self.llm = llm

    # ---- shared embedding utility ----
    def _sim(self, a: str, b: str) -> float:
        va, vb = self.embedder.encode([a, b])
        return float(np.dot(va, vb))

    def _max_sim_to_set(self, text: str, pool_vecs: np.ndarray) -> float:
        v = self.embedder.encode([text])[0]
        return float(np.max(pool_vecs @ v)) if len(pool_vecs) else 0.0

    # ---- RAG Triad: Context Relevance ----
    def context_relevance(self, query, result: AnswerResult) -> float:
        if not result.retrieved:
            return 0.0
        if self.use_llm:
            return self._llm_context_relevance(query, result)
        # math proxy: mean cosine of retrieved chunks to the query
        return float(mean(max(0.0, rc.dense_score) for rc in result.retrieved))

    # ---- RAG Triad: Groundedness / Faithfulness ----
    def groundedness(self, result: AnswerResult) -> float:
        if result.is_fallback or not result.used_chunks:
            return float("nan")
        if self.use_llm:
            return self._llm_groundedness(result)
        # math proxy: every answer sentence should be close to some context sentence
        ctx_sents = []
        for rc in result.used_chunks:
            ctx_sents.extend(split_sentences(rc.chunk["answer"]))
        if not ctx_sents:
            return 0.0
        ctx_vecs = self.embedder.encode(ctx_sents)
        ans_sents = split_sentences(result.answer) or [result.answer]
        scores = [self._max_sim_to_set(s, ctx_vecs) for s in ans_sents]
        return float(mean(scores))

    # ---- RAG Triad: Answer Relevance ----
    def answer_relevance(self, query, result: AnswerResult) -> float:
        if result.is_fallback:
            return float("nan")
        if self.use_llm:
            return self._llm_answer_relevance(query, result)
        return max(0.0, self._sim(query, result.answer))

    # ---- Answer Correctness vs the ideal answer ----
    def answer_correctness(self, result: AnswerResult, expected: str) -> float:
        if result.is_fallback:
            return float("nan")
        if self.use_llm:
            return self._llm_correctness(result, expected)
        return max(0.0, self._sim(expected, result.answer))

    # ---------------- optional LLM feedback functions ----------------
    def _ask_score(self, instruction: str) -> float:
        raw = self.llm.complete(
            "You are a strict evaluator. Reply with ONLY a number from 0 to 10.",
            instruction, max_tokens=5)
        m = re.search(r"[\d.]+", raw)
        return min(1.0, float(m.group()) / 10.0) if m else 0.0

    def _llm_context_relevance(self, query, result):
        ctx = "\n".join(f"- {rc.chunk['answer'][:400]}" for rc in result.retrieved)
        return self._ask_score(
            f"How relevant is this retrieved context to the QUESTION?\n"
            f"QUESTION: {query}\nCONTEXT:\n{ctx}\nScore 0-10:")

    def _llm_groundedness(self, result):
        ctx = "\n".join(rc.chunk["answer"] for rc in result.used_chunks)
        return self._ask_score(
            f"Is every claim in the ANSWER supported by the CONTEXT (no made-up "
            f"facts)?\nCONTEXT:\n{ctx}\nANSWER:\n{result.answer}\nScore 0-10:")

    def _llm_answer_relevance(self, query, result):
        return self._ask_score(
            f"How well does the ANSWER address the QUESTION?\nQUESTION: {query}\n"
            f"ANSWER: {result.answer}\nScore 0-10:")

    def _llm_correctness(self, result, expected):
        return self._ask_score(
            f"How well does the ANSWER match the REFERENCE answer in meaning?\n"
            f"REFERENCE: {expected}\nANSWER: {result.answer}\nScore 0-10:")


# =====================================================================
#  RETRIEVAL + CITATION METRICS  (reference-based, no model needed)
# =====================================================================
def retrieval_metrics(result: AnswerResult, expected_ids: list[str], k: int) -> dict:
    retrieved_ids = [rc.id for rc in result.retrieved][:k]
    gold = set(expected_ids)
    if not gold:
        return {"hit_rate": float("nan"), "mrr": float("nan"),
                "context_precision": float("nan"), "context_recall": float("nan")}
    hit = any(rid in gold for rid in retrieved_ids)
    mrr = 0.0
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold:
            mrr = 1.0 / rank
            break
    n_rel = sum(1 for rid in retrieved_ids if rid in gold)
    precision = n_rel / len(retrieved_ids) if retrieved_ids else 0.0
    recall = n_rel / len(gold)
    return {"hit_rate": float(hit), "mrr": mrr,
            "context_precision": precision, "context_recall": recall}


def citation_metrics(result: AnswerResult, expected_ids: list[str]) -> dict:
    if result.is_fallback or not result.citations:
        return {"citation_grounding": float("nan"),
                "citation_source_precision": float("nan"),
                "citation_source_recall": float("nan"),
                "n_citations": 0}
    # grounding: does each quoted line actually occur in its cited source chunk?
    chunk_by_id = {rc.id: rc.chunk for rc in result.retrieved}
    grounded = 0
    for c in result.citations:
        src = chunk_by_id.get(c.chunk_id, {})
        haystack = _norm(src.get("answer", ""))
        grounded += 1 if _token_overlap(c.quote, haystack) >= 0.8 else 0
    grounding = grounded / len(result.citations)
    # cited sources vs expected sources
    cited_ids = {c.chunk_id for c in result.citations}
    gold = set(expected_ids)
    if gold:
        inter = len(cited_ids & gold)
        c_prec = inter / len(cited_ids) if cited_ids else 0.0
        c_rec = inter / len(gold)
    else:
        c_prec = c_rec = float("nan")
    return {"citation_grounding": grounding,
            "citation_source_precision": c_prec,
            "citation_source_recall": c_rec,
            "n_citations": len(result.citations)}


# =====================================================================
#  DRIVER
# =====================================================================
def evaluate_item(engine: ResponseEngine, judge: Judge, item: dict, k: int) -> dict:
    t0 = time.perf_counter()
    result = engine.answer(item["question"], top_k=k)
    wall_ms = round((time.perf_counter() - t0) * 1000, 1)

    answerable = item.get("answerable", True)
    expected_ids = item.get("expected_source_ids", [])

    # operational: fallback correctness
    should_fallback = not answerable
    fallback_correct = (result.is_fallback == should_fallback)

    metrics = {
        # RAG Triad
        "context_relevance": judge.context_relevance(item["question"], result),
        "groundedness": judge.groundedness(result),
        "answer_relevance": judge.answer_relevance(item["question"], result),
        # answer quality
        "answer_correctness": judge.answer_correctness(result, item.get("expected_answer", "")),
        # retrieval
        **retrieval_metrics(result, expected_ids, k),
        # citations / grounding
        **citation_metrics(result, expected_ids),
        # operational
        "fallback_correct": float(fallback_correct),
        "latency_ms_total": result.timings_ms.get("total", wall_ms),
        "latency_ms_retrieval": result.timings_ms.get("retrieval", 0.0),
        "latency_ms_generation": result.timings_ms.get("generation", 0.0),
    }
    return {
        "id": item["id"],
        "question": item["question"],
        "type": item.get("type", ""),
        "difficulty": item.get("difficulty", ""),
        "answerable": answerable,
        "expected_answer": item.get("expected_answer", ""),
        "result": result.to_dict(),
        "metrics": metrics,
        "is_fallback": result.is_fallback,
        "fallback_correct": fallback_correct,
    }


def _nanmean(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return round(float(mean(vals)), 4) if vals else float("nan")


def aggregate(per_item: list[dict]) -> dict:
    keys = ["context_relevance", "groundedness", "answer_relevance",
            "answer_correctness", "hit_rate", "mrr", "context_precision",
            "context_recall", "citation_grounding", "citation_source_precision",
            "citation_source_recall", "fallback_correct",
            "latency_ms_total", "latency_ms_retrieval", "latency_ms_generation"]
    agg = {k: _nanmean([it["metrics"][k] for it in per_item]) for k in keys}

    # group the RAG Triad nicely + a single headline score
    triad = {
        "context_relevance": agg["context_relevance"],
        "groundedness": agg["groundedness"],
        "answer_relevance": agg["answer_relevance"],
    }
    triad_vals = [v for v in triad.values() if not np.isnan(v)]
    rag_triad_score = round(float(mean(triad_vals)), 4) if triad_vals else float("nan")

    # by question type
    by_type: dict[str, dict] = {}
    types = sorted({it["type"] for it in per_item})
    for t in types:
        subset = [it for it in per_item if it["type"] == t]
        by_type[t] = {
            "n": len(subset),
            "answer_correctness": _nanmean([s["metrics"]["answer_correctness"] for s in subset]),
            "context_recall": _nanmean([s["metrics"]["context_recall"] for s in subset]),
            "fallback_correct": _nanmean([s["metrics"]["fallback_correct"] for s in subset]),
        }

    answerable_items = [it for it in per_item if it["answerable"]]
    return {
        "n_items": len(per_item),
        "n_answerable": len(answerable_items),
        "n_fallback_expected": len(per_item) - len(answerable_items),
        "rag_triad": triad,
        "rag_triad_score": rag_triad_score,
        "all_metrics": agg,
        "by_type": by_type,
    }


def run_evaluation(engine: ResponseEngine, judge: Judge, golden: list[dict],
                   k: int = config.TOP_K, progress=None) -> dict:
    per_item = []
    for i, item in enumerate(golden):
        per_item.append(evaluate_item(engine, judge, item, k))
        if progress:
            progress(i + 1, len(golden), item["id"])
    return {"per_item": per_item, "aggregate": aggregate(per_item)}


def build_judge(retriever) -> Judge:
    """Auto-select judge backend based on config + key availability."""
    use_llm = False
    llm = None
    backend = config.JUDGE_BACKEND
    want_llm = backend == "llm" or (backend == "auto" and config.LLM_PROVIDER in ("openai", "anthropic"))
    if want_llm:
        try:
            from .llm import LLMClient, llm_available
            if llm_available():
                llm = LLMClient()
                use_llm = True
        except Exception:
            use_llm = False
    return Judge(retriever.index.embedder, use_llm=use_llm, llm=llm)
