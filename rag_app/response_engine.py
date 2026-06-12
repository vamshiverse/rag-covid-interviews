"""Response Engine: grounded answers with citations + a fallback system.

Pipeline for a single question:
    1. Retrieve top-k chunks via the hybrid retriever.
    2. Relevance gate -> if the best chunk is not similar enough to the query,
       REFUSE and return a fallback answer that states *why* (anti-hallucination).
    3. Generate a grounded answer:
         * extractive mode (default, no key): stitch the most query-relevant
           sentences from retrieved chunks -> the answer is, by construction,
           supported by the sources.
         * llm mode: an LLM answers strictly from the provided context and is
           instructed to fall back if the context is insufficient.
    4. Attach citations (quoted lines + doctor / file attribution).

Both grounding (citations) and fallback are first-class -> the two levers that
reduce hallucination and make the system trustworthy.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

from . import config
from .ingestion import _normalize_with_map
from .retriever import HybridRetriever, RetrievedChunk

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"^[QA]:\s*", "", text)
    sents = [s.strip() for s in _SENT_RE.split(text) if len(s.strip()) > 2]
    return sents


def locate_lines(chunk: dict, sentence: str) -> tuple[int | None, int | None]:
    """Map a cited sentence back to its literal source line range (1-indexed in
    the original PDF/DOCX). Returns (None, None) if it can't be located."""
    raw = chunk.get("raw_answer")
    base = chunk.get("answer_line_start")
    if not raw or not base:
        return (None, None)
    normalized, idx_map = _normalize_with_map(raw)
    pos = normalized.find(sentence)
    if pos < 0 or not idx_map:
        return (None, None)
    raw = raw.replace("\r", "\n")
    raw_start = idx_map[pos]
    raw_end = idx_map[min(pos + len(sentence) - 1, len(idx_map) - 1)]
    start_line = base + raw[:raw_start].count("\n")
    end_line = base + raw[:raw_end + 1].count("\n")
    return (start_line, end_line)


@dataclass
class Citation:
    chunk_id: str
    doctor: str
    country: str
    source_file: str
    quote: str
    relevance: float
    start_line: int | None = None
    end_line: int | None = None


@dataclass
class AnswerResult:
    query: str
    answer: str
    citations: list[Citation]
    used_chunks: list[RetrievedChunk]
    retrieved: list[RetrievedChunk]
    is_fallback: bool
    fallback_reason: str
    confidence: float
    provider: str
    timings_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "is_fallback": self.is_fallback,
            "fallback_reason": self.fallback_reason,
            "confidence": round(self.confidence, 3),
            "provider": self.provider,
            "citations": [c.__dict__ for c in self.citations],
            "used_chunk_ids": [c.id for c in self.used_chunks],
            "retrieved_chunk_ids": [c.id for c in self.retrieved],
            "timings_ms": self.timings_ms,
        }


FALLBACK_TOKEN = "INSUFFICIENT_CONTEXT"

LLM_SYSTEM = (
    "You are a careful research assistant answering questions about a corpus of "
    "COVID-19 doctor interview transcripts. You must answer ONLY using the "
    "provided context passages. Ground every claim in the context and quote the "
    "exact supporting lines. If the context does not contain enough information "
    f"to answer, reply with exactly '{FALLBACK_TOKEN}: <short reason>' and nothing "
    "else. Never use outside knowledge.")


class ResponseEngine:
    def __init__(self, retriever: HybridRetriever, provider: str | None = None,
                 fallback_min_sim: float | None = None,
                 top_k: int = config.TOP_K):
        self.retriever = retriever
        self.provider = (provider or config.LLM_PROVIDER).lower()
        # resolve the fallback threshold for the active embedding backend
        self.fallback_min_sim = (fallback_min_sim if fallback_min_sim is not None
                                 else config.fallback_threshold(retriever.index.embedder.kind))
        # optional hybrid fallback gate (dense + sparse, OFF by default)
        self.hybrid_gate = config.HYBRID_FALLBACK_GATE
        self.sparse_min = config.SPARSE_FALLBACK_THRESHOLD
        self.top_k = top_k
        self._llm = None
        if self.provider in ("openai", "anthropic"):
            from .llm import LLMClient, llm_available
            if llm_available(self.provider):
                self._llm = LLMClient(self.provider)
            else:
                self.provider = "extractive"   # key missing -> degrade

    # ------------------------------------------------------ public API ------
    def answer(self, query: str, top_k: int | None = None) -> AnswerResult:
        top_k = top_k or self.top_k
        timings = {}

        t0 = time.perf_counter()
        retrieved = self.retriever.search(query, top_k=top_k)
        # Global best signals drive the fallback gate (a truer "is anything in
        # the corpus relevant?" check than the fused top-k).
        best_sim = self.retriever.max_dense_similarity(query) if retrieved else 0.0
        best_sparse = (self.retriever.max_sparse_similarity(query)
                       if (retrieved and self.hybrid_gate) else 0.0)
        timings["retrieval"] = round((time.perf_counter() - t0) * 1000, 1)

        confidence = float(np.clip(best_sim, 0.0, 1.0))

        # ---- Fallback gate (anti-hallucination) ----
        # Default: dense-only. Hybrid (opt-in): abstain only if BOTH the dense
        # and the sparse (BM25) best-match fall below their thresholds — if either
        # search finds something relevant, proceed.
        dense_fail = best_sim < self.fallback_min_sim
        if self.hybrid_gate:
            sparse_fail = best_sparse < self.sparse_min
            do_fallback = (not retrieved) or (dense_fail and sparse_fail)
        else:
            do_fallback = (not retrieved) or dense_fail

        if do_fallback:
            if self.hybrid_gate:
                reason = (
                    "Neither search finds interview passages relevant to this "
                    f"question — dense best {best_sim:.2f} < {self.fallback_min_sim:.2f} "
                    f"AND BM25 best {best_sparse:.1f} < {self.sparse_min:.1f}. "
                    "It may be outside the scope of these COVID-19 doctor interviews.")
            else:
                reason = (
                    "The knowledge base does not contain interview passages that are "
                    f"semantically relevant to this question (best match similarity "
                    f"{best_sim:.2f} < threshold {self.fallback_min_sim:.2f}). "
                    "It may be outside the scope of these COVID-19 doctor interviews.")
            timings["generation"] = 0.0
            timings["total"] = timings["retrieval"]
            return AnswerResult(query, _fallback_text(reason), [], [], retrieved,
                                True, reason, confidence,
                                self.provider, timings)

        # ---- Generation ----
        t1 = time.perf_counter()
        if self._llm is not None:
            answer, citations, is_fb, fb_reason, used = self._generate_llm(query, retrieved)
        else:
            answer, citations, is_fb, fb_reason, used = self._generate_extractive(query, retrieved)
        timings["generation"] = round((time.perf_counter() - t1) * 1000, 1)
        timings["total"] = round(timings["retrieval"] + timings["generation"], 1)

        return AnswerResult(query, answer, citations, used, retrieved,
                            is_fb, fb_reason, confidence, self.provider, timings)

    # ------------------------------------------------ extractive generator --
    def _generate_extractive(self, query, retrieved):
        """Stitch the most query-relevant sentences from the top chunks."""
        qv = self.retriever.index.encode_query(query)

        scored = []   # (sim, chunk_idx_in_retrieved, sentence)
        for ci, rc in enumerate(retrieved):
            sents = split_sentences(rc.chunk["answer"])
            if not sents:
                continue
            svecs = self.retriever.index.embedder.encode(sents)
            sims = svecs @ qv
            for s, sim in zip(sents, sims):
                scored.append((float(sim), ci, s))
        scored.sort(key=lambda x: -x[0])

        # take the best few sentences but keep them grounded in 1-2 top sources
        top_sents = scored[:4]
        used_idx = sorted({ci for _, ci, _ in top_sents})
        used = [retrieved[i] for i in used_idx]

        # build a readable, source-attributed answer
        citations, lines = [], []
        seen = set()
        for sim, ci, sent in top_sents:
            rc = retrieved[ci]
            key = (rc.id, sent[:40])
            if key in seen:
                continue
            seen.add(key)
            sl, el = locate_lines(rc.chunk, sent)
            citations.append(Citation(rc.id, rc.chunk["doctor"], rc.chunk["country"],
                                       rc.chunk["source_file"], sent, round(sim, 3),
                                       sl, el))
            lines.append(sent)

        docs = sorted({retrieved[ci].chunk["doctor"] for _, ci, _ in top_sents})
        lead = ("Based on the interview" + ("s" if len(docs) > 1 else "") +
                " with " + ", ".join(docs) + ":")
        body = " ".join(lines)
        answer = f"{lead}\n\n{body}"
        return answer, citations, False, "", used

    # ------------------------------------------------------ llm generator ---
    def _generate_llm(self, query, retrieved):
        context = "\n\n".join(
            f"[{i+1}] (Dr. {rc.chunk['doctor'].replace('Dr. ', '')}, "
            f"{rc.chunk['country']}, {rc.chunk['source_file']})\n{rc.chunk['answer']}"
            for i, rc in enumerate(retrieved))
        user = (f"Context passages:\n{context}\n\nQuestion: {query}\n\n"
                "Answer grounded ONLY in the context. After your answer, add a "
                "'Citations:' section quoting the exact supporting lines with their "
                "[number].")
        raw = self._llm.complete(LLM_SYSTEM, user)

        if raw.startswith(FALLBACK_TOKEN):
            reason = raw.split(":", 1)[1].strip() if ":" in raw else "Context insufficient."
            return _fallback_text(reason), [], True, reason, []

        # map quoted [n] back to chunks for citation objects
        used_nums = sorted({int(n) for n in re.findall(r"\[(\d+)\]", raw)
                            if 1 <= int(n) <= len(retrieved)})
        used = [retrieved[n - 1] for n in used_nums] or retrieved[:1]
        qv = self.retriever.index.encode_query(query)
        citations = []
        for rc in used:
            sents = split_sentences(rc.chunk["answer"]) or [rc.chunk["answer"]]
            svecs = self.retriever.index.embedder.encode(sents)
            best = int(np.argmax(svecs @ qv))
            sl, el = locate_lines(rc.chunk, sents[best])
            citations.append(Citation(rc.id, rc.chunk["doctor"], rc.chunk["country"],
                                       rc.chunk["source_file"], sents[best],
                                       round(float((svecs @ qv)[best]), 3), sl, el))
        return raw, citations, False, "", used


def _fallback_text(reason: str) -> str:
    return ("I'm not able to answer this question from the available interview "
            f"transcripts. Reason: {reason}")
