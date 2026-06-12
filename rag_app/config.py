"""Central configuration and the provider switch.

The whole system is provider-agnostic. By default it runs with ZERO API keys:
  * embeddings  -> local `sentence-transformers` if installed, else TF-IDF
  * generation  -> deterministic *extractive* engine (returns the grounded
                   doctor answer + citations straight from the retrieved chunk)
  * judging     -> math-based metrics (semantic similarity / overlap)

Set environment variables to upgrade to a hosted LLM (answers AND an
LLM-as-judge RAG Triad), e.g.:

  $env:RAG_LLM_PROVIDER = "openai"      # or "anthropic"
  $env:OPENAI_API_KEY   = "sk-..."
  $env:RAG_EMBEDDING_BACKEND = "openai" # optional

Everything degrades gracefully if a key/library is missing.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths -----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Interview Transcripts (PDFs, Docx)"
INDEX_DIR = PROJECT_ROOT / "storage" / "index"
GOLDEN_PATH = PROJECT_ROOT / "rag_app" / "golden_dataset.json"

INDEX_DIR.mkdir(parents=True, exist_ok=True)


def _env(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val not in (None, "") else default


# ----------------------------------------------------------- providers ------
# generation backend: "extractive" (no key) | "openai" | "anthropic"
LLM_PROVIDER = _env("RAG_LLM_PROVIDER", "extractive").lower()

# embedding backend: "auto" | "sentence-transformers" | "tfidf" | "openai"
EMBEDDING_BACKEND = _env("RAG_EMBEDDING_BACKEND", "auto").lower()

# judge backend: "auto" -> use LLM if a provider key is set, else "math"
JUDGE_BACKEND = _env("RAG_JUDGE_BACKEND", "auto").lower()

# model names (only used when the matching provider is active)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_CHAT_MODEL = _env("RAG_OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = _env("RAG_OPENAI_EMBED_MODEL", "text-embedding-3-small")
ANTHROPIC_CHAT_MODEL = _env("RAG_ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-6")
ST_EMBED_MODEL = _env("RAG_ST_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------------------------------------- retrieval ------
TOP_K = int(_env("RAG_TOP_K", "5"))            # chunks returned to the answerer
CANDIDATE_K = int(_env("RAG_CANDIDATE_K", "20"))  # candidates pulled per method
RRF_K = int(_env("RAG_RRF_K", "60"))           # reciprocal-rank-fusion constant
DENSE_WEIGHT = float(_env("RAG_DENSE_WEIGHT", "1.0"))
SPARSE_WEIGHT = float(_env("RAG_SPARSE_WEIGHT", "1.0"))

# ----------------------------------------------------------- fallback -------
# If the best retrieved chunk's similarity to the query is below this, the
# response engine refuses to answer and explains why (anti-hallucination).
# Calibrated PER backend because cosine scales differently: dense semantic
# vectors sit ~0.3-0.7 for good matches, while sparse TF-IDF cosine is much
# lower for the same match. "auto" lets the engine pick by embedder kind.
FALLBACK_MIN_SIMILARITY = _env("RAG_FALLBACK_MIN_SIM", "auto")
# NOTE: a single similarity threshold robustly rejects clearly off-topic
# queries, but cannot separate "hard-but-answerable" from "on-topic-but-the-
# specific-fact-is-absent" (e.g. asking for a drug dose the transcripts never
# state). That second layer is the LLM groundedness check used in 'llm' mode.
FALLBACK_THRESHOLD_BY_BACKEND = {
    "sentence-transformers": 0.40,
    "openai": 0.40,
    "tfidf": 0.06,
}


def fallback_threshold(embedder_kind: str) -> float:
    """Resolve the fallback similarity threshold for the active embedder."""
    if FALLBACK_MIN_SIMILARITY != "auto":
        return float(FALLBACK_MIN_SIMILARITY)
    return FALLBACK_THRESHOLD_BY_BACKEND.get(embedder_kind, 0.20)


# --- HYBRID fallback gate (ON by default) -----------------------------------
# The gate runs on BOTH searches independently and abstains only if the dense
# (semantic) best-match AND the sparse (BM25) best-match both fall below their
# thresholds; if EITHER modality finds something relevant, it proceeds with that
# search. ("Fall back only when both fail.") Set RAG_HYBRID_FALLBACK=0 to revert
# to the dense-only gate.
#
# NOTE — calibrated on this corpus: BM25 best-scores fire on common words, so the
# 3 off-topic golden questions score BM25 11.8-13.6, right inside the real-question
# range (8.8-23.8). The hybrid gate therefore trades a bit of Fallback Correctness
# (~0.89 dense-only -> ~0.83) for never refusing a question that either search can
# support. This trade-off is accepted by design; the real fix for on-topic-but-
# absent questions is the LLM groundedness check in 'llm' mode.
HYBRID_FALLBACK_GATE = _env("RAG_HYBRID_FALLBACK", "1").lower() in ("1", "true", "yes", "on")
# BM25 best-score threshold for the sparse side of the hybrid gate. Unlike the
# dense cosine (0-1), BM25 is unbounded and corpus-specific — tune per corpus.
SPARSE_FALLBACK_THRESHOLD = float(_env("RAG_SPARSE_FALLBACK_MIN", "10.0"))


def active_config() -> dict:
    """A small dict for display in the UI / logs."""
    return {
        "llm_provider": LLM_PROVIDER,
        "embedding_backend": EMBEDDING_BACKEND,
        "judge_backend": JUDGE_BACKEND,
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "rrf_k": RRF_K,
        "fallback_min_similarity": FALLBACK_MIN_SIMILARITY,
        "fallback_thresholds_by_backend": FALLBACK_THRESHOLD_BY_BACKEND,
        "hybrid_fallback_gate": HYBRID_FALLBACK_GATE,
        "sparse_fallback_threshold": SPARSE_FALLBACK_THRESHOLD,
        "openai_key_present": bool(OPENAI_API_KEY),
        "anthropic_key_present": bool(ANTHROPIC_API_KEY),
    }
