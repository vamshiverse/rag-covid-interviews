"""Hybrid Search & Retrieval Engine.

Combines a *dense* (semantic) ranking with a *sparse* (BM25 / lexical) ranking
using Reciprocal Rank Fusion (RRF). For this corpus that pairing matters:

  * Dense embeddings catch paraphrased / conceptual questions
    ("how did doctors cope with stress?" -> "moral distress", "burnout").
  * BM25 nails exact clinical terms & acronyms that embeddings often blur
    ("MIS-C", "SpO2 92%", "monoclonal antibody", "mid-day meal scheme").

RRF is robust because it fuses *ranks*, not raw scores, so the two very
differently-scaled signals combine cleanly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config
from .vectorstore import HybridIndex, tokenize


@dataclass
class RetrievedChunk:
    chunk: dict
    fused_score: float
    dense_score: float          # cosine similarity to query (0..1-ish)
    sparse_score: float         # raw BM25 score
    dense_rank: int | None      # 1-based rank in dense list (None if absent)
    sparse_rank: int | None
    debug: dict = field(default_factory=dict)

    @property
    def id(self):
        return self.chunk["id"]


class HybridRetriever:
    def __init__(self, index: HybridIndex,
                 candidate_k: int = config.CANDIDATE_K,
                 rrf_k: int = config.RRF_K,
                 dense_weight: float = config.DENSE_WEIGHT,
                 sparse_weight: float = config.SPARSE_WEIGHT):
        self.index = index
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    # ----------------------------------------------------- raw signals ------
    def _dense_scores(self, query: str) -> np.ndarray:
        qv = self.index.encode_query(query)
        return self.index.dense_matrix @ qv          # cosine (rows normalised)

    def _sparse_scores(self, query: str) -> np.ndarray:
        return np.asarray(self.index.bm25.get_scores(tokenize(query)),
                          dtype=np.float32)

    def max_dense_similarity(self, query: str) -> float:
        """Best semantic similarity of the query to ANY chunk in the corpus.
        Used by the fallback gate as a global 'is anything relevant?' signal,
        independent of which chunks survive rank fusion."""
        return float(np.max(self._dense_scores(query)))

    @staticmethod
    def _rank_map(scores: np.ndarray, k: int) -> dict[int, int]:
        """positions of top-k -> 1-based rank."""
        top = np.argsort(-scores)[:k]
        return {int(pos): rank + 1 for rank, pos in enumerate(top)}

    # ----------------------------------------------------- hybrid search ----
    def search(self, query: str, top_k: int = config.TOP_K) -> list[RetrievedChunk]:
        dense = self._dense_scores(query)
        sparse = self._sparse_scores(query)

        dense_ranks = self._rank_map(dense, self.candidate_k)
        sparse_ranks = self._rank_map(sparse, self.candidate_k)

        # Reciprocal Rank Fusion over the union of both candidate sets.
        fused: dict[int, float] = {}
        for pos, r in dense_ranks.items():
            fused[pos] = fused.get(pos, 0.0) + self.dense_weight / (self.rrf_k + r)
        for pos, r in sparse_ranks.items():
            fused[pos] = fused.get(pos, 0.0) + self.sparse_weight / (self.rrf_k + r)

        ordered = sorted(fused.items(), key=lambda x: -x[1])[:top_k]
        results = []
        for pos, score in ordered:
            results.append(RetrievedChunk(
                chunk=self.index.chunks[pos],
                fused_score=float(score),
                dense_score=float(dense[pos]),
                sparse_score=float(sparse[pos]),
                dense_rank=dense_ranks.get(pos),
                sparse_rank=sparse_ranks.get(pos),
            ))
        return results

    # ----------------------------------------------------- single-method ---
    def search_single(self, query: str, method: str, top_k: int = config.TOP_K):
        """For UI comparison: 'dense' or 'sparse' only."""
        scores = self._dense_scores(query) if method == "dense" else self._sparse_scores(query)
        top = np.argsort(-scores)[:top_k]
        out = []
        for rank, pos in enumerate(top):
            pos = int(pos)
            out.append(RetrievedChunk(
                chunk=self.index.chunks[pos],
                fused_score=float(scores[pos]),
                dense_score=float(self._dense_scores(query)[pos]) if method == "sparse" else float(scores[pos]),
                sparse_score=float(scores[pos]) if method == "sparse" else 0.0,
                dense_rank=rank + 1 if method == "dense" else None,
                sparse_rank=rank + 1 if method == "sparse" else None,
            ))
        return out
