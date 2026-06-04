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
                 sparse_weight: float = config.SPARSE_WEIGHT,
                 use_mmr: bool = config.USE_MMR,
                 mmr_lambda: float = config.MMR_LAMBDA):
        self.index = index
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda

    # --------------------------------------------------- metadata facets ----
    def facet_values(self, field: str) -> list[str]:
        """Distinct values of a metadata field, for UI filter dropdowns."""
        vals = set()
        for c in self.index.chunks:
            v = c.get(field)
            if isinstance(v, list):
                vals.update(v)
            elif v:
                vals.add(v)
        return sorted(vals)

    def _matching_positions(self, filters: dict | None) -> set[int] | None:
        """Positions whose metadata satisfy ALL filters. None = no filter."""
        if not filters:
            return None
        keep = set()
        for i, c in enumerate(self.index.chunks):
            ok = True
            for field, wanted in filters.items():
                if wanted in (None, "", [], "All"):
                    continue
                wanted_set = set(wanted) if isinstance(wanted, (list, tuple, set)) else {wanted}
                have = c.get(field)
                have_set = set(have) if isinstance(have, list) else {have}
                if not (wanted_set & have_set):
                    ok = False
                    break
            if ok:
                keep.add(i)
        return keep

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

    @staticmethod
    def _rank_map_over(scores: np.ndarray, allowed: set[int] | None, k: int) -> dict[int, int]:
        order = np.argsort(-scores)
        if allowed is not None:
            order = [p for p in order if int(p) in allowed]
        return {int(pos): rank + 1 for rank, pos in enumerate(order[:k])}

    def _mmr_select(self, pool: list[int], fused: dict[int, float], top_k: int,
                    lambda_: float) -> list[int]:
        """Maximal Marginal Relevance over the candidate pool.

        Relevance = the HYBRID (fused) score, min-max normalised to [0,1] so it
        is on the same scale as the cosine redundancy term. Redundancy = max
        cosine similarity (dense space) to an already-selected chunk. This keeps
        the BM25 contribution in the ranking and only trades a little relevance
        to avoid near-duplicate chunks.
        """
        M = self.index.dense_matrix
        vals = [fused[p] for p in pool]
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        rel = {p: (fused[p] - lo) / span for p in pool}
        selected: list[int] = []
        remaining = list(pool)
        while remaining and len(selected) < top_k:
            if not selected:
                best = max(remaining, key=lambda p: rel[p])
            else:
                def mmr_score(p):
                    redundancy = max(float(M[p] @ M[s]) for s in selected)
                    return lambda_ * rel[p] - (1.0 - lambda_) * redundancy
                best = max(remaining, key=mmr_score)
            selected.append(best)
            remaining.remove(best)
        return selected

    # ----------------------------------------------------- hybrid search ----
    def search(self, query: str, top_k: int = config.TOP_K,
               filters: dict | None = None,
               use_mmr: bool | None = None,
               mmr_lambda: float | None = None) -> list[RetrievedChunk]:
        use_mmr = self.use_mmr if use_mmr is None else use_mmr
        mmr_lambda = self.mmr_lambda if mmr_lambda is None else mmr_lambda

        dense = self._dense_scores(query)
        sparse = self._sparse_scores(query)
        allowed = self._matching_positions(filters)   # None unless filtering

        # Rank each signal (restricted to metadata-matching chunks if filtering).
        dense_ranks = self._rank_map_over(dense, allowed, self.candidate_k)
        sparse_ranks = self._rank_map_over(sparse, allowed, self.candidate_k)

        # Reciprocal Rank Fusion over the union of both candidate sets.
        fused: dict[int, float] = {}
        for pos, r in dense_ranks.items():
            fused[pos] = fused.get(pos, 0.0) + self.dense_weight / (self.rrf_k + r)
        for pos, r in sparse_ranks.items():
            fused[pos] = fused.get(pos, 0.0) + self.sparse_weight / (self.rrf_k + r)
        if not fused:
            return []

        # Candidate pool ranked by fused relevance, then optionally MMR-diversified.
        pool = [p for p, _ in sorted(fused.items(), key=lambda x: -x[1])][:self.candidate_k]
        if use_mmr and len(pool) > 1:
            chosen = self._mmr_select(pool, fused, top_k, mmr_lambda)
        else:
            chosen = pool[:top_k]

        results = []
        for pos in chosen:
            results.append(RetrievedChunk(
                chunk=self.index.chunks[pos],
                fused_score=float(fused.get(pos, 0.0)),
                dense_score=float(dense[pos]),
                sparse_score=float(sparse[pos]),
                dense_rank=dense_ranks.get(pos),
                sparse_rank=sparse_ranks.get(pos),
                debug={"mmr": use_mmr},
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
