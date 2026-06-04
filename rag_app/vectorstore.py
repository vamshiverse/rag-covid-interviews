"""Persistent hybrid index = dense embeddings + BM25 + chunks/metadata.

This is the "Search & Retrieval Engine" storage layer. It stores the processed
documents together with their extracted metadata and embeddings, and can be
saved to / loaded from disk so the (relatively expensive) ingestion +
embedding step only runs when documents change.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .embeddings import build_embedder, TfidfEmbedder

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HybridIndex:
    def __init__(self, chunks, dense_matrix, embedder, embedder_note=""):
        self.chunks = chunks
        self.dense_matrix = dense_matrix            # (N, d) L2-normalised
        self.embedder = embedder
        self.embedder_note = embedder_note
        self._tokenized = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self._tokenized)
        self.id_to_pos = {c["id"]: i for i, c in enumerate(chunks)}

    # --------------------------------------------------------------- build --
    @classmethod
    def build(cls, chunks, backend: str | None = None):
        embedder, note = build_embedder(backend)
        texts = [c["text"] for c in chunks]
        embedder.fit(texts)
        dense = embedder.encode(texts).astype(np.float32)
        return cls(chunks, dense, embedder, note)

    # --------------------------------------------------------- query helpers
    def encode_query(self, query: str) -> np.ndarray:
        return self.embedder.encode([query])[0].astype(np.float32)

    # -------------------------------------------------------- persistence ---
    def save(self, index_dir: Path | None = None):
        d = Path(index_dir or config.INDEX_DIR)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)
        np.save(d / "dense.npy", self.dense_matrix)
        meta = {"backend": self.embedder.kind, "name": self.embedder.name,
                "note": self.embedder_note, "n_chunks": len(self.chunks),
                "dim": int(self.dense_matrix.shape[1])}
        with open(d / "embedder_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if isinstance(self.embedder, TfidfEmbedder):
            with open(d / "tfidf.pkl", "wb") as f:
                pickle.dump(self.embedder, f)

    @classmethod
    def load(cls, index_dir: Path | None = None):
        d = Path(index_dir or config.INDEX_DIR)
        with open(d / "chunks.json", encoding="utf-8") as f:
            chunks = json.load(f)
        dense = np.load(d / "dense.npy")
        with open(d / "embedder_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        backend = meta["backend"]
        if backend == "tfidf":
            with open(d / "tfidf.pkl", "rb") as f:
                embedder = pickle.load(f)
        elif backend == "sentence-transformers":
            from .embeddings import SentenceTransformerEmbedder
            embedder = SentenceTransformerEmbedder()
        elif backend == "openai":
            from .embeddings import OpenAIEmbedder
            embedder = OpenAIEmbedder()
        else:
            raise ValueError(f"Unknown persisted backend: {backend}")
        return cls(chunks, dense, embedder, meta.get("note", ""))

    @staticmethod
    def exists(index_dir: Path | None = None) -> bool:
        d = Path(index_dir or config.INDEX_DIR)
        return (d / "chunks.json").exists() and (d / "dense.npy").exists()
