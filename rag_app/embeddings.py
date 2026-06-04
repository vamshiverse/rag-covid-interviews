"""Pluggable embedding backend.

Priority (when EMBEDDING_BACKEND == "auto"):
    1. sentence-transformers  (local, free, semantic)   <- preferred
    2. TF-IDF                 (local, free, lexical)     <- always-available fallback
    3. openai                 (only if RAG_EMBEDDING_BACKEND=openai + key)

The embedder exposes a uniform interface:
    .fit(corpus_texts)             # needed by TF-IDF; no-op for the others
    .encode(list[str]) -> ndarray  # L2-normalised row vectors
    .name                          # human-readable backend id
"""
from __future__ import annotations

import numpy as np

from . import config


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class SentenceTransformerEmbedder:
    kind = "sentence-transformers"

    def __init__(self, model_name: str = config.ST_EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers:{model_name.split('/')[-1]}"

    def fit(self, corpus_texts):  # no fitting needed
        return self

    def encode(self, texts):
        vecs = self.model.encode(list(texts), normalize_embeddings=True,
                                 show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


class TfidfEmbedder:
    """Lexical fallback. Treated as the 'dense' vector when no model is present."""
    kind = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2),
            max_features=20000, sublinear_tf=True)
        self.name = "tfidf:1-2gram"
        self._fitted = False

    def fit(self, corpus_texts):
        self.vectorizer.fit(list(corpus_texts))
        self._fitted = True
        return self

    def encode(self, texts):
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.encode called before .fit()")
        mat = self.vectorizer.transform(list(texts)).toarray().astype(np.float32)
        return _normalize_rows(mat)


class OpenAIEmbedder:
    kind = "openai"

    def __init__(self, model_name: str = config.OPENAI_EMBED_MODEL):
        from openai import OpenAI
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model_name = model_name
        self.name = f"openai:{model_name}"

    def fit(self, corpus_texts):
        return self

    def encode(self, texts):
        texts = [t.replace("\n", " ") for t in texts]
        out, batch = [], 256
        for i in range(0, len(texts), batch):
            resp = self.client.embeddings.create(model=self.model_name,
                                                 input=texts[i:i + batch])
            out.extend(d.embedding for d in resp.data)
        return _normalize_rows(np.asarray(out, dtype=np.float32))


def build_embedder(backend: str | None = None):
    """Factory that honours config + graceful fallback. Returns (embedder, note)."""
    backend = (backend or config.EMBEDDING_BACKEND).lower()
    note = ""

    if backend == "openai":
        if not config.OPENAI_API_KEY:
            note = "openai requested but OPENAI_API_KEY missing -> falling back"
            backend = "auto"
        else:
            return OpenAIEmbedder(), "openai embeddings"

    if backend in ("sentence-transformers", "st"):
        return SentenceTransformerEmbedder(), "sentence-transformers"

    if backend == "tfidf":
        return TfidfEmbedder(), "tfidf (forced)"

    # ---- auto ----
    try:
        emb = SentenceTransformerEmbedder()
        return emb, (note + " | " if note else "") + "auto -> sentence-transformers"
    except Exception as exc:
        return TfidfEmbedder(), (note + " | " if note else "") + \
            f"auto -> tfidf (sentence-transformers unavailable: {type(exc).__name__})"
