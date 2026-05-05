"""
src/vector/embeddings.py
────────────────────────────────────────────────────────────────────────────
Local sentence-transformer embeddings (no API key required).
Model: all-MiniLM-L6-v2  (22 MB, 384-dimensional)
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

from config import config

log = logging.getLogger(__name__)


class EmbeddingModel:
    """Thin wrapper around SentenceTransformer."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        log.info(f"Loading embedding model '{config.EMBEDDING_MODEL}' ...")
        self._model = SentenceTransformer(config.EMBEDDING_MODEL)
        log.info("Embedding model loaded.")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of strings. Returns (n, dim) float32 array."""
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0].tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts).tolist()

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()
