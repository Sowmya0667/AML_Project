"""
src/vector/vector_store.py
────────────────────────────────────────────────────────────────────────────
ChromaDB-backed vector store for judgment text chunks.

Stores:
  - chunk text (document)
  - chunk metadata (case_number, case_name, date, petitioner, respondent)
  - embedding vectors (sentence-transformers)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from config import config
from src.vector.embeddings import EmbeddingModel
from src.ingestion.txt_parser import TextChunk

log = logging.getLogger(__name__)


class LegalVectorStore:

    def __init__(self):
        self.emb = EmbeddingModel()
        self._client = chromadb.PersistentClient(
            path=config.CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            f"ChromaDB '{config.CHROMA_COLLECTION}' — "
            f"{self._col.count()} chunks indexed"
        )

    # ── Index ─────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[TextChunk]) -> None:
        if not chunks:
            return

        ids       = [c.chunk_id for c in chunks]
        docs      = [c.text     for c in chunks]
        metas     = [
            {
                "case_number":  c.case_number,
                "case_name":    c.case_name,
                "chunk_index":  c.chunk_index,
                "total_chunks": c.total_chunks,
                "date":         c.date,
                "petitioner":   c.petitioner,
                "respondent":   c.respondent,
            }
            for c in chunks
        ]
        embeddings = self.emb.embed_documents(docs)

        self._col.upsert(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metas,
        )

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """
        Semantic search.
        Returns list of dicts: {text, metadata, score}
        score = cosine similarity (higher = more relevant).
        """
        if self._col.count() == 0:
            return []
            
        qe = self.emb.embed_query(query_text)
        kwargs: Dict[str, Any] = {
            "query_embeddings": [qe],
            "n_results": min(top_k, max(1, self._col.count())),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            res = self._col.query(**kwargs)
            out = []
            for doc, meta, dist in zip(
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
            ):
                out.append({"text": doc, "metadata": meta, "score": 1.0 - dist})
            return out
        except Exception as e:
            log.warning(f"ChromaDB query failed (DB might be locked or empty): {e}")
            return []

    # ── Stats ─────────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self._col.count()

    def list_cases(self) -> List[str]:
        all_meta = self._col.get(include=["metadatas"])["metadatas"]
        seen, cases = set(), []
        for m in all_meta:
            cn = m.get("case_number", "")
            if cn and cn not in seen:
                seen.add(cn)
                cases.append(cn)
        return cases
