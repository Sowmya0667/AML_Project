"""
src/retrieval/hybrid_retriever.py
────────────────────────────────────────────────────────────────────────────
Hybrid GraphRAG retriever.

Pipeline
────────
1.  Detect query intent  (statute / judge / concept / general)
2.  Graph retrieval      → targeted Cypher traversal on Neo4j
3.  Vector retrieval     → semantic search on ChromaDB
4.  Reciprocal Rank Fusion (RRF) → unified ranking
5.  Return top-K RetrievedChunk objects
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import config
from src.graph.graph_queries  import GraphQueryEngine
from src.vector.vector_store  import LegalVectorStore

log = logging.getLogger(__name__)


# ─── Output models ───────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    text:        str
    case_number: str
    case_name:   str
    date:        str
    source:      str       # "graph" | "vector" | "hybrid"
    score:       float
    metadata:    Dict      = field(default_factory=dict)


@dataclass
class RetrievalResult:
    query:       str
    chunks:      List[RetrievedChunk]
    graph_hits:  int
    vector_hits: int


# ─── Intent detection ────────────────────────────────────────────────────────

_STATUTE_RE  = re.compile(
    r"section\s+\d+|article\s+\d+|ipc|crpc|cpc|iea|"
    r"constitution|act\s+\d{4}|order\s+\d+\s+rule",
    re.I,
)
_JUDGE_RE    = re.compile(r"\bjustice\s+[a-z]+", re.I)
_CONCEPTS    = {
    "res judicata", "natural justice", "estoppel", "fundamental rights",
    "judicial review", "locus standi", "habeas corpus", "mandamus",
    "certiorari", "quo warranto", "due process", "stare decisis",
    "doctrine of precedent", "promissory estoppel", "legitimate expectation",
}


def _statute(q: str)  -> Optional[str]: m = _STATUTE_RE.search(q);  return m.group() if m else None
def _judge(q: str)    -> Optional[str]: m = _JUDGE_RE.search(q);    return m.group() if m else None
def _concept(q: str)  -> Optional[str]:
    ql = q.lower()
    return next((c for c in _CONCEPTS if c in ql), None)


# ─── Reciprocal Rank Fusion ───────────────────────────────────────────────────

def _rrf(lists: List[List[Dict]], id_key: str = "case_number", k: int = 60) -> List[Dict]:
    scores: Dict[str, float] = {}
    items:  Dict[str, Dict]  = {}
    for lst in lists:
        for rank, item in enumerate(lst, 1):
            iid = item.get(id_key, "")
            if not iid:
                continue
            scores[iid] = scores.get(iid, 0.0) + 1.0 / (k + rank)
            items[iid]  = item
    return sorted(items.values(), key=lambda x: scores.get(x.get(id_key, ""), 0), reverse=True)


# ─── Retriever ───────────────────────────────────────────────────────────────

class HybridRetriever:
    """Combines Neo4j graph traversal + ChromaDB semantic search."""

    def __init__(self):
        self.graph  = GraphQueryEngine()
        self.vector = LegalVectorStore()

    def retrieve(self, query: str) -> RetrievalResult:
        log.info(f"Hybrid retrieve: '{query[:80]}'")

        # ── Graph leg ─────────────────────────────────────────────────────────
        graph_rows  = self._graph_retrieve(query)
        graph_chunks = [self._graph_to_chunk(r) for r in graph_rows]

        # ── Vector leg ────────────────────────────────────────────────────────
        vec_rows   = self.vector.query(query, top_k=config.VECTOR_TOP_K)
        vec_chunks = [self._vec_to_chunk(r) for r in vec_rows]

        # ── Fuse ──────────────────────────────────────────────────────────────
        all_chunks = self._fuse(graph_chunks, vec_chunks)
        top        = all_chunks[: config.FINAL_TOP_K]

        return RetrievalResult(
            query       = query,
            chunks      = top,
            graph_hits  = len(graph_chunks),
            vector_hits = len(vec_chunks),
        )

    # ── Graph strategy ────────────────────────────────────────────────────────

    def _graph_retrieve(self, query: str) -> List[Dict]:
        results: List[Dict] = []

        s = _statute(query)
        j = _judge(query)
        c = _concept(query)

        if s: results += self.graph.by_statute(s, config.GRAPH_TOP_K)
        if j: results += self.graph.by_judge(j,   config.GRAPH_TOP_K)
        if c: results += self.graph.by_concept(c, config.GRAPH_TOP_K)

        # Always supplement with full-text search
        results += self.graph.fulltext(query, config.GRAPH_TOP_K)

        # Deduplicate
        seen, deduped = set(), []
        for r in results:
            cn = r.get("case_number", "")
            if cn and cn not in seen:
                seen.add(cn)
                deduped.append(r)
        return deduped[: config.GRAPH_TOP_K]

    # ── Converters ────────────────────────────────────────────────────────────

    def _graph_to_chunk(self, r: Dict) -> RetrievedChunk:
        text = (
            f"Case: {r.get('case_name','')}\n"
            f"Date: {r.get('date','')}\n"
            f"Area: {r.get('subject_matter','')}\n"
            f"Outcome: {r.get('outcome','')}\n"
            f"Holdings: {r.get('holdings','')}"
        )
        return RetrievedChunk(
            text=text, case_number=r.get("case_number",""),
            case_name=r.get("case_name",""), date=r.get("date",""),
            source="graph", score=r.get("score", 0.5), metadata=r,
        )

    def _vec_to_chunk(self, r: Dict) -> RetrievedChunk:
        meta = r.get("metadata", {})
        return RetrievedChunk(
            text=r.get("text",""), case_number=meta.get("case_number",""),
            case_name=meta.get("case_name",""), date=meta.get("date",""),
            source="vector", score=r.get("score", 0.0), metadata=meta,
        )

    # ── RRF fusion ────────────────────────────────────────────────────────────

    def _fuse(
        self,
        graph: List[RetrievedChunk],
        vector: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        # Use text hash as unique chunk ID to prevent chunk collapsing
        g_dicts = [{"chunk_id": str(hash(c.text)), "_c": c} for c in graph]
        v_dicts = [{"chunk_id": str(hash(c.text)), "_c": c} for c in vector]
        fused   = _rrf([g_dicts, v_dicts], id_key="chunk_id")

        seen_chunks: Dict[str, bool]    = {}
        output: List[RetrievedChunk] = []
        for d in fused:
            chunk: RetrievedChunk = d.get("_c")
            if chunk is None:
                continue
            cid = str(hash(chunk.text))
            if cid in seen_chunks:
                chunk.source = "hybrid"
            seen_chunks[cid] = True
            output.append(chunk)

        # Append any vector-only results not already present
        known_chunks = {str(hash(c.text)) for c in output}
        for c in vector:
            cid = str(hash(c.text))
            if cid not in known_chunks:
                output.append(c)
                known_chunks.add(cid)

        return output
