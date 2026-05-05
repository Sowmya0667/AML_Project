"""
api.py — FastAPI REST backend for Legal GraphRAG
Run: uvicorn api:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import config

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Legal GraphRAG API",
    description="Hybrid GraphRAG for Indian Supreme Court Judgments",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─── Request / Response models ───────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:       str
    graph_top_k: int  = 5
    vector_top_k:int  = 5
    final_top_k: int  = 6

class QueryResponse(BaseModel):
    query:       str
    answer:      str
    sources:     List[dict]
    graph_hits:  int
    vector_hits: int
    model_used:  str

class IngestRequest(BaseModel):
    text:        str
    case_number: str
    case_name:   str

class GraphSearchRequest(BaseModel):
    query:       str
    search_type: str = "fulltext"   # fulltext | statute | judge | concept
    limit:       int = 10


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """Full RAG pipeline: retrieve + generate."""
    try:
        config.GRAPH_TOP_K  = req.graph_top_k
        config.VECTOR_TOP_K = req.vector_top_k
        config.FINAL_TOP_K  = req.final_top_k

        from src.generation.llm_chain import LegalRAGChain
        r = LegalRAGChain().query(req.query)
        return QueryResponse(**vars(r))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/query/stream")
def stream_query(req: QueryRequest):
    """Stream LLM response as Server-Sent Events."""
    try:
        from src.generation.llm_chain import LegalRAGChain
        chain = LegalRAGChain()
        def gen():
            for tok in chain.stream_query(req.query):
                yield f"data: {tok}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/retrieve")
def retrieve_only(req: QueryRequest):
    """Return retrieved chunks without generating an answer."""
    try:
        from src.retrieval.hybrid_retriever import HybridRetriever
        result = HybridRetriever().retrieve(req.query)
        return {
            "query":       result.query,
            "graph_hits":  result.graph_hits,
            "vector_hits": result.vector_hits,
            "chunks": [
                {
                    "text":        c.text[:500],
                    "case_number": c.case_number,
                    "case_name":   c.case_name,
                    "date":        c.date,
                    "source":      c.source,
                    "score":       c.score,
                }
                for c in result.chunks
            ],
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/ingest/text")
def ingest_text(req: IngestRequest, bg: BackgroundTasks):
    """Ingest a judgment provided as raw text (runs in background)."""
    def _run():
        from src.ingestion.data_loader import IngestionPipeline
        from src.ingestion.txt_parser  import JudgmentMetadata
        pipeline = IngestionPipeline()
        try:
            meta = JudgmentMetadata(
                file_path="<api>", raw_text=req.text,
                case_number=req.case_number, case_name=req.case_name,
            )
            chunks = pipeline.parser._chunk(meta)
            ents   = pipeline.extractor.extract(req.text[:3500], req.case_number)
            pipeline.graph.add_judgment(meta, ents)
            pipeline.vector.add_chunks(chunks)
        finally:
            pipeline.close()
    bg.add_task(_run)
    return {"status": "queued", "case_number": req.case_number}


@app.get("/graph/stats")
def graph_stats():
    try:
        from src.graph.graph_queries import GraphQueryEngine
        return GraphQueryEngine().stats()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/graph/case/{case_number}")
def case_detail(case_number: str):
    try:
        from src.graph.graph_queries import GraphQueryEngine
        gq = GraphQueryEngine()
        return {
            "case_number": case_number,
            "cites":       gq.cited_cases(case_number),
            "cited_by":    gq.citing_cases(case_number),
            "related":     gq.related(case_number),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/graph/search")
def graph_search(req: GraphSearchRequest):
    try:
        from src.graph.graph_queries import GraphQueryEngine
        gq = GraphQueryEngine()
        fn = {
            "statute":  gq.by_statute,
            "judge":    gq.by_judge,
            "concept":  gq.by_concept,
            "fulltext": gq.fulltext,
        }.get(req.search_type, gq.fulltext)
        return {"results": fn(req.query, req.limit)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/vector/stats")
def vector_stats():
    try:
        from src.vector.vector_store import LegalVectorStore
        vs = LegalVectorStore()
        return {"total_chunks": vs.count(), "cases": vs.list_cases()}
    except Exception as e:
        raise HTTPException(500, str(e))
