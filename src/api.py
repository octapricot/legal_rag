"""
api.py — FastAPI web interface for the Legal RAG system.

Endpoints:
  POST /query   → generate a legal brief
  GET  /health  → check index status
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from .retrieval.retriever import LegalRetriever
from .generation.generator import generate_brief

app = FastAPI(
    title="Legal RAG — GDPR Research Assistant",
    description="AI-powered GDPR legal brief generator",
    version="1.0.0",
)

# Singleton retriever (loaded once on startup)
_retriever: Optional[LegalRetriever] = None


def get_retriever() -> LegalRetriever:
    global _retriever
    if _retriever is None:
        _retriever = LegalRetriever()
    return _retriever


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    top_k: int = 8
    doc_type_filter: Optional[str] = None   # "regulation" | "guideline" | None


class QueryResponse(BaseModel):
    query: str
    brief_markdown: str
    sources_used: list[str]
    warnings: list[str] = []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    try:
        retriever = get_retriever()
        collection = retriever._get_collection()
        count = collection.count()
        return {"status": "ok", "chunks_indexed": count}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    retriever = get_retriever()
    chunks = retriever.retrieve(
        request.query,
        top_k=request.top_k,
        doc_type_filter=request.doc_type_filter,
    )

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant sources found for this query.")

    brief = generate_brief(request.query, chunks)
    sources = [c.citation_key for c in chunks]

    return QueryResponse(
        query=request.query,
        brief_markdown=brief,
        sources_used=sources,
    )
