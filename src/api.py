"""
api.py — FastAPI web interface for the Legal RAG system.

Endpoints:
  GET  /         → serves the frontend UI
  POST /query    → generate a legal brief
  GET  /health   → check index status
"""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .retrieval.retriever import LegalRetriever
from .generation.generator import generate_brief

app = FastAPI(
    title="Legal RAG — GDPR Research Assistant",
    description="AI-powered GDPR legal brief generator",
    version="1.0.0",
)

FRONTEND_DIR = Path(__file__).parent / "frontend"

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
    min_rerank_score: float = 4.5           # relevance threshold


class QueryResponse(BaseModel):
    query: str
    brief_markdown: str
    sources_used: list[str]
    warnings: list[str] = []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


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
        min_rerank_score=request.min_rerank_score,
    )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No sufficiently relevant sources found for this query. "
                   "Try rephrasing using GDPR-specific terminology."
        )

    brief = generate_brief(request.query, chunks)
    sources = [c.citation_key for c in chunks]

    import re
    warnings: list[str] = []
    # Only extract citation warnings when brief parsed successfully
    if not brief.startswith("# LEGAL BRIEF\n\n**Error"):
        try:
            suffix_re = re.compile(r"\s+\[(v\d+|part \d+)\]$")
            valid_keys = {c.citation_key for c in chunks}
            valid_norm = {suffix_re.sub("", k).strip() for k in valid_keys}
            cited = re.findall(r'\[([^\]]{10,})\]', brief)
            for key in cited:
                if key not in valid_keys and suffix_re.sub("", key).strip() not in valid_norm:
                    if any(c in key for c in ["Art.", "Recital", "Guidelines", "GDPR", "EDPB"]):
                        warnings.append(f"Citation not in retrieved sources: '{key}'")
        except Exception:
            pass

    return QueryResponse(
        query=request.query,
        brief_markdown=brief,
        sources_used=sources,
        warnings=warnings,
    )
