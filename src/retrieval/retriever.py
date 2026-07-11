"""
retriever.py

Hybrid retrieval pipeline:
  1. BM25 keyword retrieval (top-K)
  2. Dense semantic retrieval via ChromaDB (top-K)
  3. Reciprocal Rank Fusion (RRF) to combine both
  4. Cross-encoder reranker to pick the final top-N

Also handles:
  - Query term expansion (legal terminology mapping)
  - Automatic inclusion of cross-referenced articles
"""

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent
INDEX_DIR = PROJECT_ROOT / os.getenv("INDEX_DIR", "data/index")
BM25_PATH = INDEX_DIR / "bm25.pkl"
CHROMA_DIR = INDEX_DIR / "chroma"
COLLECTION_NAME = "legal_corpus"

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBED_INSTRUCTION = "Represent this sentence for searching relevant passages: "

BM25_TOP_K = int(os.getenv("BM25_TOP_K", 50))
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", 50))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 8))
RRF_K = 60  # Standard RRF constant


# ---------------------------------------------------------------------------
# Term expansion dictionary
# Maps common user language to GDPR official terminology
# ---------------------------------------------------------------------------

TERM_EXPANSION: dict[str, list[str]] = {
    "data retention": ["storage limitation", "kept no longer than necessary", "retention period", "storage period"],
    "right to be forgotten": ["right to erasure", "erasure", "Article 17"],
    "consent": ["freely given", "specific", "informed", "unambiguous indication", "withdraw consent"],
    "data breach": ["personal data breach", "breach notification", "Article 33", "Article 34", "supervisory authority"],
    "privacy by design": ["data protection by design", "data protection by default", "Article 25"],
    "data minimisation": ["adequate, relevant and limited", "minimum necessary", "data minimization"],
    "legitimate interest": ["legitimate interests", "Article 6(1)(f)", "necessity test", "balancing test"],
    "data portability": ["right to data portability", "Article 20", "structured, commonly used"],
    "dpo": ["data protection officer", "Article 37", "Article 38", "Article 39"],
    "dpia": ["data protection impact assessment", "Article 35", "high risk"],
    "special categories": ["sensitive data", "Article 9", "health data", "biometric data", "racial origin"],
    "automated decision making": ["automated processing", "profiling", "Article 22", "legal effects"],
    "international transfers": ["third country", "adequacy decision", "standard contractual clauses", "Chapter V"],
    "controller": ["data controller", "determines the purposes", "Article 4(7)"],
    "processor": ["data processor", "on behalf of", "Article 4(8)", "Article 28"],
    "lawful basis": ["legal basis", "Article 6", "lawfulness of processing"],
    "transparency": ["information to be provided", "Article 13", "Article 14", "privacy notice"],
    "right of access": ["Article 15", "access to personal data", "copy of personal data"],
    "rectification": ["right to rectification", "Article 16", "inaccurate personal data"],
    "restriction": ["restriction of processing", "Article 18"],
    "objection": ["right to object", "Article 21", "direct marketing"],
    "accountability": ["Article 5(2)", "demonstrate compliance", "records of processing"],
}


def _expand_query(query: str) -> str:
    """Append relevant legal terms to the query for better BM25 recall."""
    query_lower = query.lower()
    expansions = []
    for trigger, terms in TERM_EXPANSION.items():
        if trigger in query_lower:
            expansions.extend(terms)
    if expansions:
        # Deduplicate and append
        unique = list(dict.fromkeys(expansions))
        return query + " " + " ".join(unique)
    return query


# ---------------------------------------------------------------------------
# Retrieved chunk container
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    citation_key: str
    doc_id: str
    doc_type: str
    source_name: str
    full_name: str
    date: str
    section_type: str
    article_number: Optional[int]
    paragraph_number: Optional[int]
    point: Optional[str]
    recital_number: Optional[int]
    section_number: Optional[str]
    section_heading: Optional[str]
    related_articles: list[int]
    rrf_score: float = 0.0
    rerank_score: float = 0.0


def _meta_to_chunk(chunk_id: str, text: str, meta: dict) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        citation_key=meta.get("citation_key", ""),
        doc_id=meta.get("doc_id", ""),
        doc_type=meta.get("doc_type", ""),
        source_name=meta.get("source_name", ""),
        full_name=meta.get("full_name", ""),
        date=meta.get("date", ""),
        section_type=meta.get("section_type", ""),
        article_number=meta.get("article_number") if meta.get("article_number", -1) >= 0 else None,
        paragraph_number=meta.get("paragraph_number") if meta.get("paragraph_number", -1) >= 0 else None,
        point=meta.get("point") or None,
        recital_number=meta.get("recital_number") if meta.get("recital_number", -1) >= 0 else None,
        section_number=meta.get("section_number") or None,
        section_heading=meta.get("section_heading") or None,
        related_articles=json.loads(meta.get("related_articles", "[]")),
    )


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def _rrf_fusion(
    bm25_ids: list[str],
    dense_ids: list[str],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion of two ranked lists.
    Returns a list of (chunk_id, rrf_score) sorted descending.
    """
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(bm25_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------

class LegalRetriever:
    def __init__(self):
        self._embedder: Optional[SentenceTransformer] = None
        self._reranker: Optional[CrossEncoder] = None
        self._collection: Optional[chromadb.Collection] = None
        self._bm25_index: Optional[dict] = None

    # -- Lazy loaders --------------------------------------------------------

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            print(f"Loading embedder: {EMBEDDING_MODEL}")
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedder

    def _get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            print(f"Loading reranker: {RERANKER_MODEL}")
            self._reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
        return self._reranker

    def _get_collection(self) -> chromadb.Collection:
        if self._collection is None:
            client = chromadb.PersistentClient(
                path=str(CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = client.get_collection(COLLECTION_NAME)
        return self._collection

    def _get_bm25(self) -> dict:
        if self._bm25_index is None:
            if not BM25_PATH.exists():
                raise FileNotFoundError(
                    f"BM25 index not found at {BM25_PATH}. Run ingestion first."
                )
            with open(BM25_PATH, "rb") as f:
                self._bm25_index = pickle.load(f)
        return self._bm25_index

    # -- Retrieval steps -----------------------------------------------------

    def _bm25_retrieve(self, query: str, top_k: int = BM25_TOP_K) -> list[str]:
        """Return top-k chunk IDs from BM25 search."""
        bm25_data = self._get_bm25()
        tokenized_query = query.lower().split()
        scores = bm25_data["bm25"].get_scores(tokenized_query)
        ids = bm25_data["ids"]
        ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
        return [chunk_id for chunk_id, _ in ranked[:top_k]]

    def _dense_retrieve(self, query: str, top_k: int = DENSE_TOP_K) -> list[str]:
        """Return top-k chunk IDs from dense semantic search in ChromaDB."""
        embedder = self._get_embedder()
        query_embedding = embedder.encode(
            EMBED_INSTRUCTION + query, normalize_embeddings=True
        ).tolist()
        collection = self._get_collection()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["distances"],
        )
        return results["ids"][0]

    def _fetch_chunks(self, chunk_ids: list[str]) -> dict[str, RetrievedChunk]:
        """Fetch full chunk data from ChromaDB by ID."""
        if not chunk_ids:
            return {}
        collection = self._get_collection()
        results = collection.get(
            ids=chunk_ids,
            include=["documents", "metadatas"],
        )
        chunks = {}
        for cid, text, meta in zip(results["ids"], results["documents"], results["metadatas"]):
            chunks[cid] = _meta_to_chunk(cid, text, meta)
        return chunks

    def _add_cross_references(
        self, chunks: dict[str, RetrievedChunk], top_ids: list[str]
    ) -> list[str]:
        """
        For retrieved regulation chunks, also include cross-referenced articles
        if they are not already in the result set.
        Returns augmented list of chunk IDs.
        """
        cross_ref_article_nums = set()
        for cid in top_ids:
            if cid in chunks and chunks[cid].related_articles:
                cross_ref_article_nums.update(chunks[cid].related_articles)

        if not cross_ref_article_nums:
            return top_ids

        # Fetch one representative chunk per cross-referenced article
        collection = self._get_collection()
        existing_articles = {chunks[cid].article_number for cid in top_ids if cid in chunks}
        new_articles = cross_ref_article_nums - existing_articles

        extra_ids = []
        for art_num in new_articles:
            results = collection.get(
                where={"article_number": art_num},
                include=["metadatas"],
                limit=1,
            )
            if results["ids"]:
                extra_ids.extend(results["ids"][:1])

        return top_ids + extra_ids

    def _rerank(
        self, query: str, chunks: dict[str, RetrievedChunk], candidate_ids: list[str]
    ) -> list[RetrievedChunk]:
        """Cross-encoder reranking of the candidate pool."""
        reranker = self._get_reranker()
        candidate_chunks = [chunks[cid] for cid in candidate_ids if cid in chunks]
        if not candidate_chunks:
            return []

        pairs = [(query, chunk.text) for chunk in candidate_chunks]
        scores = reranker.predict(pairs)

        for chunk, score in zip(candidate_chunks, scores):
            chunk.rerank_score = float(score)

        return sorted(candidate_chunks, key=lambda c: c.rerank_score, reverse=True)

    # -- Public API ----------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = RERANK_TOP_K,
        doc_type_filter: Optional[str] = None,
        min_rerank_score: float = 4.5,
    ) -> list[RetrievedChunk]:
        """
        Full hybrid retrieval pipeline.

        Args:
            query: Natural language legal query.
            top_k: Maximum number of chunks to return (actual count may be lower).
            doc_type_filter: "regulation" or "guideline" to restrict corpus.
            min_rerank_score: Minimum cross-encoder score to include a chunk.
                              Chunks below this are dropped even if in top_k.
                              Set to 0.0 to disable filtering.

        Returns:
            List of RetrievedChunk objects, ranked by reranker score.
            May be shorter than top_k if few chunks are sufficiently relevant.
        """
        expanded_query = _expand_query(query)

        # Step 1: BM25 + Dense retrieval
        bm25_ids = self._bm25_retrieve(expanded_query, BM25_TOP_K)
        dense_ids = self._dense_retrieve(expanded_query, DENSE_TOP_K)

        # Step 2: RRF fusion
        fused = _rrf_fusion(bm25_ids, dense_ids)
        # Take more than needed to account for cross-refs and filter
        candidate_ids = [cid for cid, _ in fused[:top_k * 4]]

        # Apply RRF scores
        rrf_score_map = {cid: score for cid, score in fused}

        # Step 3: Fetch candidates
        chunks = self._fetch_chunks(candidate_ids)

        # Step 4: Apply doc_type filter if requested
        if doc_type_filter:
            candidate_ids = [
                cid for cid in candidate_ids
                if cid in chunks and chunks[cid].doc_type == doc_type_filter
            ]

        # Step 5: Add cross-referenced articles
        candidate_ids = self._add_cross_references(chunks, candidate_ids)
        # Fetch any newly added cross-ref chunks
        new_ids = [cid for cid in candidate_ids if cid not in chunks]
        if new_ids:
            chunks.update(self._fetch_chunks(new_ids))

        # Apply RRF scores to retrieved chunks
        for cid, chunk in chunks.items():
            chunk.rrf_score = rrf_score_map.get(cid, 0.0)

        # Step 6: Rerank
        reranked = self._rerank(query, chunks, candidate_ids)

        # Step 7: Apply relevance threshold — drop chunks below min_rerank_score
        if min_rerank_score > 0:
            reranked = [c for c in reranked if c.rerank_score >= min_rerank_score]

        return reranked[:top_k]


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    retriever = LegalRetriever()
    test_query = "What are the requirements for valid consent under GDPR?"
    print(f"Query: {test_query}\n")
    results = retriever.retrieve(test_query, top_k=5)
    for i, chunk in enumerate(results):
        print(f"[{i+1}] {chunk.citation_key} (rerank: {chunk.rerank_score:.3f})")
        print(f"    {chunk.text[:150]!r}")
        print()
