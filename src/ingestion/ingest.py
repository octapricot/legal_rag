"""
ingest.py

Ingestion pipeline runner. Reads all documents from the manifest, parses them,
chunks them, embeds them, and stores them in ChromaDB + a BM25 pickle index.

Usage:
    python -m src.ingestion.ingest                  # ingest all active docs
    python -m src.ingestion.ingest --doc-id gdpr_2016_679   # single doc
    python -m src.ingestion.ingest --reset          # wipe and rebuild from scratch
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

from .legal_parser import parse_document
from .legal_chunker import chunk_sections, chunk_stats, LegalChunk

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DIR = PROJECT_ROOT / os.getenv("DATA_RAW_DIR", "data/raw")
INDEX_DIR = PROJECT_ROOT / os.getenv("INDEX_DIR", "data/index")
BM25_PATH = INDEX_DIR / "bm25.pkl"
CHROMA_DIR = INDEX_DIR / "chroma"
MANIFEST_PATH = RAW_DIR / "manifest.json"

COLLECTION_NAME = "legal_corpus"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBED_INSTRUCTION = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------------------
# GDPR cross-reference map (article → articles it references)
# Built from common knowledge of the regulation structure.
# Used to link related articles at retrieval time.
# ---------------------------------------------------------------------------

GDPR_CROSS_REFS: dict[int, list[int]] = {
    4: [5, 6, 7, 9],        # Definitions → key obligations
    5: [6, 7, 9],           # Principles → lawful bases, consent, special cats
    6: [4, 5, 7, 8, 9],     # Lawful basis → definitions, principles, consent
    7: [4, 5, 6],           # Consent conditions
    9: [6, 4, 22],          # Special categories
    12: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22],  # Transparency obligations → rights
    13: [12, 5, 6],
    14: [12, 5, 6],
    15: [12],               # Right of access
    17: [12, 18],           # Right to erasure
    22: [9, 6],             # Automated decision-making
    25: [5, 24, 32],        # DPbD → principles, controller obligations, security
    32: [5, 25],            # Security
    33: [34, 32, 4],        # Breach notification
    34: [33],
    35: [36, 37, 25],       # DPIA
    36: [35],
    37: [38, 39],           # DPO
    44: [45, 46, 47, 48, 49],  # Transfers general → specific tools
    83: [5, 6, 7, 9],      # Fines → key obligations
}


def _enrich_cross_refs(chunks: list[LegalChunk]) -> None:
    """Add related_articles metadata to chunks based on the cross-reference map."""
    for chunk in chunks:
        if chunk.doc_type == "regulation" and chunk.article_number is not None:
            chunk.related_articles = GDPR_CROSS_REFS.get(chunk.article_number, [])


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _load_embedder() -> SentenceTransformer:
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


def _embed_chunks(
    embedder: SentenceTransformer,
    chunks: list[LegalChunk],
    batch_size: int = 32,
) -> list[list[float]]:
    """Embed all chunks with the BGE instruction prefix."""
    texts = [EMBED_INSTRUCTION + chunk.text for chunk in chunks]
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        batch_embeddings = embedder.encode(batch, normalize_embeddings=True).tolist()
        embeddings.extend(batch_embeddings)
    return embeddings


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

def _get_chroma_collection(reset: bool = False) -> chromadb.Collection:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def _chunk_to_metadata(chunk: LegalChunk) -> dict:
    """Convert a LegalChunk to a flat dict for ChromaDB metadata storage."""
    return {
        "doc_id": chunk.doc_id,
        "doc_type": chunk.doc_type,
        "source_name": chunk.source_name,
        "full_name": chunk.full_name,
        "date": chunk.date,
        "section_type": chunk.section_type,
        "recital_number": chunk.recital_number or -1,
        "article_number": chunk.article_number or -1,
        "article_title": chunk.article_title or "",
        "paragraph_number": chunk.paragraph_number or -1,
        "point": chunk.point or "",
        "section_number": chunk.section_number or "",
        "section_heading": chunk.section_heading or "",
        "citation_key": chunk.citation_key,
        "token_estimate": chunk.token_estimate,
        "related_articles": json.dumps(chunk.related_articles),
    }


def _store_in_chroma(
    collection: chromadb.Collection,
    chunks: list[LegalChunk],
    embeddings: list[list[float]],
    doc_id: str,
) -> None:
    """Upsert chunks for a single document into ChromaDB."""
    # Remove existing entries for this doc first (for re-ingestion)
    existing = collection.get(where={"doc_id": doc_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    ids = [chunk._chunk_id for chunk in chunks]  # type: ignore[attr-defined]
    documents = [chunk.text for chunk in chunks]
    metadatas = [_chunk_to_metadata(chunk) for chunk in chunks]

    # ChromaDB has a limit of 5461 items per upsert call
    batch_size = 500
    for i in tqdm(range(0, len(ids), batch_size), desc="Storing in ChromaDB"):
        collection.upsert(
            ids=ids[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

def _rebuild_bm25(collection: chromadb.Collection) -> None:
    """
    Rebuild the BM25 index from all documents currently in ChromaDB.
    Must be called after any insert/update so the index stays in sync.
    """
    print("Rebuilding BM25 index...")
    # Fetch all docs from Chroma (may be large — ChromaDB loads everything into memory)
    all_docs = collection.get(include=["documents", "metadatas"])

    corpus_ids = all_docs["ids"]
    corpus_texts = all_docs["documents"]
    corpus_metadatas = all_docs["metadatas"]

    tokenized = [text.lower().split() for text in corpus_texts]
    bm25 = BM25Okapi(tokenized)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "ids": corpus_ids,
            "metadatas": corpus_metadatas,
            "texts": corpus_texts,
        }, f)

    print(f"  BM25 index saved → {BM25_PATH} ({len(corpus_ids)} documents)")


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------

def _already_ingested(collection: chromadb.Collection, doc_id: str) -> bool:
    """Return True if this doc already has chunks in the index."""
    results = collection.get(where={"doc_id": doc_id}, limit=1)
    return len(results["ids"]) > 0


def ingest_document(
    doc_meta: dict,
    embedder: SentenceTransformer,
    collection: chromadb.Collection,
) -> None:
    pdf_path = RAW_DIR / doc_meta["path"]
    if not pdf_path.exists():
        print(f"  [SKIP] File not found: {pdf_path}")
        return

    print(f"\n{'='*60}")
    print(f"Processing: {doc_meta['source_name']}")
    print(f"  Path: {pdf_path.name}")
    print(f"  Status: {doc_meta.get('status', 'active')}")

    sections = parse_document(pdf_path, doc_meta)
    if not sections:
        print("  [SKIP] No sections extracted.")
        return

    chunks = chunk_sections(sections)
    _enrich_cross_refs(chunks)

    stats = chunk_stats(chunks)
    print(f"  Chunks: {stats['total_chunks']} "
          f"(avg {stats.get('avg_tokens', '?')} tokens, "
          f"max {stats.get('max_tokens', '?')} tokens)")

    embeddings = _embed_chunks(embedder, chunks)
    _store_in_chroma(collection, chunks, embeddings, doc_meta["doc_id"])
    print(f"  ✓ Stored in ChromaDB")


def run_ingestion(target_doc_id: Optional[str] = None, reset: bool = False) -> None:
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    # Filter
    if target_doc_id:
        manifest = [m for m in manifest if m["doc_id"] == target_doc_id]
        if not manifest:
            print(f"No document with doc_id='{target_doc_id}' found in manifest.")
            return

    # Only active docs (skip superseded)
    active = [m for m in manifest if m.get("status", "active") == "active"]
    superseded = [m for m in manifest if m.get("status") == "superseded"]
    if superseded:
        print(f"Skipping {len(superseded)} superseded document(s).")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    embedder = _load_embedder()
    collection = _get_chroma_collection(reset=reset)

    print(f"\nIngesting {len(active)} document(s)...")
    for i, doc_meta in enumerate(active):
        print(f"\n[{i+1}/{len(active)}] {doc_meta['source_name']}")
        if not reset and _already_ingested(collection, doc_meta["doc_id"]):
            print(f"  [SKIP] Already in index — resuming from next document.")
            continue
        ingest_document(doc_meta, embedder, collection)

    # Always rebuild BM25 after ingestion
    _rebuild_bm25(collection)

    total = collection.count()
    print(f"\n{'='*60}")
    print(f"Ingestion complete. Total chunks in index: {total}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest legal documents into the RAG index.")
    parser.add_argument("--doc-id", type=str, default=None, help="Ingest a single document by doc_id")
    parser.add_argument("--reset", action="store_true", help="Wipe the existing index before ingesting")
    args = parser.parse_args()

    run_ingestion(target_doc_id=args.doc_id, reset=args.reset)
