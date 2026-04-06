"""
legal_chunker.py

Converts ParsedSection objects into LegalChunk objects suitable for embedding.

Key decisions:
- GDPR article points are already atomic — one chunk per point.
- Long sections (>750 tokens estimated) are split at sentence boundaries.
- Each chunk carries a citation_key generated here (used verbatim in briefs).
- Short adjacent chunks from the same article are merged if total < 300 tokens.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from .legal_parser import ParsedSection


# ---------------------------------------------------------------------------
# Token estimation (rough: 1 token ≈ 4 chars for English legal text)
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


MAX_TOKENS = 350    # Hard ceiling — BGE model limit is 512 real tokens; our estimator
                    # (chars/4) undercounts legal text, so 350 estimated ≈ 450 real tokens
MIN_TOKENS = 60     # Below this, try to merge with neighbour
TARGET_TOKENS = 250


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------

@dataclass
class LegalChunk:
    """One indexable unit for embedding and retrieval."""

    # Provenance
    doc_id: str
    doc_type: str
    source_name: str
    full_name: str
    date: str

    # Structure (regulation)
    section_type: str
    recital_number: Optional[int] = None
    article_number: Optional[int] = None
    article_title: Optional[str] = None
    paragraph_number: Optional[int] = None
    point: Optional[str] = None

    # Structure (guideline)
    section_number: Optional[str] = None
    section_heading: Optional[str] = None

    # Content
    text: str = ""
    citation_key: str = ""
    char_count: int = 0
    token_estimate: int = 0

    # For cross-reference linking (populated later by ingest.py)
    related_articles: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Citation key generation
# ---------------------------------------------------------------------------

def _make_citation_key(section: ParsedSection, sub_index: Optional[int] = None) -> str:
    """
    Generate a human-readable citation key for the chunk.

    Examples:
      GDPR Recital 39
      GDPR Art. 5
      GDPR Art. 5(1)
      GDPR Art. 5(1)(e)
      EDPB Guidelines 05/2020 on Consent, §3.2
      EDPB Guidelines 05/2020 on Consent, §3.2 [part 2]
    """
    if section.doc_type == "regulation":
        if section.section_type == "recital":
            return f"{section.source_name} Recital {section.recital_number}"
        key = f"{section.source_name} Art. {section.article_number}"
        if section.paragraph_number is not None:
            key += f"({section.paragraph_number})"
            if section.point:
                key += f"({section.point})"
        if sub_index is not None and sub_index > 0:
            key += f" [part {sub_index + 1}]"
        return key
    else:
        # Guideline
        base = section.source_name
        if section.section_number and section.section_number != "0":
            base += f", §{section.section_number}"
        elif section.section_heading and section.section_heading != "Full Document":
            base += f" — {section.section_heading}"
        if sub_index is not None and sub_index > 0:
            base += f" [part {sub_index + 1}]"
        return base


# ---------------------------------------------------------------------------
# Text splitting utilities
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\(])")


def _split_at_sentences(text: str, max_tokens: int = MAX_TOKENS) -> list[str]:
    """
    Split text into sub-chunks at sentence boundaries, respecting max_tokens.
    Never breaks in the middle of a sentence.
    """
    sentences = _SENTENCE_END.split(text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if _estimate_tokens(candidate) <= max_tokens:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If a single sentence exceeds the limit, include it anyway (can't split further)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


# ---------------------------------------------------------------------------
# Core chunker
# ---------------------------------------------------------------------------

def _section_to_chunks(section: ParsedSection) -> list[LegalChunk]:
    """Convert one ParsedSection into one or more LegalChunk objects."""
    text = section.text.strip()
    if not text:
        return []

    token_count = _estimate_tokens(text)

    # Common kwargs shared by all chunks from this section
    base_kwargs = dict(
        doc_id=section.doc_id,
        doc_type=section.doc_type,
        source_name=section.source_name,
        full_name=section.full_name,
        date=section.date,
        section_type=section.section_type,
        recital_number=section.recital_number,
        article_number=section.article_number,
        article_title=section.article_title,
        paragraph_number=section.paragraph_number,
        point=section.point,
        section_number=section.section_number,
        section_heading=section.section_heading,
    )

    if token_count <= MAX_TOKENS:
        # Single chunk
        citation_key = _make_citation_key(section)
        return [LegalChunk(
            **base_kwargs,
            text=text,
            citation_key=citation_key,
            char_count=len(text),
            token_estimate=token_count,
        )]

    # Section is too long — split at sentence boundaries
    sub_texts = _split_at_sentences(text, MAX_TOKENS)
    chunks = []
    for i, sub_text in enumerate(sub_texts):
        citation_key = _make_citation_key(section, sub_index=i)
        chunks.append(LegalChunk(
            **base_kwargs,
            text=sub_text,
            citation_key=citation_key,
            char_count=len(sub_text),
            token_estimate=_estimate_tokens(sub_text),
        ))
    return chunks


def _merge_short_chunks(chunks: list[LegalChunk]) -> list[LegalChunk]:
    """
    Merge consecutive chunks from the same article/section if both are short.
    This prevents tiny chunks (e.g. single-sentence definitions) from fragmenting retrieval.
    Only merges within the same article_number (regulation) or section_number (guideline).
    """
    if not chunks:
        return chunks

    merged: list[LegalChunk] = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        # Look ahead
        if (
            i + 1 < len(chunks)
            and current.token_estimate < MIN_TOKENS
            and chunks[i + 1].article_number == current.article_number
            and chunks[i + 1].section_number == current.section_number
            and chunks[i + 1].doc_id == current.doc_id
        ):
            next_chunk = chunks[i + 1]
            combined_text = current.text + "\n" + next_chunk.text
            combined_tokens = _estimate_tokens(combined_text)
            if combined_tokens <= TARGET_TOKENS:
                # Merge: use current's citation key, combined text
                merged.append(LegalChunk(
                    **{k: getattr(current, k) for k in [
                        "doc_id", "doc_type", "source_name", "full_name", "date",
                        "section_type", "recital_number", "article_number", "article_title",
                        "paragraph_number", "point", "section_number", "section_heading",
                        "citation_key", "related_articles",
                    ]},
                    text=combined_text,
                    char_count=len(combined_text),
                    token_estimate=combined_tokens,
                ))
                i += 2
                continue
        merged.append(current)
        i += 1

    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_sections(sections: list[ParsedSection]) -> list[LegalChunk]:
    """
    Convert a list of ParsedSection objects into LegalChunk objects.
    This is the main entry point called by ingest.py.
    """
    all_chunks: list[LegalChunk] = []

    for section in sections:
        chunks = _section_to_chunks(section)
        all_chunks.extend(chunks)

    all_chunks = _merge_short_chunks(all_chunks)
    _deduplicate_citation_keys(all_chunks)

    # Assign stable chunk IDs (used as ChromaDB document IDs)
    for i, chunk in enumerate(all_chunks):
        chunk_id = f"{chunk.doc_id}_{i:05d}"
        # Store as attribute for use by ingest.py
        chunk._chunk_id = chunk_id  # type: ignore[attr-defined]

    return all_chunks


def _deduplicate_citation_keys(chunks: list[LegalChunk]) -> None:
    """
    Ensure every citation_key is unique within the chunk list.
    If duplicates exist (e.g. from ambiguous article matching or split recitals),
    append a numeric suffix: "GDPR Art. 11(1)" → "GDPR Art. 11(1) [v2]", etc.
    Mutates in place.
    """
    from collections import Counter
    seen: dict[str, int] = {}
    for chunk in chunks:
        key = chunk.citation_key
        if key in seen:
            seen[key] += 1
            chunk.citation_key = f"{key} [v{seen[key]}]"
        else:
            seen[key] = 1


def chunk_stats(chunks: list[LegalChunk]) -> dict:
    """Return summary statistics about a list of chunks (useful for debugging)."""
    if not chunks:
        return {}
    token_counts = [c.token_estimate for c in chunks]
    return {
        "total_chunks": len(chunks),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "avg_tokens": sum(token_counts) // len(token_counts),
        "by_type": {
            t: len([c for c in chunks if c.section_type == t])
            for t in set(c.section_type for c in chunks)
        },
        "by_doc": {
            d: len([c for c in chunks if c.doc_id == d])
            for d in set(c.doc_id for c in chunks)
        },
    }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from pathlib import Path
    from .legal_parser import parse_document

    project_root = Path(__file__).parent.parent.parent
    manifest_path = project_root / "data" / "raw" / "manifest.json"

    with open(manifest_path) as f:
        manifest = json.load(f)

    doc_meta = manifest[0]
    pdf_path = project_root / "data" / "raw" / doc_meta["path"]

    print(f"Parsing: {doc_meta['source_name']}")
    sections = parse_document(pdf_path, doc_meta)
    chunks = chunk_sections(sections)

    stats = chunk_stats(chunks)
    print(f"\nChunk stats: {stats}")

    print("\nFirst 5 chunks:")
    for c in chunks[:5]:
        print(f"  [{c.citation_key}] ~{c.token_estimate} tokens")
        print(f"  {c.text[:100]!r}")
        print()
