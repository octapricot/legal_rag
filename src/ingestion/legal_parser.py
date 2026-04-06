"""
legal_parser.py

Parses GDPR regulation and EDPB guideline PDFs into structured document objects.
Each document yields a list of ParsedSection objects carrying raw text and
structural metadata. The chunker (legal_chunker.py) consumes these.

Key challenges handled here:
- GDPR EUR-Lex two-column layout
- Footnote stripping
- Ligature / encoding artifacts (via ftfy)
- Recital vs Article detection
- EDPB guideline section heading detection
"""

import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ftfy
import pdfplumber


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ParsedSection:
    """One logical section extracted from a PDF (article, recital, or guideline section)."""
    doc_id: str
    doc_type: str          # "regulation" | "guideline"
    source_name: str
    full_name: str
    date: str

    section_type: str      # "recital" | "article" | "definition" | "section" | "annex"

    # Regulation-specific
    recital_number: Optional[int] = None
    article_number: Optional[int] = None
    paragraph_number: Optional[int] = None
    point: Optional[str] = None          # e.g. "e" for point (e)
    article_title: Optional[str] = None

    # Guideline-specific
    section_number: Optional[str] = None   # e.g. "3.2.1"
    section_heading: Optional[str] = None

    page_number: Optional[int] = None
    text: str = ""

    # Filled in by chunker
    citation_key: str = ""


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Fix encoding artifacts, normalize whitespace, strip footnote markers."""
    text = ftfy.fix_text(text)
    # Remove soft hyphens and dehyphenate line breaks
    text = text.replace("\u00ad", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse multiple spaces / odd whitespace into single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse more than two newlines into two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_footnotes(text: str) -> str:
    """
    Remove inline footnote markers (superscript numbers at word boundaries).
    EUR-Lex PDFs embed footnote numbers inline like "data¹" or "data (1)".
    """
    # Superscript unicode digits
    superscripts = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    text = text.translate(superscripts)
    # Remove footnote markers at word ends (e.g. "data1 shall" → "data shall")
    # Use [a-zA-Z] lookbehind (not \w) to avoid stripping digits from "Article 10" etc.
    text = re.sub(r"(?<=[a-zA-Z])\d{1,2}(?=[\s,.])", "", text)
    return text


# ---------------------------------------------------------------------------
# GDPR Regulation Parser
# ---------------------------------------------------------------------------

# Patterns that identify structural elements in the GDPR text
_RECITAL_START = re.compile(r"^\s*\((\d{1,3})\)\s+", re.MULTILINE)
_ARTICLE_START = re.compile(
    r"^\s*Article\s+(\d{1,3})\s*\n([^\n]{3,80})\n",
    re.MULTILINE | re.IGNORECASE,
)
_PARAGRAPH_START = re.compile(r"^\s*(\d+)\.\s+", re.MULTILINE)
_POINT_START = re.compile(r"^\s*\(([a-z])\)\s+", re.MULTILINE)


def _extract_gdpr_text(pdf_path: Path) -> str:
    """
    Extract raw text from GDPR PDF (EUR-Lex, single-column layout).
    Each page is extracted in full; pdfplumber preserves reading order correctly.
    """
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            pages_text.append(text)
    return "\n".join(pages_text)


def _split_gdpr_into_recitals(full_text: str) -> list[tuple[int, str]]:
    """
    Return list of (recital_number, text) tuples.

    GDPR recitals are numbered (1)-(173) sequentially.
    We only accept a match as a recital if:
    - It is sequential (each number = previous + 1)
    - The text between it and the next match is at least 50 characters
      (filters out footnote references like "(4) OJ C 229...")
    """
    results = []
    matches = list(_RECITAL_START.finditer(full_text))
    expected_num = 1

    for i, match in enumerate(matches):
        num = int(match.group(1))
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        text = full_text[start:end].strip()
        # Strip the leading "(N)" marker
        text = re.sub(r"^\s*\(\d+\)\s+", "", text, count=1)

        # Skip if too short (footnote artifact) or not the expected sequential number
        if len(text) < 50:
            continue
        if num != expected_num:
            # Allow catching up if we missed a number (rare edge case)
            if num < expected_num or num > expected_num + 3:
                continue
        expected_num = num + 1
        results.append((num, text))
    return results


def _split_gdpr_into_articles(full_text: str) -> list[tuple[int, str, str]]:
    """Return list of (article_number, title, text) tuples."""
    results = []
    matches = list(_ARTICLE_START.finditer(full_text))
    for i, match in enumerate(matches):
        num = int(match.group(1))
        title = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        text = full_text[start:end].strip()
        results.append((num, title, text))
    return results


def _parse_article_body(article_text: str) -> list[tuple[Optional[int], Optional[str], str]]:
    """
    Break an article body into (paragraph_number, point, text) tuples.
    Returns at minimum [(None, None, full_text)] if no sub-structure found.
    """
    # Remove the "Article N\nTitle\n" header line
    article_text = re.sub(r"^Article\s+\d+\s*\n[^\n]+\n", "", article_text, flags=re.IGNORECASE)

    results = []

    # Try to find numbered paragraphs first
    para_matches = list(_PARAGRAPH_START.finditer(article_text))
    if not para_matches:
        # No paragraphs — entire article is one unit
        return [(None, None, article_text.strip())]

    for p_idx, p_match in enumerate(para_matches):
        para_num = int(p_match.group(1))
        p_start = p_match.start()
        p_end = para_matches[p_idx + 1].start() if p_idx + 1 < len(para_matches) else len(article_text)
        para_text = article_text[p_start:p_end].strip()

        # Check for lettered points within this paragraph
        point_matches = list(_POINT_START.finditer(para_text))
        if not point_matches:
            results.append((para_num, None, para_text))
            continue

        # Include the paragraph intro (text before first point)
        intro_text = para_text[: point_matches[0].start()].strip()
        if intro_text:
            results.append((para_num, None, intro_text))

        for q_idx, q_match in enumerate(point_matches):
            point_letter = q_match.group(1)
            q_start = q_match.start()
            q_end = point_matches[q_idx + 1].start() if q_idx + 1 < len(point_matches) else len(para_text)
            point_text = para_text[q_start:q_end].strip()
            results.append((para_num, point_letter, point_text))

    return results


def parse_gdpr(pdf_path: Path, doc_meta: dict) -> list[ParsedSection]:
    """
    Full parse of the GDPR regulation PDF.
    Returns a flat list of ParsedSection objects.
    """
    sections: list[ParsedSection] = []

    print(f"  Extracting text from GDPR PDF...")
    raw_text = _extract_gdpr_text(pdf_path)
    raw_text = clean_text(raw_text)
    raw_text = remove_footnotes(raw_text)

    # --- Find the boundary between recitals and articles ---
    # The GDPR text starts with recitals (1)-(173), then "HAS ADOPTED THIS REGULATION:"
    # before Article 1.
    adoption_marker = re.search(
        r"HAVE ADOPTED THIS REGULATION\s*:", raw_text, re.IGNORECASE
    )
    if adoption_marker:
        recital_text = raw_text[: adoption_marker.start()]
        article_text = raw_text[adoption_marker.end():]
    else:
        # Fallback: split at first "Article 1"
        first_article = re.search(r"\bArticle\s+1\b", raw_text, re.IGNORECASE)
        recital_text = raw_text[: first_article.start()] if first_article else ""
        article_text = raw_text[first_article.start():] if first_article else raw_text

    # --- Parse recitals ---
    print(f"  Parsing recitals...")
    recitals = _split_gdpr_into_recitals(recital_text)
    for num, text in recitals:
        sections.append(ParsedSection(
            doc_id=doc_meta["doc_id"],
            doc_type=doc_meta["doc_type"],
            source_name=doc_meta["source_name"],
            full_name=doc_meta["full_name"],
            date=doc_meta["date"],
            section_type="recital",
            recital_number=num,
            text=text,
        ))

    # --- Parse articles ---
    print(f"  Parsing articles...")
    articles = _split_gdpr_into_articles(article_text)
    for art_num, art_title, art_body in articles:
        sub_parts = _parse_article_body(art_body)
        for para_num, point_letter, text in sub_parts:
            sections.append(ParsedSection(
                doc_id=doc_meta["doc_id"],
                doc_type=doc_meta["doc_type"],
                source_name=doc_meta["source_name"],
                full_name=doc_meta["full_name"],
                date=doc_meta["date"],
                section_type="article",
                article_number=art_num,
                article_title=art_title,
                paragraph_number=para_num,
                point=point_letter,
                text=text,
            ))

    print(f"  → {len([s for s in sections if s.section_type == 'recital'])} recitals, "
          f"{len([s for s in sections if s.section_type == 'article'])} article sections")
    return sections


# ---------------------------------------------------------------------------
# EDPB Guideline Parser
# ---------------------------------------------------------------------------

_SECTION_HEADING = re.compile(
    r"^(\d+(?:\.\d+)*)\s{1,4}([A-Z][^\n]{3,80})\n",
    re.MULTILINE,
)
_ANNEX_START = re.compile(r"^\s*Annex\b", re.MULTILINE | re.IGNORECASE)


def _extract_guideline_text(pdf_path: Path) -> str:
    """Extract text from a single-column EDPB guideline PDF."""
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            pages_text.append(text)
    return "\n".join(pages_text)


def _split_guideline_into_sections(full_text: str, doc_meta: dict) -> list[ParsedSection]:
    """
    Split a guideline into sections based on numbered headings (1, 1.1, 1.1.1 …).
    Falls back to paragraph-level splitting if no headings are found.
    """
    sections: list[ParsedSection] = []

    matches = list(_SECTION_HEADING.finditer(full_text))

    if len(matches) < 3:
        # Fallback: treat the whole document as one section
        sections.append(ParsedSection(
            doc_id=doc_meta["doc_id"],
            doc_type=doc_meta["doc_type"],
            source_name=doc_meta["source_name"],
            full_name=doc_meta["full_name"],
            date=doc_meta["date"],
            section_type="section",
            section_number=None,
            section_heading="Full Document",
            text=full_text.strip(),
        ))
        return sections

    # Include any preamble before the first section heading
    preamble = full_text[: matches[0].start()].strip()
    if len(preamble) > 100:
        sections.append(ParsedSection(
            doc_id=doc_meta["doc_id"],
            doc_type=doc_meta["doc_type"],
            source_name=doc_meta["source_name"],
            full_name=doc_meta["full_name"],
            date=doc_meta["date"],
            section_type="section",
            section_number="0",
            section_heading="Introduction / Preamble",
            text=preamble,
        ))

    for i, match in enumerate(matches):
        sec_num = match.group(1)
        sec_heading = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        text = full_text[start:end].strip()

        # Detect annexes
        sec_type = "annex" if _ANNEX_START.match(sec_heading) else "section"

        sections.append(ParsedSection(
            doc_id=doc_meta["doc_id"],
            doc_type=doc_meta["doc_type"],
            source_name=doc_meta["source_name"],
            full_name=doc_meta["full_name"],
            date=doc_meta["date"],
            section_type=sec_type,
            section_number=sec_num,
            section_heading=sec_heading,
            text=text,
        ))

    return sections


def parse_guideline(pdf_path: Path, doc_meta: dict) -> list[ParsedSection]:
    """Full parse of an EDPB guideline PDF."""
    print(f"  Extracting text from guideline PDF...")
    raw_text = _extract_guideline_text(pdf_path)
    raw_text = clean_text(raw_text)
    raw_text = remove_footnotes(raw_text)

    sections = _split_guideline_into_sections(raw_text, doc_meta)
    print(f"  → {len(sections)} sections")
    return sections


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def parse_document(pdf_path: Path, doc_meta: dict) -> list[ParsedSection]:
    """
    Route to the correct parser based on doc_type in the manifest entry.
    Returns a list of ParsedSection objects ready for chunking.
    """
    doc_type = doc_meta.get("doc_type", "guideline")
    if doc_type == "regulation":
        return parse_gdpr(pdf_path, doc_meta)
    else:
        return parse_guideline(pdf_path, doc_meta)


# ---------------------------------------------------------------------------
# Quick test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    manifest_path = project_root / "data" / "raw" / "manifest.json"

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Test with the first document in the manifest
    doc_meta = manifest[0]
    pdf_path = project_root / "data" / "raw" / doc_meta["path"]

    print(f"Parsing: {doc_meta['source_name']}")
    sections = parse_document(pdf_path, doc_meta)

    print(f"\nTotal sections: {len(sections)}")
    print("\nFirst 3 sections:")
    for s in sections[:3]:
        print(f"  [{s.section_type}] recital={s.recital_number} art={s.article_number} "
              f"para={s.paragraph_number} point={s.point}")
        print(f"  Text (first 120 chars): {s.text[:120]!r}")
        print()
