"""
generator.py

Takes a query + list of RetrievedChunk objects and generates a structured
legal brief using an LLM (Ollama local or Anthropic API).

Output is validated with Pydantic and rendered as a Markdown string.
"""

import json
import os
import re
from datetime import date
from typing import Optional

import ollama
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

from ..retrieval.retriever import RetrievedChunk

load_dotenv()

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# ---------------------------------------------------------------------------
# Known valid GDPR structural elements (for citation validation)
# ---------------------------------------------------------------------------

VALID_GDPR_ARTICLES = set(range(1, 100))     # Articles 1–99
VALID_GDPR_RECITALS = set(range(1, 174))     # Recitals 1–173


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------

class CitedProvision(BaseModel):
    citation_key: str
    provision_title: str
    relevance: str          # "Primary" | "Supporting" | "Interpretive"
    verbatim_text: str


class LegalBriefOutput(BaseModel):
    query: str
    applicable_provisions: list[CitedProvision]
    analysis: str = ""
    source_summary: str = ""
    verbatim_record: str = ""
    retrieval_metadata: dict = {}

    model_config = {"extra": "ignore"}  # silently drop unknown fields like "notes"

    @field_validator("applicable_provisions")
    @classmethod
    def must_have_provisions(cls, v):
        if not v:
            raise ValueError("Brief must cite at least one provision.")
        return v

    @field_validator("source_summary", "verbatim_record", mode="before")
    @classmethod
    def coerce_to_string(cls, v):
        """Accept list-of-dicts from Mistral and serialise to readable text."""
        if isinstance(v, list):
            lines = []
            for item in v:
                if isinstance(item, dict):
                    key = item.get("citation_key", "")
                    rel = item.get("relevance", "")
                    desc = item.get("description", item.get("summary", ""))
                    lines.append(f"- **{key}** ({rel}): {desc}")
                else:
                    lines.append(str(item))
            return "\n".join(lines)
        if not isinstance(v, str):
            return str(v)
        return v


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _format_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into the numbered [SOURCE N] block for the prompt."""
    lines = []
    for i, chunk in enumerate(chunks):
        lines.append(f"[SOURCE {i+1}]")
        lines.append(f"Citation: {chunk.citation_key}")
        lines.append(f"Document: {chunk.full_name}")
        lines.append(f"Type: {'Binding Provision' if chunk.doc_type == 'regulation' else 'Interpretive Guideline'}")
        lines.append(f"Text: {chunk.text}")
        lines.append("")
    return "\n".join(lines)


SYSTEM_PROMPT = """You are a legal research assistant specializing in EU data protection law (GDPR and EDPB guidelines).
Your task is to produce a structured legal brief based ONLY on the source passages provided.

STRICT RULES:
1. Never make legal claims not supported by the provided source passages.
2. Use verbatim quotations — do not paraphrase normative text.
3. Every factual or legal claim in the Analysis must be followed by a citation in square brackets like [GDPR Art. 5(1)(e)] or [EDPB Guidelines 05/2020 on Consent, §3] — use the exact Citation key from the source, wrapped in [ ]. Never write "citation key:" as text.
4. If the provided passages do not contain sufficient information to answer a specific aspect, state explicitly: "Insufficient source material to address [specific aspect]."
5. Do not invent or guess article numbers, recital numbers, or guideline names.
6. Distinguish clearly between BINDING provisions (GDPR Articles) and INTERPRETIVE guidance (Recitals, EDPB Guidelines).
7. Cite the exact Citation key as provided in the source — do not modify it.

Your output MUST be valid JSON matching this schema:
{
  "applicable_provisions": [
    {
      "citation_key": "exact citation key from source",
      "provision_title": "short title of the provision",
      "relevance": "Primary | Supporting | Interpretive",
      "verbatim_text": "exact quoted text from the source"
    }
  ],
  "analysis": "Full analysis text with inline [citation key] references",
  "source_summary": "Bullet-point summary of each cited source and its relevance",
  "verbatim_record": "Full verbatim text of every cited provision, labelled by citation key"
}"""


USER_PROMPT_TEMPLATE = """LEGAL QUERY: {query}

SOURCE PASSAGES:
{formatted_sources}

Generate a legal brief in the JSON format specified. Use the exact citation keys from the source passages."""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_ollama(system: str, user: str) -> str:
    client = ollama.Client(host=OLLAMA_BASE_URL)
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0},
    )
    return response["message"]["content"]


def _call_anthropic(system: str, user: str) -> str:
    import anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY")
    model = os.getenv("API_MODEL", "claude-haiku-4-5-20251001")
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=0,
    )
    return message.content[0].text


def _call_llm(system: str, user: str) -> str:
    if LLM_BACKEND == "api":
        return _call_anthropic(system, user)
    return _call_ollama(system, user)


# ---------------------------------------------------------------------------
# Citation validator
# ---------------------------------------------------------------------------

def _validate_citations(
    output: LegalBriefOutput,
    retrieved_chunks: list[RetrievedChunk],
) -> list[str]:
    """
    Check that every citation_key in the output actually appears in retrieved chunks.
    Returns a list of warning strings (empty = all good).
    """
    # Normalize keys by stripping deduplication/split suffixes like "[v2]" or "[part 3]"
    _suffix = re.compile(r"\s+\[(v\d+|part \d+)\]$")
    def _norm(key: str) -> str:
        return _suffix.sub("", key).strip()

    valid_keys = {chunk.citation_key for chunk in retrieved_chunks}
    valid_keys_norm = {_norm(k) for k in valid_keys}
    warnings = []
    for provision in output.applicable_provisions:
        key = provision.citation_key
        if key not in valid_keys and _norm(key) not in valid_keys_norm:
            warnings.append(f"HALLUCINATION WARNING: '{key}' was not in the retrieved sources.")
        # Check GDPR article numbers
        if "Art." in key:
            art_match = re.search(r"Art\.\s*(\d+)", key)
            if art_match:
                art_num = int(art_match.group(1))
                if art_num not in VALID_GDPR_ARTICLES:
                    warnings.append(f"INVALID ARTICLE: '{key}' references Art. {art_num} which does not exist in GDPR.")
        if "Recital" in key:
            rec_match = re.search(r"Recital\s+(\d+)", key)
            if rec_match:
                rec_num = int(rec_match.group(1))
                if rec_num not in VALID_GDPR_RECITALS:
                    warnings.append(f"INVALID RECITAL: '{key}' references Recital {rec_num} which does not exist in GDPR.")
    return warnings


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def _render_brief(
    query: str,
    output: LegalBriefOutput,
    retrieved_chunks: list[RetrievedChunk],
    warnings: list[str],
    today: str,
) -> str:
    """Render a LegalBriefOutput as a formatted Markdown string."""

    # Header
    lines = [
        "# LEGAL BRIEF",
        "",
        "## Query",
        query,
        "",
        "## Jurisdiction & Scope",
        "- **Legal Framework:** Regulation (EU) 2016/679 (GDPR)",
    ]

    guideline_names = list({
        c.source_name for c in retrieved_chunks if c.doc_type == "guideline"
    })
    if guideline_names:
        lines.append(f"- **Applicable Guidelines:** {', '.join(guideline_names)}")

    lines += [
        f"- **Brief Date:** {today}",
        "- **Disclaimer:** This brief is generated by an AI system for research purposes only "
        "and does not constitute legal advice.",
        "",
    ]

    # Applicable provisions table
    lines += [
        "## Applicable Legal Provisions",
        "",
        "| Citation | Provision Title | Relevance |",
        "|----------|----------------|-----------|",
    ]
    for p in output.applicable_provisions:
        lines.append(f"| {p.citation_key} | {p.provision_title} | {p.relevance} |")

    lines += ["", "## Analysis", ""]
    analysis = output.analysis.strip()
    if not analysis:
        # Synthesise a basic analysis from the provisions list
        analysis = (
            "The following provisions are directly applicable to this query. "
            "See the Key Verbatim Provisions section for the exact text of each.\n\n"
        )
        for p in output.applicable_provisions:
            analysis += f"- **{p.provision_title}** [{p.citation_key}]: {p.verbatim_text[:200]}...\n"
    lines.append(analysis)

    # Verbatim key provisions
    lines += ["", "## Key Verbatim Provisions", ""]
    for p in output.applicable_provisions:
        lines.append(f"**{p.citation_key}**")
        lines.append(f"> {p.verbatim_text}")
        lines.append("")

    # Source summary (only if LLM produced it)
    if output.source_summary.strip():
        lines += ["## Source Summary", ""]
        lines.append(output.source_summary)

    # Verbatim record (only if LLM produced it)
    if output.verbatim_record.strip():
        lines += ["", "## Full Verbatim Record", ""]
        lines.append(output.verbatim_record)

    # Warnings
    if warnings:
        lines += ["", "## ⚠️ Validation Warnings", ""]
        for w in warnings:
            lines.append(f"- {w}")

    # Retrieval metadata
    lines += [
        "",
        "## Retrieval Metadata",
        "",
        f"- **Chunks retrieved:** {len(retrieved_chunks)}",
        f"- **Retrieval method:** Hybrid BM25 + Dense + Cross-encoder reranking",
        f"- **Embedding model:** {os.getenv('EMBEDDING_MODEL', 'BAAI/bge-large-en-v1.5')}",
        f"- **Generation model:** {OLLAMA_MODEL if LLM_BACKEND == 'ollama' else os.getenv('API_MODEL', 'claude-haiku-4-5-20251001')}",
        "- **Sources used:**",
    ]
    for chunk in retrieved_chunks:
        lines.append(f"  - {chunk.citation_key} (rerank score: {chunk.rerank_score:.3f})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON cleanup
# ---------------------------------------------------------------------------

def _clean_llm_json(raw: str) -> str:
    """
    Robustly extract a JSON object from LLM output that may contain:
    - Markdown code fences (```json ... ```)
    - Extra fields with unquoted array values (e.g. "source_passages": [SOURCE_7])
    - Trailing commas before closing braces
    """
    text = raw.strip()

    # Strip markdown code fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:]
            if part.strip().startswith("{"):
                text = part.strip()
                break

    # Remove lines containing unquoted array values like: "field": [SOURCE_1, SOURCE_2]
    # These have [...] where the contents are not quoted strings
    text = re.sub(r'"[^"]+"\s*:\s*\[[^\]"]*\]', '""', text)

    # Remove trailing commas before } or ]
    text = re.sub(r",(\s*[}\]])", r"\1", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_brief(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    save_to: Optional[str] = None,
) -> str:
    """
    Generate a legal brief from a query and retrieved chunks.

    Args:
        query: The original user query.
        retrieved_chunks: Chunks returned by LegalRetriever.retrieve().
        save_to: Optional file path to save the output Markdown.

    Returns:
        Formatted Markdown brief string.
    """
    if not retrieved_chunks:
        return "# LEGAL BRIEF\n\nNo relevant sources found in the knowledge base for this query."

    formatted_sources = _format_chunks_for_prompt(retrieved_chunks)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        query=query,
        formatted_sources=formatted_sources,
    )

    # Call the LLM
    print("Calling LLM...")
    raw_response = _call_llm(SYSTEM_PROMPT, user_prompt)

    # Parse JSON output
    json_text = _clean_llm_json(raw_response)

    try:
        parsed = json.loads(json_text)
        parsed["query"] = query
        parsed["retrieval_metadata"] = {
            "chunks_retrieved": len(retrieved_chunks),
            "chunk_ids": [c.chunk_id for c in retrieved_chunks],
        }
        output = LegalBriefOutput(**parsed)
    except Exception as e:
        # If JSON parsing fails, return raw response with error note
        return (
            f"# LEGAL BRIEF\n\n**Error parsing structured output:** {e}\n\n"
            f"## Raw LLM Response\n\n{raw_response}"
        )

    # Validate citations
    warnings = _validate_citations(output, retrieved_chunks)
    if warnings:
        for w in warnings:
            print(f"  WARNING: {w}")

    today = date.today().isoformat()
    brief_md = _render_brief(query, output, retrieved_chunks, warnings, today)

    if save_to:
        with open(save_to, "w", encoding="utf-8") as f:
            f.write(brief_md)
        print(f"Brief saved to: {save_to}")

    return brief_md


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from ..retrieval.retriever import LegalRetriever

    query = "What are the conditions for lawful processing of personal data?"
    retriever = LegalRetriever()
    chunks = retriever.retrieve(query, top_k=6)
    brief = generate_brief(query, chunks)
    print(brief)
