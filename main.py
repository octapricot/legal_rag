"""
main.py — CLI entry point for the Legal RAG system.

Commands:
  ingest    Build / update the vector index from PDFs
  query     Ask a legal question and get a brief
  serve     Start a FastAPI server (optional)

Examples:
  python main.py ingest --reset
  python main.py ingest --doc-id gdpr_2016_679
  python main.py query "What are the conditions for valid consent under GDPR?"
  python main.py query "..." --save output/my_brief.md --top-k 10
  python main.py serve
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ingest command
# ---------------------------------------------------------------------------

def cmd_ingest(args):
    from src.ingestion.ingest import run_ingestion
    run_ingestion(target_doc_id=args.doc_id, reset=args.reset)


# ---------------------------------------------------------------------------
# Query command
# ---------------------------------------------------------------------------

def cmd_query(args):
    from src.retrieval.retriever import LegalRetriever
    from src.generation.generator import generate_brief

    query = args.query
    print(f"\nQuery: {query}\n")

    retriever = LegalRetriever()
    chunks = retriever.retrieve(
        query,
        top_k=args.top_k,
        doc_type_filter=args.filter,
    )

    if not chunks:
        print("No relevant sources found.")
        sys.exit(0)

    print(f"Retrieved {len(chunks)} chunks. Generating brief...\n")

    save_to = args.save
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)

    brief = generate_brief(query, chunks, save_to=save_to)
    print(brief)


# ---------------------------------------------------------------------------
# Serve command
# ---------------------------------------------------------------------------

def cmd_serve(args):
    import uvicorn
    from src.api import app
    uvicorn.run(app, host=args.host, port=args.port)


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="legal-rag",
        description="AI-powered GDPR legal research assistant",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- ingest --
    p_ingest = subparsers.add_parser("ingest", help="Build or update the document index")
    p_ingest.add_argument(
        "--doc-id", type=str, default=None,
        help="Only ingest a specific document (by doc_id from manifest)"
    )
    p_ingest.add_argument(
        "--reset", action="store_true",
        help="Wipe existing index before ingesting"
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # -- query --
    p_query = subparsers.add_parser("query", help="Ask a legal question")
    p_query.add_argument("query", type=str, help="Natural language legal query")
    p_query.add_argument(
        "--top-k", type=int, default=8,
        help="Number of source chunks to use (default: 8)"
    )
    p_query.add_argument(
        "--filter", type=str, choices=["regulation", "guideline"], default=None,
        help="Restrict retrieval to regulation-only or guideline-only"
    )
    p_query.add_argument(
        "--save", type=str, default=None,
        help="Save the brief as a Markdown file at this path"
    )
    p_query.set_defaults(func=cmd_query)

    # -- serve --
    p_serve = subparsers.add_parser("serve", help="Start the FastAPI server")
    p_serve.add_argument("--host", type=str, default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
