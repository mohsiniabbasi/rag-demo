"""Ingest a document into the pgvector store.

    python -m scripts.ingest data/booking-policy.md

Load → naive chunk → Gemini embed → store (idempotent per source).
Requires DATABASE_URL and GOOGLE_API_KEY in .env.
"""
import argparse
from pathlib import Path

from app.chunking import chunk_document
from app.config import settings
from app.embeddings import embed_texts
from app import store


def ingest_file(path: str, strategy: str | None = None) -> int:
    strategy = strategy or settings.chunk_strategy
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    chunks = chunk_document(text, p.name, strategy=strategy)
    if not chunks:
        print(f"no content in {path}")
        return 0
    print(f"{p.name}: {len(chunks)} chunks ({strategy}) -> embedding...")
    embeddings = embed_texts([c.content for c in chunks])
    n = store.replace_source(p.name, chunks, embeddings)
    print(f"stored {n} chunks. total rows in store: {store.count()}")
    return n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a doc into pgvector")
    parser.add_argument("path", help="path to a text/markdown file")
    parser.add_argument("--strategy", choices=["structured", "naive"], default=None,
                        help="chunking strategy (default: settings.chunk_strategy)")
    args = parser.parse_args()
    ingest_file(args.path, strategy=args.strategy)
