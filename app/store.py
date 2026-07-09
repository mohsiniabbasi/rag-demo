"""pgvector store — write chunks + embeddings, and similarity search.

Embeddings are passed as pgvector text literals ('[v1,v2,...]') with an
explicit `::vector` cast. Postgres will implicitly cast an array when
*assigning* to a vector column, but the `<=>` operator will not — so the
cast is required for search to work.

Search is used by M2's /ask endpoint; it lives here so ingestion and
retrieval share one code path.
"""
from psycopg.rows import dict_row
from psycopg.types.json import Json

from app.chunking import Chunk
from app.db import get_connection


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


def replace_source(source: str, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
    """Delete any existing rows for `source`, then insert the new chunks.
    Idempotent — re-ingesting a doc won't create duplicates. Returns row count.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must be the same length")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("delete from documents where source = %s", (source,))
            for chunk, embedding in zip(chunks, embeddings):
                cur.execute(
                    """
                    insert into documents (source, chunk_index, content, metadata, embedding)
                    values (%s, %s, %s, %s, %s::vector)
                    """,
                    (chunk.source, chunk.chunk_index, chunk.content, Json(chunk.metadata), _vec_literal(embedding)),
                )
        conn.commit()
    return len(chunks)


def search(query_embedding: list[float], top_k: int) -> list[dict]:
    """Return the top_k most similar chunks by cosine similarity.
    `score` is 1 - cosine_distance (higher = closer)."""
    lit = _vec_literal(query_embedding)
    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                select source, chunk_index, content, metadata,
                       1 - (embedding <=> %s::vector) as score
                from documents
                order by embedding <=> %s::vector
                limit %s
                """,
                (lit, lit, top_k),
            )
            return cur.fetchall()


def count() -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("select count(*) from documents")
            return cur.fetchone()[0]
