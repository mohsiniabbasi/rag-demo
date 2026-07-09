-- Pivot Bureau RAG demo — pgvector schema.
-- Run once against your Supabase Postgres (SQL editor or psql).

create extension if not exists vector;

create table if not exists documents (
    id          bigserial primary key,
    source      text        not null,   -- e.g. 'booking-policy.md'
    chunk_index int         not null,   -- position within the source doc
    content     text        not null,   -- the chunk text
    metadata    jsonb       not null default '{}'::jsonb,
    embedding   vector(768),            -- Gemini gemini-embedding-001 (truncated to 768); must match settings.embedding_dim
    created_at  timestamptz not null default now()
);

-- No ANN index by design. For this corpus size, exact KNN via sequential scan
-- (order by embedding <=> query) is correct and fast into the tens of thousands
-- of chunks. An approximate index only helps at scale — and ivfflat with a large
-- `lists` on few rows silently returns incomplete/empty results (default
-- probes = 1 hits an empty list). When the corpus grows large, add HNSW instead:
--   create index on documents using hnsw (embedding vector_cosine_ops);
drop index if exists documents_embedding_idx;
