# Pivot Bureau RAG Demo — "chat with your business documents"

A grounded assistant that answers customer questions **only** from a business's
own documents (FAQs, pricing, policies) — with citations — and says "not in our
docs" instead of hallucinating.

**Stack:** FastAPI · Neon Postgres + pgvector · Gemini `gemini-embedding-001` (768-dim) · Gemini Flash-lite / Claude Haiku

## Pipeline
documents → chunking → embeddings → pgvector store → top-k retrieval (+ metadata) → grounded generation with citations

## Milestones
- [x] **M0 — skeleton & health** ✓ (`/health` boots without DB, config loads)
- [x] **M1 — ingestion** ✓ (live on Neon; sample doc embedded + retrievable by cosine similarity)
- [x] **M2 — retrieval + grounded answer** ✓ (`/ask` live: cited answers + refuse-and-handoff verified)
- [x] **M3 — the war story** ✓ (structure-aware chunking + section metadata; before/after in [`WAR-STORY.md`](WAR-STORY.md), reproduce with `python -m scripts.eval_chunking`)
- [x] **M4 — minimal chat UI** ✓ (served at `/`; sample questions, citation chips, amber handoff state, light/dark)
- [ ] M5 — deploy to `rag.pivotbureau.com` + public repo ← *next*
- [ ] M6 — add to Upwork / Fiverr portfolio

## Run locally (M0)
```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# Git Bash / *nix:      source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in DATABASE_URL etc.
uvicorn app.main:app --reload
```
Then open http://127.0.0.1:8000/health — it reports service status and whether
the database is reachable (the app boots fine even before the DB is configured).

## Database setup (Neon)
Create a free project at [neon.tech](https://neon.tech), open the SQL editor, and
run [`schema.sql`](schema.sql) (enables `pgvector` and creates the `documents`
table). Copy the connection string (it includes `?sslmode=require`) into `.env`
as `DATABASE_URL`.

> `embedding vector(768)` matches Gemini `text-embedding-004`. If you switch
> embedding models, update the column dimension and `EMBEDDING_DIM` to match.

## Ingest a document (M1)
```bash
python -m scripts.ingest data/booking-policy.md
```
Loads → structure-aware chunks → Gemini-embeds → stores in pgvector (idempotent per source).

## Deploy (Render)
Uses the [`Dockerfile`](Dockerfile) via the [`render.yaml`](render.yaml) blueprint:
1. Render → **New + → Blueprint** → connect this repo.
2. Set the two secret env vars in the dashboard: `DATABASE_URL` (Neon) and `GOOGLE_API_KEY` (Gemini). The rest come from the blueprint.
3. Deploy → Render gives a `*.onrender.com` URL. Add `rag.pivotbureau.com` as a custom domain, then CNAME it in Cloudflare.

Note: on the free tier both Render and Neon spin down when idle, so the first request after a lull cold-starts (~30–60s).
