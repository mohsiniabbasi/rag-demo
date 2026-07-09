# What broke, and how I fixed it

Three real problems hit while building this RAG demo end-to-end. Each is the
kind of thing that demos fine on a slide and fails in production — so they're
worth writing down.

---

## 1. Retrieval returned *nothing* — the ivfflat index

**Symptom.** With the pipeline working and two chunks stored, some questions
came back with **zero** retrieved rows, while others returned one — on a
`LIMIT 2` query against a 2-row table, which should always return 2 rows.

**Root cause.** The schema created an `ivfflat` approximate-nearest-neighbour
index with `lists = 100`. ivfflat buckets vectors into lists and, by default,
probes only **one** list per query (`ivfflat.probes = 1`). On a tiny table most
of those 100 lists are empty, so a query whose vector fell into an empty list
matched nothing. The index was silently dropping results.

**Fix.** Removed the ANN index. For this corpus size, exact KNN via sequential
scan (`order by embedding <=> query`) is correct and fast well into the tens of
thousands of chunks. An approximate index only earns its keep at scale — and
then the right choice is HNSW, not ivfflat-with-large-lists on sparse data. The
schema documents when to add it.

**Lesson.** An ANN index trades recall for speed. On small data that trade is
all cost and no benefit, and the failure is invisible — no error, just missing
answers.

---

## 2. Similarity search errored — `vector <=> double precision[]`

**Symptom.** Inserts worked, but the search query raised
`operator does not exist: vector <=> double precision[]`.

**Root cause.** Postgres implicitly casts an array to `vector` when *assigning*
to a `vector` column (so inserts were fine), but the `<=>` distance operator has
no such implicit cast. The query embedding was being sent as a `float8[]`.

**Fix.** Pass embeddings as pgvector text literals (`'[v1,v2,...]'`) with an
explicit `::vector` cast in the SQL, for both insert and search.

**Lesson.** "It inserted fine" doesn't mean the type is right for every
operation. The write path and the query path can disagree.

---

## 3. Wrong chunk retrieved — naive fixed-size chunking

**Symptom.** Asking *"is there wifi?"* retrieved the **Payment and
cancellation** passage — the wrong topic entirely — and other questions matched
a large, blended chunk with the relevant sentence buried inside.

**Root cause.** The M1 baseline chunker split the document into fixed 120-word
windows, ignoring structure. That merged unrelated sections into one chunk and
split single topics across the boundary. WiFi (one line under *Amenities*) ended
up inside a window dominated by payment terms, so the chunk's embedding pointed
at "payments", not "wifi".

**Fix.** Structure-aware chunking: split on Markdown headings so each section
stays whole, and store the section title in metadata (which also enables
metadata filtering later). One section per chunk → tight, on-topic embeddings.

**Evidence.** Same questions, same embeddings, top-1 retrieval, `scripts/eval_chunking.py`:

| Question | Naive (score) | Structured (score) |
|---|---|---|
| do you allow dogs? | chunk #0 — 0.614 | **Pets — 0.727** |
| how much is late checkout? | chunk #0 — 0.570 | **Check-in and check-out — 0.719** |
| what deposit is needed to book? | chunk #1 — 0.575 | **Payment and cancellation — 0.683** |
| is there wifi? | **Payment and cancellation (wrong) — 0.587** | **Amenities — 0.651** |

Structured chunking scored higher on every question and, on the WiFi question,
returned the *correct* section where naive returned the wrong one.

**Lesson.** Retrieval quality is set at ingestion, not at query time. If the
chunk boundaries don't respect the document's structure, no amount of prompt
tuning downstream will recover the answer — the right text was never retrieved.
