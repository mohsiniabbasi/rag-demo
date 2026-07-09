"""M3 evidence: compare naive vs structured chunking on the same questions.

For each strategy we ingest the sample doc, then retrieve the single best chunk
for a set of targeted questions and print its similarity score + what section it
came from. Structured (heading-aware) chunks should score higher and return
tight, on-topic passages instead of big blended ones.

    python -m scripts.eval_chunking
"""
from scripts.ingest import ingest_file
from app.embeddings import embed_text
from app import store

QUESTIONS = [
    "do you allow dogs?",
    "how much is late checkout?",
    "what deposit is needed to book?",
    "is there wifi?",
]


def top_chunk(question: str) -> dict:
    rows = store.search(embed_text(question), top_k=1)
    return rows[0]


def run() -> None:
    results: dict[str, list[dict]] = {}
    for strategy in ("naive", "structured"):
        print(f"\n=== ingesting with {strategy} chunking ===")
        ingest_file("data/booking-policy.md", strategy=strategy)
        results[strategy] = [top_chunk(q) for q in QUESTIONS]

    print("\n" + "=" * 78)
    print(f"{'question':<32}{'naive':<23}{'structured':<23}")
    print("-" * 78)
    for i, q in enumerate(QUESTIONS):
        n, s = results["naive"][i], results["structured"][i]
        n_tag = f"#{n['chunk_index']} ({n['score']:.3f})"
        s_tag = f"{(s['metadata'].get('section') or '?')[:14]} ({s['score']:.3f})"
        print(f"{q:<32}{n_tag:<23}{s_tag:<23}")

    print("\n--- best-chunk content per question ---")
    for i, q in enumerate(QUESTIONS):
        print(f"\nQ: {q}")
        n, s = results["naive"][i], results["structured"][i]
        print(f"  naive      [{n['score']:.3f}] {n['content'][:100].replace(chr(10),' ')}...")
        print(f"  structured [{s['score']:.3f}] {s['content'][:100].replace(chr(10),' ')}...")


if __name__ == "__main__":
    run()
