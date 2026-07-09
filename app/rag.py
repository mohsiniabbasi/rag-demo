"""RAG orchestration — embed the question, retrieve, then generate a grounded
answer. This is the single entry point the /ask endpoint (and any future
channel) calls.
"""
from app import store
from app.config import settings
from app.embeddings import embed_text
from app.generate import generate_answer


def answer_question(question: str, top_k: int | None = None) -> dict:
    k = top_k or settings.top_k
    contexts = store.search(embed_text(question), k)
    answer = generate_answer(question, contexts)
    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "score": round(float(c["score"]), 3),
            }
            for c in contexts
        ],
    }
