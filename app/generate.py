"""Grounded generation — turn retrieved passages into a cited answer.

The system instruction enforces the "grounded or silent" rule: answer only
from the supplied context, and hand off rather than guess when the answer
isn't there. This is what makes the demo safe to put in front of customers.
"""
import httpx

from app.config import settings

_GEMINI_GEN = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

SYSTEM = (
    "You are a helpful support assistant for a small business. "
    "Answer the customer's question using ONLY the context passages provided below. "
    "If the answer is not in the context, say you don't have that information and "
    "will pass the question to the team - do NOT guess or use outside knowledge. "
    "Keep the answer short, warm and plain. "
    "At the end, on a new line starting 'Sources:', list the [source#index] tags "
    "you actually used (or 'Sources: none' if you handed off)."
)


def build_prompt(question: str, contexts: list[dict]) -> str:
    if contexts:
        context_block = "\n\n".join(
            f"[{c['source']}#{c['chunk_index']}] {c['content']}" for c in contexts
        )
    else:
        context_block = "(no relevant passages found)"
    return (
        f"{SYSTEM}\n\n"
        f"Context passages:\n{context_block}\n\n"
        f"Customer question: {question}\n\n"
        f"Answer:"
    )


def generate_answer(question: str, contexts: list[dict]) -> str:
    if settings.llm_provider != "gemini":
        raise NotImplementedError(
            f"llm_provider '{settings.llm_provider}' not implemented (this build wires 'gemini')"
        )
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY not set - cannot generate")

    url = _GEMINI_GEN.format(model=settings.llm_model)
    with httpx.Client(timeout=60) as client:
        resp = client.post(
            url,
            params={"key": settings.google_api_key},
            json={
                "contents": [{"parts": [{"text": build_prompt(question, contexts)}]}],
                "generationConfig": {"temperature": 0.2},
            },
        )
        resp.raise_for_status()
        data = resp.json()

    candidates = data.get("candidates", [])
    if not candidates:
        # Safety filter or empty response — fail safe by handing off.
        return "I don't have that information right now, so I'll pass your question to the team."
    return candidates[0]["content"]["parts"][0]["text"].strip()
