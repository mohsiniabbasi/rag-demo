"""Embeddings — turn text into vectors.

Default provider is Gemini (`text-embedding-004`, 768 dims), which reuses the
existing GOOGLE_API_KEY. The function signature is provider-agnostic so M2's
query embedding and M1's document embedding share one code path.
"""
import httpx

from app.config import settings

_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, preserving order. Raises with a clear message
    if the provider isn't configured."""
    if not texts:
        return []
    if settings.embedding_provider == "gemini":
        return _embed_gemini(texts)
    raise NotImplementedError(
        f"embedding_provider '{settings.embedding_provider}' not implemented "
        "(this build wires up 'gemini')"
    )


def embed_text(text: str) -> list[float]:
    """Convenience single-text embed (used by the query side in M2)."""
    return embed_texts([text])[0]


def _embed_gemini(texts: list[str]) -> list[list[float]]:
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY not set - cannot embed with Gemini")
    model = settings.embedding_model
    url = _GEMINI_ENDPOINT.format(model=model)
    out: list[list[float]] = []
    with httpx.Client(timeout=30) as client:
        for text in texts:
            resp = client.post(
                url,
                params={"key": settings.google_api_key},
                json={
                    "model": f"models/{model}",
                    "content": {"parts": [{"text": text}]},
                    # gemini-embedding-001 is 3072-dim natively; request a
                    # truncated size to match the pgvector column.
                    "outputDimensionality": settings.embedding_dim,
                },
            )
            resp.raise_for_status()
            values = resp.json()["embedding"]["values"]
            if len(values) != settings.embedding_dim:
                raise ValueError(
                    f"embedding dim {len(values)} != configured "
                    f"{settings.embedding_dim} — update EMBEDDING_DIM and the "
                    "schema's vector() column to match"
                )
            out.append(values)
    return out
