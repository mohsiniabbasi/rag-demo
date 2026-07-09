"""Chunking — split a document into passages for embedding.

Two strategies, same `Chunk` shape so the rest of the pipeline is unchanged:

- `naive_chunks`   — fixed word windows with overlap. The M1 baseline; it
  ignores document structure, so unrelated sections get blended into one chunk
  and a single topic can be split across two. That dilutes the embedding and is
  the failure the M3 war-story fixes.
- `structured_chunks` — splits on Markdown headings so each section stays whole,
  and records the section title in metadata (also enables metadata filtering).

`chunk_document` dispatches by strategy name.
"""
import re
from dataclasses import dataclass, field

_HEADING = re.compile(r"^#{1,6}\s")


@dataclass
class Chunk:
    source: str
    chunk_index: int
    content: str
    metadata: dict = field(default_factory=dict)


def naive_chunks(
    text: str,
    source: str,
    *,
    words_per_chunk: int = 120,
    overlap: int = 20,
) -> list[Chunk]:
    """Fixed-size word windows with overlap. Deliberately structure-blind."""
    if words_per_chunk <= 0:
        raise ValueError("words_per_chunk must be positive")
    if overlap < 0 or overlap >= words_per_chunk:
        raise ValueError("overlap must be >= 0 and < words_per_chunk")

    words = text.split()
    if not words:
        return []

    step = words_per_chunk - overlap
    chunks: list[Chunk] = []
    for i, start in enumerate(range(0, len(words), step)):
        window = words[start : start + words_per_chunk]
        if not window:
            break
        chunks.append(
            Chunk(
                source=source,
                chunk_index=i,
                content=" ".join(window),
                metadata={"strategy": "naive", "words": len(window)},
            )
        )
        if start + words_per_chunk >= len(words):
            break
    return chunks


def structured_chunks(
    text: str,
    source: str,
    *,
    max_words: int = 180,
) -> list[Chunk]:
    """Split on Markdown headings; one section per chunk, section title in
    metadata. A section longer than `max_words` is sub-split with the naive
    windower but keeps its section tag."""
    sections: list[tuple[str | None, list[str]]] = []
    title: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        if _HEADING.match(line):
            if body:
                sections.append((title, body))
            title = line.lstrip("#").strip()
            body = [line]
        else:
            body.append(line)
    if body:
        sections.append((title, body))

    chunks: list[Chunk] = []
    idx = 0
    for sec_title, sec_lines in sections:
        # skip heading-only sections (e.g. the document H1 with no body)
        has_body = any(ln.strip() and not _HEADING.match(ln) for ln in sec_lines)
        if not has_body:
            continue
        content = "\n".join(sec_lines).strip()
        words = content.split()
        if len(words) > max_words:
            for sub in naive_chunks(content, source, words_per_chunk=max_words, overlap=20):
                chunks.append(
                    Chunk(source, idx, sub.content,
                          {"strategy": "structured", "section": sec_title or "", "words": sub.metadata["words"]})
                )
                idx += 1
        else:
            chunks.append(
                Chunk(source, idx, content,
                      {"strategy": "structured", "section": sec_title or "", "words": len(words)})
            )
            idx += 1
    return chunks


def chunk_document(text: str, source: str, strategy: str = "structured") -> list[Chunk]:
    if strategy == "naive":
        return naive_chunks(text, source)
    if strategy == "structured":
        return structured_chunks(text, source)
    raise ValueError(f"unknown chunk strategy: {strategy}")
