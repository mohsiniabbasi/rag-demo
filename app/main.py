"""Pivot Bureau RAG demo — 'chat with your business documents'.

M0: skeleton + health. Ingestion (M1), retrieval + grounded answers (M2)
are added in later milestones.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app import db, rag
from app.config import settings

app = FastAPI(title="Pivot Bureau RAG Demo", version="0.1.0")

_INDEX_HTML = (Path(__file__).resolve().parent.parent / "web" / "index.html").read_text(encoding="utf-8")


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    """Answer a question grounded in the ingested documents, with citations."""
    return rag.answer_question(req.question, top_k=req.top_k)


@app.get("/health")
def health() -> dict:
    db_ok, db_detail = db.check_connection()
    return {
        "status": "ok",
        "service": "rag-demo",
        "version": app.version,
        "database": {"connected": db_ok, "detail": db_detail},
        "config": {
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "top_k": settings.top_k,
        },
    }


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return _INDEX_HTML
