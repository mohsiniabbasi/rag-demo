"""Central configuration, loaded from environment / .env.

Nothing here requires a value at import time — the app must be able to boot
(and answer /health) even before the database or API keys are configured.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(_ENV_PATH), extra="ignore")

    # Supabase Postgres (pgvector). URI form from Supabase → Database → Connection string.
    database_url: str | None = None

    # Embeddings (M1). Dimension must match the `documents.embedding` column.
    embedding_provider: str = "gemini"
    embedding_model: str = "gemini-embedding-001"  # 3072 native; we request 768
    embedding_dim: int = 768
    openai_api_key: str | None = None  # only if EMBEDDING_PROVIDER=openai

    # Generation LLM (M2). Cheap + fast by default.
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash-lite"
    google_api_key: str | None = None

    # Chunking (M3): 'structured' (heading-aware) or 'naive' (fixed windows)
    chunk_strategy: str = "structured"

    # Retrieval
    top_k: int = 4


settings = Settings()
