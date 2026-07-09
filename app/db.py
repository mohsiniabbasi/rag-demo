"""Database access for the pgvector store.

Connections are created lazily so the service can start without a database —
`check_connection()` is safe to call from /health and never raises.
"""
import psycopg

from app.config import settings


def get_connection() -> psycopg.Connection:
    """Open a new Postgres connection. Raises if DATABASE_URL is unset."""
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not configured")
    return psycopg.connect(settings.database_url)


def check_connection() -> tuple[bool, str]:
    """Return (reachable, detail). Never raises — for use in /health."""
    if not settings.database_url:
        return False, "DATABASE_URL not set"
    try:
        with psycopg.connect(settings.database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("select 1")
                cur.fetchone()
        return True, "reachable"
    except Exception as exc:  # noqa: BLE001 - health check reports, never crashes
        return False, str(exc)
