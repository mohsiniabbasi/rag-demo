"""Apply schema.sql to the configured database (idempotent).

    python -m scripts.setup_db

psycopg's extended protocol runs one statement per execute, so we strip
comments and split the script on ';'. schema.sql stays the source of truth.
"""
import re
from pathlib import Path

from app.db import get_connection


def apply_schema(path: str = "schema.sql") -> int:
    raw = Path(path).read_text(encoding="utf-8")
    no_comments = re.sub(r"--[^\n]*", "", raw)
    statements = [s.strip() for s in no_comments.split(";") if s.strip()]
    with get_connection() as conn:
        with conn.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)
        conn.commit()
    print(f"applied {len(statements)} statement(s) from {path}")
    return len(statements)


if __name__ == "__main__":
    apply_schema()
