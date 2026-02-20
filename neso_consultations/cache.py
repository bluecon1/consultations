from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SummaryCache:
    def __init__(self, path: Path) -> None:
        """Initialise a lightweight SQLite cache for summary payloads.

        Input:
            path: SQLite file path where cache rows should be stored.
        """
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._schema_ready = False

    def _connect(self) -> sqlite3.Connection:
        """Open a tuned SQLite connection to the configured cache file."""
        conn = sqlite3.connect(self._path, timeout=30)
        # Improve lock tolerance on shared infra and concurrent readers/writers.
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _ensure_schema(self) -> None:
        """Create the cache table if it does not already exist."""
        if self._schema_ready:
            return

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with self._connect() as conn:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS summary_cache (
                            cache_key TEXT PRIMARY KEY,
                            payload TEXT NOT NULL,
                            created_at TEXT NOT NULL
                        )
                        """
                    )
                    conn.commit()
                self._schema_ready = True
                return
            except sqlite3.OperationalError as exc:
                last_error = exc
                if "locked" not in str(exc).lower() or attempt >= 2:
                    raise
                time.sleep(0.5 * (attempt + 1))

        if last_error is not None:
            raise last_error

    def make_key(self, *, approach: str, target_id: str, model: str, data_fingerprint: str) -> str:
        """Build a deterministic cache key from request identity fields.

        Inputs:
            approach: Logical summarisation mode, e.g. `approach_1`.
            target_id: Response ID or question ID.
            model: LLM model identifier.
            data_fingerprint: Hash tied to source CSV state.

        Output:
            SHA-256 cache key string.
        """
        raw = f"{approach}|{target_id}|{model}|{data_fingerprint}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        """Fetch a cached summary payload by key.

        Input:
            cache_key: Key returned by `make_key`.

        Output:
            Parsed payload dictionary, or `None` when cache miss/invalid JSON.
        """
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM summary_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()

        if not row:
            return None

        try:
            parsed = json.loads(row[0])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None

        return None

    def set(self, cache_key: str, payload: Any) -> None:
        """Insert or update a summary payload under a cache key.

        Inputs:
            cache_key: Stable key from `make_key`.
            payload: Serializable object or dataclass graph.
        """
        self._ensure_schema()
        encoded = json.dumps(_to_serializable(payload), ensure_ascii=True)
        created_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO summary_cache (cache_key, payload, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload = excluded.payload,
                    created_at = excluded.created_at
                """,
                (cache_key, encoded, created_at),
            )
            conn.commit()


def _to_serializable(value: Any) -> Any:
    """Recursively convert nested dataclasses to JSON-serialisable objects."""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    return value
