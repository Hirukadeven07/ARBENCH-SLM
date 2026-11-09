"""Persistent, thread-safe LLM response cache (SQLite).

Every request (model + messages + options + format) is hashed; the response and
its timing statistics are stored. Re-running the pipeline therefore resumes
exactly where it stopped – essential for multi-hour CPU benchmarks.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


def request_key(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class LLMCache:
    def __init__(self, path: str | Path | None):
        self.enabled = path is not None
        self._lock = threading.Lock()
        if not self.enabled:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=60)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, model TEXT, response TEXT, created REAL)"
        )
        self._conn.commit()

    def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        with self._lock:
            row = self._conn.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, model: str, response: dict) -> None:
        if not self.enabled:
            return
        import time

        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (key, model, response, created) VALUES (?, ?, ?, ?)",
                (key, model, json.dumps(response, ensure_ascii=False), time.time()),
            )
            self._conn.commit()

    def close(self) -> None:
        if self.enabled:
            with self._lock:
                self._conn.close()
