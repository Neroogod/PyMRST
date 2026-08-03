"""Transactional durable metadata for Research Studio projects and runs."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import threading


class StudioMetadataStore:
    """Small SQLite ledger; numerical arrays remain in the result stores."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript("""
          CREATE TABLE IF NOT EXISTS studio_metadata (
            kind TEXT NOT NULL, identifier TEXT NOT NULL,
            payload TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(kind, identifier));
          CREATE TABLE IF NOT EXISTS studio_events (
            run_id TEXT NOT NULL, ordinal INTEGER NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY(run_id, ordinal));
          PRAGMA user_version=1;
        """)
        self._connection.commit()

    def put(self, kind, identifier, payload):
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO studio_metadata(kind,identifier,payload) VALUES(?,?,?) "
                "ON CONFLICT(kind,identifier) DO UPDATE SET payload=excluded.payload, "
                "updated_at=CURRENT_TIMESTAMP", (kind, identifier, encoded))

    def append_event(self, run_id, ordinal, payload):
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO studio_events(run_id,ordinal,payload) VALUES(?,?,?)",
                (run_id, int(ordinal), encoded))

    def get(self, kind, identifier):
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM studio_metadata WHERE kind=? AND identifier=?",
                (kind, identifier)).fetchone()
        return None if row is None else json.loads(row[0])

    def list(self, kind):
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM studio_metadata WHERE kind=? ORDER BY identifier",
                (kind,)).fetchall()
        return tuple(json.loads(row[0]) for row in rows)

    def events(self, run_id):
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM studio_events WHERE run_id=? ORDER BY ordinal",
                (run_id,)).fetchall()
        return tuple(json.loads(row[0]) for row in rows)

    def close(self):
        with self._lock:
            self._connection.close()


__all__ = ["StudioMetadataStore"]
