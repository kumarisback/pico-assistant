"""Persistent memory: notes with semantic recall, a user profile, and conversation history.

SQLite + brute-force cosine similarity. A personal assistant accumulates thousands
of notes, not millions, so a vector database would be pure overhead here.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT    NOT NULL,
    tags       TEXT    NOT NULL DEFAULT '',
    created_at TEXT    NOT NULL,
    embedding  BLOB
);
CREATE TABLE IF NOT EXISTS profile (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reminders (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    fire_at    TEXT    NOT NULL,
    text       TEXT    NOT NULL,
    repeat     TEXT    NOT NULL DEFAULT '',
    fired      INTEGER NOT NULL DEFAULT 0,
    created_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reminders_pending ON reminders (fired, fire_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Note:
    id: int
    text: str
    tags: str
    created_at: str
    score: float = 0.0


class Memory:
    def __init__(self, cfg) -> None:
        db_path = Path(cfg.path("memory", "db_path"))
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False + a lock: asyncio.to_thread uses a thread pool,
        # so connections are touched from varying threads but never concurrently.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()

        m = cfg["memory"]
        self._top_k = int(m["recall_top_k"])
        self._min_score = float(m["recall_min_score"])
        self._history_turns = int(m["history_turns"])

        from fastembed import TextEmbedding

        log.info("loading embedding model %s", m["embed_model"])
        self._embedder = TextEmbedding(
            model_name=m["embed_model"], cache_dir=str(cfg.path("memory", "embed_cache_dir"))
        )
        log.info("memory ready at %s", db_path)

    # ---------- embeddings ----------

    def _embed_sync(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self._embedder.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-9, None)

    # ---------- notes ----------

    def _remember_sync(self, text: str, tags: str) -> int:
        vec = self._embed_sync([text])[0]
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO notes (text, tags, created_at, embedding) VALUES (?, ?, ?, ?)",
                (text, tags, _now(), vec.tobytes()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    async def remember(self, text: str, tags: str = "") -> int:
        return await asyncio.to_thread(self._remember_sync, text, tags)

    def _recall_sync(self, query: str, limit: int) -> list[Note]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, text, tags, created_at, embedding FROM notes"
            ).fetchall()
        if not rows:
            return []

        matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        scores = matrix @ self._embed_sync([query])[0]

        order = np.argsort(-scores)[:limit]
        results = []
        for i in order:
            if float(scores[i]) < self._min_score:
                continue
            r = rows[int(i)]
            results.append(
                Note(
                    id=r["id"],
                    text=r["text"],
                    tags=r["tags"],
                    created_at=r["created_at"],
                    score=round(float(scores[i]), 3),
                )
            )
        return results

    async def recall(self, query: str, limit: int | None = None) -> list[Note]:
        return await asyncio.to_thread(self._recall_sync, query, limit or self._top_k)

    def _forget_sync(self, note_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            self._conn.commit()
            return cur.rowcount > 0

    async def forget(self, note_id: int) -> bool:
        return await asyncio.to_thread(self._forget_sync, note_id)

    # ---------- profile ----------

    def set_profile(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO profile (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "updated_at = excluded.updated_at",
                (key, value, _now()),
            )
            self._conn.commit()

    def get_profile(self) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute("SELECT key, value FROM profile").fetchall()
        return {r["key"]: r["value"] for r in rows}

    # ---------- conversation history ----------

    def add_turn(self, role: str, content: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO turns (role, content, created_at) VALUES (?, ?, ?)",
                (role, content, _now()),
            )
            self._conn.commit()

    def history(self) -> list[dict[str, str]]:
        """Most recent turns, oldest first, ready to splice into a chat request."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM turns ORDER BY id DESC LIMIT ?",
                (self._history_turns * 2,),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def clear_history(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM turns")
            self._conn.commit()

    # ---------- reminders ----------

    def add_reminder(self, fire_at: datetime, text: str, repeat: str = "") -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO reminders (fire_at, text, repeat, created_at) VALUES (?, ?, ?, ?)",
                (fire_at.isoformat(), text, repeat, _now()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def pending_reminders(self, until: datetime | None = None) -> list[dict]:
        sql = "SELECT * FROM reminders WHERE fired = 0"
        params: list = []
        if until:
            sql += " AND fire_at <= ?"
            params.append(until.isoformat())
        sql += " ORDER BY fire_at"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def mark_fired(self, reminder_id: int) -> None:
        with self._lock:
            self._conn.execute("UPDATE reminders SET fired = 1 WHERE id = ?", (reminder_id,))
            self._conn.commit()

    def cancel_reminder(self, reminder_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE reminders SET fired = 1 WHERE id = ? AND fired = 0", (reminder_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def export(self) -> str:
        return json.dumps(
            {"profile": self.get_profile(), "notes": [n.text for n in self._recall_sync("", 1000)]},
            indent=2,
        )
