"""Persistent interaction and user-feedback storage for EnBot."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "feedback" / "enbot_feedback.db"
ALLOWED_FEEDBACK_CATEGORIES = {
    "事实错误", "回答不完整", "回答不相关", "型号混淆", "数值或单位错误",
    "链接或页码错误", "应拒答但没有拒答", "错误拒答", "其他",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_feedback_db_path() -> Path:
    configured = os.getenv("ENBOT_FEEDBACK_DB_PATH", "").strip()
    if not configured:
        return DEFAULT_DB_PATH
    path = Path(configured).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


class FeedbackStore:
    """Store chat interactions and one updatable feedback record per request."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else resolve_feedback_db_path()
        self._initialized = False
        self._lock = threading.RLock()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS interactions (
                        request_id TEXT PRIMARY KEY,
                        session_hash TEXT NOT NULL,
                        query TEXT NOT NULL,
                        response TEXT NOT NULL,
                        sources_json TEXT NOT NULL DEFAULT '[]',
                        retrieval_backend TEXT,
                        duration_seconds REAL NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        request_id TEXT NOT NULL UNIQUE,
                        rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
                        categories_json TEXT NOT NULL DEFAULT '[]',
                        comment TEXT NOT NULL DEFAULT '',
                        expected_answer TEXT NOT NULL DEFAULT '',
                        review_status TEXT NOT NULL DEFAULT 'pending',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (request_id)
                            REFERENCES interactions(request_id) ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS idx_feedback_rating
                        ON feedback(rating);
                    CREATE INDEX IF NOT EXISTS idx_feedback_review_status
                        ON feedback(review_status);
                    """
                )
            self._initialized = True

    @staticmethod
    def _session_hash(session_id: str) -> str:
        return hashlib.sha256(session_id.encode("utf-8")).hexdigest()

    def record_interaction(
        self, *, request_id: str, session_id: str, query: str, response: str,
        sources: list[str], retrieval_backend: str | None,
        duration_seconds: float,
    ) -> None:
        self.initialize()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO interactions (
                    request_id, session_hash, query, response, sources_json,
                    retrieval_backend, duration_seconds, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    session_hash = excluded.session_hash,
                    query = excluded.query,
                    response = excluded.response,
                    sources_json = excluded.sources_json,
                    retrieval_backend = excluded.retrieval_backend,
                    duration_seconds = excluded.duration_seconds,
                    created_at = excluded.created_at
                """,
                (
                    request_id, self._session_hash(session_id), query, response,
                    json.dumps(sources, ensure_ascii=False), retrieval_backend,
                    max(0.0, float(duration_seconds)), utc_now(),
                ),
            )

    def submit_feedback(
        self,
        *,
        request_id: str,
        rating: str,
        categories: list[str],
        comment: str,
        expected_answer: str,
    ) -> dict[str, Any]:
        self.initialize()
        invalid = sorted(set(categories) - ALLOWED_FEEDBACK_CATEGORIES)
        if invalid:
            raise ValueError("Unsupported feedback categories: " + ", ".join(invalid))
        now = utc_now()
        review_status = "accepted" if rating == "up" else "pending"
        with self._lock, self._connect() as connection:
            interaction = connection.execute(
                "SELECT request_id FROM interactions WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if interaction is None:
                raise KeyError(request_id)
            connection.execute(
                """
                INSERT INTO feedback (
                    request_id, rating, categories_json, comment,
                    expected_answer, review_status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    rating = excluded.rating,
                    categories_json = excluded.categories_json,
                    comment = excluded.comment,
                    expected_answer = excluded.expected_answer,
                    review_status = excluded.review_status,
                    updated_at = excluded.updated_at
                """,
                (
                    request_id,
                    rating,
                    json.dumps(categories, ensure_ascii=False),
                    comment,
                    expected_answer,
                    review_status,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                """
                SELECT id, request_id, rating, review_status, created_at, updated_at
                FROM feedback WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()
        return dict(row)

    def stats(self) -> dict[str, Any]:
        self.initialize()
        with self._lock, self._connect() as connection:
            interactions = connection.execute(
                "SELECT COUNT(*) AS count FROM interactions"
            ).fetchone()["count"]
            totals = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN rating = 'up' THEN 1 ELSE 0 END) AS up_count,
                       SUM(CASE WHEN rating = 'down' THEN 1 ELSE 0 END) AS down_count,
                       SUM(CASE WHEN review_status = 'pending' THEN 1 ELSE 0 END)
                           AS pending_count
                FROM feedback
                """
            ).fetchone()
            category_rows = connection.execute(
                "SELECT categories_json FROM feedback WHERE rating = 'down'"
            ).fetchall()
        total = int(totals["total"] or 0)
        up_count = int(totals["up_count"] or 0)
        down_count = int(totals["down_count"] or 0)
        category_counts: dict[str, int] = {}
        for row in category_rows:
            for category in json.loads(row["categories_json"] or "[]"):
                category_counts[category] = category_counts.get(category, 0) + 1
        return {
            "interactions_total": int(interactions),
            "feedback_total": total,
            "up_count": up_count,
            "down_count": down_count,
            "satisfaction_rate": round(up_count / total, 6) if total else None,
            "pending_review_count": int(totals["pending_count"] or 0),
            "category_counts": dict(sorted(category_counts.items())),
        }

    def export(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        self.initialize()
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT f.id, f.request_id, f.rating, f.categories_json,
                       f.comment, f.expected_answer, f.review_status,
                       f.created_at AS feedback_created_at, f.updated_at,
                       i.session_hash, i.query, i.response, i.sources_json,
                       i.retrieval_backend, i.duration_seconds,
                       i.created_at AS interaction_created_at
                FROM feedback AS f
                JOIN interactions AS i ON i.request_id = f.request_id
                ORDER BY f.updated_at DESC
                LIMIT ?
                """,
                (max(1, min(int(limit), 10000)),),
            ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            record = dict(row)
            record["categories"] = json.loads(record.pop("categories_json") or "[]")
            record["sources"] = json.loads(record.pop("sources_json") or "[]")
            records.append(record)
        return records


feedback_store = FeedbackStore()
