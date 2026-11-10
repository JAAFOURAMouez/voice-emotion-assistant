"""Local SQLite history: sessions and interactions.

Only the transcript, the reply, the emotion scores and the timestamp are stored
(plus the path of the turn audio when STORE_RAW_AUDIO is explicitly enabled).
Everything stays in a single local file that can be erased at any time.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from voxemo.emotion.types import EmotionEstimate, ProsodyFeatures

SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    ended_at TEXT
);
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    timestamp TEXT NOT NULL,
    transcript TEXT NOT NULL,
    reply TEXT NOT NULL,
    label TEXT,
    confidence REAL,
    scores_json TEXT,
    features_json TEXT,
    arousal REAL,
    model_scores_json TEXT,
    text_terms_json TEXT,
    audio_path TEXT
);
CREATE INDEX IF NOT EXISTS interactions_timestamp ON interactions(timestamp);
CREATE INDEX IF NOT EXISTS interactions_session ON interactions(session_id);
"""


@dataclass(frozen=True)
class Interaction:
    id: int
    session_id: int
    timestamp: datetime
    transcript: str
    reply: str
    estimate: EmotionEstimate | None
    audio_path: Path | None = None


@dataclass(frozen=True)
class SessionInfo:
    id: int
    started_at: datetime
    ended_at: datetime | None
    interaction_count: int


def _now() -> datetime:
    return datetime.now(UTC)


def _to_text(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return moment.astimezone(UTC).isoformat()


def _from_text(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _estimate_from_row(row: sqlite3.Row) -> EmotionEstimate | None:
    if row["label"] is None:
        return None
    return EmotionEstimate(
        label=row["label"],
        confidence=row["confidence"],
        scores=json.loads(row["scores_json"]),
        arousal=row["arousal"],
        features=ProsodyFeatures(**json.loads(row["features_json"])),
        model_scores=json.loads(row["model_scores_json"] or "{}"),
        text_terms=tuple(json.loads(row["text_terms_json"] or "[]")),
    )


class InteractionStore:
    """Thread-safe access to the history database (one connection, opened lazily)."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._connection: sqlite3.Connection | None = None
        self._lock = Lock()

    # --- connection ---

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._connection is None:
                self._connection = self._open()
            with self._connection:  # commit on success, roll back on error
                yield self._connection

    def _open(self) -> sqlite3.Connection:
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        # Erased rows are overwritten on disk, not just unlinked ("effacer mon historique").
        connection.execute("PRAGMA secure_delete = ON")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"history database {self.path} is from a newer version")
        connection.executescript(_SCHEMA)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        return connection

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def __enter__(self) -> InteractionStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # --- writing ---

    def start_session(self, started_at: datetime | None = None) -> int:
        with self._db() as db:
            cursor = db.execute(
                "INSERT INTO sessions (started_at) VALUES (?)", (_to_text(started_at or _now()),)
            )
            return int(cursor.lastrowid or 0)

    def end_session(self, session_id: int, ended_at: datetime | None = None) -> None:
        with self._db() as db:
            db.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ?",
                (_to_text(ended_at or _now()), session_id),
            )

    def add_interaction(
        self,
        session_id: int,
        *,
        transcript: str,
        reply: str,
        estimate: EmotionEstimate | None,
        timestamp: datetime | None = None,
        audio_path: str | Path | None = None,
    ) -> int:
        values: dict[str, Any] = {
            "session_id": session_id,
            "timestamp": _to_text(timestamp or _now()),
            "transcript": transcript,
            "reply": reply,
            "audio_path": str(audio_path) if audio_path else None,
        }
        if estimate is not None:
            values |= {
                "label": estimate.label,
                "confidence": estimate.confidence,
                "scores_json": json.dumps(estimate.scores),
                "features_json": json.dumps(dataclasses.asdict(estimate.features)),
                "arousal": estimate.arousal,
                "model_scores_json": json.dumps(estimate.model_scores),
                "text_terms_json": json.dumps(list(estimate.text_terms), ensure_ascii=False),
            }
        columns = ", ".join(values)
        placeholders = ", ".join(f":{name}" for name in values)
        with self._db() as db:
            cursor = db.execute(
                f"INSERT INTO interactions ({columns}) VALUES ({placeholders})", values
            )
            return int(cursor.lastrowid or 0)

    def clear_history(self) -> int:
        """Erase every session and interaction, and the saved turn audio; returns the count."""
        with self._db() as db:
            paths = [r[0] for r in db.execute("SELECT audio_path FROM interactions") if r[0]]
            count = db.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            db.execute("DELETE FROM interactions")
            db.execute("DELETE FROM sessions")
        for path in paths:
            Path(path).unlink(missing_ok=True)
        return int(count)

    # --- reading ---

    def interactions(
        self,
        *,
        since: datetime | None = None,
        session_id: int | None = None,
        limit: int | None = None,
    ) -> list[Interaction]:
        """Interactions in chronological order (the most recent ``limit`` ones if given)."""
        clauses, params = [], []
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(_to_text(since))
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM interactions {where} ORDER BY timestamp DESC, id DESC"
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        with self._db() as db:
            rows = db.execute(query, params).fetchall()
        return [
            Interaction(
                id=row["id"],
                session_id=row["session_id"],
                timestamp=_from_text(row["timestamp"]),
                transcript=row["transcript"],
                reply=row["reply"],
                estimate=_estimate_from_row(row),
                audio_path=Path(row["audio_path"]) if row["audio_path"] else None,
            )
            for row in reversed(rows)
        ]

    def recent_estimates(self, limit: int) -> list[EmotionEstimate]:
        """Latest estimates, oldest first: they seed the user's voice baseline."""
        return [i.estimate for i in self.interactions(limit=limit) if i.estimate is not None]

    def sessions(self) -> list[SessionInfo]:
        with self._db() as db:
            rows = db.execute(
                "SELECT s.id, s.started_at, s.ended_at, COUNT(i.id) AS n FROM sessions s "
                "LEFT JOIN interactions i ON i.session_id = s.id "
                "GROUP BY s.id ORDER BY s.started_at, s.id"
            ).fetchall()
        return [
            SessionInfo(
                id=row["id"],
                started_at=_from_text(row["started_at"]),
                ended_at=_from_text(row["ended_at"]) if row["ended_at"] else None,
                interaction_count=row["n"],
            )
            for row in rows
        ]
