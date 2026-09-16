"""Disposable Stage-1 SQLite store with transparent in-process cosine retrieval."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .gmail_client import Email


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    gmail_message_id: str
    gmail_thread_id: str
    subject: str
    sender: str
    received_at: str
    text: str
    score: float

    @property
    def gmail_url(self) -> str:
        return f"https://mail.google.com/mail/u/0/#all/{self.gmail_message_id}"


class StageOneStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                gmail_message_id TEXT PRIMARY KEY,
                gmail_thread_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                sender TEXT NOT NULL,
                recipients TEXT NOT NULL,
                received_at TEXT NOT NULL,
                labels_json TEXT NOT NULL,
                original_body TEXT NOT NULL,
                cleaned_body TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                gmail_message_id TEXT NOT NULL REFERENCES messages(gmail_message_id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_dimension INTEGER NOT NULL,
                UNIQUE(gmail_message_id, ordinal)
            );
            CREATE INDEX IF NOT EXISTS chunks_by_message ON chunks(gmail_message_id);
            """
        )

    def close(self) -> None:
        self.connection.close()

    def status(self) -> tuple[int, int]:
        message_count = self.connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        chunk_count = self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return message_count, chunk_count

    def upsert_email(
        self,
        email: Email,
        cleaned_body: str,
        chunks: Sequence[str],
        vectors: np.ndarray,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("every chunk needs exactly one embedding")
        labels_json = ",".join(email.labels)
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO messages (
                    gmail_message_id, gmail_thread_id, subject, sender, recipients,
                    received_at, labels_json, original_body, cleaned_body
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(gmail_message_id) DO UPDATE SET
                    gmail_thread_id=excluded.gmail_thread_id, subject=excluded.subject,
                    sender=excluded.sender, recipients=excluded.recipients,
                    received_at=excluded.received_at, labels_json=excluded.labels_json,
                    original_body=excluded.original_body, cleaned_body=excluded.cleaned_body
                """,
                (
                    email.gmail_message_id,
                    email.gmail_thread_id,
                    email.subject,
                    email.sender,
                    email.recipients,
                    email.received_at,
                    labels_json,
                    email.body,
                    cleaned_body,
                ),
            )
            self.connection.execute(
                "DELETE FROM chunks WHERE gmail_message_id = ?", (email.gmail_message_id,)
            )
            self.connection.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, gmail_message_id, ordinal, text, embedding, embedding_dimension
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        f"{email.gmail_message_id}:{ordinal}",
                        email.gmail_message_id,
                        ordinal,
                        chunk,
                        np.asarray(vector, dtype=np.float32).tobytes(),
                        int(vector.shape[0]),
                    )
                    for ordinal, (chunk, vector) in enumerate(zip(chunks, vectors))
                ],
            )

    def retrieve(self, query_vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        rows = self.connection.execute(
            """
            SELECT c.chunk_id, c.gmail_message_id, c.text, c.embedding, c.embedding_dimension,
                   m.gmail_thread_id, m.subject, m.sender, m.received_at
            FROM chunks AS c JOIN messages AS m ON m.gmail_message_id = c.gmail_message_id
            """
        ).fetchall()
        if not rows:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        candidates: list[RetrievedChunk] = []
        for row in rows:
            vector = np.frombuffer(row["embedding"], dtype=np.float32, count=row["embedding_dimension"])
            if vector.shape != query.shape:
                raise RuntimeError(
                    "Embedding dimension changed. Delete the disposable Stage-1 database and re-ingest."
                )
            candidates.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    gmail_message_id=row["gmail_message_id"],
                    gmail_thread_id=row["gmail_thread_id"],
                    subject=row["subject"],
                    sender=row["sender"],
                    received_at=row["received_at"],
                    text=row["text"],
                    score=float(np.dot(query, vector)),
                )
            )
        return sorted(candidates, key=lambda chunk: chunk.score, reverse=True)[:top_k]

