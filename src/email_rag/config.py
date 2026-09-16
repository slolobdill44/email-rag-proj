"""Configuration comes only from environment variables, never hard-coded user paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _path_from_env(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


@dataclass(frozen=True)
class Settings:
    client_secrets_file: Path
    token_file: Path
    database_path: Path
    embedding_model: str
    ollama_base_url: str
    ollama_model: str

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            client_secrets_file=_path_from_env(
                "GMAIL_CLIENT_SECRETS_FILE", "config/credentials.json"
            ),
            token_file=_path_from_env("GMAIL_TOKEN_FILE", "data/gmail-token.json"),
            database_path=_path_from_env("RAG_DB_PATH", "data/stage1.sqlite3"),
            embedding_model=os.environ.get(
                "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            ollama_base_url=os.environ.get(
                "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
            ).rstrip("/"),
            ollama_model=os.environ.get("OLLAMA_MODEL", "qwen3:8b"),
        )

