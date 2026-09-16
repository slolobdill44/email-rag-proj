# Local private email RAG — Stage 1

This is the deliberately small proof of the email RAG loop:

```text
Gmail (read-only) -> cleaned email chunks -> local embeddings -> SQLite -> retrieve -> local Ollama answer + citations
```

Email content is sent only to Gmail (to fetch it), the local embedding process, the local SQLite database, and Ollama at `127.0.0.1`. The first embedding-model download fetches public model weights; it does not upload email content.

## 1. Prerequisites

- Python 3.11 or newer
- [Ollama](https://ollama.com/) running locally
- A Google Cloud project with the Gmail API enabled and a **Desktop app** OAuth client

In Google Cloud, add your Gmail account as an OAuth test user if the consent screen is in Testing. Download the OAuth client JSON and save it as `config/credentials.json`. This file is deliberately git-ignored.

The app requests only `https://www.googleapis.com/auth/gmail.readonly`. It cannot label, trash, send, unsubscribe, or otherwise alter email.

## 2. Install and configure

```bash
cd /Users/lobby/email_proj
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
mkdir -p config data
cp .env.example .env
set -a; source .env; set +a
ollama pull qwen3:8b
```

The program reads environment variables rather than hard-coded paths. The `source .env` command above exports the development defaults. In a container, Compose will supply the same variables. To override just one value:

```bash
export OLLAMA_MODEL=qwen3:8b
```

## 3. Build the tiny index

```bash
email-rag ingest --limit 100
```

The first run opens a browser for Google OAuth, creates `data/gmail-token.json`, downloads the public embedding weights if needed, and indexes the 100 most-recent messages currently carrying Gmail's `INBOX` label.

## 4. Ask a grounded question

```bash
email-rag ask "Which projects had deadlines mentioned recently?"
```

The answer is constrained to retrieved email excerpts. Source IDs such as `[S1]` are printed with subject, sender, date, and a direct Gmail link after the answer.

## Useful checks

```bash
email-rag status
email-rag retrieve "invoice from August" --top-k 5
```

`retrieve` is intentionally included as a retrieval-quality check: it shows exactly what the answer model receives, before an LLM is involved.

## Stage-1 limits (intentional)

- SQLite similarity search scans all chunks in memory. That is fine for roughly 100 messages and is replaced by pgvector in Stage 2.
- Quoted history and signatures are removed with conservative heuristics. The original extracted body is kept in SQLite for inspection.
- This project supports only local Ollama answering; it deliberately has no cloud-model configuration.
