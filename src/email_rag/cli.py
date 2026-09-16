"""CLI entry point for the Stage-1 RAG proof of concept."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .cleaning import clean_for_retrieval, retrieval_chunks
from .config import Settings
from .embeddings import LocalEmbedder
from .gmail_client import GmailInbox
from .ollama import answer_with_context
from .store import RetrievedChunk, StageOneStore


def _print_sources(results: Sequence[RetrievedChunk]) -> None:
    print("\nSources:")
    for index, result in enumerate(results, start=1):
        print(
            f"[S{index}] {result.subject} — {result.sender} — {result.received_at} "
            f"(score {result.score:.3f})\n     {result.gmail_url}"
        )


def _retrieve(question: str, top_k: int, settings: Settings) -> list[RetrievedChunk]:
    embedder = LocalEmbedder(settings.embedding_model)
    store = StageOneStore(settings.database_path)
    try:
        message_count, _ = store.status()
        if message_count == 0:
            raise RuntimeError("No indexed email. Run `email-rag ingest --limit 100` first.")
        return store.retrieve(embedder.embed([question])[0], top_k)
    finally:
        store.close()


def command_ingest(args: argparse.Namespace, settings: Settings) -> None:
    emails = GmailInbox(settings).newest_inbox(args.limit)
    print(f"Fetched {len(emails)} most-recent Inbox messages through Gmail read-only access.")
    embedder = LocalEmbedder(settings.embedding_model)
    store = StageOneStore(settings.database_path)
    try:
        indexed = 0
        skipped = 0
        for position, email in enumerate(emails, start=1):
            cleaned_body = clean_for_retrieval(email.body)
            chunks = retrieval_chunks(
                subject=email.subject,
                sender=email.sender,
                received_at=email.received_at,
                cleaned_body=cleaned_body,
            )
            if not chunks:
                skipped += 1
                print(f"[{position}/{len(emails)}] Skipped empty body: {email.subject}")
                continue
            store.upsert_email(email, cleaned_body, chunks, embedder.embed(chunks))
            indexed += 1
            print(f"[{position}/{len(emails)}] Indexed {len(chunks)} chunk(s): {email.subject}")
        messages, chunks = store.status()
    finally:
        store.close()
    print(f"Done. Indexed {indexed} emails this run ({skipped} empty); database has {messages} emails / {chunks} chunks.")


def command_status(settings: Settings) -> None:
    store = StageOneStore(settings.database_path)
    try:
        messages, chunks = store.status()
    finally:
        store.close()
    print(f"Stage-1 index: {messages} messages, {chunks} chunks at {settings.database_path}")


def command_retrieve(args: argparse.Namespace, settings: Settings) -> None:
    results = _retrieve(args.question, args.top_k, settings)
    if not results:
        print("No chunks found.")
        return
    for index, result in enumerate(results, start=1):
        print(f"\n[S{index}] score={result.score:.3f}\n{result.text}")
    _print_sources(results)


def command_ask(args: argparse.Namespace, settings: Settings) -> None:
    results = _retrieve(args.question, args.top_k, settings)
    if not results:
        print("No chunks found.")
        return
    context = "\n\n".join(f"[S{index}]\n{result.text}" for index, result in enumerate(results, start=1))
    answer = answer_with_context(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        question=args.question,
        context=context,
    )
    print(f"\nAnswer:\n{answer}")
    _print_sources(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local, read-only Gmail RAG — Stage 1")
    subcommands = parser.add_subparsers(dest="command", required=True)

    ingest = subcommands.add_parser("ingest", help="Fetch and index recent Inbox messages")
    ingest.add_argument("--limit", type=int, default=100, help="Number of Inbox messages (default: 100)")

    ask = subcommands.add_parser("ask", help="Answer a question from retrieved email excerpts")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=6)

    retrieve = subcommands.add_parser("retrieve", help="Inspect retrieved chunks without using an LLM")
    retrieve.add_argument("question")
    retrieve.add_argument("--top-k", type=int, default=6)

    subcommands.add_parser("status", help="Show local index counts")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings.from_environment()
    try:
        if args.command == "ingest":
            command_ingest(args, settings)
        elif args.command == "ask":
            command_ask(args, settings)
        elif args.command == "retrieve":
            command_retrieve(args, settings)
        elif args.command == "status":
            command_status(settings)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

