"""Read-only Gmail API access and MIME body extraction."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from .config import Settings

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


@dataclass(frozen=True)
class Email:
    gmail_message_id: str
    gmail_thread_id: str
    subject: str
    sender: str
    recipients: str
    received_at: str
    labels: tuple[str, ...]
    body: str


def _decode_body(data: str) -> str:
    return base64.urlsafe_b64decode(data.encode("ascii") + b"===").decode(
        "utf-8", errors="replace"
    )


def _walk_parts(part: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield part
    for child in part.get("parts", []):
        yield from _walk_parts(child)


def _extract_body(payload: dict[str, Any]) -> str:
    """Prefer text/plain, then fall back to text/html. Attachments are intentionally ignored."""
    parts = list(_walk_parts(payload))
    for mime_type in ("text/plain", "text/html"):
        matching = [part for part in parts if part.get("mimeType") == mime_type]
        decoded = [
            _decode_body(part["body"]["data"])
            for part in matching
            if part.get("body", {}).get("data")
        ]
        if decoded:
            return "\n".join(decoded)
    return ""


class GmailInbox:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _service(self) -> Any:
        credentials: Credentials | None = None
        if self.settings.token_file.exists():
            credentials = Credentials.from_authorized_user_file(
                str(self.settings.token_file), SCOPES
            )

        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            else:
                if not self.settings.client_secrets_file.exists():
                    raise FileNotFoundError(
                        "Gmail OAuth client file not found at "
                        f"{self.settings.client_secrets_file}. Save a Desktop-app OAuth "
                        "client JSON there, then run ingest again."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.settings.client_secrets_file), SCOPES
                )
                credentials = flow.run_local_server(port=0)
            self.settings.token_file.parent.mkdir(parents=True, exist_ok=True)
            self.settings.token_file.write_text(credentials.to_json(), encoding="utf-8")

        return build("gmail", "v1", credentials=credentials, cache_discovery=False)

    def newest_inbox(self, limit: int) -> list[Email]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        service = self._service()
        message_refs = (
            service.users()
            .messages()
            .list(userId="me", labelIds=["INBOX"], maxResults=limit)
            .execute()
            .get("messages", [])
        )
        emails: list[Email] = []
        for ref in message_refs:
            message = (
                service.users()
                .messages()
                .get(userId="me", id=ref["id"], format="full")
                .execute()
            )
            headers = {
                header["name"].lower(): header["value"]
                for header in message.get("payload", {}).get("headers", [])
            }
            internal_ms = int(message.get("internalDate", "0"))
            received_at = datetime.fromtimestamp(
                internal_ms / 1000, tz=timezone.utc
            ).isoformat()
            emails.append(
                Email(
                    gmail_message_id=message["id"],
                    gmail_thread_id=message["threadId"],
                    subject=headers.get("subject", "(no subject)"),
                    sender=headers.get("from", "(unknown sender)"),
                    recipients=headers.get("to", ""),
                    received_at=received_at,
                    labels=tuple(message.get("labelIds", [])),
                    body=_extract_body(message.get("payload", {})),
                )
            )
        return emails

