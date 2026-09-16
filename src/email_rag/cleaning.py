"""Conservative, explainable email cleanup and word-based chunking."""

from __future__ import annotations

import re
from html.parser import HTMLParser


class _HTMLTextExtractor(HTMLParser):
    """Small dependency-free HTML-to-text converter for Gmail HTML parts."""

    _BREAK_TAGS = {"p", "div", "br", "li", "tr", "blockquote", "hr"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignore_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._ignore_depth += 1
        if tag in self._BREAK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignore_depth:
            self._ignore_depth -= 1
        if tag in self._BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignore_depth:
            self.parts.append(data)


def html_to_text(value: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(value)
    return "".join(parser.parts)


_REPLY_START = re.compile(r"(?im)^on .{1,300}wrote:\s*$")
_FORWARDED_START = re.compile(r"(?im)^[- ]{0,8}original message[- ]{0,8}$")
_SIGNATURE_START = re.compile(r"(?m)^-- \s*$")
_MOBILE_SIGNATURE = re.compile(r"(?im)^sent from (?:my )?(?:iphone|ipad|android|mobile device).*$")


def clean_for_retrieval(body: str) -> str:
    """Remove common reply-history/signature noise without modifying the stored original."""
    text = html_to_text(body) if "<" in body and ">" in body else body
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Lines introduced with '>' are almost always quoted history. Keep authorship only.
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(">"))

    # Stop at the first conventional reply/forward separator. This favours the newest message.
    separator_positions = [
        match.start()
        for expression in (_REPLY_START, _FORWARDED_START, _SIGNATURE_START, _MOBILE_SIGNATURE)
        if (match := expression.search(text))
    ]
    if separator_positions:
        text = text[: min(separator_positions)]

    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_words: int = 220, overlap_words: int = 40) -> list[str]:
    """Make modest, overlapping chunks so an email's meaning survives retrieval boundaries."""
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")
    words = text.split()
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]

    step = max_words - overlap_words
    return [" ".join(words[start : start + max_words]) for start in range(0, len(words), step)]


def retrieval_chunks(
    *, subject: str, sender: str, received_at: str, cleaned_body: str
) -> list[str]:
    """Prefix each body segment with context that helps search and later citations."""
    header = f"Subject: {subject}\nFrom: {sender}\nDate: {received_at}\n\n"
    return [header + body_chunk for body_chunk in chunk_text(cleaned_body)]

