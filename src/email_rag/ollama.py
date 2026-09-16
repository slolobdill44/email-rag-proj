"""Small standard-library client for Ollama's local chat API."""

from __future__ import annotations

import json
from urllib.error import URLError
from urllib.request import Request, urlopen


class OllamaError(RuntimeError):
    pass


def answer_with_context(*, base_url: str, model: str, question: str, context: str) -> str:
    system_prompt = """You answer questions only from the supplied email excerpts.
If the excerpts do not support an answer, say exactly that you cannot determine it from the retrieved emails.
Do not use outside knowledge. Cite every factual claim using the supplied source markers such as [S1].
Do not invent citations. Be concise and do not reveal chain-of-thought or internal reasoning."""
    body = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0.1},
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Question: {question}\n\nRetrieved email excerpts:\n{context}",
            },
        ],
    }
    request = Request(
        f"{base_url}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except URLError as error:
        raise OllamaError(
            f"Could not reach local Ollama at {base_url}: {error.reason}. "
            "Start Ollama and ensure OLLAMA_BASE_URL points to it."
        ) from error
    except TimeoutError as error:
        raise OllamaError("Local Ollama did not respond within 180 seconds.") from error
    try:
        return str(payload["message"]["content"]).strip()
    except KeyError as error:
        raise OllamaError(f"Unexpected Ollama response: {payload}") from error

