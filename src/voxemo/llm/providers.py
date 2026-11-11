"""Chat models beyond the local Ollama: any OpenAI-compatible API, with the user's own key.

Opt-in only: the default model stays local. When the user picks a provider, the text
of the conversation (transcripts, replies, the cautious emotion hint in the system
prompt) is sent to that provider; audio never leaves the machine.

No extra dependency: the OpenAI chat-completions protocol is spoken with httpx, and
most providers expose it (OpenAI, Mistral, Groq, OpenRouter, Google Gemini, Anthropic's
compatibility endpoint, local servers such as LM Studio or vLLM).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from voxemo import texts
from voxemo.llm.client import LLMUnavailableError
from voxemo.llm.prompts import ChatMessage

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


@dataclass(frozen=True)
class Provider:
    id: str
    name: str
    base_url: str
    """Empty for "custom": the user gives it."""
    needs_key: bool = True
    extra_headers: tuple[tuple[str, str], ...] = ()


PROVIDERS: dict[str, Provider] = {
    p.id: p
    for p in (
        Provider("openai", "OpenAI", "https://api.openai.com/v1"),
        Provider(
            "anthropic",
            "Anthropic",
            "https://api.anthropic.com/v1",
            extra_headers=(("anthropic-version", "2023-06-01"),),
        ),
        Provider("mistral", "Mistral AI", "https://api.mistral.ai/v1"),
        Provider(
            "gemini", "Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai"
        ),
        Provider("groq", "Groq", "https://api.groq.com/openai/v1"),
        Provider("openrouter", "OpenRouter", "https://openrouter.ai/api/v1"),
        Provider("custom", "Autre (compatible OpenAI)", "", needs_key=False),
    )
}


def check_base_url(url: str) -> str:
    """A provider URL: https, or plain http only towards this machine (keys travel in it)."""
    url = url.strip().rstrip("/")
    parsed = urlparse(url)
    if not parsed.hostname or parsed.scheme not in {"http", "https"}:
        raise ValueError(texts.LLM_BAD_URL)
    if parsed.scheme == "http" and parsed.hostname not in LOOPBACK_HOSTS:
        raise ValueError(texts.LLM_HTTP_URL)
    return url


def key_hint(key: str) -> str:
    """What the interface may show of a stored key: never the key itself."""
    return f"…{key[-4:]}" if len(key) >= 12 else "…"


class RemoteChatModel:
    """`ChatModel` speaking the OpenAI chat-completions protocol (streamed, SSE)."""

    local = False

    def __init__(
        self,
        provider: Provider,
        model: str,
        *,
        api_key: str = "",
        base_url: str = "",
        timeout_s: float = 120.0,
        temperature: float = 0.7,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = check_base_url(base_url or provider.base_url)
        self.timeout_s = timeout_s
        self.temperature = temperature
        self._transport = transport
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            headers = dict(self.provider.extra_headers)
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                if self.provider.id == "anthropic":
                    headers["x-api-key"] = self.api_key
            self._client = httpx.Client(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.timeout_s, connect=10.0),
                transport=self._transport,
            )
        return self._client

    @contextmanager
    def _user_facing_errors(self) -> Iterator[None]:
        name = self.provider.name
        try:
            yield
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code in (401, 403):
                message = texts.LLM_KEY_REJECTED.format(provider=name)
            elif code == 404:
                message = texts.LLM_MODEL_UNKNOWN.format(provider=name, model=self.model)
            elif code == 429:
                message = texts.LLM_QUOTA.format(provider=name)
            else:
                message = texts.LLM_PROVIDER_ERROR.format(provider=name, code=code)
            raise LLMUnavailableError(message) from exc
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError(
                texts.LLM_PROVIDER_TIMEOUT.format(provider=name, timeout=self.timeout_s)
            ) from exc
        except (httpx.TransportError, ConnectionError) as exc:
            raise LLMUnavailableError(texts.LLM_PROVIDER_UNREACHABLE.format(provider=name)) from exc

    def stream_chat(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        body = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "temperature": self.temperature,
        }
        with (
            self._user_facing_errors(),
            self.client.stream("POST", "/chat/completions", json=body) as response,
        ):
            if response.is_error:
                response.read()
                response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue  # SSE comments (": keep-alive"), event names, blank lines
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices") or ():
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        yield content

    def list_models(self) -> list[str]:
        """Model ids offered by the provider (``GET /models``)."""
        with self._user_facing_errors():
            response = self.client.get("/models", timeout=15.0)
            response.raise_for_status()
            data = response.json().get("data") or []
        ids = {str(m["id"]).removeprefix("models/") for m in data if m.get("id")}
        return sorted(ids)

    def check(self) -> str | None:
        """No network call (the status is polled): only what is missing locally."""
        if self.provider.needs_key and not self.api_key:
            return texts.LLM_KEY_MISSING.format(provider=self.provider.name)
        return None

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
