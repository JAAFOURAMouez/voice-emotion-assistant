"""Streaming chat with a local Ollama model, plus conversation memory."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import httpx
import ollama

from voxemo import texts
from voxemo.config import Settings
from voxemo.emotion.types import EmotionEstimate
from voxemo.llm.prompts import ChatMessage, build_messages, system_prompt
from voxemo.wellbeing.safety import mentions_helpline, mentions_severe_distress

if TYPE_CHECKING:
    from collections.abc import Mapping


WARM_UP_TIMEOUT_S = 300.0


class LLMUnavailableError(RuntimeError):
    """The LLM could not produce a reply; the message is user-facing (French)."""


class ChatModel(Protocol):
    """Anything able to stream a chat completion, chunk by chunk."""

    def stream_chat(self, messages: Sequence[ChatMessage]) -> Iterator[str]: ...


@dataclass(frozen=True)
class LocalModel:
    """A chat model installed in Ollama."""

    name: str
    size_gb: float
    parameters: str
    thinking: bool
    """Reasoning model: its "thinking" is switched off to keep replies fast."""


def list_local_models(host: str, timeout_s: float = 3.0) -> list[LocalModel]:
    """Chat models installed in Ollama (embedding-only models left out).

    Raises :class:`LLMUnavailableError` when Ollama cannot be reached.
    """
    client = ollama.Client(host=host, timeout=timeout_s)
    try:
        listing = client.list()
    except (httpx.HTTPError, ConnectionError, ollama.ResponseError) as exc:
        raise LLMUnavailableError(texts.OLLAMA_UNREACHABLE.format(host=host)) from exc
    models = []
    for entry in listing.models:
        if not entry.model:
            continue
        capabilities = _capabilities(client, entry.model)
        if capabilities is None:  # older Ollama: guess from the name
            capabilities = {"embedding"} if "embed" in entry.model else {"completion"}
        if "completion" not in capabilities:
            continue
        details = entry.details
        models.append(
            LocalModel(
                name=entry.model.removesuffix(":latest"),
                size_gb=round((entry.size or 0) / 1e9, 1),
                parameters=(details.parameter_size or "") if details else "",
                thinking="thinking" in capabilities,
            )
        )
    return sorted(models, key=lambda m: m.name)


def _capabilities(client: ollama.Client, model: str) -> set[str] | None:
    try:
        found = client.show(model).capabilities
    except (httpx.HTTPError, ConnectionError, ollama.ResponseError):
        return None
    return set(found) if found else None


class OllamaChatModel:
    """`ChatModel` backed by a local Ollama server. The HTTP client is created lazily."""

    local = True

    def __init__(
        self,
        host: str,
        model: str,
        *,
        timeout_s: float = 120.0,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        self.host = host
        self.model = model
        self.timeout_s = timeout_s
        self.options = dict(options or {})
        self._client: ollama.Client | None = None
        self._warm_client: ollama.Client | None = None
        self._think: bool | str | None = "unknown"

    @classmethod
    def from_settings(cls, settings: Settings) -> OllamaChatModel:
        return cls(
            settings.ollama_host,
            settings.ollama_model,
            timeout_s=settings.ollama_timeout_s,
            options={"temperature": settings.ollama_temperature},
        )

    @property
    def client(self) -> ollama.Client:
        if self._client is None:
            self._client = ollama.Client(host=self.host, timeout=self.timeout_s)
        return self._client

    @contextmanager
    def _user_facing_errors(self) -> Iterator[None]:
        """Turn client/transport failures into :class:`LLMUnavailableError` (French)."""
        try:
            yield
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise LLMUnavailableError(
                    texts.OLLAMA_MODEL_MISSING.format(model=self.model)
                ) from exc
            raise LLMUnavailableError(texts.OLLAMA_ERROR.format(detail=exc.error)) from exc
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError(texts.OLLAMA_TIMEOUT.format(timeout=self.timeout_s)) from exc
        except (httpx.TransportError, ConnectionError) as exc:
            raise LLMUnavailableError(texts.OLLAMA_UNREACHABLE.format(host=self.host)) from exc

    def _think_option(self) -> bool | None:
        """``False`` for reasoning models (qwen3, deepseek-r1…): a voice reply cannot
        wait for a long hidden reasoning. ``None`` (Ollama's default) otherwise."""
        if self._think == "unknown":
            capabilities = _capabilities(self.client, self.model) or set()
            self._think = False if "thinking" in capabilities else None
        return self._think  # type: ignore[return-value]

    def stream_chat(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        with self._user_facing_errors():
            for chunk in self.client.chat(
                model=self.model,
                messages=payload,
                stream=True,
                options=self.options,
                think=self._think_option(),
            ):
                if content := chunk.message.content:
                    yield content

    def check(self, timeout_s: float = 3.0) -> str | None:
        """Quick health check: ``None`` if the server answers and has the model.

        Otherwise a user-facing message (unreachable server, model not pulled).
        """
        try:
            listing = ollama.Client(host=self.host, timeout=timeout_s).list()
        except (httpx.HTTPError, ConnectionError, ollama.ResponseError):
            return texts.OLLAMA_UNREACHABLE.format(host=self.host)
        names = {m.model for m in listing.models if m.model}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in names and self.model not in names:
            return texts.OLLAMA_MODEL_MISSING.format(model=self.model)
        return None

    def warm_up(self, messages: Sequence[ChatMessage] = ()) -> None:
        """Load the model into memory and pre-process ``messages`` (generating one token).

        Ollama keeps the processed prompt in its cache: warming up with the system
        prompt saves its evaluation (several seconds on CPU) on the first real turn.
        """
        payload = [{"role": m.role, "content": m.content} for m in messages]
        if self._warm_client is None:
            # Loading the model can take long on a cold start: allow more than a reply.
            timeout = max(self.timeout_s, WARM_UP_TIMEOUT_S)
            self._warm_client = ollama.Client(host=self.host, timeout=timeout)
        with self._user_facing_errors():
            self._warm_client.chat(
                model=self.model,
                messages=payload,
                stream=False,
                think=self._think_option(),
                options={**self.options, "num_predict": 1},
            )


class Conversation:
    """Keeps the dialogue history and adapts each turn to the estimated emotion."""

    def __init__(self, model: ChatModel, settings: Settings) -> None:
        self.model = model
        self.settings = settings
        self._history: list[ChatMessage] = []

    @property
    def history(self) -> tuple[ChatMessage, ...]:
        return tuple(self._history)

    def reset(self) -> None:
        self._history.clear()

    def warm_up(self) -> None:
        """Let the model load and cache the system prompt, if it supports it."""
        warm_up = getattr(self.model, "warm_up", None)
        if warm_up is not None:
            warm_up([ChatMessage("system", system_prompt(self.settings.language))])

    def add_exchange(self, user_text: str, reply: str) -> None:
        """Record an exchange the LLM did not write (e.g. a guided exercise)."""
        self._history += [ChatMessage("user", user_text), ChatMessage("assistant", reply)]
        del self._history[: -2 * self.settings.llm_history_turns]

    def reply_stream(
        self,
        user_text: str,
        estimate: EmotionEstimate | None = None,
        *,
        note: str | None = None,
        suffix: str = "",
    ) -> Iterator[str]:
        """Stream the assistant reply; the history is updated once the reply is complete.

        When the user's words suggest severe distress and the model failed to point to
        support resources, a fixed safety message is appended (the LLM is not trusted
        alone with this rule). ``note`` is an extra system instruction for this turn;
        ``suffix`` is fixed text added after the reply (and kept in the history).
        """
        messages = build_messages(
            user_text,
            self._history,
            estimate,
            threshold=self.settings.emotion_confidence_threshold,
            language=self.settings.language,
            note=note,
        )
        parts: list[str] = []
        for chunk in self.model.stream_chat(messages):
            parts.append(chunk)
            yield chunk

        if mentions_severe_distress(user_text) and not mentions_helpline("".join(parts)):
            fallback = ("\n\n" if parts else "") + texts.SAFETY_FALLBACK
            parts.append(fallback)
            yield fallback

        if suffix:
            extra = ("\n\n" if parts else "") + suffix
            parts.append(extra)
            yield extra

        self.add_exchange(user_text, "".join(parts).strip())

    def reply(self, user_text: str, estimate: EmotionEstimate | None = None) -> str:
        return "".join(self.reply_stream(user_text, estimate)).strip()
