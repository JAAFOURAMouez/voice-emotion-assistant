"""Which chat model answers: a local Ollama model, or a provider chosen by the user.

The choice and the user's API keys are kept in a small JSON file on this machine
(``LLM_CONFIG_PATH``, readable by its owner only). Keys are never sent back to the
browser: the interface only learns whether one is stored and its last characters.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from voxemo import texts
from voxemo.config import Settings
from voxemo.llm.client import ChatModel, LocalModel, OllamaChatModel, list_local_models
from voxemo.llm.prompts import ChatMessage
from voxemo.llm.providers import PROVIDERS, RemoteChatModel, check_base_url, key_hint

logger = logging.getLogger(__name__)

OLLAMA = "ollama"
MAX_KEY_LENGTH = 512
MAX_MODEL_LENGTH = 200


@dataclass(frozen=True)
class LlmChoice:
    provider: str = OLLAMA
    model: str = ""

    @property
    def local(self) -> bool:
        return self.provider == OLLAMA


@dataclass
class LlmConfig:
    choice: LlmChoice | None = None
    keys: dict[str, str] = field(default_factory=dict)
    base_urls: dict[str, str] = field(default_factory=dict)


class LlmConfigStore:
    """The JSON file holding the choice and the keys (owner-only permissions)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> LlmConfig:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return LlmConfig()
        except (OSError, ValueError):
            logger.warning("unreadable LLM settings in %s, ignored", self.path)
            return LlmConfig()
        saved = raw.get("choice")
        choice = None
        if isinstance(saved, dict):
            fields = LlmChoice.__annotations__
            choice = LlmChoice(**{k: str(v) for k, v in saved.items() if k in fields})
        return LlmConfig(
            choice=choice,
            keys={str(k): str(v) for k, v in (raw.get("keys") or {}).items()},
            base_urls={str(k): str(v) for k, v in (raw.get("base_urls") or {}).items()},
        )

    def save(self, config: LlmConfig) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "choice": asdict(config.choice) if config.choice else None,
            "keys": config.keys,
            "base_urls": config.base_urls,
        }
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)
        with contextlib.suppress(OSError):  # e.g. a Windows drive mounted in WSL
            os.chmod(self.path, 0o600)


def validate_key(key: str) -> str:
    key = key.strip()
    if not key or len(key) > MAX_KEY_LENGTH or any(c.isspace() for c in key):
        raise ValueError(texts.LLM_BAD_KEY)
    return key


def validate_model(model: str) -> str:
    model = model.strip()
    if not model or len(model) > MAX_MODEL_LENGTH or any(c.isspace() for c in model):
        raise ValueError(texts.LLM_BAD_MODEL)
    return model


class LlmSelector:
    """A `ChatModel` whose backend can be switched while the server runs.

    Every conversation shares it, so a new choice applies from the next reply on.
    """

    def __init__(
        self,
        settings: Settings,
        store: LlmConfigStore,
        *,
        ollama_factory: Callable[[str], ChatModel] | None = None,
        remote_factory: Callable[..., ChatModel] | None = None,
        lister: Callable[[], list[LocalModel]] | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self._ollama_factory = ollama_factory or self._ollama
        self._remote_factory = remote_factory or self._remote
        self._lister = lister or (lambda: list_local_models(settings.ollama_host))
        self._lock = threading.Lock()
        self.config = store.load()
        self.choice = LlmChoice(OLLAMA, settings.ollama_model)
        self.current: ChatModel = self._ollama_factory(settings.ollama_model)
        saved = self.config.choice
        if saved is not None:
            try:
                self._activate(saved)
            except ValueError as exc:
                logger.warning("saved LLM choice ignored: %s", exc)

    # --- factories ---

    def _ollama(self, model: str) -> ChatModel:
        s = self.settings
        return OllamaChatModel(
            s.ollama_host,
            model,
            timeout_s=s.ollama_timeout_s,
            options={"temperature": s.ollama_temperature},
        )

    def _remote(self, provider_id: str, model: str, api_key: str, base_url: str) -> ChatModel:
        return RemoteChatModel(
            PROVIDERS[provider_id],
            model,
            api_key=api_key,
            base_url=base_url,
            timeout_s=self.settings.ollama_timeout_s,
            temperature=self.settings.ollama_temperature,
        )

    # --- ChatModel ---

    def stream_chat(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        model = self.current  # a switch mid-reply does not cut it
        yield from model.stream_chat(messages)

    def check(self) -> str | None:
        check = getattr(self.current, "check", None)
        return check() if check is not None else None

    def warm_up(self, messages: Sequence[ChatMessage] = ()) -> None:
        warm_up = getattr(self.current, "warm_up", None)
        if warm_up is not None:
            warm_up(messages)

    # --- choosing ---

    def _build(self, choice: LlmChoice) -> ChatModel:
        if choice.local:
            return self._ollama_factory(validate_model(choice.model))
        if not self.settings.allow_cloud_llm:
            raise ValueError(texts.LLM_CLOUD_DISABLED)
        provider = PROVIDERS.get(choice.provider)
        if provider is None:
            raise ValueError(texts.LLM_UNKNOWN_PROVIDER)
        key = self.config.keys.get(provider.id, "")
        if provider.needs_key and not key:
            raise ValueError(texts.LLM_KEY_MISSING.format(provider=provider.name))
        base_url = self.config.base_urls.get(provider.id, "")
        if provider.id == "custom" and not base_url:
            raise ValueError(texts.LLM_BAD_URL)
        return self._remote_factory(provider.id, validate_model(choice.model), key, base_url)

    def _activate(self, choice: LlmChoice) -> None:
        model = self._build(choice)
        with self._lock:
            old, self.current, self.choice = self.current, model, choice
        close = getattr(old, "close", None)
        if close is not None and old is not model:
            close()

    def select(
        self,
        provider: str,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> LlmChoice:
        """Switch to ``model`` of ``provider``; a new key or URL is stored first.

        Raises ``ValueError`` with a user-facing message when the choice is invalid.
        """
        if provider != OLLAMA and provider not in PROVIDERS:
            raise ValueError(texts.LLM_UNKNOWN_PROVIDER)
        model = validate_model(model)
        if provider == OLLAMA:
            installed = {m.name for m in self.local_models()}
            if installed and model not in installed and f"{model}:latest" not in installed:
                raise ValueError(texts.OLLAMA_MODEL_MISSING.format(model=model))
        keys, urls = dict(self.config.keys), dict(self.config.base_urls)
        if api_key:
            self.config.keys[provider] = validate_key(api_key)
        if base_url:
            self.config.base_urls[provider] = check_base_url(base_url)
        choice = LlmChoice(provider, model)
        try:
            self._activate(choice)
        except ValueError:
            self.config.keys, self.config.base_urls = keys, urls  # nothing half-applied
            raise
        self.config.choice = choice
        self.store.save(self.config)
        return choice

    def forget_key(self, provider: str) -> None:
        """Delete a stored key; if that provider was answering, go back to Ollama."""
        self.config.keys.pop(provider, None)
        if self.choice.provider == provider:
            self._activate(LlmChoice(OLLAMA, self.settings.ollama_model))
            self.config.choice = None
        self.store.save(self.config)

    def remote_models(
        self, provider: str, *, api_key: str | None = None, base_url: str | None = None
    ) -> list[str]:
        """Ask a provider which models it offers (with a new key, or the stored one)."""
        found = PROVIDERS.get(provider)
        if found is None:
            raise ValueError(texts.LLM_UNKNOWN_PROVIDER)
        key = validate_key(api_key) if api_key else self.config.keys.get(provider, "")
        if found.needs_key and not key:
            raise ValueError(texts.LLM_KEY_MISSING.format(provider=found.name))
        url = check_base_url(base_url) if base_url else self.config.base_urls.get(provider, "")
        if provider == "custom" and not url:
            raise ValueError(texts.LLM_BAD_URL)
        model = self._remote_factory(provider, "-", key, url)
        try:
            return list(model.list_models())  # type: ignore[attr-defined]
        finally:
            close = getattr(model, "close", None)
            if close is not None:
                close()

    def local_models(self) -> list[LocalModel]:
        try:
            return self._lister()
        except Exception:  # Ollama down: the status already says so
            return []

    # --- what the interface may know ---

    @property
    def provider_name(self) -> str:
        if self.choice.local:
            return "Ollama"
        return PROVIDERS[self.choice.provider].name

    def state(self) -> dict[str, Any]:
        """Current choice, for the status message (no secret)."""
        return {
            "provider": self.choice.provider,
            "provider_name": self.provider_name,
            "model": self.choice.model,
            "local": self.choice.local,
        }

    def describe(self) -> dict[str, Any]:
        """Everything the settings panel needs; keys only as a hint (``…a1b2``)."""
        keys = self.config.keys
        return {
            "current": self.state(),
            "cloud_allowed": self.settings.allow_cloud_llm,
            "ollama_models": [asdict(m) for m in self.local_models()],
            "providers": [
                {
                    "id": p.id,
                    "name": p.name,
                    "base_url": self.config.base_urls.get(p.id, p.base_url),
                    "needs_key": p.needs_key,
                    "has_key": p.id in keys,
                    "key_hint": key_hint(keys[p.id]) if p.id in keys else None,
                }
                for p in PROVIDERS.values()
            ],
        }


def privacy_notice(selector: object | None) -> str:
    """The privacy line for the current model."""
    if isinstance(selector, LlmSelector) and not selector.choice.local:
        return texts.UI_PRIVACY_CLOUD.format(provider=selector.provider_name)
    return texts.UI_PRIVACY
