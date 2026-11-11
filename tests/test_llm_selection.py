"""Choosing the chat model: OpenAI-compatible providers, stored keys, local Ollama models."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterator, Sequence
from pathlib import Path

import httpx
import pytest

from voxemo.config import Settings
from voxemo.llm.client import LLMUnavailableError, LocalModel
from voxemo.llm.prompts import ChatMessage
from voxemo.llm.providers import PROVIDERS, RemoteChatModel, check_base_url, key_hint
from voxemo.llm.selection import LlmConfigStore, LlmSelector, privacy_notice

KEY = "sk-test-0123456789abcd"
MESSAGES = [ChatMessage("system", "Sois bref."), ChatMessage("user", "Bonjour")]


def _sse(*chunks: str) -> bytes:
    lines = [": keep-alive", ""]
    for chunk in chunks:
        lines += [f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}", ""]
    lines += ["data: [DONE]", ""]
    return "\n".join(lines).encode()


def _remote(handler, provider: str = "openai", **kw) -> RemoteChatModel:
    return RemoteChatModel(
        PROVIDERS[provider], "gpt-test", api_key=KEY, transport=httpx.MockTransport(handler), **kw
    )


def test_streams_the_reply_and_sends_the_key() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=_sse("Bonjour", ", je vous écoute."))

    reply = "".join(_remote(handler).stream_chat(MESSAGES))
    assert reply == "Bonjour, je vous écoute."
    request = seen[0]
    assert request.url == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    body = json.loads(request.content)
    assert body["stream"] is True and body["model"] == "gpt-test"
    assert body["messages"][0] == {"role": "system", "content": "Sois bref."}


def test_anthropic_gets_its_own_headers() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=_sse("ok"))

    list(_remote(handler, "anthropic").stream_chat(MESSAGES))
    assert seen[0].headers["x-api-key"] == KEY
    assert seen[0].headers["anthropic-version"]


@pytest.mark.parametrize(
    ("status", "words"),
    [(401, "refuse la clé"), (404, "ne connaît pas"), (429, "quota"), (500, "erreur (500)")],
)
def test_http_errors_become_french_messages_without_the_key(status: int, words: str) -> None:
    model = _remote(lambda _r: httpx.Response(status, json={"error": {"message": KEY}}))
    with pytest.raises(LLMUnavailableError, match=r".*") as info:
        list(model.stream_chat(MESSAGES))
    assert words in str(info.value) and KEY not in str(info.value)


def test_network_failure_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    with pytest.raises(LLMUnavailableError, match="Impossible de joindre OpenAI"):
        list(_remote(handler).stream_chat(MESSAGES))


def test_lists_models_without_gemini_prefix() -> None:
    data = {"data": [{"id": "models/gemini-flash"}, {"id": "b"}, {"id": "a"}]}
    model = _remote(lambda _r: httpx.Response(200, json=data), "gemini")
    assert model.list_models() == ["a", "b", "gemini-flash"]


def test_base_url_rules() -> None:
    assert check_base_url("http://localhost:1234/v1/") == "http://localhost:1234/v1"
    assert check_base_url("https://example.org/v1") == "https://example.org/v1"
    with pytest.raises(ValueError, match="en clair"):
        check_base_url("http://example.org/v1")
    with pytest.raises(ValueError, match="invalide"):
        check_base_url("ftp://example.org")


def test_key_hint_never_shows_the_key() -> None:
    assert key_hint(KEY) == "…abcd"
    assert key_hint("short") == "…"


# --- the selector ---


class Recorder:
    """Stands for a chat model; remembers how it was built."""

    def __init__(self, *args: str) -> None:
        self.args = args
        self.closed = False

    def stream_chat(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        yield f"réponse de {self.args[-1] if len(self.args) == 1 else self.args[1]}"

    def list_models(self) -> list[str]:
        return ["m1", "m2"]

    def close(self) -> None:
        self.closed = True


def _selector(tmp_path: Path, **settings) -> LlmSelector:
    s = Settings(_env_file=None, llm_config_path=tmp_path / "llm.json", **settings)
    return LlmSelector(
        s,
        LlmConfigStore(s.llm_config_path),
        ollama_factory=lambda model: Recorder(model),
        remote_factory=lambda *a: Recorder(*a),
        lister=lambda: [
            LocalModel("llama3.2", 2.0, "3.2B", False),
            LocalModel("qwen3", 5.0, "8B", True),
        ],
    )


def test_default_is_the_local_model(tmp_path: Path) -> None:
    selector = _selector(tmp_path)
    assert selector.state() == {
        "provider": "ollama",
        "provider_name": "Ollama",
        "model": "llama3.2",
        "local": True,
    }
    assert "cette machine" in privacy_notice(selector)
    assert "".join(selector.stream_chat(MESSAGES)) == "réponse de llama3.2"


def test_switch_to_another_installed_ollama_model(tmp_path: Path) -> None:
    selector = _selector(tmp_path)
    selector.select("ollama", "qwen3")
    assert "".join(selector.stream_chat(MESSAGES)) == "réponse de qwen3"
    with pytest.raises(ValueError, match="pas installé"):
        selector.select("ollama", "absent")


def test_provider_with_key_is_saved_privately_and_survives_a_restart(tmp_path: Path) -> None:
    selector = _selector(tmp_path)
    selector.select("openai", "gpt-test", api_key=KEY)
    assert selector.state()["local"] is False
    assert "envoyé à OpenAI" in privacy_notice(selector)
    assert "audio reste" in privacy_notice(selector)

    path = tmp_path / "llm.json"
    if os.name == "posix" and not str(path).startswith("/mnt/"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    restarted = _selector(tmp_path)
    assert restarted.state()["model"] == "gpt-test"
    assert restarted.current.args == ("openai", "gpt-test", KEY, "")


def test_describe_never_contains_the_key(tmp_path: Path) -> None:
    selector = _selector(tmp_path)
    selector.select("mistral", "mistral-small", api_key=KEY)
    described = json.dumps(selector.describe())
    assert KEY not in described
    mistral = next(p for p in selector.describe()["providers"] if p["id"] == "mistral")
    assert mistral["has_key"] is True and mistral["key_hint"] == "…abcd"
    assert [m["name"] for m in selector.describe()["ollama_models"]] == ["llama3.2", "qwen3"]


def test_provider_without_key_is_refused_and_nothing_changes(tmp_path: Path) -> None:
    selector = _selector(tmp_path)
    with pytest.raises(ValueError, match="Aucune clé"):
        selector.select("groq", "llama-x")
    assert selector.state()["provider"] == "ollama"
    with pytest.raises(ValueError, match="en clair"):
        selector.select("custom", "m", base_url="http://example.org/v1")
    assert "custom" not in selector.config.base_urls


def test_custom_local_server_needs_no_key(tmp_path: Path) -> None:
    selector = _selector(tmp_path)
    selector.select("custom", "local-model", base_url="http://localhost:1234/v1")
    assert selector.current.args == ("custom", "local-model", "", "http://localhost:1234/v1")


def test_forgetting_the_key_in_use_goes_back_to_ollama(tmp_path: Path) -> None:
    selector = _selector(tmp_path)
    selector.select("openai", "gpt-test", api_key=KEY)
    remote = selector.current
    selector.forget_key("openai")
    assert selector.state()["provider"] == "ollama" and remote.closed
    assert KEY not in (tmp_path / "llm.json").read_text()


def test_cloud_can_be_disabled_by_the_server(tmp_path: Path) -> None:
    selector = _selector(tmp_path, allow_cloud_llm=False)
    with pytest.raises(ValueError, match="désactivés"):
        selector.select("openai", "gpt-test", api_key=KEY)
    assert selector.describe()["cloud_allowed"] is False


def test_remote_models_with_a_key_not_yet_saved(tmp_path: Path) -> None:
    selector = _selector(tmp_path)
    assert selector.remote_models("openai", api_key=KEY) == ["m1", "m2"]
    assert "openai" not in selector.config.keys  # only stored when chosen


def test_corrupt_settings_file_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "llm.json").write_text("{not json")
    assert _selector(tmp_path).state()["model"] == "llama3.2"


def test_distress_rule_still_applies_with_another_model(tmp_path: Path) -> None:
    """The safety fallback lives in Conversation, whatever model answers."""
    from voxemo.llm.client import Conversation

    selector = _selector(tmp_path)
    selector.select("openai", "gpt-test", api_key=KEY)
    reply = Conversation(selector, selector.settings).reply("Je veux mourir, je n'en peux plus.")
    assert "3114" in reply
