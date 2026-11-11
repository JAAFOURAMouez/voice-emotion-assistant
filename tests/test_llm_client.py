"""Tests for the Ollama client wrapper, the conversation and the text chat loop."""

import io
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import httpx
import ollama
import pytest

from conftest import FakeChatModel, make_estimate
from voxemo import texts
from voxemo.config import Settings
from voxemo.llm.__main__ import ChatSession, main
from voxemo.llm.client import Conversation, LLMUnavailableError, OllamaChatModel
from voxemo.llm.prompts import ChatMessage

# --- OllamaChatModel (the real ollama.Client is replaced by a stub) ---


class _StubClient:
    def __init__(
        self,
        chunks: list[Any] | None = None,
        error: Exception | None = None,
        capabilities: list[str] | None = None,
    ) -> None:
        self.chunks = chunks or []
        self.error = error
        self.capabilities = capabilities or ["completion"]
        self.kwargs: dict[str, Any] = {}

    def show(self, model: str) -> Any:
        return SimpleNamespace(capabilities=self.capabilities)

    def chat(self, **kwargs: Any) -> Iterator[Any]:
        self.kwargs = kwargs
        yield from self.chunks
        if self.error:
            raise self.error


def _chunk(text: str) -> ollama.ChatResponse:
    return ollama.ChatResponse(message=ollama.Message(role="assistant", content=text))


def _model_with(stub: _StubClient) -> OllamaChatModel:
    model = OllamaChatModel("http://localhost:11434", "llama3.2", options={"temperature": 0.2})
    model._client = stub  # type: ignore[assignment]
    model._warm_client = stub  # type: ignore[assignment]
    return model


def test_ollama_streams_content_and_sends_messages() -> None:
    stub = _StubClient([_chunk("Bon"), _chunk(""), _chunk("jour")])
    chunks = list(_model_with(stub).stream_chat([ChatMessage("user", "Salut")]))
    assert chunks == ["Bon", "jour"]
    assert stub.kwargs["model"] == "llama3.2"
    assert stub.kwargs["stream"] is True
    assert stub.kwargs["messages"] == [{"role": "user", "content": "Salut"}]
    assert stub.kwargs["options"] == {"temperature": 0.2}
    assert stub.kwargs["think"] is None  # not a reasoning model: Ollama's default


def test_reasoning_models_do_not_think_before_a_voice_reply() -> None:
    stub = _StubClient([_chunk("Oui")], capabilities=["completion", "thinking"])
    assert list(_model_with(stub).stream_chat([ChatMessage("user", "Salut")])) == ["Oui"]
    assert stub.kwargs["think"] is False


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ollama.ResponseError("model not found", 404), "ollama pull llama3.2"),
        (ollama.ResponseError("boom", 500), "boom"),
        (httpx.ConnectError("refused"), "Impossible de joindre Ollama"),
        (ConnectionError("refused"), "Impossible de joindre Ollama"),
        (httpx.ReadTimeout("slow"), "trop de temps"),
    ],
)
def test_ollama_errors_become_user_messages(error: Exception, expected: str) -> None:
    with pytest.raises(LLMUnavailableError, match=expected):
        list(_model_with(_StubClient(error=error)).stream_chat([ChatMessage("user", "x")]))


def test_client_is_created_lazily() -> None:
    model = OllamaChatModel("http://localhost:11434", "llama3.2")
    assert model._client is None


def test_from_settings(settings: Settings) -> None:
    model = OllamaChatModel.from_settings(settings)
    assert (model.host, model.model) == (settings.ollama_host, settings.ollama_model)
    assert model.options == {"temperature": settings.ollama_temperature}


# --- Conversation ---


def test_conversation_streams_and_records_history(settings: Settings) -> None:
    fake = FakeChatModel("Bonjour à vous.")
    conversation = Conversation(fake, settings)
    assert conversation.reply("Bonjour") == "Bonjour à vous."
    assert conversation.history == (
        ChatMessage("user", "Bonjour"),
        ChatMessage("assistant", "Bonjour à vous."),
    )
    conversation.reply("Ça va ?")
    assert fake.calls[1][1:3] == list(conversation.history[:2])


def test_conversation_passes_emotion_hint(settings: Settings) -> None:
    fake = FakeChatModel()
    Conversation(fake, settings).reply("J'ai trop de choses à faire", make_estimate("stress"))
    assert "la voix suggère une certaine tension" in fake.calls[0][-2].content


def test_hint_is_not_stored_in_history(settings: Settings) -> None:
    conversation = Conversation(FakeChatModel(), settings)
    conversation.reply("Bof", make_estimate("tristesse", 0.9))
    assert all(m.role != "system" for m in conversation.history)


def test_history_is_bounded(settings: Settings) -> None:
    conversation = Conversation(FakeChatModel(), settings)
    for i in range(10):
        conversation.reply(f"message {i}")
    assert len(conversation.history) == 2 * settings.llm_history_turns
    assert conversation.history[0] == ChatMessage("user", "message 7")


def test_failed_turn_leaves_history_unchanged(settings: Settings) -> None:
    class Broken:
        def stream_chat(self, messages: object) -> Iterator[str]:
            raise LLMUnavailableError("down")
            yield ""

    conversation = Conversation(Broken(), settings)
    with pytest.raises(LLMUnavailableError):
        conversation.reply("Bonjour")
    assert conversation.history == ()


def test_distress_reply_always_points_to_helpline(settings: Settings) -> None:
    fake = FakeChatModel("Essayons une respiration profonde.")
    reply = Conversation(fake, settings).reply("Je veux en finir, je n'en peux plus")
    assert "3114" in reply
    assert texts.SAFETY_FALLBACK in reply


def test_no_duplicate_safety_message_when_model_complies(settings: Settings) -> None:
    fake = FakeChatModel("Je suis là. Vous pouvez appeler le 3114, jour et nuit.")
    reply = Conversation(fake, settings).reply("J'ai envie de mourir")
    assert texts.SAFETY_FALLBACK not in reply


# --- Text chat loop ---


def _reader(lines: list[str]) -> Callable[[str], str]:
    """Simulate a terminal: return the lines one by one, then end of input."""
    inputs = iter(lines)

    def read_line(_prompt: str) -> str:
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError from None

    return read_line


def _run_session(lines: list[str], fake: FakeChatModel, settings: Settings) -> str:
    out = io.StringIO()
    ChatSession(Conversation(fake, settings)).run(_reader(lines), out)
    return out.getvalue()


def test_chat_session_commands(settings: Settings) -> None:
    fake = FakeChatModel("Je comprends.")
    output = _run_session(
        ["/emotion tristesse 0,8", "Rien ne va", "/emotion bof", "/reset", "/quit", "jamais lu"],
        fake,
        settings,
    )
    assert "Indice vocal simulé : tristesse (confiance 0,8)" in output
    assert "Assistant : Je comprends." in output
    assert texts.CLI_EMOTION_USAGE in output
    assert texts.CLI_RESET in output
    assert output.rstrip().endswith(texts.CLI_GOODBYE)
    assert len(fake.calls) == 1
    assert "de la tristesse" in fake.calls[0][-2].content


def test_chat_session_reports_llm_errors(settings: Settings) -> None:
    class Down:
        def stream_chat(self, messages: object) -> Iterator[str]:
            raise LLMUnavailableError("Ollama est arrêté")
            yield ""

    out = io.StringIO()
    ChatSession(Conversation(Down(), settings)).run(_reader(["Bonjour"]), out)
    assert "Ollama est arrêté" in out.getvalue()


def test_main_accepts_simulated_emotion(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeChatModel("Prenons le temps.")
    monkeypatch.setattr("builtins.input", _reader(["Je suis débordé"]))
    assert main(["--emotion", "stress", "--confidence", "0.9"], settings=settings, model=fake) == 0
    assert "Prenons le temps." in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--confidence", "1.5"], settings=settings, model=fake)
    assert "confiance 0,9" in fake.calls[0][-2].content


def test_warm_up_preprocesses_messages_with_one_token() -> None:
    stub = _StubClient()
    calls: list[dict[str, Any]] = []
    stub.chat = lambda **kw: calls.append(kw)  # type: ignore[method-assign]
    _model_with(stub).warm_up([ChatMessage("system", "Tu es un assistant.")])
    assert calls == [
        {
            "model": "llama3.2",
            "messages": [{"role": "system", "content": "Tu es un assistant."}],
            "stream": False,
            "think": None,
            "options": {"temperature": 0.2, "num_predict": 1},
        }
    ]


def test_conversation_warm_up_is_optional(settings: Settings) -> None:
    Conversation(FakeChatModel(), settings).warm_up()  # model without warm_up: no-op


def test_warm_up_maps_errors() -> None:
    stub = _StubClient()

    def refuse(**_: Any) -> None:
        raise httpx.ConnectError("refused")

    stub.chat = refuse  # type: ignore[method-assign]
    with pytest.raises(LLMUnavailableError, match="Impossible de joindre Ollama"):
        _model_with(stub).warm_up()


def test_local_model_list_leaves_out_embedding_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from voxemo.llm import client as client_module

    class Lister:
        def __init__(self, **_kw: Any) -> None: ...

        def list(self) -> ollama.ListResponse:
            return ollama.ListResponse(
                models=[
                    ollama.ListResponse.Model(model="qwen3:8b", size=5_200_000_000),
                    ollama.ListResponse.Model(model="llama3.2:latest", size=2_000_000_000),
                    ollama.ListResponse.Model(model="nomic-embed-text:latest", size=1),
                ]
            )

        def show(self, model: str) -> SimpleNamespace:
            caps = {
                "qwen3:8b": ["completion", "thinking"],
                "nomic-embed-text:latest": ["embedding"],
            }
            return SimpleNamespace(capabilities=caps.get(model, ["completion"]))

    monkeypatch.setattr(client_module.ollama, "Client", Lister)
    models = client_module.list_local_models("http://localhost:11434")
    assert [(m.name, m.thinking, m.size_gb) for m in models] == [
        ("llama3.2", False, 2.0),
        ("qwen3:8b", True, 5.2),
    ]
