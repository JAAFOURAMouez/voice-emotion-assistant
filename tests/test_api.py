"""Tests for the web API: REST routes and the conversation WebSocket (fake models)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from conftest import SAMPLE_RATE, EnergyDetector, FakeChatModel, make_estimate, silence, tone
from test_pipeline import FailingChatModel, FakeAnalyzer, FakeTranscriber
from voxemo.api.app import create_app, spoken_texts
from voxemo.api.services import Services
from voxemo.api.session import origin_allowed
from voxemo.config import Settings
from voxemo.tracking.store import InteractionStore
from voxemo.wellbeing.exercises import get_exercise


class FakeSpeaker:
    sample_rate = 22_050

    def __init__(self) -> None:
        self.said: list[str] = []

    def synthesize(self, text: str) -> np.ndarray:
        self.said.append(text)
        return np.zeros(220, dtype=np.float32)


def _services(tmp_path: Path, settings: Settings, **kw: Any) -> Services:
    return Services(
        settings=settings,
        transcriber=kw.get("transcriber", FakeTranscriber("Bonjour")),
        analyzer=kw.get("analyzer", FakeAnalyzer()),
        chat_model=kw.get("model", FakeChatModel("Bonjour. Je vous écoute.")),
        detector=EnergyDetector(),
        store=InteractionStore(tmp_path / "h.db"),
        speaker=kw.get("speaker"),
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, web_dist=tmp_path / "no-dist", llm_history_turns=3)


def _client(tmp_path: Path, settings: Settings, **kw: Any) -> tuple[TestClient, Services]:
    services = _services(tmp_path, settings, **kw)
    app = create_app(
        settings, services_factory=lambda _s: services, warm=lambda _s: None, background=False
    )
    return TestClient(app), services


def _pcm(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


SPEECH = _pcm(np.concatenate([silence(0.3), tone(1.0), silence(0.8)]))


def _receive_until_done(ws) -> tuple[list[dict], list[bytes]]:
    events, audio = [], []
    while True:
        message = ws.receive()
        if message.get("bytes") is not None:
            audio.append(message["bytes"])
            continue
        event = json.loads(message["text"])
        events.append(event)
        if event["type"] == "done":
            return events, audio


def _voice_turn(ws, pcm: bytes = SPEECH) -> tuple[list[dict], list[bytes]]:
    ws.send_json({"type": "start", "sample_rate": SAMPLE_RATE})
    assert ws.receive_json()["type"] == "recording"
    for start in range(0, len(pcm), 4096):
        ws.send_bytes(pcm[start : start + 4096])
    ws.send_json({"type": "end"})
    return _receive_until_done(ws)


def _types(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


# --- REST ---


def test_status_reports_ready_and_local(tmp_path: Path, settings: Settings) -> None:
    client, _ = _client(tmp_path, settings)
    with client:
        body = client.get("/api/status").json()
    assert body["state"] == "ready"
    assert body["llm"] == {"ok": True, "message": None}
    assert body["local_only"] is True and "cette machine" in body["notice"]
    assert body["sample_rate"] == 16_000


def test_status_shows_llm_problem(tmp_path: Path, settings: Settings) -> None:
    class Down(FakeChatModel):
        def check(self) -> str:
            return "Impossible de joindre Ollama"

    client, _ = _client(tmp_path, settings, model=Down())
    with client:
        assert client.get("/api/status").json()["llm"] == {
            "ok": False,
            "message": "Impossible de joindre Ollama",
        }


def test_no_external_docs_page(tmp_path: Path, settings: Settings) -> None:
    client, _ = _client(tmp_path, settings)
    with client:
        assert client.get("/docs").status_code == 404
        assert "npm run build" in client.get("/").text


def test_static_frontend_is_served(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<p>voxemo web</p>", encoding="utf-8")
    client, _ = _client(tmp_path, Settings(_env_file=None, web_dist=dist))
    with client:
        assert "voxemo web" in client.get("/").text
        assert client.get("/api/status").status_code == 200


def test_history_real_demo_and_clear(tmp_path: Path, settings: Settings) -> None:
    client, services = _client(tmp_path, settings)
    session = services.store.start_session()
    services.store.add_interaction(
        session, transcript="Je suis stressé", reply="", estimate=make_estimate("stress", 0.7)
    )
    with client:
        body = client.get("/api/history").json()
        assert body["simulated"] is False and body["notice"] is None
        [item] = body["interactions"]
        assert item["transcript"] == "Je suis stressé" and item["label"] == "stress"
        assert len(body["days"]) == 7 and len(body["trend"]) == 4

        demo = client.get("/api/history", params={"demo": True}).json()
        assert demo["simulated"] is True and "simulé" in demo["notice"]
        assert len(demo["interactions"]) >= 14

        assert client.delete("/api/history").json() == {"deleted": 1}
        assert client.get("/api/history").json()["interactions"] == []
    assert services.store.interactions() == []  # the demo was never stored


def test_exercises_and_their_voice(tmp_path: Path, settings: Settings) -> None:
    speaker = FakeSpeaker()
    client, _ = _client(tmp_path, settings, speaker=speaker)
    with client:
        listing = client.get("/api/exercises").json()
        assert listing[0]["id"] == "coherence_cardiaque"
        first = listing[0]
        assert first["steps"][0] == {"instruction": "Inspirez", "seconds": 5, "kind": "inspire"}
        assert first["cycles"] == 12 and first["total_seconds"] == 120
        assert first["spoken"] == spoken_texts(get_exercise("coherence_cardiaque"))
        audio = client.get("/api/exercises/coherence_cardiaque/speech/0")
        assert audio.status_code == 200 and audio.content[:4] == b"RIFF"
        assert audio.headers["content-type"] == "audio/wav"
        client.get("/api/exercises/coherence_cardiaque/speech/0")
        assert speaker.said == ["Inspirez"]  # cached
        assert client.get("/api/exercises/coherence_cardiaque/speech/99").status_code == 404
        assert client.get("/api/exercises/yoga").status_code == 404


def test_exercise_voice_unavailable(tmp_path: Path, settings: Settings) -> None:
    client, _ = _client(tmp_path, settings)
    with client:
        assert client.get("/api/exercises/respiration_carree/speech/0").status_code == 503


# --- WebSocket ---


def test_voice_turn_streams_every_step_in_order(tmp_path: Path, settings: Settings) -> None:
    speaker = FakeSpeaker()
    client, services = _client(
        tmp_path, settings, analyzer=FakeAnalyzer(make_estimate("stress", 0.7)), speaker=speaker
    )
    with client, client.websocket_connect("/api/turn") as ws:
        status = ws.receive_json()
        assert status["type"] == "status" and status["state"] == "ready" and status["voice"]
        events, audio = _voice_turn(ws)
    kinds = _types(events)
    assert kinds[:4] == ["transcript", "emotion", "reply_start", "reply_chunk"]
    assert kinds[-2:] == ["reply_end", "done"]
    emotion = events[1]
    assert emotion["label"] == "stress" and emotion["confidence"] == 0.7
    assert emotion["phrase"].startswith("Voix plus tendue que d'habitude")
    assert emotion["learning"] is True
    reply = "".join(e["text"] for e in events if e["type"] == "reply_chunk")
    assert reply.strip() == "Bonjour. Je vous écoute."
    # Spoken sentence by sentence, each header followed by its WAV.
    headers = [e for e in events if e["type"] == "speech"]
    assert [h["index"] for h in headers] == [0, 1]
    assert all(a[:4] == b"RIFF" for a in audio) and len(audio) == 2
    assert speaker.said == ["Bonjour.", "Je vous écoute."]
    assert events[-1]["timings"]["total_s"] >= 0
    [stored] = services.store.interactions()
    assert stored.transcript == "Bonjour" and stored.estimate is not None


def test_voice_can_be_turned_off(tmp_path: Path, settings: Settings) -> None:
    client, _ = _client(tmp_path, settings, speaker=FakeSpeaker())
    with client, client.websocket_connect("/api/turn") as ws:
        ws.receive_json()
        ws.send_json({"type": "voice", "enabled": False})
        events, audio = _voice_turn(ws)
    assert audio == [] and "speech" not in _types(events)


def test_silence_is_reported(tmp_path: Path, settings: Settings) -> None:
    client, _ = _client(tmp_path, settings)
    with client, client.websocket_connect("/api/turn") as ws:
        ws.receive_json()
        events, _ = _voice_turn(ws, _pcm(silence(1.5)))
    assert _types(events) == ["silence", "done"]


def test_unintelligible_turn(tmp_path: Path, settings: Settings) -> None:
    client, services = _client(tmp_path, settings, transcriber=FakeTranscriber(""))
    with client, client.websocket_connect("/api/turn") as ws:
        ws.receive_json()
        events, _ = _voice_turn(ws)
    assert _types(events) == ["not_understood", "done"]
    assert services.store.interactions() == []


def test_typed_message(tmp_path: Path, settings: Settings) -> None:
    client, _ = _client(tmp_path, settings)
    with client, client.websocket_connect("/api/turn") as ws:
        ws.receive_json()
        ws.send_json({"type": "text", "text": "  Bonjour  "})
        events, _ = _receive_until_done(ws)
    assert events[0] == {"type": "transcript", "text": "Bonjour", "typed": True}
    assert "emotion" not in _types(events)  # no voice, no vocal cue


def test_too_long_recording_is_refused(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, max_turn_s=1.0)
    client, _ = _client(tmp_path, settings)
    with client, client.websocket_connect("/api/turn") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        ws.receive_json()
        ws.send_bytes(_pcm(tone(7.0)))
        error = ws.receive_json()
    assert error["type"] == "error" and error["code"] == "too_long"


def test_bad_messages(tmp_path: Path, settings: Settings) -> None:
    client, _ = _client(tmp_path, settings)
    with client, client.websocket_connect("/api/turn") as ws:
        ws.receive_json()
        ws.send_text("not json")
        assert ws.receive_json()["code"] == "bad_message"
        ws.send_json({"type": "dance"})
        assert ws.receive_json()["code"] == "bad_message"
        ws.send_json({"type": "start", "sample_rate": 44_100})
        assert ws.receive_json()["code"] == "bad_rate"


def test_llm_unreachable_is_reported(tmp_path: Path, settings: Settings) -> None:
    client, _ = _client(tmp_path, settings, model=FailingChatModel())
    with client, client.websocket_connect("/api/turn") as ws:
        ws.receive_json()
        events, _ = _voice_turn(ws)
    error = next(e for e in events if e["type"] == "error")
    assert error["code"] == "llm" and "Ollama" in error["message"]


def test_distress_reply_points_to_helplines(tmp_path: Path, settings: Settings) -> None:
    client, _ = _client(
        tmp_path,
        settings,
        transcriber=FakeTranscriber("Je veux mourir"),
        model=FakeChatModel("Respirons ensemble."),
    )
    with client, client.websocket_connect("/api/turn") as ws:
        ws.receive_json()
        events, _ = _voice_turn(ws)
    assert "3114" in next(e for e in events if e["type"] == "reply_end")["text"]
    assert "offer" not in _types(events)


def test_exercise_offer_then_accept(tmp_path: Path, settings: Settings) -> None:
    client, _ = _client(
        tmp_path,
        settings,
        analyzer=FakeAnalyzer(make_estimate("stress", 0.8)),
        model=FakeChatModel("D'accord."),
    )
    with client, client.websocket_connect("/api/turn") as ws:
        ws.receive_json()
        first, _ = _voice_turn(ws)
        second, _ = _voice_turn(ws)
        assert "offer" not in _types(first)
        offer = next(e for e in second if e["type"] == "offer")
        assert "Voulez-vous essayer ?" in offer["text"]
        assert offer["exercise"]["steps"]
        ws.send_json({"type": "offer", "accept": True})
        third, _ = _receive_until_done(ws)
    exercise = next(e for e in third if e["type"] == "exercise")
    assert exercise["id"] == offer["exercise"]["id"]
    assert next(e for e in third if e["type"] == "reply_end")["text"].startswith("Très bien.")


def test_reset_starts_a_new_conversation(tmp_path: Path, settings: Settings) -> None:
    model = FakeChatModel("Oui.")
    client, _ = _client(tmp_path, settings, model=model)
    with client, client.websocket_connect("/api/turn") as ws:
        ws.receive_json()
        _voice_turn(ws)
        ws.send_json({"type": "reset"})
        assert ws.receive_json()["type"] == "done"
        _voice_turn(ws)
    assert len(model.calls[1]) == len(model.calls[0])  # no history carried over


def test_models_loading_then_ready(tmp_path: Path, settings: Settings) -> None:
    release = threading.Event()
    services = _services(tmp_path, settings)

    def slow_factory(_settings: Settings) -> Services:
        release.wait(5)
        return services

    app = create_app(settings, services_factory=slow_factory, warm=lambda _s: None)
    with TestClient(app) as client:
        assert client.get("/api/status").json()["state"] == "loading"
        assert client.get("/api/history").status_code == 503
        with client.websocket_connect("/api/turn") as ws:
            assert ws.receive_json()["state"] == "loading"
            ws.send_json({"type": "text", "text": "Bonjour"})
            error = ws.receive_json()
            assert error["code"] == "loading" and "Chargement" in error["message"]
            release.set()
            assert ws.receive_json() == {
                "type": "status",
                "state": "ready",
                "message": None,
                "llm": {"ok": True, "message": None},
                "voice": False,
                "warnings": [],
            }


def test_loading_failure_is_shown(tmp_path: Path, settings: Settings) -> None:
    def broken(_settings: Settings) -> Services:
        raise OSError("disk full")

    app = create_app(settings, services_factory=broken, warm=lambda _s: None, background=False)
    with TestClient(app) as client:
        body = client.get("/api/status").json()
    assert body["state"] == "error" and "disk full" in body["message"]


def test_foreign_origin_cannot_open_the_websocket(tmp_path: Path, settings: Settings) -> None:
    client, _ = _client(tmp_path, settings)
    with (
        client,
        pytest.raises(WebSocketDisconnect) as info,
        client.websocket_connect("/api/turn", headers={"origin": "https://evil.example"}) as ws,
    ):
        ws.receive_json()
    assert info.value.code == 1008


@pytest.mark.parametrize(
    ("origin", "host", "extra", "allowed"),
    [
        (None, "localhost:8000", "", True),
        ("http://localhost:5173", "localhost:8000", "", True),
        ("http://127.0.0.1:8000", "127.0.0.1:8000", "", True),
        ("https://voxemo.example.org", "voxemo.example.org", "", True),
        ("https://evil.example", "voxemo.example.org", "", False),
        ("https://evil.example", "localhost:8000", "https://evil.example/", True),
        ("http://192.168.1.20:8000", "localhost:8000", "", False),
    ],
)
def test_origin_rules(origin, host, extra, allowed) -> None:
    assert origin_allowed(origin, host, extra) is allowed


def test_turn_completes_even_if_history_cannot_be_saved(tmp_path: Path, settings: Settings) -> None:
    client, services = _client(tmp_path, settings)

    def broken(*_args: Any, **_kwargs: Any) -> int:
        raise OSError("read-only database")

    services.store.start_session = broken  # type: ignore[method-assign]
    with client, client.websocket_connect("/api/turn") as ws:
        ws.receive_json()
        events, _ = _voice_turn(ws)
    assert _types(events)[-2:] == ["reply_end", "done"]
