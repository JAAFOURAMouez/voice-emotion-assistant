"""Tests for text-to-speech: text preparation, sentence streaming and the CLI wiring."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from voxemo.audio.recorder import AudioDeviceError
from voxemo.config import Settings
from voxemo.tts import speaker as speaker_module
from voxemo.tts.speaker import (
    PiperSpeaker,
    SoundDevicePlayer,
    SpeechQueue,
    create_speech,
    prepare_for_speech,
    split_sentences,
)


class FakeSpeaker:
    sample_rate = 22_050

    def __init__(self, error: Exception | None = None) -> None:
        self.texts: list[str] = []
        self.error = error

    def synthesize(self, text: str) -> np.ndarray:
        if self.error:
            raise self.error
        self.texts.append(text)
        return np.full(len(text), 0.1, dtype=np.float32)


class FakePlayer:
    def __init__(self, delay: float = 0.0, error: Exception | None = None) -> None:
        self.played: list[int] = []
        self.delay = delay
        self.error = error
        self.stopped = threading.Event()

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        if self.error:
            raise self.error
        self.stopped.clear()
        self.stopped.wait(self.delay)
        self.played.append(len(audio))

    def stop(self) -> None:
        self.stopped.set()


@pytest.fixture
def queue_parts():
    speaker, player = FakeSpeaker(), FakePlayer()
    speech = SpeechQueue(speaker, player)
    yield speech, speaker, player
    speech.close()


# --- text ---


def test_prepare_removes_markdown_links_and_emojis() -> None:
    text = "**Bonjour** 😊 ! Voir https://example.org pour `plus`."
    assert prepare_for_speech(text) == "Bonjour ! Voir pour plus."


def test_helpline_is_read_as_a_phone_number() -> None:
    assert prepare_for_speech("Appelez le 3114, ou le 15.") == "Appelez le 31 14, ou le 15."
    assert prepare_for_speech("Call 3114", "en") == "Call 3114"


def test_split_sentences_keeps_the_unfinished_rest() -> None:
    sentences, rest = split_sentences("Bonjour. Comment allez-vous ? Je")
    assert sentences == ["Bonjour.", "Comment allez-vous ?"]
    assert rest == "Je"


def test_split_on_newlines_and_long_clauses() -> None:
    assert split_sentences("Un\nDeux\n") == (["Un", "Deux"], "")
    long = "mot " * 40 + ", " + "suite " * 20
    sentences, rest = split_sentences(long)
    assert len(sentences) == 1 and sentences[0].endswith(",")
    assert rest.startswith("suite")


def test_decimal_points_do_not_split() -> None:
    assert split_sentences("Il est 3.5 fois plus") == ([], "Il est 3.5 fois plus")


# --- queue ---


def test_streamed_chunks_are_spoken_sentence_by_sentence(queue_parts) -> None:
    speech, speaker, player = queue_parts
    for chunk in ["Bon", "jour. Je vous ", "écoute. Et", " ensuite"]:
        speech.feed(chunk)
    assert speech.wait(2)
    assert speaker.texts == ["Bonjour.", "Je vous écoute."]
    speech.flush()
    assert speech.wait(2)
    assert speaker.texts[-1] == "Et ensuite"
    assert len(player.played) == 3


def test_first_sentence_plays_before_the_reply_ends(queue_parts) -> None:
    speech, _, player = queue_parts
    speech.feed("Première phrase. Deuxième")
    deadline = time.monotonic() + 2
    while not player.played and time.monotonic() < deadline:
        time.sleep(0.01)
    assert player.played, "the first sentence should already be playing"


def test_say_cleans_text_and_skips_empty(queue_parts) -> None:
    speech, speaker, _ = queue_parts
    speech.say("** ** 😊")
    speech.say("Appelez le 3114.")
    assert speech.wait(2)
    assert speaker.texts == ["Appelez le 31 14."]


def test_cancel_drops_queued_speech() -> None:
    speaker, player = FakeSpeaker(), FakePlayer(delay=5)
    speech = SpeechQueue(speaker, player)
    try:
        speech.say("Une. Deux. Trois. Quatre.")
        time.sleep(0.1)
        speech.cancel()
        assert speech.wait(2)
        assert len(player.played) <= 1
    finally:
        speech.close()


@pytest.mark.parametrize(
    ("speaker", "player"),
    [
        (FakeSpeaker(RuntimeError("voice broken")), FakePlayer()),
        (
            FakeSpeaker(),
            FakePlayer(error=AudioDeviceError("Impossible d'utiliser le haut-parleur")),
        ),
    ],
)
def test_failures_disable_speech_without_blocking(speaker, player) -> None:
    speech = SpeechQueue(speaker, player)
    try:
        speech.say("Bonjour. Encore.")
        assert speech.wait(2)
        assert not speech.enabled
        assert speech.error
        speech.say("Ignoré.")  # no longer queued
        assert speech.wait(0.5)
    finally:
        speech.close()


def test_warm_up_failure_is_reported_not_raised() -> None:
    speech = SpeechQueue(FakeSpeaker(RuntimeError("no voice")), FakePlayer())
    try:
        speech.warm_up()
        assert speech.error is not None and "no voice" in speech.error
    finally:
        speech.close()


# --- Piper and sounddevice wrappers ---


def test_piper_speaker_is_lazy_and_uses_speed(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    calls: dict[str, Any] = {}

    class FakeVoice:
        config = SimpleNamespace(sample_rate=22_050)

        def synthesize(self, text: str, config: Any) -> list[Any]:
            calls["length_scale"] = config.length_scale
            return [SimpleNamespace(audio_float_array=np.ones(3, dtype=np.float32))]

    speaker = PiperSpeaker("fr_FR-test", tmp_path, speed=1.25)
    (tmp_path / "fr_FR-test.onnx").write_bytes(b"")
    import piper

    loaded = []
    monkeypatch.setattr(
        piper.PiperVoice, "load", staticmethod(lambda p: loaded.append(p) or FakeVoice())
    )
    assert speaker._voice is None
    audio = speaker.synthesize("Bonjour")
    speaker.synthesize("Encore")
    assert len(audio) == 3 and speaker.sample_rate == 22_050
    assert loaded == [tmp_path / "fr_FR-test.onnx"]  # loaded once
    assert calls["length_scale"] == pytest.approx(0.8)


def test_missing_voice_is_downloaded_once(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import piper
    import piper.download_voices as dl

    downloads, messages = [], []

    def fake_download(voice: str, directory) -> None:
        downloads.append(voice)
        (directory / f"{voice}.onnx").write_bytes(b"")

    monkeypatch.setattr(dl, "download_voice", fake_download)
    monkeypatch.setattr(piper.PiperVoice, "load", staticmethod(lambda p: SimpleNamespace()))
    PiperSpeaker("fr_FR-x", tmp_path / "voices", on_download=messages.append).load()
    assert downloads == ["fr_FR-x"]
    assert "Téléchargement de la voix fr_FR-x" in messages[0]


def test_failed_download_is_user_facing(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import piper.download_voices as dl

    def offline(voice: str, directory) -> None:
        raise OSError("network unreachable")

    monkeypatch.setattr(dl, "download_voice", offline)
    with pytest.raises(speaker_module.TTSUnavailableError, match="Synthèse vocale désactivée"):
        PiperSpeaker("fr_FR-x", tmp_path).load()


class _FakeStream:
    latency = 0.0

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.blocks: list[int] = []
        self.closed = False

    def start(self) -> None:
        pass

    def write(self, data: np.ndarray) -> None:
        assert data.ndim == 2 and data.shape[1] == 1
        self.blocks.append(len(data))

    def close(self) -> None:
        self.closed = True


def _fake_sd(streams: list[_FakeStream], error: Exception | None = None) -> SimpleNamespace:
    class PortAudioError(Exception):
        pass

    def open_stream(**kwargs: Any) -> _FakeStream:
        if error:
            raise PortAudioError(str(error))
        streams.append(_FakeStream(**kwargs))
        return streams[-1]

    return SimpleNamespace(PortAudioError=PortAudioError, OutputStream=open_stream)


def test_player_writes_blocks_and_reuses_the_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    streams: list[_FakeStream] = []
    monkeypatch.setattr(speaker_module, "import_sounddevice", lambda: _fake_sd(streams))
    player = SoundDevicePlayer("3")
    player.play(np.zeros(22_050, dtype=np.float32), 22_050)
    player.play(np.zeros(100, dtype=np.float32), 22_050)
    assert len(streams) == 1
    assert streams[0].kwargs["device"] == 3 and streams[0].kwargs["samplerate"] == 22_050
    assert sum(streams[0].blocks) == 22_150 and max(streams[0].blocks) == 2205
    player.play(np.zeros(10, dtype=np.float32), 16_000)  # new rate: new stream
    assert len(streams) == 2 and streams[0].closed
    player.close()
    assert streams[1].closed


def test_player_stop_interrupts_between_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    streams: list[_FakeStream] = []
    monkeypatch.setattr(speaker_module, "import_sounddevice", lambda: _fake_sd(streams))
    player = SoundDevicePlayer()
    original_open = player._open

    def open_then_stop(rate: int) -> Any:
        stream = original_open(rate)
        stream.write = lambda data: (stream.blocks.append(len(data)), player.stop())
        return stream

    player._open = open_then_stop  # type: ignore[method-assign]
    player.play(np.zeros(22_050 * 5, dtype=np.float32), 22_050)
    assert len(streams[0].blocks) == 1


def test_player_maps_portaudio_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_sd([], error=RuntimeError("no device"))
    monkeypatch.setattr(speaker_module, "import_sounddevice", lambda: fake)
    with pytest.raises(AudioDeviceError, match="haut-parleur"):
        SoundDevicePlayer().play(np.zeros(10, dtype=np.float32), 22_050)


def test_create_speech_respects_settings() -> None:
    assert create_speech(Settings(_env_file=None, tts_enabled=False)) is None
    assert create_speech(Settings(_env_file=None), enabled=False) is None
    speech = create_speech(Settings(_env_file=None, tts_speed=1.2))
    assert speech is not None
    try:
        assert isinstance(speech.speaker, PiperSpeaker) and speech.speaker.speed == 1.2
    finally:
        speech.close()
