"""Tests for the Whisper wrapper (fake model) and the transcription command."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from conftest import SAMPLE_RATE, EnergyDetector, silence, tone
from voxemo.audio.recorder import AudioArray
from voxemo.config import Settings
from voxemo.stt.__main__ import transcribe_file
from voxemo.stt.transcriber import Transcription, WhisperTranscriber, is_hallucination


class _FakeWhisperModel:
    def __init__(self, *texts: str, language: str = "fr") -> None:
        self.texts = texts
        self.language = language
        self.kwargs: dict[str, Any] = {}

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[Any, Any]:
        self.kwargs = kwargs
        segments = (SimpleNamespace(text=t) for t in self.texts)
        return segments, SimpleNamespace(language=self.language, language_probability=0.97)


def _transcriber(model: _FakeWhisperModel, **kwargs: Any) -> WhisperTranscriber:
    transcriber = WhisperTranscriber(**kwargs)
    transcriber._model = model
    return transcriber


def test_joins_segments_and_reports_language() -> None:
    model = _FakeWhisperModel(" Bonjour,", " comment ça va ? ")
    result = _transcriber(model).transcribe(silence(2.0))
    assert result == Transcription("Bonjour, comment ça va ?", "fr", 0.97, 2.0)
    assert model.kwargs["language"] == "fr"
    assert model.kwargs["vad_filter"] is False
    assert model.kwargs["temperature"] == 0.0  # no slow fallback re-decoding


def test_drops_known_hallucinations() -> None:
    model = _FakeWhisperModel(" Sous-titres réalisés par la communauté d’Amara.org")
    result = _transcriber(model).transcribe(silence(1.0))
    assert result.is_empty


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Merci d'avoir regardé !", True),
        ("Thank you for watching.", True),
        ("Merci d'avoir regardé mon dossier", False),
        ("Bonjour", False),
    ],
)
def test_is_hallucination(text: str, expected: bool) -> None:
    assert is_hallucination(text) is expected


def test_rejects_wrong_sample_rate() -> None:
    with pytest.raises(ValueError, match="16000"):
        _transcriber(_FakeWhisperModel()).transcribe(silence(1.0), 44_100)


def test_model_is_loaded_lazily() -> None:
    assert WhisperTranscriber()._model is None


@pytest.mark.parametrize(
    ("whisper_language", "expected"), [("", "fr"), ("auto", None), ("EN", "en")]
)
def test_language_from_settings(whisper_language: str, expected: str | None) -> None:
    settings = Settings(_env_file=None, whisper_language=whisper_language)
    transcriber = WhisperTranscriber.from_settings(settings)
    assert transcriber.language == expected
    assert (transcriber.model_size, transcriber.compute_type) == ("small", "int8")


# --- Command line ---


class _RecordingTranscriber:
    def __init__(self) -> None:
        self.durations: list[float] = []

    def transcribe(self, audio: AudioArray, sample_rate: int = SAMPLE_RATE) -> Transcription:
        self.durations.append(len(audio) / sample_rate)
        return Transcription(f"tour {len(self.durations)}", "fr", 0.9, len(audio) / sample_rate)


def test_transcribe_file_prints_each_turn(tmp_path: Path) -> None:
    path = tmp_path / "two_turns.wav"
    signal = np.concatenate([silence(0.5), tone(1.0), silence(1.5), tone(1.0), silence(1.0)])
    sf.write(path, signal, SAMPLE_RATE)
    transcriber = _RecordingTranscriber()
    out = io.StringIO()
    transcribe_file(str(path), Settings(_env_file=None), EnergyDetector(), transcriber, out)
    lines = out.getvalue().splitlines()
    assert len(lines) == 2
    assert lines[0].endswith("tour 1")
    assert ", fr, " in lines[1]
