"""Real VAD, Whisper and emotion models. Run with: pytest -m integration

The speech sample is synthesized locally with the Piper voice when it is present
in PIPER_VOICES_DIR (download: python -m piper.download_voices fr_FR-siwis-medium
--data-dir data/voices); otherwise the speech tests are skipped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import SAMPLE_RATE, silence
from voxemo.audio.recorder import ArraySource
from voxemo.audio.vad import SileroSpeechDetector, iter_turns
from voxemo.config import Settings
from voxemo.emotion.classifier import MODEL_CLASSES, Wav2Vec2EmotionClassifier
from voxemo.emotion.fusion import EmotionAnalyzer, FusionConfig, VoiceBaseline
from voxemo.stt.transcriber import WhisperTranscriber

pytestmark = pytest.mark.integration

SENTENCE = "Bonjour, je voudrais organiser ma semaine de travail."


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture(scope="module")
def speech(settings: Settings) -> np.ndarray:
    model_path = Path(settings.piper_voices_dir) / f"{settings.piper_voice}.onnx"
    if not model_path.exists():
        pytest.skip(f"Piper voice not found: {model_path}")
    import librosa
    from piper import PiperVoice

    voice = PiperVoice.load(model_path)
    chunks = list(voice.synthesize(SENTENCE))
    audio = np.concatenate([c.audio_float_array for c in chunks])
    audio = librosa.resample(audio, orig_sr=chunks[0].sample_rate, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32)


def test_silero_ignores_noise() -> None:
    noise = np.random.default_rng(0).normal(0, 0.01, 3 * SAMPLE_RATE).astype(np.float32)
    turns = list(iter_turns(ArraySource(noise, SAMPLE_RATE), SileroSpeechDetector()))
    assert turns == []


def test_silero_finds_one_turn(speech: np.ndarray) -> None:
    signal = np.concatenate([silence(1.0), speech, silence(1.5)])
    turns = list(iter_turns(ArraySource(signal, SAMPLE_RATE), SileroSpeechDetector()))
    assert len(turns) == 1
    assert turns[0].duration_s == pytest.approx(len(speech) / SAMPLE_RATE, abs=0.8)


def test_whisper_transcribes_french(settings: Settings, speech: np.ndarray) -> None:
    result = WhisperTranscriber.from_settings(settings).transcribe(speech)
    assert result.language == "fr"
    words = result.text.lower()
    assert "bonjour" in words
    assert "semaine" in words


def test_wav2vec2_classifier_outputs_distribution(settings: Settings, speech: np.ndarray) -> None:
    probs = Wav2Vec2EmotionClassifier.from_settings(settings).predict(speech, SAMPLE_RATE)
    assert set(probs) == set(MODEL_CLASSES)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)


def test_analyzer_on_real_speech(settings: Settings, speech: np.ndarray) -> None:
    analyzer = EmotionAnalyzer(
        Wav2Vec2EmotionClassifier.from_settings(settings), FusionConfig.from_settings(settings)
    )
    first = analyzer.analyze(speech, SAMPLE_RATE)
    assert 80 < first.features.pitch_mean_hz < 400
    assert 1.0 < first.features.speech_rate < 10.0
    # Same voice again, with that turn as history: it should look like the usual voice.
    again = analyzer.analyze(speech, SAMPLE_RATE, VoiceBaseline([first.features] * 5))
    assert again.arousal == pytest.approx(0.5, abs=0.05)
