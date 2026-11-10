"""Shared, heavy resources of the server (models, database), built once.

Each WebSocket connection gets its own :class:`VoicePipeline` (conversation,
voice baseline, exercise rule) that shares these models.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from voxemo import texts
from voxemo.audio.recorder import AudioArray, encode_wav
from voxemo.audio.vad import (
    SegmenterConfig,
    SileroSpeechDetector,
    SpeechDetector,
    SpeechTurn,
    speech_turn,
)
from voxemo.config import Settings
from voxemo.emotion.classifier import Wav2Vec2EmotionClassifier
from voxemo.emotion.fusion import EmotionAnalyzer, FusionConfig
from voxemo.llm.client import ChatModel, Conversation, OllamaChatModel
from voxemo.pipeline import VoiceAnalyzer, VoicePipeline
from voxemo.stt.transcriber import Transcriber, WhisperTranscriber
from voxemo.tracking.store import InteractionStore
from voxemo.tts.speaker import PiperSpeaker, Speaker, prepare_for_speech
from voxemo.wellbeing.exercises import ExerciseAdvisor

LoadState = Literal["loading", "ready", "error"]


@dataclass
class Services:
    settings: Settings
    transcriber: Transcriber
    analyzer: VoiceAnalyzer
    chat_model: ChatModel
    detector: SpeechDetector
    store: InteractionStore
    speaker: Speaker | None = None
    warnings: list[str] = field(default_factory=list)
    _detector_lock: threading.Lock = field(default_factory=threading.Lock)
    _speech_lock: threading.Lock = field(default_factory=threading.Lock)

    def new_pipeline(self) -> VoicePipeline:
        settings = self.settings
        return VoicePipeline(
            self.transcriber,
            self.analyzer,
            Conversation(self.chat_model, settings),
            baseline_window=settings.baseline_window,
            history=self.store.recent_estimates(settings.baseline_window),
            fusion_config=FusionConfig.from_settings(settings),
            language=settings.language,
            audio_dir=settings.audio_dir if settings.store_raw_audio else None,
            advisor=ExerciseAdvisor.from_settings(settings),
        )

    def turn_from_pcm(self, pcm: bytes) -> SpeechTurn | None:
        """16-bit little-endian mono PCM at ``sample_rate`` → speech turn, or ``None``."""
        audio = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2")
        return self.turn_from_audio(audio.astype(np.float32) / 32768.0)

    def turn_from_audio(self, audio: AudioArray) -> SpeechTurn | None:
        """Push-to-talk recording: one turn covering all its speech (``None``: silence)."""
        with self._detector_lock:  # the VAD keeps state between frames
            return speech_turn(
                audio,
                self.settings.sample_rate,
                self.detector,
                SegmenterConfig.from_settings(self.settings),
            )

    def speech_wav(self, text: str) -> bytes | None:
        """Spoken text as WAV bytes, or ``None`` if speech is unavailable or empty."""
        if self.speaker is None:
            return None
        spoken = prepare_for_speech(text, self.settings.language)
        if not spoken:
            return None
        try:
            audio = self.speaker.synthesize(spoken)
            rate = self.speaker.sample_rate
        except Exception as exc:
            with self._speech_lock:
                self.warnings.append(texts.TTS_UNAVAILABLE.format(detail=exc))
                self.speaker = None
            return None
        return encode_wav(audio, rate)

    def llm_problem(self) -> str | None:
        """User-facing message when the LLM cannot answer right now."""
        check = getattr(self.chat_model, "check", None)
        return check() if check is not None else None


def build_services(settings: Settings) -> Services:
    fusion = FusionConfig.from_settings(settings)
    return Services(
        settings=settings,
        transcriber=WhisperTranscriber.from_settings(settings),
        analyzer=EmotionAnalyzer(Wav2Vec2EmotionClassifier.from_settings(settings), fusion),
        chat_model=OllamaChatModel.from_settings(settings),
        detector=SileroSpeechDetector(settings.sample_rate),
        store=InteractionStore(settings.db_path),
        speaker=PiperSpeaker.from_settings(settings) if settings.tts_enabled else None,
    )


def warm_up(services: Services) -> None:
    """Load every model now. LLM problems are not kept as warnings: the status
    checks Ollama live (it may be started, or finish loading, later)."""
    with services.new_pipeline() as pipeline:
        pipeline.warm_up(services.settings.sample_rate)
    if services.speaker is not None:
        services.speech_wav("Bonjour.")
