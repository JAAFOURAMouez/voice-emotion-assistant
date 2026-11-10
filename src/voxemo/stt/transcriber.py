"""Speech-to-text with faster-whisper (local, CPU-friendly int8)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol

import numpy as np

from voxemo.audio.recorder import AudioArray
from voxemo.config import Settings

WHISPER_SAMPLE_RATE = 16_000

# Phrases Whisper tends to invent on silence or noise (learnt from subtitled videos).
_HALLUCINATIONS = frozenset(
    {
        "sous-titres realises par la communaute d'amara.org",
        "sous-titrage st' 501",
        "sous-titrage societe radio-canada",
        "sous-titres par la communaute d'amara.org",
        "merci d'avoir regarde",
        "merci d'avoir regarde cette video",
        "abonnez-vous",
        "thank you for watching",
        "thanks for watching",
        "you",
    }
)


@dataclass(frozen=True)
class Transcription:
    text: str
    language: str
    language_probability: float
    duration_s: float

    @property
    def is_empty(self) -> bool:
        return not self.text


class Transcriber(Protocol):
    def transcribe(
        self, audio: AudioArray, sample_rate: int = WHISPER_SAMPLE_RATE
    ) -> Transcription: ...


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower().replace("’", "'"))
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[\s.!?…]+$", "", plain.strip())


def is_hallucination(text: str) -> bool:
    return _normalize(text) in _HALLUCINATIONS


class WhisperTranscriber:
    """`Transcriber` backed by faster-whisper; the model is loaded once, on first use."""

    def __init__(
        self,
        model_size: str = "small",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = "fr",
        beam_size: int = 1,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self._model: Any = None
        self._lock = Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> WhisperTranscriber:
        choice = settings.whisper_language.strip().lower()
        language = None if choice == "auto" else (choice or settings.language)
        return cls(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            language=language,
            beam_size=settings.whisper_beam_size,
        )

    @property
    def model(self) -> Any:
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                options = {"device": self.device, "compute_type": self.compute_type}
                try:  # local copy first: no request to the Hugging Face Hub
                    self._model = WhisperModel(self.model_size, local_files_only=True, **options)
                except Exception:  # not downloaded yet (huggingface_hub raises various errors)
                    self._model = WhisperModel(self.model_size, **options)
        return self._model

    def transcribe(
        self, audio: AudioArray, sample_rate: int = WHISPER_SAMPLE_RATE
    ) -> Transcription:
        if sample_rate != WHISPER_SAMPLE_RATE:
            raise ValueError(f"Whisper expects {WHISPER_SAMPLE_RATE} Hz audio, got {sample_rate}")
        segments, info = self.model.transcribe(
            np.asarray(audio, dtype=np.float32),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,  # turns are already cut by our own VAD
            condition_on_previous_text=False,
            # No temperature fallback: each retry re-decodes the whole turn (seconds
            # on CPU) for little gain on short conversational turns.
            temperature=0.0,
        )
        texts = [s.text.strip() for s in segments]
        text = " ".join(t for t in texts if t and not is_hallucination(t))
        return Transcription(
            text=text,
            language=info.language,
            language_probability=float(info.language_probability),
            duration_s=len(audio) / sample_rate,
        )
