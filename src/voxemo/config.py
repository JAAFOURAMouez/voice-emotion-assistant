"""Application settings, read from environment variables and an optional `.env` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class Settings(BaseSettings):
    """All tunable parameters of the assistant.

    Every field maps to an upper-case environment variable of the same name
    (e.g. ``ollama_model`` ← ``OLLAMA_MODEL``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- LLM (Ollama) ---
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_timeout_s: float = Field(default=120.0, gt=0)
    """Generous: the first call also loads the model into memory."""
    ollama_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    llm_history_turns: int = Field(default=10, ge=1)
    """Number of past exchanges (user + assistant) sent back to the model."""
    allow_remote_ollama: bool = False
    """Opt-in escape hatch: by default Ollama must run on this machine."""

    # --- Language ---
    language: str = Field(default="fr", min_length=2, max_length=5)

    # --- Audio ---
    sample_rate: int = Field(default=16_000, gt=0)
    store_raw_audio: bool = False
    """Raw audio is never kept unless explicitly enabled."""
    audio_dir: Path = Path("data/audio")
    audio_input_device: str = ""
    """Microphone name or index for sounddevice; empty = system default."""

    # --- Voice activity detection (speech turns) ---
    vad_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    vad_min_speech_ms: int = Field(default=250, ge=0)
    vad_min_silence_ms: int = Field(default=800, ge=0)
    """Silence that ends a speech turn."""
    vad_speech_pad_ms: int = Field(default=300, ge=0)
    max_turn_s: float = Field(default=30.0, gt=0)
    listen_timeout_s: float = Field(default=15.0, gt=0)
    """Give up waiting for the user to start speaking after this delay."""

    # --- Speech-to-text ---
    whisper_model: str = "small"
    whisper_device: Literal["cpu", "cuda", "auto"] = "cpu"
    whisper_compute_type: str = "int8"
    whisper_language: str = ""
    """Empty = same as LANGUAGE (faster, more robust); "auto" = automatic detection."""
    whisper_beam_size: int = Field(default=1, ge=1)
    """1 (greedy) keeps CPU latency low; 5 is slightly more accurate."""

    # --- Emotion ---
    emotion_model: str = "superb/wav2vec2-base-superb-er"
    emotion_device: Literal["cpu", "cuda"] = "cpu"
    emotion_model_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    """Share of the wav2vec2 model in the final scores; the rest comes from prosody."""
    emotion_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    emotion_text_gain: float = Field(default=5.0, ge=0.0)
    """Weight of explicit emotion words in the transcript ("très énervé"); 0 = voice only."""
    baseline_min_turns: int = Field(default=5, ge=1)
    """Below this many past turns, the user baseline is unreliable and confidence is lowered."""
    baseline_window: int = Field(default=20, ge=1)
    """Number of previous turns used for the rolling voice baseline."""

    # --- Well-being exercises ---
    exercise_consecutive_turns: int = Field(default=2, ge=1)
    exercise_cooldown_minutes: float = Field(default=15.0, ge=0)

    # --- Text-to-speech ---
    piper_voice: str = "fr_FR-siwis-medium"
    piper_voices_dir: Path = Path("data/voices")
    tts_enabled: bool = True
    tts_speed: float = Field(default=1.0, gt=0.25, le=4.0)
    """Speaking rate: above 1 is faster, below 1 slower."""
    audio_output_device: str = ""
    """Speaker name or index for sounddevice; empty = system default."""

    # --- Web interface (API) ---
    api_host: str = "127.0.0.1"
    """Interface the server listens on; 127.0.0.1 = this machine only."""
    api_port: int = Field(default=8000, gt=0, lt=65536)
    allowed_origins: str = ""
    """Extra browser origins allowed to open the WebSocket, comma-separated
    (e.g. https://voxemo.example.org behind a reverse proxy). Same-origin and
    localhost pages are always allowed."""
    web_dist: Path = Path("web/dist")
    """Built frontend served at / (``npm run build`` in web/)."""

    # --- Storage ---
    db_path: Path = Path("data/voxemo.db")

    @field_validator("ollama_host")
    @classmethod
    def _check_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"OLLAMA_HOST must be an http(s) URL, got {value!r}")
        return value.rstrip("/")

    @field_validator("language")
    @classmethod
    def _normalize_language(cls, value: str) -> str:
        return value.strip().lower()

    @model_validator(mode="after")
    def _enforce_local_llm(self) -> Settings:
        host = urlparse(self.ollama_host).hostname
        if host not in _LOOPBACK_HOSTS and not self.allow_remote_ollama:
            raise ValueError(
                f"OLLAMA_HOST points to {host!r}, which is not this machine. "
                "Transcripts must stay local; set ALLOW_REMOTE_OLLAMA=true only "
                "for a server you control."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loaded once."""
    return Settings()
