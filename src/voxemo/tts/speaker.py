"""Text-to-speech with Piper, spoken sentence by sentence while the reply streams.

Synthesis and playback run in two background threads: the first sentence is spoken
as soon as it is complete, while the next ones are synthesized. Everything runs
locally; the voice model is downloaded once (Hugging Face, free) if missing.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from voxemo import texts
from voxemo.audio.recorder import AudioArray, AudioDeviceError, import_sounddevice
from voxemo.config import Settings

logger = logging.getLogger(__name__)


class Speaker(Protocol):
    """Text → audio."""

    @property
    def sample_rate(self) -> int: ...

    def synthesize(self, text: str) -> AudioArray: ...


class AudioPlayer(Protocol):
    def play(self, audio: AudioArray, sample_rate: int) -> None:
        """Play and block until done (or until :meth:`stop`)."""
        ...

    def stop(self) -> None: ...


class TTSUnavailableError(RuntimeError):
    """The voice could not be loaded or played; the message is user-facing."""


class PiperSpeaker:
    """`Speaker` backed by a Piper ONNX voice, loaded once on first use."""

    def __init__(
        self,
        voice: str = "fr_FR-siwis-medium",
        voices_dir: str | Path = "data/voices",
        *,
        speed: float = 1.0,
        on_download: Callable[[str], None] | None = None,
    ) -> None:
        self.voice = voice
        self.voices_dir = Path(voices_dir)
        self.speed = speed
        self.on_download = on_download
        self._voice: Any = None
        self._lock = threading.Lock()

    @classmethod
    def from_settings(
        cls, settings: Settings, on_download: Callable[[str], None] | None = None
    ) -> PiperSpeaker:
        return cls(
            settings.piper_voice,
            settings.piper_voices_dir,
            speed=settings.tts_speed,
            on_download=on_download,
        )

    @property
    def model_path(self) -> Path:
        return self.voices_dir / f"{self.voice}.onnx"

    def load(self) -> Any:
        with self._lock:
            if self._voice is None:
                from piper import PiperVoice

                if not self.model_path.exists():
                    self._download()
                self._voice = PiperVoice.load(self.model_path)
        return self._voice

    def _download(self) -> None:
        from piper.download_voices import download_voice

        if self.on_download:
            self.on_download(texts.TTS_DOWNLOADING.format(voice=self.voice))
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        try:
            download_voice(self.voice, self.voices_dir)
        except Exception as exc:  # network, unknown voice name…
            raise TTSUnavailableError(texts.TTS_UNAVAILABLE.format(detail=exc)) from exc

    @property
    def sample_rate(self) -> int:
        return int(self.load().config.sample_rate)

    def synthesize(self, text: str) -> AudioArray:
        from piper import SynthesisConfig

        voice = self.load()
        config = SynthesisConfig(length_scale=1.0 / self.speed)
        chunks = [c.audio_float_array for c in voice.synthesize(text, config)]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)


class SoundDevicePlayer:
    """`AudioPlayer` on the system output (PortAudio), with a blocking output stream.

    The blocking API is used on purpose: callback playback (``sounddevice.play``)
    hangs with the WSLg PulseAudio server. Audio is written in short blocks so that
    :meth:`stop` takes effect quickly; the stream stays open between sentences.
    """

    BLOCK_S = 0.1

    def __init__(self, device: str | int | None = None) -> None:
        self.device = int(device) if isinstance(device, str) and device.isdigit() else device
        self._stream: Any = None
        self._stream_rate = 0
        self._stop = threading.Event()

    def _open(self, sample_rate: int) -> Any:
        sd = import_sounddevice()
        if self._stream is not None and self._stream_rate == sample_rate:
            return self._stream
        self.close()
        try:
            stream = sd.OutputStream(
                samplerate=sample_rate, channels=1, dtype="float32", device=self.device or None
            )
            stream.start()
        except sd.PortAudioError as exc:
            raise AudioDeviceError(texts.TTS_OUTPUT_UNAVAILABLE.format(detail=exc)) from exc
        self._stream, self._stream_rate = stream, sample_rate
        return stream

    def play(self, audio: AudioArray, sample_rate: int) -> None:
        self._stop.clear()
        stream = self._open(sample_rate)
        block = max(1, int(self.BLOCK_S * sample_rate))
        signal = np.asarray(audio, dtype=np.float32).reshape(-1, 1)
        try:
            for start in range(0, len(signal), block):
                if self._stop.is_set():
                    return
                stream.write(signal[start : start + block])
        except Exception as exc:
            self.close()
            raise AudioDeviceError(texts.TTS_OUTPUT_UNAVAILABLE.format(detail=exc)) from exc
        # write() returns once the data is buffered: wait for it to be heard.
        self._stop.wait(float(getattr(stream, "latency", 0.0) or 0.0))

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.close()
            self._stream = None


# --- text preparation ---

_MARKDOWN = re.compile(r"[*_`#>|~]+")
_URL = re.compile(r"https?://\S+")
_EMOJI = re.compile("[\U0001f000-\U0001faff☀-➿️]")
_SENTENCE_END = re.compile(r"(?<=[.!?…:;])\s+|\n+")
MAX_SENTENCE_CHARS = 220
_SPOKEN_NUMBERS = {"fr": {"3114": "31 14"}}  # read as a French phone number


def prepare_for_speech(text: str, language: str = "fr") -> str:
    """Remove what should not be read aloud (markdown, links, emojis)."""
    cleaned = _URL.sub("", text)
    cleaned = _MARKDOWN.sub("", cleaned)
    cleaned = _EMOJI.sub("", cleaned)
    for written, spoken in _SPOKEN_NUMBERS.get(language, {}).items():
        cleaned = re.sub(rf"\b{written}\b", spoken, cleaned)
    return re.sub(r"[ \t]+", " ", cleaned).strip()


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Complete sentences found in ``buffer``, and the unfinished rest."""
    parts = _SENTENCE_END.split(buffer)
    rest = parts.pop() if parts else ""
    sentences = [p.strip() for p in parts if p.strip()]
    # A very long sentence without punctuation is cut at a comma to start speaking.
    if len(rest) > MAX_SENTENCE_CHARS and "," in rest[:MAX_SENTENCE_CHARS]:
        cut = rest.rindex(",", 0, MAX_SENTENCE_CHARS) + 1
        sentences.append(rest[:cut].strip())
        rest = rest[cut:].lstrip()
    return sentences, rest


# --- streaming speech ---

_STOP = object()


class SpeechQueue:
    """Speaks text as it arrives, sentence by sentence, in the background.

    If the voice or the speaker fails, speech is disabled for the session and
    ``error`` holds a user-facing message; the conversation goes on in text.
    """

    def __init__(self, speaker: Speaker, player: AudioPlayer, language: str = "fr") -> None:
        self.speaker = speaker
        self.player = player
        self.language = language
        self.error: str | None = None
        self._buffer = ""
        self._texts: queue.Queue[Any] = queue.Queue()
        self._audio: queue.Queue[Any] = queue.Queue()
        self._pending = 0
        self._idle = threading.Condition()
        self._generation = 0
        self._threads = [
            threading.Thread(target=self._synthesis_loop, name="tts-synth", daemon=True),
            threading.Thread(target=self._playback_loop, name="tts-play", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    @property
    def enabled(self) -> bool:
        return self.error is None

    def warm_up(self) -> None:
        """Load the voice now (errors disable speech instead of raising)."""
        try:
            self.speaker.synthesize("Bonjour.")
        except Exception as exc:
            self._disable(exc)

    # --- feeding text ---

    def feed(self, chunk: str) -> None:
        """Add streamed text; complete sentences start being spoken."""
        if not self.enabled:
            return
        self._buffer += chunk
        sentences, self._buffer = split_sentences(self._buffer)
        for sentence in sentences:
            self._enqueue(sentence)

    def flush(self) -> None:
        """Speak whatever is left in the buffer (end of a reply)."""
        rest, self._buffer = self._buffer.strip(), ""
        if rest and self.enabled:
            self._enqueue(rest)

    def say(self, text: str) -> None:
        """Speak a complete text (after any text already queued)."""
        self.feed(text)
        self.flush()

    def _enqueue(self, sentence: str) -> None:
        spoken = prepare_for_speech(sentence, self.language)
        if not spoken:
            return
        with self._idle:
            self._pending += 1
        self._texts.put((self._generation, spoken))

    # --- waiting / stopping ---

    def wait(self, timeout: float | None = None) -> bool:
        """Block until everything queued has been spoken; False on timeout."""
        with self._idle:
            return self._idle.wait_for(lambda: self._pending == 0, timeout)

    def cancel(self) -> None:
        """Drop queued speech and stop the current sentence."""
        self._buffer = ""
        self._generation += 1
        self.player.stop()

    def close(self) -> None:
        self.cancel()
        self._texts.put(_STOP)
        for thread in self._threads:
            thread.join(timeout=2)
        close = getattr(self.player, "close", None)
        if close is not None:
            close()

    # --- workers ---

    def _done_one(self) -> None:
        with self._idle:
            self._pending -= 1
            self._idle.notify_all()

    def _disable(self, exc: Exception) -> None:
        if self.error is None:
            detail = str(exc) or type(exc).__name__
            self.error = (
                detail
                if isinstance(exc, (TTSUnavailableError, AudioDeviceError))
                else (texts.TTS_UNAVAILABLE.format(detail=detail))
            )
            logger.warning("speech disabled: %s", exc)

    def _synthesis_loop(self) -> None:
        while (item := self._texts.get()) is not _STOP:
            generation, sentence = item
            audio = None
            if generation == self._generation and self.enabled:
                try:
                    audio = self.speaker.synthesize(sentence)
                except Exception as exc:
                    self._disable(exc)
            self._audio.put((generation, audio))
        self._audio.put(_STOP)

    def _playback_loop(self) -> None:
        while (item := self._audio.get()) is not _STOP:
            generation, audio = item
            try:
                if audio is not None and len(audio) and generation == self._generation:
                    self.player.play(audio, self.speaker.sample_rate)
            except Exception as exc:
                self._disable(exc)
            finally:
                self._done_one()


def create_speech(settings: Settings, enabled: bool = True) -> SpeechQueue | None:
    """The speech queue configured by the settings, or ``None`` when disabled."""
    if not (enabled and settings.tts_enabled):
        return None
    return SpeechQueue(
        PiperSpeaker.from_settings(settings),
        SoundDevicePlayer(settings.audio_output_device or None),
        settings.language,
    )
