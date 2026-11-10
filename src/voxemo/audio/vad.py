"""Voice activity detection: split an audio stream into speech turns.

The speech probability comes from Silero VAD (bundled with the ``silero-vad``
package, no download). The turn logic itself (:class:`TurnSegmenter`) is model-free
so it can be tested with synthetic signals and a fake detector.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from voxemo.audio.recorder import ArraySource, AudioArray, AudioSource
from voxemo.config import Settings

SILERO_FRAME_SIZE = {16_000: 512, 8_000: 256}


class SpeechDetector(Protocol):
    """Gives the probability that a fixed-size frame contains speech."""

    frame_size: int

    def speech_probability(self, frame: AudioArray) -> float: ...

    def reset(self) -> None: ...


class SileroSpeechDetector:
    """Silero VAD (ONNX), loaded lazily on first use."""

    def __init__(self, sample_rate: int = 16_000) -> None:
        if sample_rate not in SILERO_FRAME_SIZE:
            raise ValueError("Silero VAD supports 8 kHz and 16 kHz only")
        self.sample_rate = sample_rate
        self.frame_size = SILERO_FRAME_SIZE[sample_rate]
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from silero_vad import load_silero_vad

            self._model = load_silero_vad(onnx=True)
        return self._model

    def speech_probability(self, frame: AudioArray) -> float:
        import torch

        return float(self.model(torch.from_numpy(frame), self.sample_rate).item())

    def reset(self) -> None:
        if self._model is not None:
            self._model.reset_states()


@dataclass(frozen=True)
class SpeechTurn:
    """One user utterance, kept in memory only."""

    audio: AudioArray
    sample_rate: int
    start_s: float
    """Offset of the turn from the start of listening."""

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sample_rate

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


@dataclass(frozen=True)
class SegmenterConfig:
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 800
    speech_pad_ms: int = 300
    max_turn_s: float = 30.0

    @classmethod
    def from_settings(cls, settings: Settings) -> SegmenterConfig:
        return cls(
            threshold=settings.vad_threshold,
            min_speech_ms=settings.vad_min_speech_ms,
            min_silence_ms=settings.vad_min_silence_ms,
            speech_pad_ms=settings.vad_speech_pad_ms,
            max_turn_s=settings.max_turn_s,
        )


class TurnSegmenter:
    """Frame-by-frame state machine turning speech probabilities into turns.

    A turn starts when a frame exceeds ``threshold`` (with ``speech_pad_ms`` of audio
    kept before it) and ends after ``min_silence_ms`` below ``threshold - 0.15``
    (hysteresis, as in Silero). Turns with less than ``min_speech_ms`` of speech are
    dropped as noise; turns longer than ``max_turn_s`` are cut.
    """

    def __init__(
        self, detector: SpeechDetector, sample_rate: int, config: SegmenterConfig | None = None
    ) -> None:
        self.detector = detector
        self.sample_rate = sample_rate
        self.config = config or SegmenterConfig()
        frame_ms = 1000 * detector.frame_size / sample_rate
        self._pad_frames = round(self.config.speech_pad_ms / frame_ms)
        self._min_speech_frames = max(1, round(self.config.min_speech_ms / frame_ms))
        self._min_silence_frames = max(1, round(self.config.min_silence_ms / frame_ms))
        self._max_frames = max(1, round(1000 * self.config.max_turn_s / frame_ms))
        self._end_threshold = max(0.0, self.config.threshold - 0.15)
        self.reset()

    def reset(self) -> None:
        self.detector.reset()
        self._preroll: deque[AudioArray] = deque(maxlen=self._pad_frames or 1)
        self._frames: list[AudioArray] = []
        self._speech_frames = 0
        self._silence_run = 0
        self._start_frame = 0
        self._frame_index = 0

    @property
    def in_speech(self) -> bool:
        return bool(self._frames)

    def push(self, frame: AudioArray) -> SpeechTurn | None:
        """Feed exactly one frame of ``detector.frame_size`` samples."""
        probability = self.detector.speech_probability(frame)
        index = self._frame_index
        self._frame_index += 1

        if not self._frames:
            if probability >= self.config.threshold:
                preroll = list(self._preroll) if self._pad_frames else []
                self._frames = [*preroll, frame]
                self._start_frame = index - len(preroll)
                self._speech_frames = 1
                self._silence_run = 0
            else:
                self._preroll.append(frame)
            return None

        self._frames.append(frame)
        if probability >= self.config.threshold:
            self._speech_frames += 1
        self._silence_run = self._silence_run + 1 if probability < self._end_threshold else 0

        if self._silence_run >= self._min_silence_frames or len(self._frames) >= self._max_frames:
            return self._close()
        return None

    def flush(self) -> SpeechTurn | None:
        """End of stream: return the pending turn, if any."""
        return self._close() if self._frames else None

    def _close(self) -> SpeechTurn | None:
        # Keep only `speech_pad_ms` of the trailing silence.
        trailing = max(0, self._silence_run - self._pad_frames)
        frames = self._frames[: len(self._frames) - trailing]
        enough_speech = self._speech_frames >= self._min_speech_frames
        start = self._start_frame
        self._frames = []
        self._preroll.clear()
        self._speech_frames = self._silence_run = 0
        if not enough_speech:
            return None
        frame_s = self.detector.frame_size / self.sample_rate
        return SpeechTurn(np.concatenate(frames), self.sample_rate, start * frame_s)


def reframe(blocks: Iterable[AudioArray], frame_size: int) -> Iterator[AudioArray]:
    """Regroup blocks of any length into frames of exactly ``frame_size`` samples."""
    pending = np.zeros(0, dtype=np.float32)
    for block in blocks:
        pending = np.concatenate([pending, block]) if len(pending) else block
        n_full = len(pending) // frame_size
        for i in range(n_full):
            yield pending[i * frame_size : (i + 1) * frame_size]
        pending = pending[n_full * frame_size :]
    if len(pending):
        yield np.pad(pending, (0, frame_size - len(pending)))


def iter_turns(
    source: AudioSource, detector: SpeechDetector, config: SegmenterConfig | None = None
) -> Iterator[SpeechTurn]:
    """Yield every speech turn of a (finite or live) source."""
    segmenter = TurnSegmenter(detector, source.sample_rate, config)
    for frame in reframe(source.blocks(), detector.frame_size):
        if turn := segmenter.push(frame):
            yield turn
    if turn := segmenter.flush():
        yield turn


def listen_for_turn(
    source: AudioSource,
    detector: SpeechDetector,
    config: SegmenterConfig | None = None,
    timeout_s: float = 15.0,
) -> SpeechTurn | None:
    """Wait for the user to speak and return the first complete turn.

    Returns ``None`` if nobody starts speaking within ``timeout_s`` (measured in audio
    time, so it behaves the same with a live microphone and a replayed file). The
    source is closed as soon as the turn is complete.
    """
    segmenter = TurnSegmenter(detector, source.sample_rate, config)
    timeout_frames = timeout_s * source.sample_rate / detector.frame_size
    blocks = source.blocks()
    try:
        for count, frame in enumerate(reframe(blocks, detector.frame_size), start=1):
            if turn := segmenter.push(frame):
                return turn
            if not segmenter.in_speech and count >= timeout_frames:
                return None
        return segmenter.flush()
    finally:
        close = getattr(blocks, "close", None)
        if close:
            close()


def speech_turn(
    audio: AudioArray,
    sample_rate: int,
    detector: SpeechDetector,
    config: SegmenterConfig | None = None,
) -> SpeechTurn | None:
    """One turn spanning all the speech of a recording (pauses inside are kept).

    For push-to-talk recordings: the user decides when the turn ends, so internal
    pauses belong to it (they matter for the pause ratio). ``None`` without speech.
    """
    detector.reset()
    turns = list(iter_turns(ArraySource(audio, sample_rate), detector, config))
    if not turns:
        return None
    start = int(turns[0].start_s * sample_rate)
    end = int(turns[-1].end_s * sample_rate)
    return SpeechTurn(np.asarray(audio[start:end], dtype=np.float32), sample_rate, turns[0].start_s)
