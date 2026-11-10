"""Audio input: microphone capture (sounddevice) and in-memory / file sources.

Every source yields float32 mono blocks at the configured sample rate (16 kHz by
default). Nothing is written to disk here: raw audio stays in memory.
"""

from __future__ import annotations

import queue
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import BinaryIO, Protocol

import numpy as np
import numpy.typing as npt

from voxemo import texts

AudioArray = npt.NDArray[np.float32]


class AudioDeviceError(RuntimeError):
    """The microphone cannot be used; the message is user-facing (French)."""


class AudioSource(Protocol):
    """A stream of float32 mono audio blocks of arbitrary length."""

    sample_rate: int

    def blocks(self) -> Iterator[AudioArray]: ...


def to_mono_float32(audio: npt.ArrayLike) -> AudioArray:
    """Convert (frames,) or (frames, channels) audio, int or float, to float32 mono in [-1, 1]."""
    array = np.asarray(audio)
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float32) / np.iinfo(array.dtype).max
    array = array.astype(np.float32, copy=False)
    if array.ndim == 2:
        array = array.mean(axis=1)
    elif array.ndim != 1:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {array.shape}")
    return np.ascontiguousarray(array)


class ArraySource:
    """Replays an in-memory signal block by block (tests, audio files)."""

    def __init__(self, audio: npt.ArrayLike, sample_rate: int, block_size: int = 512) -> None:
        self.audio = to_mono_float32(audio)
        self.sample_rate = sample_rate
        self.block_size = block_size

    @classmethod
    def from_file(cls, path: str | Path, sample_rate: int = 16_000) -> ArraySource:
        return cls(load_audio(path, sample_rate), sample_rate)

    def blocks(self) -> Iterator[AudioArray]:
        for start in range(0, len(self.audio), self.block_size):
            yield self.audio[start : start + self.block_size]


def load_audio(path: str | Path | BinaryIO, sample_rate: int = 16_000) -> AudioArray:
    """Read an audio file (path or file object) as float32 mono at ``sample_rate``."""
    import soundfile as sf

    source = str(path) if isinstance(path, (str, Path)) else path
    data, file_rate = sf.read(source, dtype="float32", always_2d=False)
    audio = to_mono_float32(data)
    if file_rate != sample_rate:
        import soxr

        audio = soxr.resample(audio, file_rate, sample_rate)
    return audio.astype(np.float32, copy=False)


def import_sounddevice() -> ModuleType:
    try:
        import sounddevice as sd
    except OSError as exc:  # PortAudio missing
        raise AudioDeviceError(texts.PORTAUDIO_MISSING) from exc
    return sd


class MicrophoneSource:
    """Live microphone capture. Recording runs only while ``blocks()`` is being consumed."""

    def __init__(
        self, sample_rate: int = 16_000, block_size: int = 512, device: str | int | None = None
    ) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = int(device) if isinstance(device, str) and device.isdigit() else device
        self.overflows = 0

    def blocks(self) -> Iterator[AudioArray]:
        sd = import_sounddevice()
        buffer: queue.Queue[AudioArray] = queue.Queue()

        def callback(indata: np.ndarray, _frames: int, _time: object, status: object) -> None:
            if status:
                self.overflows += 1
            buffer.put(indata[:, 0].copy())

        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                dtype="float32",
                device=self.device or None,
                callback=callback,
            )
        except sd.PortAudioError as exc:
            raise AudioDeviceError(texts.MICROPHONE_UNAVAILABLE.format(detail=exc)) from exc

        with stream:
            while True:
                yield buffer.get()


def save_wav(audio: AudioArray, sample_rate: int, path: str | Path) -> Path:
    """Write a mono 16-bit WAV file (parent folders are created)."""
    import soundfile as sf

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_16")
    return target


def encode_wav(audio: AudioArray, sample_rate: int) -> bytes:
    """Mono 16-bit WAV bytes (e.g. to play in a browser)."""
    import io

    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(audio, dtype=np.float32), sample_rate, format="WAV")
    return buffer.getvalue()
