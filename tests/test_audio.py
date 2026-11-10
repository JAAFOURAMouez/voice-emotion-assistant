"""Tests for audio sources and speech-turn segmentation (synthetic signals, fake VAD)."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from conftest import SAMPLE_RATE, EnergyDetector, silence, tone
from voxemo.audio import recorder
from voxemo.audio.recorder import (
    ArraySource,
    AudioDeviceError,
    MicrophoneSource,
    load_audio,
    to_mono_float32,
)
from voxemo.audio.vad import (
    SegmenterConfig,
    TurnSegmenter,
    iter_turns,
    listen_for_turn,
    reframe,
)
from voxemo.config import Settings

FRAME_S = 512 / SAMPLE_RATE

# --- Sources ---


def test_to_mono_float32_handles_int16_and_stereo() -> None:
    stereo = np.array([[32767, 0], [-32767, 0]], dtype=np.int16)
    mono = to_mono_float32(stereo)
    assert mono.dtype == np.float32
    np.testing.assert_allclose(mono, [0.5, -0.5], atol=1e-4)


def test_to_mono_float32_rejects_3d() -> None:
    with pytest.raises(ValueError):
        to_mono_float32(np.zeros((2, 2, 2)))


def test_array_source_yields_blocks() -> None:
    blocks = list(ArraySource(np.arange(1100, dtype=np.float32), SAMPLE_RATE).blocks())
    assert [len(b) for b in blocks] == [512, 512, 76]


def test_load_audio_resamples(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    sf.write(path, np.zeros((48_000, 2), dtype=np.float32), 48_000)
    audio = load_audio(path, SAMPLE_RATE)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert abs(len(audio) - SAMPLE_RATE) <= 1


class _FakeStream:
    def __init__(self, *, callback: Any, blocksize: int, **kwargs: Any) -> None:
        self.callback = callback
        self.blocksize = blocksize
        self.kwargs = kwargs
        self.open = False

    def __enter__(self) -> _FakeStream:
        self.open = True
        for i in range(3):
            self.callback(
                np.full((self.blocksize, 1), i, dtype=np.float32), self.blocksize, None, None
            )
        return self

    def __exit__(self, *exc: object) -> None:
        self.open = False


def _fake_sounddevice(monkeypatch: pytest.MonkeyPatch, stream_cls: Any) -> types.SimpleNamespace:
    fake = types.SimpleNamespace(
        InputStream=stream_cls, PortAudioError=type("PAErr", (Exception,), {})
    )
    monkeypatch.setattr(recorder, "import_sounddevice", lambda: fake)
    return fake


def test_microphone_source_streams_mono_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    streams: list[_FakeStream] = []

    def make(**kwargs: Any) -> _FakeStream:
        streams.append(_FakeStream(**kwargs))
        return streams[-1]

    _fake_sounddevice(monkeypatch, make)
    blocks = MicrophoneSource(SAMPLE_RATE, block_size=512, device="2").blocks()
    received = [next(blocks) for _ in range(3)]
    assert [float(b[0]) for b in received] == [0.0, 1.0, 2.0]
    assert received[0].shape == (512,)
    assert streams[0].kwargs["samplerate"] == SAMPLE_RATE
    assert streams[0].kwargs["channels"] == 1
    assert streams[0].kwargs["device"] == 2
    blocks.close()
    assert not streams[0].open  # the microphone is released


def test_microphone_error_is_user_facing(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.SimpleNamespace(PortAudioError=type("PAErr", (Exception,), {}))

    def failing(**_kwargs: Any) -> None:
        raise fake.PortAudioError("no device")

    fake.InputStream = failing
    monkeypatch.setattr(recorder, "import_sounddevice", lambda: fake)
    with pytest.raises(AudioDeviceError, match="Impossible d'ouvrir le micro"):
        next(MicrophoneSource().blocks())


def test_missing_portaudio_is_user_facing(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Finder:
        def find_spec(self, name: str, *_args: object) -> None:
            if name == "sounddevice":
                raise OSError("PortAudio library not found")

    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Finder(), *sys.meta_path])
    with pytest.raises(AudioDeviceError, match="libportaudio2"):
        recorder.import_sounddevice()


# --- Segmentation ---


def test_reframe_regroups_and_pads() -> None:
    blocks = [np.ones(300, np.float32), np.ones(900, np.float32)]
    frames = list(reframe(blocks, 512))
    assert [len(f) for f in frames] == [512, 512, 512]
    assert frames[-1][176:].sum() == 0  # zero padding at the end


def test_segments_two_turns_and_drops_blips() -> None:
    signal = np.concatenate(
        [silence(1.0), tone(1.0), silence(1.5), tone(0.1), silence(1.5), tone(0.8), silence(1.0)]
    )
    turns = list(iter_turns(ArraySource(signal, SAMPLE_RATE), EnergyDetector()))
    assert len(turns) == 2
    first, second = turns
    pad, silence_end = 0.3, 0.8
    assert first.start_s == pytest.approx(1.0 - pad, abs=2 * FRAME_S)
    assert first.duration_s == pytest.approx(pad + 1.0 + pad, abs=3 * FRAME_S)
    assert second.start_s == pytest.approx(5.1 - pad, abs=2 * FRAME_S)
    assert second.duration_s < 0.8 + 2 * pad + silence_end
    assert first.audio.dtype == np.float32


def test_turn_ending_with_stream_is_flushed() -> None:
    signal = np.concatenate([silence(0.5), tone(1.0)])
    turns = list(iter_turns(ArraySource(signal, SAMPLE_RATE), EnergyDetector()))
    assert len(turns) == 1


def test_long_turn_is_cut() -> None:
    config = SegmenterConfig(max_turn_s=2.0)
    signal = tone(5.0)
    turns = list(iter_turns(ArraySource(signal, SAMPLE_RATE), EnergyDetector(), config))
    assert turns[0].duration_s == pytest.approx(2.0, abs=FRAME_S)
    assert len(turns) == 3


def test_short_pause_does_not_split_turn() -> None:
    signal = np.concatenate([tone(1.0), silence(0.4), tone(1.0), silence(1.0)])
    turns = list(iter_turns(ArraySource(signal, SAMPLE_RATE), EnergyDetector()))
    assert len(turns) == 1


def test_segmenter_resets_detector_state() -> None:
    detector = EnergyDetector()
    TurnSegmenter(detector, SAMPLE_RATE)
    assert detector.resets == 1


def test_config_from_settings() -> None:
    settings = Settings(_env_file=None, vad_min_silence_ms=500, max_turn_s=12)
    config = SegmenterConfig.from_settings(settings)
    assert (config.min_silence_ms, config.max_turn_s) == (500, 12)


# --- Listening for one turn ---


class _TrackedSource:
    """Live-like source that records whether it was closed."""

    def __init__(self, signal: np.ndarray) -> None:
        self.inner = ArraySource(signal, SAMPLE_RATE)
        self.sample_rate = SAMPLE_RATE
        self.closed = False

    def blocks(self) -> Iterator[np.ndarray]:
        try:
            yield from self.inner.blocks()
            while True:  # a microphone never ends by itself
                yield silence(FRAME_S)
        finally:
            self.closed = True


def test_listen_returns_first_turn_and_releases_source() -> None:
    source = _TrackedSource(np.concatenate([silence(0.5), tone(1.0), silence(1.0), tone(1.0)]))
    turn = listen_for_turn(source, EnergyDetector(), timeout_s=5)
    assert turn is not None
    assert turn.duration_s == pytest.approx(1.6, abs=3 * FRAME_S)
    assert source.closed


def test_listen_times_out_when_nobody_speaks() -> None:
    source = _TrackedSource(silence(0.1))
    assert listen_for_turn(source, EnergyDetector(), timeout_s=2) is None
    assert source.closed


def test_listen_does_not_time_out_mid_sentence() -> None:
    source = _TrackedSource(np.concatenate([silence(0.8), tone(3.0)]))
    turn = listen_for_turn(source, EnergyDetector(), timeout_s=1)
    assert turn is not None
    assert turn.duration_s > 3.0
