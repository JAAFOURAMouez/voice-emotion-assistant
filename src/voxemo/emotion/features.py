"""Prosodic features of one speech turn: pitch, energy, speaking rate, pauses, jitter.

All values are raw measurements; they only become meaningful once compared with the
user's own baseline (see :mod:`voxemo.emotion.fusion`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from voxemo.emotion.types import ProsodyFeatures

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class FeatureConfig:
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    silence_db: float = -35.0
    """Frames quieter than this (relative to the loudest frame) count as pauses."""
    min_rms: float = 1e-4
    """Absolute floor: below this a frame is silent whatever the turn loudness."""
    fmin_hz: float = 65.0
    fmax_hz: float = 500.0
    pitch_hop_ms: float = 20.0
    syllable_prominence_db: float = 3.0
    min_syllable_gap_ms: float = 100.0


def frame_rms(audio: npt.NDArray[np.floating], frame: int, hop: int) -> FloatArray:
    """Root-mean-square energy of overlapping frames (no padding)."""
    signal = np.asarray(audio, dtype=np.float64)
    if len(signal) < frame:
        signal = np.pad(signal, (0, frame - len(signal)))
    windows = np.lib.stride_tricks.sliding_window_view(signal, frame)[::hop]
    return np.sqrt(np.mean(windows**2, axis=1))


def _speech_span(silent: npt.NDArray[np.bool_]) -> slice:
    """From the first to the last non-silent frame (drops VAD padding at both ends)."""
    voiced = np.flatnonzero(~silent)
    if len(voiced) == 0:
        return slice(0, 0)
    return slice(int(voiced[0]), int(voiced[-1]) + 1)


def _count_syllables(rms: FloatArray, silent: npt.NDArray[np.bool_], cfg: FeatureConfig) -> int:
    """Syllable nuclei ≈ prominent peaks of the smoothed loudness envelope."""
    from scipy.signal import find_peaks

    db = 20 * np.log10(np.maximum(rms, 1e-10))
    width = max(1, round(50 / cfg.hop_ms))
    smoothed = np.convolve(db, np.ones(width) / width, mode="same")
    peaks, _ = find_peaks(
        smoothed,
        prominence=cfg.syllable_prominence_db,
        distance=max(1, round(cfg.min_syllable_gap_ms / cfg.hop_ms)),
    )
    return int(np.count_nonzero(~silent[peaks]))


def _pitch_track(
    audio: npt.NDArray[np.floating], sample_rate: int, cfg: FeatureConfig
) -> FloatArray:
    """Fundamental frequency of voiced frames only (Hz), via librosa's pYIN."""
    import librosa

    hop = int(sample_rate * cfg.pitch_hop_ms / 1000)
    f0, voiced, _ = librosa.pyin(
        np.asarray(audio, dtype=np.float32),
        fmin=cfg.fmin_hz,
        fmax=cfg.fmax_hz,
        sr=sample_rate,
        frame_length=4 * hop,
        hop_length=hop,
    )
    return np.where(voiced, f0, np.nan).astype(np.float64)


def _jitter(f0: FloatArray) -> float:
    """Mean relative change of f0 between consecutive voiced frames (frame-level proxy)."""
    steps = np.abs(np.diff(f0))
    valid = ~np.isnan(steps)
    if not valid.any():
        return 0.0
    return float(np.mean(steps[valid]) / np.nanmean(f0))


def extract_features(
    audio: npt.NDArray[np.floating], sample_rate: int, config: FeatureConfig | None = None
) -> ProsodyFeatures:
    cfg = config or FeatureConfig()
    frame = int(sample_rate * cfg.frame_ms / 1000)
    hop = int(sample_rate * cfg.hop_ms / 1000)
    rms = frame_rms(audio, frame, hop)

    peak = float(rms.max()) if len(rms) else 0.0
    if peak < cfg.min_rms:
        return ProsodyFeatures(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    db = 20 * np.log10(np.maximum(rms, 1e-10) / peak)
    silent = (db < cfg.silence_db) | (rms < cfg.min_rms)
    span = _speech_span(silent)
    rms, silent = rms[span], silent[span]
    speaking_s = np.count_nonzero(~silent) * cfg.hop_ms / 1000

    f0 = _pitch_track(audio, sample_rate, cfg)
    voiced_f0 = f0[~np.isnan(f0)]

    return ProsodyFeatures(
        pitch_mean_hz=float(voiced_f0.mean()) if len(voiced_f0) else 0.0,
        pitch_std_hz=float(voiced_f0.std()) if len(voiced_f0) > 1 else 0.0,
        energy_rms=float(rms[~silent].mean()),
        speech_rate=float(_count_syllables(rms, silent, cfg) / speaking_s) if speaking_s else 0.0,
        pause_ratio=float(silent.mean()),
        jitter=min(1.0, _jitter(f0)),
    )
