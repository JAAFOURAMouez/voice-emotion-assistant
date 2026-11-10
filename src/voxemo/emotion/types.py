"""Central data contract shared by every downstream module (prompts, exercises, tracking)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, get_args

EmotionLabel = Literal["neutre", "joie", "tristesse", "stress"]
EMOTION_LABELS: tuple[EmotionLabel, ...] = get_args(EmotionLabel)


@dataclass(frozen=True)
class ProsodyFeatures:
    """Prosodic cues extracted from one speech turn."""

    pitch_mean_hz: float
    pitch_std_hz: float
    energy_rms: float
    speech_rate: float
    """Estimated syllable-like onsets per second."""
    pause_ratio: float
    """Share of the turn spent in silence, 0–1."""
    jitter: float
    """Approximate cycle-to-cycle pitch variation, relative (0–1)."""


@dataclass(frozen=True)
class EmotionEstimate:
    """Estimated (never diagnosed) emotion for one speech turn."""

    label: EmotionLabel
    confidence: float
    scores: dict[str, float] = field(hash=False)
    arousal: float
    features: ProsodyFeatures
    model_scores: dict[str, float] = field(default_factory=dict, hash=False)
    """Raw (uncalibrated) classifier probabilities, kept to learn the user's baseline."""
    text_terms: tuple[str, ...] = ()
    """Emotion words found in the transcript and taken into account (empty: voice only)."""

    def __post_init__(self) -> None:
        if self.label not in EMOTION_LABELS:
            raise ValueError(f"unknown emotion label {self.label!r}")
        for name in ("confidence", "arousal"):
            value = getattr(self, name)
            if not (math.isfinite(value) and 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be within [0, 1], got {value!r}")
