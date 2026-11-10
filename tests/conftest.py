"""Shared test doubles and factories."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pytest

from voxemo.config import Settings
from voxemo.emotion.types import EmotionEstimate, EmotionLabel, ProsodyFeatures
from voxemo.llm.prompts import ChatMessage


class FakeChatModel:
    """Records the messages it receives and streams canned replies."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies) or ["D'accord."]
        self.calls: list[list[ChatMessage]] = []

    def stream_chat(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        self.calls.append(list(messages))
        reply = self.replies[min(len(self.calls), len(self.replies)) - 1]
        for word in reply.split(" "):
            yield word + " "


def make_estimate(label: EmotionLabel = "stress", confidence: float = 0.7) -> EmotionEstimate:
    return EmotionEstimate(
        label=label,
        confidence=confidence,
        scores={label: confidence},
        arousal=0.6,
        features=ProsodyFeatures(180.0, 25.0, 0.05, 4.5, 0.1, 0.02),
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, llm_history_turns=3)


SAMPLE_RATE = 16_000


def tone(seconds: float, amplitude: float = 0.3, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


class EnergyDetector:
    """Fake VAD: a frame is 'speech' when its RMS exceeds a threshold."""

    frame_size = 512

    def __init__(self, rms_threshold: float = 0.05) -> None:
        self.rms_threshold = rms_threshold
        self.resets = 0

    def speech_probability(self, frame: np.ndarray) -> float:
        return 1.0 if float(np.sqrt(np.mean(frame**2))) > self.rms_threshold else 0.0

    def reset(self) -> None:
        self.resets += 1
