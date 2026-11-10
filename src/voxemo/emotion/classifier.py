"""Speech emotion recognition model (wav2vec2 fine-tuned on IEMOCAP, English).

Its outputs on French speech are approximate; they are only one input of the fusion.
"""

from __future__ import annotations

from threading import Lock
from typing import Any, Literal, Protocol

import numpy as np

from voxemo.audio.recorder import AudioArray
from voxemo.config import Settings

ModelClass = Literal["neutre", "joie", "colere", "tristesse"]
MODEL_CLASSES: tuple[ModelClass, ...] = ("neutre", "joie", "colere", "tristesse")

# Label names used by common speech-emotion checkpoints → our classes.
LABEL_MAP: dict[str, ModelClass] = {
    "neu": "neutre",
    "neutral": "neutre",
    "hap": "joie",
    "happy": "joie",
    "happiness": "joie",
    "joy": "joie",
    "exc": "joie",
    "ang": "colere",
    "angry": "colere",
    "anger": "colere",
    "sad": "tristesse",
    "sadness": "tristesse",
}

MIN_SAMPLES = 8_000  # 0.5 s at 16 kHz: shorter inputs are padded with silence


class EmotionClassifier(Protocol):
    def predict(self, audio: AudioArray, sample_rate: int) -> dict[ModelClass, float]:
        """Probabilities for each of :data:`MODEL_CLASSES` (summing to 1)."""
        ...


class Wav2Vec2EmotionClassifier:
    """`EmotionClassifier` backed by a Hugging Face audio-classification checkpoint.

    The model is loaded once, on first use. Classes the checkpoint does not know get
    probability 0; classes we do not use (e.g. "fear") are dropped before renormalizing.
    """

    def __init__(self, model_name: str = "superb/wav2vec2-base-superb-er", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._extractor: Any = None
        self._lock = Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> Wav2Vec2EmotionClassifier:
        return cls(settings.emotion_model, settings.emotion_device)

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            # Use the local copy without asking the Hugging Face Hub (offline, private);
            # download only the first time.
            try:
                self._extractor = AutoFeatureExtractor.from_pretrained(
                    self.model_name, local_files_only=True
                )
                model = AutoModelForAudioClassification.from_pretrained(
                    self.model_name, local_files_only=True
                )
            except OSError:
                self._extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
                model = AutoModelForAudioClassification.from_pretrained(self.model_name)
            self._model = model.to(self.device).eval()

    @property
    def sample_rate(self) -> int:
        self.load()
        return int(self._extractor.sampling_rate)

    def predict(self, audio: AudioArray, sample_rate: int) -> dict[ModelClass, float]:
        import torch

        self.load()
        if sample_rate != self.sample_rate:
            raise ValueError(f"model expects {self.sample_rate} Hz audio, got {sample_rate}")
        signal = np.asarray(audio, dtype=np.float32)
        if len(signal) < MIN_SAMPLES:
            signal = np.pad(signal, (0, MIN_SAMPLES - len(signal)))
        inputs = self._extractor(signal, sampling_rate=sample_rate, return_tensors="pt")
        with torch.inference_mode():
            logits = self._model(**inputs.to(self.device)).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return map_probabilities(self._model.config.id2label, probs)


def map_probabilities(id2label: dict[int, str], probs: np.ndarray) -> dict[ModelClass, float]:
    """Project checkpoint-specific labels onto :data:`MODEL_CLASSES` and renormalize."""
    scores: dict[ModelClass, float] = dict.fromkeys(MODEL_CLASSES, 0.0)
    for index, prob in enumerate(probs):
        target = LABEL_MAP.get(str(id2label[int(index)]).lower())
        if target is not None:
            scores[target] += float(prob)
    total = sum(scores.values())
    if total <= 0:
        return {**dict.fromkeys(MODEL_CLASSES, 0.0), "neutre": 1.0}
    return {k: v / total for k, v in scores.items()}
