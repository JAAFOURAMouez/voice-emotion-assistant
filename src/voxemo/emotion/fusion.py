"""Fuse model probabilities and prosody into an :class:`EmotionEstimate`.

Prosody is always judged against the user's own baseline (rolling statistics of
their previous turns), never against absolute thresholds. With little history the
baseline is shrunk towards generic population values and the confidence is lowered.

"stress" is not a model class: it is derived from high arousal (higher pitch than
usual, faster speech, fewer pauses, more jitter), the model's "anger" probability
and the absence of joy. The weights below are a starting point, meant to be tuned.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from voxemo.audio.recorder import AudioArray
from voxemo.config import Settings
from voxemo.emotion.classifier import MODEL_CLASSES, EmotionClassifier, ModelClass
from voxemo.emotion.features import FeatureConfig, extract_features
from voxemo.emotion.text_cues import TextCue
from voxemo.emotion.types import EMOTION_LABELS, EmotionEstimate, EmotionLabel, ProsodyFeatures


@dataclass(frozen=True)
class Prior:
    """Generic knowledge about one measure, used until the user's own history is rich."""

    mean: float | None
    """Population mean; ``None`` when it is meaningless (loudness depends on the mic)."""
    between_std: float
    """Spread between speakers (how unsure we are about a new user's usual value)."""
    within_std: float
    """Typical turn-to-turn spread for one speaker."""


PRIORS: dict[str, Prior] = {
    "pitch": Prior(165.0, 50.0, 20.0),  # Hz, mixed voices
    "pitch_var": Prior(25.0, 12.0, 10.0),  # Hz
    "rate": Prior(4.5, 1.2, 0.8),  # syllable-like peaks per second of speech
    "pause": Prior(0.15, 0.12, 0.07),
    "jitter": Prior(0.03, 0.015, 0.008),
    "energy": Prior(None, math.inf, 0.2),  # log10 RMS
}
PRIOR_TURNS = 3  # weight of the prior turn-to-turn spread, in "turns"
Z_CLIP = 3.0
STRESS_GATE_CENTER = 0.65  # arousal at which half of the model's "anger" counts as stress
STRESS_GATE_SLOPE = 10.0
AROUSAL_GAIN = 1.2
PROSODY_SHARPNESS = 0.7  # < 1 softens the prosody-only distribution
TEXT_MIN_RELIABILITY = 0.9  # explicit words do not depend on the voice baseline


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _softmax(logits: Mapping[EmotionLabel, float]) -> dict[EmotionLabel, float]:
    top = max(logits.values())
    exp = {k: math.exp(v - top) for k, v in logits.items()}
    total = sum(exp.values())
    return {k: v / total for k, v in exp.items()}


def _measures(f: ProsodyFeatures) -> dict[str, float | None]:
    """Values compared with the baseline; ``None`` when not measurable for this turn."""
    voiced = f.pitch_mean_hz > 0
    return {
        "pitch": f.pitch_mean_hz if voiced else None,
        "pitch_var": f.pitch_std_hz if voiced else None,
        "jitter": f.jitter if voiced else None,
        "energy": math.log10(f.energy_rms) if f.energy_rms > 0 else None,
        "rate": f.speech_rate if f.energy_rms > 0 else None,
        "pause": f.pause_ratio if f.energy_rms > 0 else None,
    }


class VoiceBaseline:
    """Rolling statistics of the user's previous turns (the current turn excluded).

    Besides prosody, it tracks the model's usual output for this user: the wav2vec2
    model (trained on English) can be biased on a given voice or language, e.g.
    always leaning towards "anger"; only deviations from that habit are evidence.
    """

    def __init__(
        self,
        history: Sequence[ProsodyFeatures] = (),
        window: int = 20,
        model_history: Sequence[Mapping[str, float]] = (),
    ) -> None:
        recent = [f for f in history if f.energy_rms > 0][-window:]
        self.n_turns = len(recent)
        self._values: dict[str, list[float]] = {}
        for features in recent:
            for name, value in _measures(features).items():
                if value is not None:
                    self._values.setdefault(name, []).append(value)
        self._model_history = [m for m in model_history if m][-window:]

    @classmethod
    def from_estimates(
        cls, estimates: Sequence[EmotionEstimate], window: int = 20
    ) -> VoiceBaseline:
        return cls([e.features for e in estimates], window, [e.model_scores for e in estimates])

    def calibrate_model(self, probs: Mapping[ModelClass, float]) -> dict[ModelClass, float]:
        """Divide by the user's usual model output (shrunk towards uniform) and renormalize."""
        n, k = len(self._model_history), PRIOR_TURNS
        uniform = 1 / len(MODEL_CLASSES)
        ratios: dict[ModelClass, float] = {}
        for cls_ in MODEL_CLASSES:
            past = sum(m.get(cls_, 0.0) for m in self._model_history)
            usual = (past + k * uniform) / (n + k)
            ratios[cls_] = probs.get(cls_, 0.0) / max(usual, 1e-3)
        total = sum(ratios.values())
        if total <= 0:
            return dict(probs)
        return {c: r / total for c, r in ratios.items()}

    def _reference(self, name: str) -> tuple[float, float] | None:
        """Expected value and spread of a measure for this user (Bayesian shrinkage).

        The usual value is the posterior mean given the population prior and the
        user's past turns; the spread adds the turn-to-turn variability and the
        remaining uncertainty about that usual value.
        """
        prior = PRIORS[name]
        values = np.asarray(self._values.get(name, []), dtype=np.float64)
        n = len(values)
        if n == 0 and prior.mean is None:
            return None  # e.g. unknown microphone level: no judgement on loudness
        k = PRIOR_TURNS
        within_var = (n * values.var() + k * prior.within_std**2) / (n + k) if n else 0.0
        within_var = max(within_var, (prior.within_std / 2) ** 2) if n else prior.within_std**2
        if prior.mean is None:
            mean, mean_var = float(values.mean()), within_var / n
        else:
            precision = 1 / prior.between_std**2 + n / within_var
            mean_var = 1 / precision
            mean = mean_var * (prior.mean / prior.between_std**2 + values.sum() / within_var)
        return float(mean), math.sqrt(within_var + mean_var)

    def z_scores(self, features: ProsodyFeatures) -> dict[str, float]:
        """How far this turn is from the user's usual voice, in standard deviations."""
        scores: dict[str, float] = {}
        for name, value in _measures(features).items():
            ref = self._reference(name)
            if value is None or ref is None:
                scores[name] = 0.0
            else:
                mean, std = ref
                scores[name] = float(np.clip((value - mean) / std, -Z_CLIP, Z_CLIP))
        return scores


@dataclass(frozen=True)
class FusionConfig:
    model_weight: float = 0.35
    """Share of the wav2vec2 probabilities in the final scores (the rest is prosody)."""
    confidence_threshold: float = 0.5
    baseline_min_turns: int = 5
    min_reliable_duration_s: float = 1.0
    text_gain: float = 5.0
    """Weight of explicit emotion words: likelihood ratio 1 + gain x strength (0 = off)."""

    @classmethod
    def from_settings(cls, settings: Settings) -> FusionConfig:
        return cls(
            model_weight=settings.emotion_model_weight,
            confidence_threshold=settings.emotion_confidence_threshold,
            baseline_min_turns=settings.baseline_min_turns,
            text_gain=settings.emotion_text_gain,
        )


def arousal_from_z(z: Mapping[str, float]) -> float:
    """Vocal activation, 0 (calm, low) to 1 (aroused); 0.5 = the user's usual level."""
    linear = (
        0.35 * z["pitch"]
        + 0.25 * z["energy"]
        + 0.20 * z["rate"]
        - 0.15 * z["pause"]
        + 0.10 * z["jitter"]
        + 0.15 * z["pitch_var"]
    )
    return _sigmoid(AROUSAL_GAIN * linear)


def prosody_scores(z: Mapping[str, float]) -> dict[EmotionLabel, float]:
    """Distribution over labels from prosody alone."""
    deviation = sum(abs(v) for v in z.values()) / len(z)
    logits: dict[EmotionLabel, float] = {
        "stress": 0.8 * z["pitch"]
        + 0.6 * z["rate"]
        - 0.6 * z["pause"]
        + 0.5 * z["jitter"]
        + 0.3 * z["energy"]
        - 0.3 * z["pitch_var"],
        "joie": 0.5 * z["pitch"] + 0.7 * z["energy"] + 0.8 * z["pitch_var"] + 0.2 * z["rate"],
        "tristesse": -0.7 * z["energy"]
        - 0.6 * z["rate"]
        - 0.6 * z["pitch_var"]
        + 0.5 * z["pause"]
        - 0.3 * z["pitch"],
        "neutre": 1.0 - 0.8 * deviation,
    }
    return _softmax({k: PROSODY_SHARPNESS * v for k, v in logits.items()})


def reliability(baseline_turns: int, duration_s: float, config: FusionConfig) -> float:
    """Multiplier (≤ 1) applied to the confidence for short history or short turns."""
    factor = 1.0
    if baseline_turns < config.baseline_min_turns:
        factor *= 0.6 + 0.4 * baseline_turns / config.baseline_min_turns
    if duration_s < config.min_reliable_duration_s:
        factor *= 0.8
    return factor


def fuse(
    model_probs: Mapping[ModelClass, float],
    features: ProsodyFeatures,
    baseline: VoiceBaseline,
    *,
    duration_s: float,
    config: FusionConfig | None = None,
) -> EmotionEstimate:
    cfg = config or FusionConfig()
    raw_probs = dict(model_probs)
    model_probs = baseline.calibrate_model(model_probs)
    z = baseline.z_scores(features)
    prosody = prosody_scores(z)
    arousal = arousal_from_z(z)
    # "Anger" from the model only counts as stress when the voice is also more
    # activated than usual; otherwise that probability mass goes back to neutral.
    anger = model_probs.get("colere", 0.0)
    anger_as_stress = anger * _sigmoid(STRESS_GATE_SLOPE * (arousal - STRESS_GATE_CENTER))
    model: dict[EmotionLabel, float] = {
        "neutre": model_probs.get("neutre", 0.0) + anger - anger_as_stress,
        "joie": model_probs.get("joie", 0.0),
        "tristesse": model_probs.get("tristesse", 0.0),
        "stress": anger_as_stress,
    }
    w = cfg.model_weight
    combined = {k: w * model[k] + (1 - w) * prosody[k] for k in EMOTION_LABELS}
    combined["stress"] *= 1.0 - 0.5 * model["joie"]  # joy in the voice argues against stress
    total = sum(combined.values())
    scores = {k: v / total for k, v in combined.items()}

    factor = reliability(baseline.n_turns, duration_s, cfg)
    best, confidence = _decide(scores, factor, cfg)
    return EmotionEstimate(
        label=best,
        confidence=round(confidence, 4),
        scores={k: round(v, 4) for k, v in scores.items()},
        arousal=round(arousal, 4),
        features=features,
        model_scores={k: round(v, 4) for k, v in raw_probs.items()},
    )


def _decide(
    scores: Mapping[EmotionLabel, float], factor: float, config: FusionConfig
) -> tuple[EmotionLabel, float]:
    """Best label and its confidence; below the threshold, fall back to neutral."""
    best: EmotionLabel = max(scores, key=scores.__getitem__)
    confidence = scores[best] * factor
    if best != "neutre" and confidence < config.confidence_threshold:
        best, confidence = "neutre", scores["neutre"] * factor
    return best, confidence


def apply_text_cue(
    estimate: EmotionEstimate,
    cue: TextCue,
    *,
    baseline_turns: int,
    duration_s: float,
    config: FusionConfig | None = None,
) -> EmotionEstimate:
    """Combine the voice estimate with emotion words of the transcript.

    Each label found in the words multiplies its voice score by a likelihood ratio,
    so explicit words weigh a lot but a clearly different voice still counts. With
    little voice history, the voice scores are first shrunk towards uniform (the
    voice is less trustworthy), and the low-history penalty on the confidence is
    relaxed: words do not depend on the voice baseline.
    """
    cfg = config or FusionConfig()
    if not cue or cfg.text_gain <= 0:
        return estimate
    voice_trust = reliability(baseline_turns, duration_s, cfg)
    uniform = 1.0 / len(EMOTION_LABELS)
    combined = {
        k: (voice_trust * estimate.scores.get(k, 0.0) + (1 - voice_trust) * uniform)
        * (1.0 + cfg.text_gain * cue.strengths.get(k, 0.0))
        for k in EMOTION_LABELS
    }
    total = sum(combined.values())
    if total <= 0:
        return estimate
    scores = {k: v / total for k, v in combined.items()}
    factor = max(voice_trust, TEXT_MIN_RELIABILITY)
    best, confidence = _decide(scores, factor, cfg)
    return dataclasses.replace(
        estimate,
        label=best,
        confidence=round(confidence, 4),
        scores={k: round(v, 4) for k, v in scores.items()},
        text_terms=cue.terms,
    )


class EmotionAnalyzer:
    """Audio of one turn → :class:`EmotionEstimate` (prosody + model + fusion)."""

    def __init__(
        self,
        classifier: EmotionClassifier,
        config: FusionConfig | None = None,
        feature_config: FeatureConfig | None = None,
    ) -> None:
        self.classifier = classifier
        self.config = config or FusionConfig()
        self.feature_config = feature_config

    def analyze(
        self, audio: AudioArray, sample_rate: int, baseline: VoiceBaseline | None = None
    ) -> EmotionEstimate:
        features = extract_features(audio, sample_rate, self.feature_config)
        probs = self.classifier.predict(audio, sample_rate)
        return fuse(
            probs,
            features,
            baseline or VoiceBaseline(),
            duration_s=len(audio) / sample_rate,
            config=self.config,
        )
