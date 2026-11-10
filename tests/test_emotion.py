"""Tests for prosodic features, the classifier wrapper and the fusion rules."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from conftest import SAMPLE_RATE, silence, tone
from voxemo.config import Settings
from voxemo.emotion.classifier import (
    MODEL_CLASSES,
    Wav2Vec2EmotionClassifier,
    map_probabilities,
)
from voxemo.emotion.features import extract_features, frame_rms
from voxemo.emotion.fusion import (
    EmotionAnalyzer,
    FusionConfig,
    VoiceBaseline,
    arousal_from_z,
    fuse,
    reliability,
)
from voxemo.emotion.types import EmotionEstimate, ProsodyFeatures


def _vibrato(seconds: float, depth_hz: float, base_hz: float = 200.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    f0 = base_hz + depth_hz * np.sin(2 * np.pi * 3 * t)
    return (0.3 * np.sin(2 * np.pi * np.cumsum(f0) / SAMPLE_RATE)).astype(np.float32)


def _bursts(n: int, on_s: float, off_s: float, amplitude: float = 0.3) -> np.ndarray:
    one = np.concatenate([tone(on_s, amplitude, 180.0), silence(off_s)])
    return np.concatenate([silence(0.3), np.tile(one, n), silence(0.3)])


# --- Features (synthetic signals) ---


def test_frame_rms_of_constant_signal() -> None:
    rms = frame_rms(np.full(1600, 0.5, dtype=np.float32), frame=400, hop=160)
    np.testing.assert_allclose(rms, 0.5)


def test_pitch_of_pure_tone() -> None:
    features = extract_features(tone(1.5, freq=200.0), SAMPLE_RATE)
    assert features.pitch_mean_hz == pytest.approx(200.0, rel=0.03)
    assert features.pitch_std_hz < 2.0
    assert features.jitter < 0.01


def test_pitch_variation_and_jitter_increase_with_vibrato() -> None:
    steady = extract_features(tone(2.0, freq=200.0), SAMPLE_RATE)
    vibrato = extract_features(_vibrato(2.0, depth_hz=40.0), SAMPLE_RATE)
    assert vibrato.pitch_std_hz > 15.0
    assert vibrato.jitter > steady.jitter


def test_energy_follows_amplitude() -> None:
    loud = extract_features(tone(1.0, amplitude=0.4), SAMPLE_RATE)
    quiet = extract_features(tone(1.0, amplitude=0.1), SAMPLE_RATE)
    assert loud.energy_rms == pytest.approx(4 * quiet.energy_rms, rel=0.05)


def test_pauses_and_rate() -> None:
    fast = extract_features(_bursts(12, on_s=0.12, off_s=0.08), SAMPLE_RATE)
    slow = extract_features(_bursts(6, on_s=0.25, off_s=0.35), SAMPLE_RATE)
    assert slow.pause_ratio > fast.pause_ratio
    assert fast.speech_rate > slow.speech_rate
    # VAD padding (leading/trailing silence) must not count as pauses
    assert extract_features(_bursts(1, 1.0, 0.0), SAMPLE_RATE).pause_ratio < 0.05


def test_silence_yields_empty_features() -> None:
    features = extract_features(silence(1.0), SAMPLE_RATE)
    assert features == ProsodyFeatures(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)


# --- Classifier wrapper (fake checkpoint) ---


def test_map_probabilities_projects_and_renormalizes() -> None:
    id2label = {0: "neu", 1: "hap", 2: "ang", 3: "sad", 4: "fear"}
    scores = map_probabilities(id2label, np.array([0.1, 0.2, 0.3, 0.2, 0.2]))
    assert set(scores) == set(MODEL_CLASSES)
    assert scores["colere"] == pytest.approx(0.375)
    assert sum(scores.values()) == pytest.approx(1.0)


def test_map_probabilities_with_unknown_labels_is_neutral() -> None:
    assert map_probabilities({0: "fear"}, np.array([1.0]))["neutre"] == 1.0


class _FakeExtractor:
    sampling_rate = SAMPLE_RATE

    def __init__(self) -> None:
        self.lengths: list[int] = []

    def __call__(self, signal: np.ndarray, **kwargs: Any) -> Any:
        import torch

        self.lengths.append(len(signal))
        values = torch.from_numpy(signal)[None]
        return SimpleNamespace(to=lambda _device: {"input_values": values})


class _FakeModel:
    config = SimpleNamespace(id2label={0: "neu", 1: "hap", 2: "ang", 3: "sad"})

    def __call__(self, input_values: Any) -> Any:
        import torch

        return SimpleNamespace(logits=torch.tensor([[0.0, 0.0, 2.0, 0.0]]))


def test_classifier_predicts_with_fake_checkpoint() -> None:
    classifier = Wav2Vec2EmotionClassifier()
    classifier._extractor, classifier._model = _FakeExtractor(), _FakeModel()
    scores = classifier.predict(silence(0.2), SAMPLE_RATE)
    assert max(scores, key=scores.__getitem__) == "colere"
    assert classifier._extractor.lengths == [8_000]  # short turns are padded


def test_classifier_rejects_wrong_rate_and_loads_lazily() -> None:
    classifier = Wav2Vec2EmotionClassifier()
    assert classifier._model is None
    classifier._extractor, classifier._model = _FakeExtractor(), _FakeModel()
    with pytest.raises(ValueError):
        classifier.predict(silence(1.0), 8_000)


def test_classifier_from_settings() -> None:
    settings = Settings(_env_file=None)
    classifier = Wav2Vec2EmotionClassifier.from_settings(settings)
    assert classifier.model_name == "superb/wav2vec2-base-superb-er"


# --- Baseline ---

USUAL = ProsodyFeatures(
    pitch_mean_hz=120.0,
    pitch_std_hz=18.0,
    energy_rms=0.05,
    speech_rate=4.0,
    pause_ratio=0.2,
    jitter=0.02,
)


def _history(n: int, seed: int = 0) -> list[ProsodyFeatures]:
    """Slightly varying turns around the user's usual voice."""
    rng = np.random.default_rng(seed)
    return [
        replace(
            USUAL,
            pitch_mean_hz=120 + rng.normal(0, 5),
            pitch_std_hz=18 + rng.normal(0, 2),
            energy_rms=0.05 * float(np.exp(rng.normal(0, 0.1))),
            speech_rate=4 + rng.normal(0, 0.3),
            pause_ratio=0.2 + rng.normal(0, 0.03),
            jitter=0.02 + rng.normal(0, 0.003),
        )
        for _ in range(n)
    ]


def test_usual_turn_is_close_to_baseline() -> None:
    z = VoiceBaseline(_history(10)).z_scores(USUAL)
    assert all(abs(v) < 1.0 for v in z.values())


def test_baseline_is_personal() -> None:
    """140 Hz is high for a 120 Hz voice, but not for a 200 Hz one."""
    turn = replace(USUAL, pitch_mean_hz=140.0)
    low_voice = VoiceBaseline(_history(10)).z_scores(turn)["pitch"]
    high_voice = VoiceBaseline([replace(f, pitch_mean_hz=200.0) for f in _history(10)])
    assert low_voice > 1.5
    assert high_voice.z_scores(turn)["pitch"] < -1.5


def test_without_history_energy_is_not_judged() -> None:
    z = VoiceBaseline().z_scores(replace(USUAL, energy_rms=0.5))
    assert z["energy"] == 0.0


def test_baseline_ignores_silent_turns_and_respects_window() -> None:
    silent = ProsodyFeatures(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert VoiceBaseline([silent, *_history(3)]).n_turns == 3
    assert VoiceBaseline(_history(30), window=20).n_turns == 20


def test_unvoiced_turn_has_neutral_pitch_scores() -> None:
    z = VoiceBaseline(_history(10)).z_scores(replace(USUAL, pitch_mean_hz=0.0, pitch_std_hz=0.0))
    assert z["pitch"] == z["pitch_var"] == z["jitter"] == 0.0


def test_model_calibration_removes_a_constant_bias() -> None:
    """A model that always says 'anger' for this voice is no evidence of anger."""
    biased = {"neutre": 0.25, "joie": 0.1, "colere": 0.6, "tristesse": 0.05}
    baseline = VoiceBaseline(_history(10), model_history=[biased] * 10)
    calibrated = baseline.calibrate_model(biased)  # type: ignore[arg-type]
    spread = max(calibrated.values()) - min(calibrated.values())
    assert spread < (0.6 - 0.05) / 2  # bias largely removed (shrinkage keeps some)
    # ...while an unusual output still stands out
    sadder = {"neutre": 0.2, "joie": 0.05, "colere": 0.35, "tristesse": 0.4}
    shifted = baseline.calibrate_model(sadder)  # type: ignore[arg-type]
    assert max(shifted, key=shifted.__getitem__) == "tristesse"


def test_calibration_without_history_keeps_probabilities() -> None:
    probs = {"neutre": 0.7, "joie": 0.1, "colere": 0.1, "tristesse": 0.1}
    calibrated = VoiceBaseline().calibrate_model(probs)  # type: ignore[arg-type]
    assert calibrated == pytest.approx(probs)


def test_biased_anger_does_not_become_stress() -> None:
    biased = {"neutre": 0.25, "joie": 0.1, "colere": 0.6, "tristesse": 0.05}
    history = [
        fuse(biased, f, VoiceBaseline(), duration_s=3.0)  # type: ignore[arg-type]
        for f in _history(10)
    ]
    baseline = VoiceBaseline.from_estimates(history)
    assert baseline.n_turns == 10
    estimate = fuse(biased, USUAL, baseline, duration_s=3.0)  # type: ignore[arg-type]
    assert estimate.label == "neutre"
    assert estimate.model_scores == biased


def test_anger_at_usual_arousal_is_not_stress() -> None:
    angry_model = {"neutre": 0.2, "joie": 0.05, "colere": 0.65, "tristesse": 0.1}
    estimate = _fuse(angry_model, USUAL)
    assert estimate.label != "stress"
    assert estimate.scores["stress"] < _fuse(angry_model, STRESSED).scores["stress"] / 2


# --- Fusion ---

NEUTRAL_MODEL = {"neutre": 0.7, "joie": 0.1, "colere": 0.1, "tristesse": 0.1}
STRESSED = replace(USUAL, pitch_mean_hz=150.0, speech_rate=5.5, pause_ratio=0.05, jitter=0.035)
SAD = replace(USUAL, pitch_mean_hz=112.0, pitch_std_hz=8.0, energy_rms=0.025, speech_rate=2.8)
HAPPY = replace(USUAL, pitch_mean_hz=140.0, pitch_std_hz=35.0, energy_rms=0.09)


def _fuse(model: dict[str, float], features: ProsodyFeatures, turns: int = 10) -> EmotionEstimate:
    return fuse(model, features, VoiceBaseline(_history(turns)), duration_s=3.0)  # type: ignore[arg-type]


def test_scores_form_a_distribution() -> None:
    estimate = _fuse(NEUTRAL_MODEL, USUAL)
    assert set(estimate.scores) == {"neutre", "joie", "tristesse", "stress"}
    assert sum(estimate.scores.values()) == pytest.approx(1.0, abs=1e-3)


def test_usual_voice_is_neutral() -> None:
    estimate = _fuse(NEUTRAL_MODEL, USUAL)
    assert estimate.label == "neutre"
    assert estimate.arousal == pytest.approx(0.5, abs=0.15)


def test_stress_from_arousal_and_anger() -> None:
    angry_model = {"neutre": 0.2, "joie": 0.05, "colere": 0.65, "tristesse": 0.1}
    estimate = _fuse(angry_model, STRESSED)
    assert estimate.label == "stress"
    assert estimate.confidence >= 0.5
    assert estimate.arousal > 0.7


def test_prosody_alone_raises_stress_score() -> None:
    calm, tense = _fuse(NEUTRAL_MODEL, USUAL), _fuse(NEUTRAL_MODEL, STRESSED)
    assert tense.scores["stress"] > 2 * calm.scores["stress"]


def test_joy_in_the_model_argues_against_stress() -> None:
    joyful_model = {"neutre": 0.1, "joie": 0.8, "colere": 0.05, "tristesse": 0.05}
    estimate = _fuse(joyful_model, STRESSED)
    # The model weighs little by default (unreliable on French): joy lowers stress
    # but a very tense voice may still win.
    assert estimate.scores["stress"] < _fuse(NEUTRAL_MODEL, STRESSED).scores["stress"]


def test_sadness_from_model_weighted_by_prosody() -> None:
    sad_model = {"neutre": 0.25, "joie": 0.05, "colere": 0.05, "tristesse": 0.65}
    sad_voice, usual_voice = _fuse(sad_model, SAD), _fuse(sad_model, USUAL)
    assert sad_voice.label == "tristesse"
    assert sad_voice.scores["tristesse"] > usual_voice.scores["tristesse"]
    assert sad_voice.arousal < 0.4


def test_joy_from_model_weighted_by_prosody() -> None:
    happy_model = {"neutre": 0.2, "joie": 0.7, "colere": 0.05, "tristesse": 0.05}
    estimate = _fuse(happy_model, HAPPY)
    assert estimate.label == "joie"
    assert estimate.scores["joie"] > _fuse(happy_model, SAD).scores["joie"]


def test_short_history_lowers_confidence() -> None:
    angry_model = {"neutre": 0.2, "joie": 0.05, "colere": 0.65, "tristesse": 0.1}
    rich, poor = _fuse(angry_model, STRESSED, turns=10), _fuse(angry_model, STRESSED, turns=1)
    assert poor.confidence < rich.confidence


def test_reliability_factor() -> None:
    config = FusionConfig()
    assert reliability(5, 3.0, config) == 1.0
    assert reliability(0, 3.0, config) == pytest.approx(0.6)
    assert reliability(10, 0.5, config) == pytest.approx(0.8)


def test_below_threshold_falls_back_to_neutral() -> None:
    hesitant = {"neutre": 0.4, "joie": 0.1, "colere": 0.4, "tristesse": 0.1}
    estimate = fuse(hesitant, STRESSED, VoiceBaseline(), duration_s=3.0)  # type: ignore[arg-type]
    assert estimate.label == "neutre"
    assert estimate.confidence < 0.5


def test_threshold_is_configurable() -> None:
    angry_model = {"neutre": 0.2, "joie": 0.05, "colere": 0.65, "tristesse": 0.1}
    strict = FusionConfig(confidence_threshold=0.99)
    estimate = fuse(angry_model, STRESSED, VoiceBaseline(_history(10)), duration_s=3, config=strict)  # type: ignore[arg-type]
    assert estimate.label == "neutre"


def test_arousal_is_bounded() -> None:
    names = ("pitch", "pitch_var", "energy", "rate", "pause", "jitter")
    assert 0.0 < arousal_from_z(dict.fromkeys(names, -3.0)) < 0.1
    assert 0.9 < arousal_from_z(dict.fromkeys(names, 3.0)) < 1.0


def test_estimate_validates_ranges() -> None:
    with pytest.raises(ValueError):
        EmotionEstimate("stress", 1.2, {}, 0.5, USUAL)
    with pytest.raises(ValueError):
        EmotionEstimate("peur", 0.5, {}, 0.5, USUAL)  # type: ignore[arg-type]


# --- Analyzer ---


class _ConstantClassifier:
    def __init__(self, probs: dict[str, float]) -> None:
        self.probs = probs
        self.calls = 0

    def predict(self, audio: np.ndarray, sample_rate: int) -> dict[str, float]:
        self.calls += 1
        return self.probs


def test_analyzer_combines_features_and_model() -> None:
    classifier = _ConstantClassifier(NEUTRAL_MODEL)
    estimate = EmotionAnalyzer(classifier).analyze(tone(1.5, freq=150.0), SAMPLE_RATE)  # type: ignore[arg-type]
    assert classifier.calls == 1
    assert estimate.features.pitch_mean_hz == pytest.approx(150.0, rel=0.03)
    assert estimate.label == "neutre"


def test_fusion_config_from_settings() -> None:
    settings = Settings(_env_file=None, emotion_model_weight=0.3, baseline_min_turns=8)
    config = FusionConfig.from_settings(settings)
    assert (config.model_weight, config.baseline_min_turns) == (0.3, 8)


def test_report_turns_uses_rolling_baseline() -> None:
    import io

    from voxemo.audio.vad import SpeechTurn
    from voxemo.emotion.__main__ import report_turns

    turns = [SpeechTurn(tone(1.2, freq=150.0 + 10 * i), SAMPLE_RATE, 2.0 * i) for i in range(3)]
    analyzer = EmotionAnalyzer(_ConstantClassifier(NEUTRAL_MODEL))  # type: ignore[arg-type]
    out = io.StringIO()
    history = report_turns(turns, analyzer, Settings(_env_file=None), out)
    lines = out.getvalue().splitlines()
    assert len(lines) == len(history) == 3
    assert "émotion estimée : neutre" in lines[0]
    assert history[-1].model_scores == NEUTRAL_MODEL
