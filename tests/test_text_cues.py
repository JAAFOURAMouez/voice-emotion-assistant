"""Tests for emotion words in the transcript and their fusion with the voice."""

from __future__ import annotations

import dataclasses

import pytest

from conftest import make_estimate
from voxemo.emotion.fusion import FusionConfig, apply_text_cue
from voxemo.emotion.text_cues import TextCue, fold, text_cues
from voxemo.emotion.types import EmotionEstimate

# --- lexicon ---


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("Je suis très énervé.", "stress"),
        ("Je suis stressée par mon examen", "stress"),
        ("j'ai peur de rater", "stress"),
        ("J'en ai marre de tout ça", "stress"),
        ("Je me sens triste aujourd'hui", "tristesse"),
        ("Je suis 0 en maths", "tristesse"),
        ("je suis nul", "tristesse"),
        ("J'ai pas le moral", "tristesse"),
        ("Je suis tellement contente !", "joie"),
        ("C'est génial, j'ai réussi", "joie"),
        ("L'entretien, c'était bien passé", "joie"),
        ("I am really worried", "stress"),
    ],
)
def test_finds_explicit_emotion_words(text: str, label: str) -> None:
    language = "en" if text.startswith("I ") else "fr"
    cue = text_cues(text, language)
    assert max(cue.strengths, key=cue.strengths.__getitem__) == label


@pytest.mark.parametrize(
    "text",
    [
        "Bonjour.",
        "Merci",
        "Je ne suis pas stressé.",
        "Je n'ai plus peur",
        "Heureusement, la réunion est finie.",
        "Il y a du monde dans le métro.",
        "Je suis sans stress depuis hier",
    ],
)
def test_ignores_neutral_and_negated_text(text: str) -> None:
    assert not text_cues(text)


def test_negated_joy_suggests_mild_sadness() -> None:
    for text, term in [
        ("Je ne suis pas très content", "pas très content"),
        ("j'ai pas réussi", "pas réussi"),
    ]:
        cue = text_cues(text)
        assert cue.strengths == {"tristesse": 0.5}
        assert cue.terms == (term,)


@pytest.mark.parametrize(
    "text", ["Putain, ça me fait chier", "ça me saoule", "c'est chiant", "quelle galère"]
)
def test_finds_familiar_annoyance(text: str) -> None:
    assert "stress" in text_cues(text).strengths


def test_negation_does_not_cross_a_clause() -> None:
    cue = text_cues("Ça ne va pas, je suis stressé")
    assert cue.strengths.get("stress") == 1.0


def test_intensifier_doubles_strength_and_terms_keep_accents() -> None:
    plain, strong = text_cues("je suis énervé"), text_cues("je suis très énervée")
    assert plain.strengths["stress"] == 1.0
    assert strong.strengths["stress"] == 2.0
    assert strong.terms == ("énervée",)


def test_strength_is_capped() -> None:
    cue = text_cues("très stressé, très angoissé, très inquiet, très tendu")
    assert cue.strengths["stress"] == 3.0
    assert len(cue.terms) == 4


def test_unknown_language_gives_no_cue() -> None:
    assert not text_cues("Ich bin sehr traurig", "de")


def test_fold_keeps_length() -> None:
    text = "Ça m’énerve, ÉNORMÉMENT"
    assert len(fold(text)) == len(text)
    assert fold(text) == "ca m'enerve, enormement"


# --- fusion with the voice estimate ---


def _voice(neutre: float = 0.4, stress: float = 0.2, joie: float = 0.2, tristesse: float = 0.2):
    base = make_estimate("neutre", 0.24)
    return dataclasses.replace(
        base, scores={"neutre": neutre, "joie": joie, "tristesse": tristesse, "stress": stress}
    )


def test_explicit_words_win_over_an_undecided_voice_at_cold_start() -> None:
    estimate = apply_text_cue(
        _voice(), text_cues("Je suis très énervé."), baseline_turns=1, duration_s=2.0
    )
    assert estimate.label == "stress"
    assert estimate.confidence >= 0.5
    assert estimate.text_terms == ("énervé",)
    assert sum(estimate.scores.values()) == pytest.approx(1.0, abs=1e-3)


def test_a_clearly_different_voice_still_counts() -> None:
    joyful_voice = _voice(neutre=0.1, stress=0.05, joie=0.8, tristesse=0.05)
    estimate = apply_text_cue(
        joyful_voice, text_cues("je suis un peu inquiet"), baseline_turns=10, duration_s=2.0
    )
    assert estimate.label == "joie"


def test_self_deprecation_overrides_a_weak_joy_estimate() -> None:
    weak_joy = _voice(neutre=0.25, stress=0.15, joie=0.45, tristesse=0.15)
    estimate = apply_text_cue(
        weak_joy, text_cues("Je suis 0 en maths"), baseline_turns=4, duration_s=1.5
    )
    assert estimate.label != "joie"


def test_without_words_the_estimate_is_unchanged() -> None:
    voice = _voice()
    assert apply_text_cue(voice, TextCue(), baseline_turns=0, duration_s=2.0) is voice


def test_text_gain_zero_disables_words() -> None:
    voice = _voice()
    cue = text_cues("très stressé")
    config = FusionConfig(text_gain=0.0)
    assert apply_text_cue(voice, cue, baseline_turns=0, duration_s=2.0, config=config) is voice


def test_raw_model_scores_and_features_are_kept() -> None:
    voice: EmotionEstimate = dataclasses.replace(_voice(), model_scores={"colere": 0.6})
    estimate = apply_text_cue(voice, text_cues("triste"), baseline_turns=0, duration_s=2.0)
    assert estimate.model_scores == {"colere": 0.6}
    assert estimate.features == voice.features
