"""Tests for the exercise catalogue, the offer rule, the answers and the guide."""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta

import pytest

from conftest import make_estimate
from voxemo.config import Settings
from voxemo.wellbeing.__main__ import main as wellbeing_main
from voxemo.wellbeing.exercises import (
    CATALOGUE,
    ExerciseAdvisor,
    classify_answer,
    get_exercise,
)
from voxemo.wellbeing.guide import guide

T0 = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
CALM = "Je voulais parler de ma journée."


def _minutes(m: float) -> datetime:
    return T0 + timedelta(minutes=m)


# --- catalogue ---


def test_catalogue_has_the_five_planned_exercises() -> None:
    assert [e.id for e in CATALOGUE] == [
        "coherence_cardiaque",
        "respiration_478",
        "respiration_carree",
        "ancrage_54321",
        "relaxation_musculaire",
    ]


@pytest.mark.parametrize("exercise", CATALOGUE, ids=lambda e: e.id)
def test_exercises_are_short_and_well_formed(exercise) -> None:
    assert exercise.steps and all(step.seconds > 0 for step in exercise.steps)
    assert 30 <= exercise.total_seconds <= 180
    assert exercise.suited_for <= {"stress", "tristesse"} and exercise.suited_for
    assert exercise.name in exercise.introduction()


def test_breathing_rhythms_match_their_names() -> None:
    assert [s.seconds for s in get_exercise("coherence_cardiaque").steps] == [5, 5]
    assert [s.seconds for s in get_exercise("respiration_478").steps] == [4, 7, 8]
    assert [s.seconds for s in get_exercise("respiration_carree").steps] == [4, 4, 4, 4]
    assert len(get_exercise("ancrage_54321").steps) == 5
    assert get_exercise("coherence_cardiaque").total_seconds == 120


def test_unknown_exercise() -> None:
    with pytest.raises(KeyError, match="known"):
        get_exercise("yoga")


def test_offer_is_cautious_and_declinable() -> None:
    text = get_exercise("coherence_cardiaque").offer("stress")
    assert "semblent un peu tendus" in text
    assert "je peux me tromper" in text
    assert "dire non" in text
    assert "vous êtes stressé" not in text.lower()


# --- offer rule ---


def test_two_consecutive_confident_turns_trigger_an_offer() -> None:
    advisor = ExerciseAdvisor()
    assert advisor.observe(make_estimate("stress", 0.7), CALM, _minutes(0)) is None
    suggestion = advisor.observe(make_estimate("stress", 0.7), CALM, _minutes(1))
    assert suggestion is not None
    assert suggestion.label == "stress"
    assert "stress" in suggestion.exercise.suited_for


def test_stress_then_sadness_counts_and_follows_the_latest() -> None:
    advisor = ExerciseAdvisor()
    advisor.observe(make_estimate("stress", 0.7), CALM, _minutes(0))
    suggestion = advisor.observe(make_estimate("tristesse", 0.7), CALM, _minutes(1))
    assert suggestion is not None and suggestion.label == "tristesse"
    assert "tristesse" in suggestion.exercise.suited_for


@pytest.mark.parametrize(
    "middle",
    [make_estimate("neutre", 0.9), make_estimate("joie", 0.8), make_estimate("stress", 0.4), None],
)
def test_streak_is_broken_by_other_turns(middle) -> None:
    advisor = ExerciseAdvisor()
    advisor.observe(make_estimate("stress", 0.7), CALM, _minutes(0))
    assert advisor.observe(middle, CALM, _minutes(1)) is None
    assert advisor.observe(make_estimate("stress", 0.7), CALM, _minutes(2)) is None


def test_cooldown_limits_offers_even_when_declined() -> None:
    advisor = ExerciseAdvisor(cooldown=timedelta(minutes=15))
    stress = make_estimate("stress", 0.8)
    advisor.observe(stress, CALM, _minutes(0))
    first = advisor.observe(stress, CALM, _minutes(1))
    assert first is not None
    for m in (2, 3, 10, 15):
        assert advisor.observe(stress, CALM, _minutes(m)) is None
    second = advisor.observe(stress, CALM, _minutes(16.5))
    assert second is not None
    assert second.exercise != first.exercise  # variety: not the same one twice in a row


def test_never_offered_after_severe_distress() -> None:
    advisor = ExerciseAdvisor()
    sad = make_estimate("tristesse", 0.9)
    advisor.observe(sad, "Je n'en peux plus", _minutes(0))
    assert advisor.observe(sad, "J'ai envie de mourir", _minutes(1)) is None
    assert advisor.distress_seen
    for m in range(2, 60, 5):  # nor later in the session
        assert advisor.observe(sad, CALM, _minutes(m)) is None


def test_from_settings() -> None:
    settings = Settings(_env_file=None, exercise_consecutive_turns=3, exercise_cooldown_minutes=5)
    advisor = ExerciseAdvisor.from_settings(settings)
    assert advisor.consecutive_turns == 3
    assert advisor.cooldown == timedelta(minutes=5)
    stress = make_estimate("stress", 0.8)
    assert advisor.observe(stress, CALM, _minutes(0)) is None
    assert advisor.observe(stress, CALM, _minutes(1)) is None
    assert advisor.observe(stress, CALM, _minutes(2)) is not None


# --- answers ---


@pytest.mark.parametrize(
    "text", ["Oui.", "Oui, d'accord", "OK", "Volontiers !", "Allons-y", "Pourquoi pas", "ouais"]
)
def test_accepting_answers(text: str) -> None:
    assert classify_answer(text) == "accept"


@pytest.mark.parametrize(
    "text",
    ["Non.", "Non merci", "Pas maintenant", "Oui mais plus tard", "Ça ira", "Une autre fois"],
)
def test_declining_answers(text: str) -> None:
    assert classify_answer(text) == "decline"


@pytest.mark.parametrize(
    "text",
    [
        "Je voulais plutôt parler de mon examen de demain matin",
        "Comment ça marche ?",
        "",
        "Non, je veux mourir",  # never a mere "no": the safety path must answer
    ],
)
def test_other_answers_are_not_classified(text: str) -> None:
    assert classify_answer(text) is None


# --- guide ---


def test_guide_runs_every_step_and_cycle() -> None:
    out, slept = io.StringIO(), []
    assert guide(get_exercise("coherence_cardiaque"), out, sleep=slept.append, interactive=False)
    text = out.getvalue()
    assert text.count("Inspirez…") == 12
    assert "Cycle 12/12" in text
    assert "C'est terminé" in text
    assert sum(slept) == 120


def test_interactive_guide_counts_down_each_second() -> None:
    out, slept = io.StringIO(), []
    guide(get_exercise("respiration_478"), out, sleep=slept.append, interactive=True)
    assert len(slept) == 4 * (4 + 7 + 8)
    assert "Retenez…  7" in out.getvalue()


def test_guide_can_be_stopped() -> None:
    out = io.StringIO()

    def interrupt(_: float) -> None:
        raise KeyboardInterrupt

    assert not guide(get_exercise("ancrage_54321"), out, sleep=interrupt, interactive=False)
    assert "Exercice interrompu" in out.getvalue()


def test_wellbeing_cli_lists_exercises(capsys: pytest.CaptureFixture[str]) -> None:
    assert wellbeing_main([]) == 0
    text = capsys.readouterr().out
    assert "coherence_cardiaque" in text and "environ 2 minutes" in text


def test_duration_text() -> None:
    assert get_exercise("respiration_478").duration_text == "environ 1 minute"
    assert get_exercise("coherence_cardiaque").duration_text == "environ 2 minutes"
