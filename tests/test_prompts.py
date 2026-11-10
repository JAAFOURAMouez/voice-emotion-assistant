"""Tests for the system prompt and emotion-dependent tone guidance."""

import pytest

from conftest import make_estimate
from voxemo.emotion.types import EMOTION_LABELS
from voxemo.llm.prompts import (
    SAFETY_REMINDER,
    TONE_GUIDANCE,
    ChatMessage,
    build_messages,
    effective_label,
    emotion_hint,
    system_prompt,
)


def test_system_prompt_contains_safety_rule() -> None:
    prompt = system_prompt()
    for number in ("3114", "15", "112"):
        assert number in prompt
    assert "ne propose pas d'exercice de respiration" in prompt


def test_system_prompt_forbids_stating_emotion_as_fact() -> None:
    prompt = system_prompt()
    assert "N'affirme jamais ce que ressent l'utilisateur" in prompt
    assert "sans jamais mentionner cette estimation" in prompt
    assert "diagnostic" in prompt


def test_system_prompt_language_is_configurable() -> None:
    assert "Réponds toujours en français" in system_prompt("fr")
    assert "Réponds toujours en anglais" in system_prompt("en")


def test_every_label_has_tone_guidance() -> None:
    assert set(TONE_GUIDANCE) == set(EMOTION_LABELS)


def test_low_confidence_falls_back_to_neutral() -> None:
    assert effective_label(make_estimate("stress", 0.49), threshold=0.5) == "neutre"
    assert effective_label(make_estimate("stress", 0.5), threshold=0.5) == "stress"
    assert effective_label(None, threshold=0.5) == "neutre"


@pytest.mark.parametrize("label", ["stress", "tristesse", "joie"])
def test_hint_is_phrased_as_a_cue(label: str) -> None:
    hint = emotion_hint(make_estimate(label, 0.72), threshold=0.5)  # type: ignore[arg-type]
    assert "la voix suggère" in hint
    assert "confiance 0,7" in hint
    assert TONE_GUIDANCE[label] in hint  # type: ignore[index]
    assert "ce n'est pas un fait" in hint


def test_no_hint_for_neutral_or_uncertain() -> None:
    assert emotion_hint(make_estimate("neutre", 0.9), threshold=0.5) == ""
    assert emotion_hint(make_estimate("tristesse", 0.3), threshold=0.5) == ""
    assert emotion_hint(None, threshold=0.5) == ""


def test_build_messages_order() -> None:
    history = [ChatMessage("user", "Bonjour"), ChatMessage("assistant", "Bonjour !")]
    messages = build_messages("J'ai trop de travail", history, make_estimate("stress", 0.8))
    assert [m.role for m in messages] == ["system", "user", "assistant", "system", "user"]
    assert "la voix suggère une certaine tension" in messages[3].content
    assert messages[-1] == ChatMessage("user", "J'ai trop de travail")


def test_build_messages_without_hint() -> None:
    messages = build_messages("Quelle heure est-il ?")
    assert [m.role for m in messages] == ["system", "user"]


def test_distress_overrides_emotion_hint() -> None:
    messages = build_messages("Je n'ai plus envie de vivre", estimate=make_estimate("stress"))
    assert messages[-2] == ChatMessage("system", SAFETY_REMINDER)
    assert "la voix suggère" not in "".join(m.content for m in messages)
    assert "3114" in SAFETY_REMINDER


def test_hint_mentions_words_when_they_were_used() -> None:
    import dataclasses

    estimate = dataclasses.replace(make_estimate("stress", 0.7), text_terms=("énervé",))
    assert "la voix et les mots employés suggèrent" in emotion_hint(estimate, threshold=0.5)
