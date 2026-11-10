"""Tests for severe-distress detection."""

import pytest

from voxemo.wellbeing.safety import mentions_helpline, mentions_severe_distress


@pytest.mark.parametrize(
    "text",
    [
        "Je pense au suicide depuis quelques jours",
        "j'ai envie d'en finir",
        "Je veux en finir avec la vie",
        "Parfois je voudrais mourir",
        "Je n'ai plus envie de vivre",
        "J'ai envie de me faire du mal",
        "Je pense à mettre fin à mes jours",
        "J’ai envie de me tuer",
        "I want to kill myself",
        "I want to die",
    ],
)
def test_detects_severe_distress(text: str) -> None:
    assert mentions_severe_distress(text)


@pytest.mark.parametrize(
    "text",
    [
        "Je suis fatigué de cette journée",
        "Je meurs de faim",
        "Il faut que j'en finisse avec ce rapport",
        "Ce film m'a tué de rire",
        "J'ai envie de vivre à la mer",
    ],
)
def test_ignores_everyday_wording(text: str) -> None:
    assert not mentions_severe_distress(text)


def test_mentions_helpline() -> None:
    assert mentions_helpline("Appelez le 3114.")
    assert not mentions_helpline("Prenez soin de vous.")
