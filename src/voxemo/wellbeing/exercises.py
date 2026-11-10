"""Breathing and relaxation exercises, and the rule deciding when to offer one.

An exercise is only ever *offered*: the user can decline, and nothing starts without
an explicit yes. It is never offered when the user's words suggest severe distress;
in that case the assistant points to people and helplines instead (see ``safety``).
The catalogue text is user-facing (French).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from voxemo import texts
from voxemo.config import Settings
from voxemo.emotion.types import EmotionEstimate, EmotionLabel
from voxemo.wellbeing.safety import mentions_severe_distress

StepKind = Literal["inspire", "expire", "hold", "free"]
"""How the visual guide animates a step (grow, shrink, stay, gentle pulse)."""


@dataclass(frozen=True)
class Step:
    instruction: str
    seconds: float
    kind: StepKind = "free"


@dataclass(frozen=True)
class Exercise:
    id: str
    name: str
    summary: str
    """One sentence presenting the exercise, read before starting."""
    steps: tuple[Step, ...]
    """One cycle; repeated ``cycles`` times."""
    cycles: int = 1
    suited_for: frozenset[EmotionLabel] = field(default_factory=frozenset)
    caution: str = ""

    @property
    def total_seconds(self) -> float:
        return self.cycles * sum(step.seconds for step in self.steps)

    @property
    def duration_text(self) -> str:
        minutes = max(1, round(self.total_seconds / 60))
        return "environ 1 minute" if minutes == 1 else f"environ {minutes} minutes"

    def introduction(self) -> str:
        """What the assistant says when the user accepts."""
        return texts.EXERCISE_STARTING.format(
            name=self.name, duration=self.duration_text, summary=self.summary, caution=self.caution
        ).strip()

    def offer(self, label: EmotionLabel) -> str:
        """The proposal itself, always declinable."""
        return texts.EXERCISE_OFFER.format(
            reason=texts.EXERCISE_REASONS[label], name=self.name, duration=self.duration_text
        )


_STRESS = frozenset({"stress"})
_SAD = frozenset({"tristesse"})
_BOTH = _STRESS | _SAD
_GENERAL_CAUTION = "Arrêtez-vous à tout moment si vous ressentez un inconfort."

CATALOGUE: tuple[Exercise, ...] = (
    Exercise(
        id="coherence_cardiaque",
        name="Cohérence cardiaque 5-5",
        summary=(
            "Inspirez pendant 5 secondes, puis expirez pendant 5 secondes, calmement, "
            "par le nez si possible."
        ),
        steps=(Step("Inspirez", 5, "inspire"), Step("Expirez", 5, "expire")),
        cycles=12,
        suited_for=_BOTH,
        caution=_GENERAL_CAUTION,
    ),
    Exercise(
        id="respiration_478",
        name="Respiration 4-7-8",
        summary=(
            "Inspirez par le nez sur 4 temps, retenez votre souffle sur 7 temps, puis "
            "expirez lentement par la bouche sur 8 temps."
        ),
        steps=(
            Step("Inspirez par le nez", 4, "inspire"),
            Step("Retenez", 7, "hold"),
            Step("Expirez", 8, "expire"),
        ),
        cycles=4,
        suited_for=_STRESS,
        caution=(
            "Si retenir votre souffle est inconfortable, raccourcissez la pause. "
            + _GENERAL_CAUTION
        ),
    ),
    Exercise(
        id="respiration_carree",
        name="Respiration carrée",
        summary="Quatre temps égaux : inspirer, retenir, expirer, retenir, 4 secondes chacun.",
        steps=(
            Step("Inspirez", 4, "inspire"),
            Step("Retenez, poumons pleins", 4, "hold"),
            Step("Expirez", 4, "expire"),
            Step("Retenez, poumons vides", 4, "hold"),
        ),
        cycles=6,
        suited_for=_STRESS,
        caution=_GENERAL_CAUTION,
    ),
    Exercise(
        id="ancrage_54321",
        name="Ancrage 5-4-3-2-1",
        summary=(
            "Revenez à l'instant présent en portant attention à vos sens, un par un, "
            "à votre rythme."
        ),
        steps=(
            Step("Nommez 5 choses que vous voyez autour de vous", 25),
            Step("Remarquez 4 choses que vous pouvez toucher", 20),
            Step("Écoutez 3 sons autour de vous", 20),
            Step("Repérez 2 odeurs", 15),
            Step("Notez 1 goût, ou une chose que vous appréciez en ce moment", 15),
        ),
        suited_for=_BOTH,
    ),
    Exercise(
        id="relaxation_musculaire",
        name="Relaxation musculaire courte",
        summary=(
            "Contractez doucement un groupe de muscles quelques secondes, puis relâchez "
            "et observez la détente."
        ),
        steps=(
            Step("Serrez les poings et les avant-bras", 5, "hold"),
            Step("Relâchez, sentez la détente", 10, "expire"),
            Step("Montez les épaules vers les oreilles", 5, "hold"),
            Step("Relâchez les épaules", 10, "expire"),
            Step("Plissez le visage, yeux et mâchoire", 5, "hold"),
            Step("Relâchez le visage", 10, "expire"),
            Step("Contractez les jambes et les pieds", 5, "hold"),
            Step("Relâchez tout le corps", 15, "expire"),
        ),
        suited_for=_BOTH,
        caution="Contractez sans forcer, et évitez toute zone douloureuse. " + _GENERAL_CAUTION,
    ),
)
_BY_ID = {exercise.id: exercise for exercise in CATALOGUE}


def get_exercise(exercise_id: str) -> Exercise:
    try:
        return _BY_ID[exercise_id]
    except KeyError:
        raise KeyError(f"unknown exercise {exercise_id!r}; known: {', '.join(_BY_ID)}") from None


# --- when to offer ---

TRIGGER_LABELS: frozenset[EmotionLabel] = frozenset({"stress", "tristesse"})


@dataclass(frozen=True)
class Suggestion:
    exercise: Exercise
    label: EmotionLabel
    """The estimated emotion that motivated the offer (stress or tristesse)."""


class ExerciseAdvisor:
    """Offers an exercise after N consecutive turns of estimated stress or sadness.

    - only estimates at or above the confidence threshold count;
    - at most one offer per cooldown period, whether accepted or declined;
    - never after words suggesting severe distress (for the rest of the session);
    - the exercise matches the most recent label and avoids repeating the last one.
    """

    def __init__(
        self,
        *,
        consecutive_turns: int = 2,
        cooldown: timedelta = timedelta(minutes=15),
        confidence_threshold: float = 0.5,
        catalogue: Sequence[Exercise] = CATALOGUE,
    ) -> None:
        self.consecutive_turns = consecutive_turns
        self.cooldown = cooldown
        self.confidence_threshold = confidence_threshold
        self.catalogue = tuple(catalogue)
        self._streak: list[EmotionLabel] = []
        self._last_offer_at: datetime | None = None
        self._last_exercise: Exercise | None = None
        self.distress_seen = False

    @classmethod
    def from_settings(cls, settings: Settings) -> ExerciseAdvisor:
        return cls(
            consecutive_turns=settings.exercise_consecutive_turns,
            cooldown=timedelta(minutes=settings.exercise_cooldown_minutes),
            confidence_threshold=settings.emotion_confidence_threshold,
        )

    def observe(
        self, estimate: EmotionEstimate | None, transcript: str, now: datetime
    ) -> Suggestion | None:
        """Record one turn; return an offer when the rule fires."""
        if mentions_severe_distress(transcript):
            self.distress_seen = True
        if self.distress_seen:
            self._streak.clear()
            return None
        if (
            estimate is None
            or estimate.label not in TRIGGER_LABELS
            or estimate.confidence < self.confidence_threshold
        ):
            self._streak.clear()
            return None
        self._streak.append(estimate.label)
        if len(self._streak) < self.consecutive_turns:
            return None
        if self._last_offer_at is not None and now - self._last_offer_at < self.cooldown:
            return None
        label = self._streak[-1]
        exercise = self._choose(label)
        if exercise is None:
            return None
        self._streak.clear()
        self._last_offer_at = now
        self._last_exercise = exercise
        return Suggestion(exercise, label)

    def _choose(self, label: EmotionLabel) -> Exercise | None:
        suited = [e for e in self.catalogue if label in e.suited_for]
        if not suited:
            return None
        if self._last_exercise in suited and len(suited) > 1:
            start = suited.index(self._last_exercise) + 1
            return suited[start % len(suited)]
        return suited[0]


# --- the user's answer to an offer ---

Answer = Literal["accept", "decline"]
MAX_ANSWER_WORDS = 6
_DECLINE = re.compile(
    r"^(non|nan|nope|pas (maintenant|tout de suite|besoin|envie|la peine|aujourd'hui)|"
    r"plus tard|une autre fois|je (ne )?prefere(rais)? pas|sans facon|ca ira|ca va aller|"
    r"no\b|not now|later)"
)
_DECLINE_ANYWHERE = re.compile(r"\b(pas maintenant|plus tard|une autre fois|non merci|not now)\b")
_ACCEPT = re.compile(
    r"^(oui|ouais|ok|okay|d'accord|dac|volontiers|allons-y|allez|vas-y|go\b|pourquoi pas|"
    r"je veux bien|avec plaisir|carrement|bien sur|on y va|c'est parti|yes|sure)"
)


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower().replace("’", "'"))
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^\w' -]+", " ", plain).strip()


def classify_answer(text: str) -> Answer | None:
    """Short yes/no answer to an exercise offer; ``None`` if it is something else.

    Words suggesting severe distress are never taken as a mere answer ("non, je veux
    mourir" must reach the safety path, not a polite acknowledgement).
    """
    if mentions_severe_distress(text):
        return None
    folded = _fold(text)
    if not folded or len(folded.split()) > MAX_ANSWER_WORDS:
        return None
    if _DECLINE.match(folded) or _DECLINE_ANYWHERE.search(folded):
        return "decline"
    if _ACCEPT.match(folded):
        return "accept"
    return None
