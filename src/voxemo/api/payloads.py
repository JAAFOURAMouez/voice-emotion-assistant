"""JSON payloads sent to the browser (plain dicts, stable field names)."""

from __future__ import annotations

import dataclasses
import math
import random
from datetime import UTC, datetime, timedelta
from typing import Any

from voxemo import texts
from voxemo.display import cautious_phrase
from voxemo.emotion.types import EMOTION_LABELS, EmotionEstimate, EmotionLabel, ProsodyFeatures
from voxemo.tracking.store import Interaction
from voxemo.tracking.trends import daily_summary, weekly_trend
from voxemo.wellbeing.exercises import Exercise, Suggestion


def estimate_payload(
    estimate: EmotionEstimate, *, baseline_turns: int, min_turns: int
) -> dict[str, Any]:
    return {
        "label": estimate.label,
        "confidence": estimate.confidence,
        "scores": estimate.scores,
        "arousal": estimate.arousal,
        "text_terms": list(estimate.text_terms),
        "phrase": cautious_phrase(estimate, baseline_turns, min_turns),
        "learning": baseline_turns < min_turns,
        "baseline_turns": baseline_turns,
        "features": dataclasses.asdict(estimate.features),
    }


def exercise_payload(exercise: Exercise) -> dict[str, Any]:
    return {
        "id": exercise.id,
        "name": exercise.name,
        "summary": exercise.summary,
        "caution": exercise.caution,
        "cycles": exercise.cycles,
        "total_seconds": exercise.total_seconds,
        "duration_text": exercise.duration_text,
        "suited_for": sorted(exercise.suited_for),
        "steps": [
            {"instruction": s.instruction, "seconds": s.seconds, "kind": s.kind}
            for s in exercise.steps
        ],
        "introduction": exercise.introduction(),
    }


def offer_payload(suggestion: Suggestion) -> dict[str, Any]:
    return {
        "label": suggestion.label,
        "text": suggestion.exercise.offer(suggestion.label),
        "exercise": exercise_payload(suggestion.exercise),
    }


def interaction_payload(item: Interaction) -> dict[str, Any]:
    estimate = item.estimate
    return {
        "id": item.id,
        "session_id": item.session_id,
        "timestamp": item.timestamp.isoformat(),
        "transcript": item.transcript,
        "label": estimate.label if estimate else None,
        "confidence": estimate.confidence if estimate else None,
        "scores": estimate.scores if estimate else None,
        "arousal": estimate.arousal if estimate else None,
        "text_terms": list(estimate.text_terms) if estimate else [],
    }


def history_payload(
    interactions: list[Interaction], *, now: datetime | None = None, simulated: bool = False
) -> dict[str, Any]:
    return {
        "simulated": simulated,
        "notice": texts.SIMULATED_HISTORY if simulated else None,
        "interactions": [interaction_payload(i) for i in interactions],
        "days": [
            {
                "day": d.day.isoformat(),
                "count": d.interaction_count,
                "shares": d.shares,
                "mean_scores": d.mean_scores,
            }
            for d in daily_summary(interactions, now=now)
        ],
        "trend": [
            {
                "label": t.label,
                "recent": t.recent,
                "previous": t.previous,
                "direction": t.direction,
            }
            for t in weekly_trend(interactions, now=now)
        ],
    }


_DEMO_FEATURES = ProsodyFeatures(150.0, 20.0, 0.02, 4.0, 0.15, 0.03)
_DEMO_TEXTS: dict[EmotionLabel, tuple[str, ...]] = {
    "neutre": ("J'ai rendez-vous à quinze heures.", "Tu peux me rappeler la liste ?"),
    "joie": ("J'ai eu le poste !", "C'était une très belle journée."),
    "tristesse": ("Je n'ai pas le moral ce soir.", "Je me sens un peu seul."),
    "stress": ("J'ai trop de choses à rendre demain.", "Je suis très stressé par l'examen."),
}


def simulated_interactions(now: datetime | None = None, seed: int = 7) -> list[Interaction]:
    """A plausible week of estimates, to preview the strata when the history is empty.

    Always returned with ``simulated: true`` and a visible notice: never mixed with
    the user's own data, never stored.
    """
    rng = random.Random(seed)
    end = now or datetime.now(UTC)
    items: list[Interaction] = []
    weights = {"neutre": 5, "joie": 2, "tristesse": 1, "stress": 2}
    for day in range(7, 0, -1):
        for k in range(rng.randint(2, 5)):
            label: EmotionLabel = rng.choices(list(weights), list(weights.values()))[0]
            confidence = round(rng.uniform(0.35, 0.9), 2)
            rest = (1 - confidence) / 3
            scores = {k2: (confidence if k2 == label else rest) for k2 in EMOTION_LABELS}
            moment = end - timedelta(days=day - 1, hours=rng.uniform(1, 12) + k)
            estimate = EmotionEstimate(
                label=label,
                confidence=confidence,
                scores={k2: round(v, 3) for k2, v in scores.items()},
                arousal=round(0.5 + 0.3 * math.tanh(rng.gauss(0, 1)), 3),
                features=_DEMO_FEATURES,
            )
            text = rng.choice(_DEMO_TEXTS[label])
            items.append(Interaction(-len(items) - 1, -day, moment, text, "", estimate))
    return sorted(items, key=lambda i: i.timestamp)
