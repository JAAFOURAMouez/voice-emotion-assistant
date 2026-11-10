"""Small formatting helpers shared by the command line and the web interface."""

from __future__ import annotations

from voxemo import texts
from voxemo.emotion.types import EmotionEstimate


def decimal(value: float, digits: int = 1) -> str:
    """French decimal comma: 0.72 → "0,72"."""
    return f"{value:.{digits}f}".replace(".", ",")


def cue_detail(estimate: EmotionEstimate) -> str:
    if estimate.label == "neutre":
        others = {k: v for k, v in estimate.scores.items() if k != "neutre"}
        if not others:
            return ""
        label = max(others, key=others.__getitem__)
        detail = texts.PIPELINE_CUE_NOTHING_MARKED.format(
            label=label, score=decimal(others[label], 2)
        )
    else:
        detail = texts.PIPELINE_CUE_CONFIDENCE.format(confidence=decimal(estimate.confidence, 2))
    if estimate.text_terms:
        detail += texts.PIPELINE_CUE_WORDS.format(terms=", ".join(estimate.text_terms))
    return detail


def cautious_phrase(estimate: EmotionEstimate, baseline_turns: int, min_turns: int) -> str:
    """A short, hedged description of the estimate ("Voix plus tendue que d'habitude")."""
    phrase = texts.EMOTION_PHRASES[estimate.label]
    extras = []
    if estimate.text_terms and estimate.label != "neutre":
        extras.append(texts.EMOTION_PHRASE_WORDS)
    if baseline_turns < min_turns:
        extras.append(texts.EMOTION_PHRASE_LEARNING)
    return phrase + (" · " + " · ".join(extras) if extras else "")
