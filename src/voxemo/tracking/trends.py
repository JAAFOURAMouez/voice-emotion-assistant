"""Aggregates over the history: per-day summary, per-session distribution, 7-day trend.

Pure functions over :class:`Interaction` records, so the interface (Streamlit) and the
command line share them. They describe *estimated* emotions, never diagnoses.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, tzinfo
from typing import Literal

from voxemo.emotion.types import EMOTION_LABELS, EmotionLabel
from voxemo.tracking.store import Interaction

TrendDirection = Literal["hausse", "baisse", "stable", "inconnu"]


def _estimated(interactions: Sequence[Interaction]) -> list[Interaction]:
    return [i for i in interactions if i.estimate is not None]


def label_shares(interactions: Sequence[Interaction]) -> dict[EmotionLabel, float]:
    """Share of each estimated label (0 for all when there is no estimate)."""
    estimated = _estimated(interactions)
    counts = Counter(i.estimate.label for i in estimated if i.estimate)
    total = len(estimated)
    return {label: (counts[label] / total if total else 0.0) for label in EMOTION_LABELS}


def mean_scores(interactions: Sequence[Interaction]) -> dict[EmotionLabel, float]:
    """Average score of each label (finer than shares: includes sub-threshold cues)."""
    estimated = _estimated(interactions)
    if not estimated:
        return dict.fromkeys(EMOTION_LABELS, 0.0)
    return {
        label: sum(i.estimate.scores.get(label, 0.0) for i in estimated if i.estimate)
        / len(estimated)
        for label in EMOTION_LABELS
    }


@dataclass(frozen=True)
class DaySummary:
    day: date
    interaction_count: int
    shares: dict[EmotionLabel, float]
    mean_scores: dict[EmotionLabel, float]


def daily_summary(
    interactions: Sequence[Interaction],
    *,
    days: int = 7,
    now: datetime | None = None,
    tz: tzinfo | None = None,
) -> list[DaySummary]:
    """One entry per calendar day (local time), oldest first, empty days included."""
    current = (now or datetime.now().astimezone()).astimezone(tz)
    today = current.date()
    by_day: dict[date, list[Interaction]] = defaultdict(list)
    for item in interactions:
        by_day[item.timestamp.astimezone(tz).date()].append(item)
    summaries = []
    for offset in range(days - 1, -1, -1):
        day = today - timedelta(days=offset)
        items = by_day.get(day, [])
        summaries.append(DaySummary(day, len(items), label_shares(items), mean_scores(items)))
    return summaries


def session_shares(interactions: Sequence[Interaction]) -> dict[int, dict[EmotionLabel, float]]:
    """Distribution of estimated labels for each session."""
    by_session: dict[int, list[Interaction]] = defaultdict(list)
    for item in interactions:
        by_session[item.session_id].append(item)
    return {sid: label_shares(items) for sid, items in sorted(by_session.items())}


@dataclass(frozen=True)
class LabelTrend:
    label: EmotionLabel
    recent: float
    """Mean score over the last ``days`` days."""
    previous: float
    """Mean score over the ``days`` days before."""
    direction: TrendDirection


def weekly_trend(
    interactions: Sequence[Interaction],
    *,
    now: datetime | None = None,
    days: int = 7,
    min_interactions: int = 3,
    tolerance: float = 0.05,
) -> list[LabelTrend]:
    """Compare the last ``days`` days with the ``days`` before, label by label.

    Mean scores are used rather than label counts: they are less jumpy with few
    interactions. ``inconnu`` when either period has too little data.
    """
    end = now or datetime.now().astimezone()
    recent_start, previous_start = end - timedelta(days=days), end - timedelta(days=2 * days)
    recent = [i for i in interactions if recent_start < i.timestamp <= end]
    previous = [i for i in interactions if previous_start < i.timestamp <= recent_start]
    enough = (
        len(_estimated(recent)) >= min_interactions
        and len(_estimated(previous)) >= min_interactions
    )
    recent_scores, previous_scores = mean_scores(recent), mean_scores(previous)
    trends = []
    for label in EMOTION_LABELS:
        delta = recent_scores[label] - previous_scores[label]
        direction: TrendDirection
        if not enough:
            direction = "inconnu"
        elif delta > tolerance:
            direction = "hausse"
        elif delta < -tolerance:
            direction = "baisse"
        else:
            direction = "stable"
        trends.append(LabelTrend(label, recent_scores[label], previous_scores[label], direction))
    return trends
