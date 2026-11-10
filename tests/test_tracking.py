"""Tests for the SQLite history, the trends and the history command line."""

from __future__ import annotations

import dataclasses
import io
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from conftest import make_estimate
from voxemo.emotion.types import EmotionEstimate, EmotionLabel
from voxemo.tracking.__main__ import clear, print_report
from voxemo.tracking.store import SCHEMA_VERSION, Interaction, InteractionStore
from voxemo.tracking.trends import (
    daily_summary,
    label_shares,
    mean_scores,
    session_shares,
    weekly_trend,
)

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path) -> InteractionStore:
    with InteractionStore(tmp_path / "data" / "history.db") as s:
        yield s


def _rich_estimate() -> EmotionEstimate:
    return dataclasses.replace(
        make_estimate("stress", 0.62),
        scores={"neutre": 0.2, "joie": 0.1, "tristesse": 0.08, "stress": 0.62},
        model_scores={"neutre": 0.3, "joie": 0.2, "colere": 0.4, "tristesse": 0.1},
        text_terms=("énervé",),
    )


# --- store ---


def test_round_trip_keeps_every_field(store: InteractionStore) -> None:
    session = store.start_session(NOW)
    estimate = _rich_estimate()
    store.add_interaction(
        session,
        transcript="Je suis énervé",
        reply="Je vous écoute.",
        estimate=estimate,
        timestamp=NOW,
    )
    [item] = store.interactions()
    assert item.session_id == session
    assert item.timestamp == NOW
    assert item.transcript == "Je suis énervé"
    assert item.reply == "Je vous écoute."
    assert item.estimate == estimate
    assert item.audio_path is None


def test_turn_without_estimate_is_stored(store: InteractionStore) -> None:
    session = store.start_session()
    store.add_interaction(session, transcript="Bonjour", reply="", estimate=None)
    assert store.interactions()[0].estimate is None
    assert store.recent_estimates(10) == []


def test_database_file_and_schema_are_created(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "voxemo.db"
    with InteractionStore(path) as s:
        s.start_session()
    assert path.exists()
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        columns = {row[1] for row in db.execute("PRAGMA table_info(interactions)")}
    expected = {"id", "session_id", "timestamp", "transcript", "reply", "label", "confidence"}
    assert expected | {"scores_json", "features_json"} <= columns
    assert "audio" not in columns  # never the raw audio itself, at most a path


def test_reopening_keeps_history(tmp_path: Path) -> None:
    path = tmp_path / "h.db"
    with InteractionStore(path) as s:
        s.add_interaction(s.start_session(), transcript="a", reply="b", estimate=make_estimate())
    with InteractionStore(path) as s:
        assert len(s.interactions()) == 1


def test_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "h.db"
    with sqlite3.connect(path) as db:
        db.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(RuntimeError, match="newer"):
        InteractionStore(path).sessions()


def test_naive_timestamps_are_rejected(store: InteractionStore) -> None:
    with pytest.raises(ValueError, match="timezone"):
        store.start_session(datetime(2026, 1, 1))


def test_filters_and_chronological_order(store: InteractionStore) -> None:
    first, second = store.start_session(NOW), store.start_session(NOW)
    for hours, session in [(3, first), (1, second), (2, first)]:
        store.add_interaction(
            session,
            transcript=f"{hours}",
            reply="",
            estimate=make_estimate(),
            timestamp=NOW - timedelta(hours=hours),
        )
    assert [i.transcript for i in store.interactions()] == ["3", "2", "1"]
    assert [i.transcript for i in store.interactions(session_id=first)] == ["3", "2"]
    assert [i.transcript for i in store.interactions(since=NOW - timedelta(hours=2))] == ["2", "1"]
    assert [i.transcript for i in store.interactions(limit=2)] == ["2", "1"]


def test_recent_estimates_seed_the_baseline(store: InteractionStore) -> None:
    session = store.start_session()
    labels: list[EmotionLabel] = ["neutre", "joie", "stress"]
    for k, label in enumerate(labels):
        store.add_interaction(
            session,
            transcript=label,
            reply="",
            estimate=make_estimate(label),
            timestamp=NOW + timedelta(minutes=k),
        )
    assert [e.label for e in store.recent_estimates(2)] == ["joie", "stress"]


def test_sessions_count_interactions(store: InteractionStore) -> None:
    session = store.start_session(NOW)
    store.start_session(NOW + timedelta(hours=1))
    store.add_interaction(session, transcript="a", reply="", estimate=None)
    store.end_session(session, NOW + timedelta(minutes=5))
    infos = store.sessions()
    assert [(s.id, s.interaction_count) for s in infos] == [(session, 1), (session + 1, 0)]
    assert infos[0].ended_at == NOW + timedelta(minutes=5)
    assert infos[1].ended_at is None


def test_clear_history_erases_everything_including_audio(
    store: InteractionStore, tmp_path: Path
) -> None:
    audio = tmp_path / "turn.wav"
    audio.write_bytes(b"RIFF")
    session = store.start_session()
    store.add_interaction(
        session, transcript="secret", reply="", estimate=make_estimate(), audio_path=audio
    )
    store.add_interaction(session, transcript="autre", reply="", estimate=None)
    assert store.clear_history() == 2
    assert store.interactions() == []
    assert store.sessions() == []
    assert not audio.exists()
    store.close()
    assert b"secret" not in Path(store.path).read_bytes()  # secure_delete


def test_store_is_usable_from_several_threads(store: InteractionStore) -> None:
    session = store.start_session()

    def write(k: int) -> None:
        store.add_interaction(session, transcript=str(k), reply="", estimate=make_estimate())

    threads = [threading.Thread(target=write, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store.interactions()) == 8


# --- trends ---


def _item(label: EmotionLabel | None, when: datetime, session: int = 1, score: float = 0.7):
    estimate = None
    if label is not None:
        scores = dict.fromkeys(("neutre", "joie", "tristesse", "stress"), (1 - score) / 3)
        scores[label] = score
        estimate = dataclasses.replace(make_estimate(label, score), scores=scores)
    return Interaction(0, session, when, "t", "r", estimate)


def test_label_shares_and_mean_scores() -> None:
    items = [_item("stress", NOW), _item("stress", NOW), _item("joie", NOW), _item(None, NOW)]
    shares = label_shares(items)
    assert shares["stress"] == pytest.approx(2 / 3)
    assert shares["joie"] == pytest.approx(1 / 3)
    assert shares["neutre"] == 0.0
    assert mean_scores(items)["stress"] == pytest.approx((0.7 + 0.7 + 0.1) / 3)


def test_empty_history_gives_zeros() -> None:
    assert set(label_shares([]).values()) == {0.0}
    assert set(mean_scores([]).values()) == {0.0}


def test_daily_summary_covers_every_day() -> None:
    items = [_item("joie", NOW - timedelta(days=1)), _item("stress", NOW - timedelta(days=1))]
    days = daily_summary(items, now=NOW, tz=UTC)
    assert len(days) == 7
    assert days[-1].day == NOW.date()
    assert days[-2].interaction_count == 2
    assert days[-2].shares["joie"] == 0.5
    assert days[0].interaction_count == 0


def test_daily_summary_uses_local_days() -> None:
    from datetime import timezone

    paris = timezone(timedelta(hours=1))
    late = datetime(2026, 11, 9, 23, 30, tzinfo=UTC)  # already the 10th in Paris
    days = daily_summary([_item("joie", late)], now=NOW, tz=paris)
    assert days[-1].interaction_count == 1


def test_session_shares() -> None:
    items = [_item("joie", NOW, 1), _item("stress", NOW, 2), _item("stress", NOW, 2)]
    result = session_shares(items)
    assert result[1]["joie"] == 1.0
    assert result[2]["stress"] == 1.0


def test_weekly_trend_detects_rising_stress() -> None:
    previous = [_item("neutre", NOW - timedelta(days=10, hours=k)) for k in range(4)]
    recent = [_item("stress", NOW - timedelta(days=2, hours=k)) for k in range(4)]
    trends = {t.label: t for t in weekly_trend(previous + recent, now=NOW)}
    assert trends["stress"].direction == "hausse"
    assert trends["neutre"].direction == "baisse"
    assert trends["stress"].recent == pytest.approx(0.7)


def test_weekly_trend_stable_and_unknown() -> None:
    same = [_item("neutre", NOW - timedelta(days=d)) for d in (1, 2, 3, 8, 9, 10)]
    assert {t.direction for t in weekly_trend(same, now=NOW)} == {"stable"}
    too_few = [_item("stress", NOW - timedelta(days=1))]
    assert {t.direction for t in weekly_trend(too_few, now=NOW)} == {"inconnu"}


def test_old_history_is_outside_the_trend_windows() -> None:
    old = [_item("stress", NOW - timedelta(days=30)) for _ in range(5)]
    assert {t.direction for t in weekly_trend(old, now=NOW)} == {"inconnu"}


# --- command line ---


def test_report_on_empty_history(store: InteractionStore) -> None:
    out = io.StringIO()
    print_report(store, out, now=NOW)
    assert "Aucun historique" in out.getvalue()


def test_report_shows_shares_days_and_cautious_wording(store: InteractionStore) -> None:
    session = store.start_session(NOW)
    for k in range(3):
        store.add_interaction(
            session,
            transcript="x",
            reply="",
            estimate=make_estimate("stress"),
            timestamp=NOW - timedelta(hours=k),
        )
    out = io.StringIO()
    print_report(store, out, now=NOW)
    text = out.getvalue()
    assert "3 échanges enregistrés sur 1 session(s)" in text
    assert "émotions estimées" in text and "pas un diagnostic" in text
    assert "stress 100%" in text
    assert "mar. 10/11 : 3 échange(s)" in text
    assert "pas encore assez d'échanges" in text


def test_clear_requires_confirmation(store: InteractionStore) -> None:
    store.add_interaction(store.start_session(), transcript="x", reply="", estimate=None)
    out = io.StringIO()
    assert clear(store, out, confirm=lambda _: "non") == 0
    assert len(store.interactions()) == 1
    assert clear(store, out, confirm=lambda _: "oui") == 1
    assert store.interactions() == []
    assert "Historique effacé (1 échanges supprimés)" in out.getvalue()


def test_clear_without_confirmation(store: InteractionStore) -> None:
    store.add_interaction(store.start_session(), transcript="x", reply="", estimate=None)
    assert clear(store, io.StringIO(), confirm=None) == 1
