"""Show or erase the history: ``python -m voxemo.tracking [--clear [--yes]]``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import datetime
from typing import TextIO

from voxemo import texts
from voxemo.config import get_settings
from voxemo.tracking.store import InteractionStore
from voxemo.tracking.trends import daily_summary, label_shares, weekly_trend


def _percent(shares: dict[str, float]) -> str:
    return ", ".join(f"{label} {share:.0%}" for label, share in shares.items())


def print_report(store: InteractionStore, out: TextIO, now: datetime | None = None) -> None:
    interactions = store.interactions()
    if not interactions:
        print(texts.TRACKING_EMPTY.format(path=store.path), file=out)
        return
    print(
        texts.TRACKING_SUMMARY.format(
            interactions=len(interactions), sessions=len(store.sessions()), path=store.path
        ),
        file=out,
    )
    print(texts.TRACKING_DISCLAIMER, file=out)
    print(texts.TRACKING_SHARES.format(shares=_percent(label_shares(interactions))), file=out)

    print(texts.TRACKING_DAYS_HEADER, file=out)
    for day in daily_summary(interactions, now=now):
        detail = ""
        if day.interaction_count:
            marked = {k: v for k, v in day.shares.items() if v > 0}
            detail = f" — {_percent(marked)}"
        print(
            texts.TRACKING_DAY.format(
                weekday=texts.WEEKDAYS[day.day.weekday()],
                day=day.day,
                count=day.interaction_count,
                detail=detail,
            ),
            file=out,
        )

    trends = weekly_trend(interactions, now=now)
    if all(t.direction == "inconnu" for t in trends):
        print(texts.TRACKING_TREND_UNKNOWN, file=out)
        return
    print(texts.TRACKING_TREND_HEADER, file=out)
    for trend in trends:
        print(
            texts.TRACKING_TREND.format(
                label=trend.label,
                recent=f"{trend.recent:.2f}".replace(".", ","),
                previous=f"{trend.previous:.2f}".replace(".", ","),
                direction=trend.direction,
            ),
            file=out,
        )


def clear(store: InteractionStore, out: TextIO, *, confirm: Callable[[str], str] | None) -> int:
    """Erase the history after confirmation (``confirm=None`` skips it); returns the count."""
    count = len(store.interactions())
    if confirm is not None:
        try:
            answer = confirm(texts.TRACKING_CONFIRM_CLEAR.format(count=count))
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer.strip().lower() not in {"oui", "o", "yes", "y"}:
            print(texts.TRACKING_CLEAR_CANCELLED, file=out)
            return 0
    deleted = store.clear_history()
    print(texts.TRACKING_CLEARED.format(count=deleted), file=out)
    return deleted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m voxemo.tracking", description=__doc__)
    parser.add_argument("--clear", action="store_true", help="effacer tout l'historique")
    parser.add_argument("--yes", action="store_true", help="ne pas demander de confirmation")
    args = parser.parse_args(argv)
    with InteractionStore(get_settings().db_path) as store:
        if args.clear:
            clear(store, sys.stdout, confirm=None if args.yes else input)
        else:
            print_report(store, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
