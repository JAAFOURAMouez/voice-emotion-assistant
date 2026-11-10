"""List or run an exercise directly: ``python -m voxemo.wellbeing [exercise_id]``."""

from __future__ import annotations

import argparse
import sys

from voxemo import texts
from voxemo.config import get_settings
from voxemo.tts.speaker import create_speech
from voxemo.wellbeing.exercises import CATALOGUE, get_exercise
from voxemo.wellbeing.guide import guide


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m voxemo.wellbeing", description=__doc__)
    parser.add_argument("exercise", nargs="?", choices=[e.id for e in CATALOGUE])
    parser.add_argument("--no-tts", action="store_true", help="ne pas lire les consignes")
    args = parser.parse_args(argv)
    if args.exercise is None:
        print(texts.EXERCISE_LIST_HEADER)
        for exercise in CATALOGUE:
            print(
                texts.EXERCISE_LIST_ITEM.format(
                    id=exercise.id, name=exercise.name, duration=exercise.duration_text
                )
            )
        return 0
    exercise = get_exercise(args.exercise)
    speech = create_speech(get_settings(), enabled=not args.no_tts)
    print(exercise.introduction())
    try:
        if speech is not None:
            speech.say(exercise.introduction())
            speech.wait()
        guide(
            exercise,
            sys.stdout,
            interactive=sys.stdout.isatty(),
            announce=speech.say if speech is not None else None,
        )
        if speech is not None:
            speech.wait()
            if speech.error:
                print(speech.error, file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        if speech is not None:
            speech.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
