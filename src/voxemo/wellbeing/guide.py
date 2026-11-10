"""Guide an exercise in the terminal, step by step, with a countdown."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TextIO

from voxemo import texts
from voxemo.wellbeing.exercises import Exercise


def guide(
    exercise: Exercise,
    out: TextIO,
    *,
    sleep: Callable[[float], None] = time.sleep,
    interactive: bool = True,
    announce: Callable[[str], None] | None = None,
) -> bool:
    """Run the exercise; returns False if the user stopped it (Ctrl+C).

    With ``interactive`` the remaining seconds are rewritten on the same line;
    otherwise each step is printed once (logs, tests). ``announce`` (e.g. speech)
    receives each instruction when its step starts, without delaying the timer.
    """
    say = announce or (lambda _text: None)
    try:
        for cycle in range(1, exercise.cycles + 1):
            if exercise.cycles > 1:
                print(texts.EXERCISE_CYCLE.format(n=cycle, total=exercise.cycles), file=out)
            for step in exercise.steps:
                label = texts.EXERCISE_STEP.format(instruction=step.instruction)
                say(step.instruction)
                seconds = round(step.seconds)
                if interactive:
                    for remaining in range(seconds, 0, -1):
                        print(f"\r{label} {remaining:>2} ", end="", file=out, flush=True)
                        sleep(1)
                    print(f"\r{label}    ", file=out, flush=True)
                else:
                    print(f"{label} ({seconds} s)", file=out, flush=True)
                    sleep(step.seconds)
    except KeyboardInterrupt:
        print(file=out)
        print(texts.EXERCISE_STOPPED, file=out, flush=True)
        say(texts.EXERCISE_STOPPED)
        return False
    print(texts.EXERCISE_DONE, file=out, flush=True)
    say(texts.EXERCISE_DONE)
    return True
