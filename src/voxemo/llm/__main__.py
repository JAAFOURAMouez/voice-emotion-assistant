"""Text-only chat loop, no microphone needed: ``python -m voxemo.llm``.

A vocal cue can be simulated with ``--emotion stress --confidence 0.7`` or the
``/emotion`` command, to check how the tone adapts.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TextIO, cast

from voxemo import texts
from voxemo.config import Settings, get_settings
from voxemo.emotion.types import EMOTION_LABELS, EmotionEstimate, EmotionLabel, ProsodyFeatures
from voxemo.llm.client import ChatModel, Conversation, LLMUnavailableError, OllamaChatModel

_NO_FEATURES = ProsodyFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def simulated_estimate(label: str, confidence: float) -> EmotionEstimate | None:
    """Build a fake estimate for manual testing; ``None`` for a neutral cue."""
    if label not in EMOTION_LABELS:
        raise ValueError(label)
    if label == "neutre":
        return None
    return EmotionEstimate(
        label=cast(EmotionLabel, label),
        confidence=confidence,
        scores={label: confidence},
        arousal=0.5,
        features=_NO_FEATURES,
    )


@dataclass
class ChatSession:
    conversation: Conversation
    estimate: EmotionEstimate | None = None

    def handle_command(self, line: str, out: TextIO) -> bool:
        """Process a ``/command``; return False when the user wants to quit."""
        command, *args = line.split()
        if command in {"/quit", "/exit", "/q"}:
            return False
        if command == "/reset":
            self.conversation.reset()
            print(texts.CLI_RESET, file=out)
        elif command == "/emotion":
            try:
                confidence = float(args[1].replace(",", ".")) if len(args) > 1 else 0.7
                self.estimate = simulated_estimate(args[0], confidence)
            except (IndexError, ValueError):
                print(texts.CLI_EMOTION_USAGE, file=out)
                return True
            if self.estimate is None:
                print(texts.CLI_EMOTION_CLEARED, file=out)
            else:
                shown = f"{confidence:.1f}".replace(".", ",")
                print(texts.CLI_EMOTION_SET.format(label=args[0], confidence=shown), file=out)
        else:
            print(texts.CLI_UNKNOWN_COMMAND.format(command=command), file=out)
        return True

    def run(self, read_line: Callable[[str], str], out: TextIO) -> None:
        print(texts.CLI_WELCOME, file=out)
        while True:
            try:
                line = read_line(texts.CLI_PROMPT).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line.startswith("/"):
                if not self.handle_command(line, out):
                    break
                continue
            print(texts.CLI_ASSISTANT, end="", file=out, flush=True)
            try:
                for chunk in self.conversation.reply_stream(line, self.estimate):
                    print(chunk, end="", file=out, flush=True)
            except LLMUnavailableError as exc:
                print(f"\n{exc}", file=out)
            print(file=out)
        print(texts.CLI_GOODBYE, file=out)


def main(
    argv: list[str] | None = None,
    *,
    settings: Settings | None = None,
    model: ChatModel | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="python -m voxemo.llm", description=__doc__)
    parser.add_argument("--emotion", choices=EMOTION_LABELS, default="neutre")
    parser.add_argument("--confidence", type=float, default=0.7)
    args = parser.parse_args(argv)
    if not 0.0 <= args.confidence <= 1.0:
        parser.error(texts.CLI_EMOTION_USAGE)

    settings = settings or get_settings()
    model = model or OllamaChatModel.from_settings(settings)
    session = ChatSession(
        Conversation(model, settings), simulated_estimate(args.emotion, args.confidence)
    )
    session.run(input, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
