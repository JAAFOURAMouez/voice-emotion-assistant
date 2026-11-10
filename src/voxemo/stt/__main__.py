"""Listen and transcribe, turn by turn: ``python -m voxemo.stt [--file audio.wav]``.

Without ``--file``, the microphone is used until Ctrl+C. Nothing is saved to disk.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import TextIO

from voxemo import texts
from voxemo.audio.recorder import ArraySource, AudioDeviceError, MicrophoneSource
from voxemo.audio.vad import (
    SegmenterConfig,
    SileroSpeechDetector,
    SpeechDetector,
    SpeechTurn,
    iter_turns,
    listen_for_turn,
)
from voxemo.config import Settings, get_settings
from voxemo.stt.transcriber import Transcriber, WhisperTranscriber


def print_transcription(turn: SpeechTurn, transcriber: Transcriber, out: TextIO) -> None:
    started = time.perf_counter()
    result = transcriber.transcribe(turn.audio, turn.sample_rate)
    elapsed = time.perf_counter() - started
    if result.is_empty:
        print(texts.STT_NOTHING_UNDERSTOOD, file=out)
        return
    print(
        texts.STT_RESULT.format(
            start=turn.start_s,
            duration=turn.duration_s,
            language=result.language,
            elapsed=elapsed,
            text=result.text,
        ),
        file=out,
        flush=True,
    )


def transcribe_file(
    path: str,
    settings: Settings,
    detector: SpeechDetector,
    transcriber: Transcriber,
    out: TextIO,
) -> None:
    source = ArraySource.from_file(path, settings.sample_rate)
    for turn in iter_turns(source, detector, SegmenterConfig.from_settings(settings)):
        print_transcription(turn, transcriber, out)


def transcribe_microphone(
    settings: Settings, detector: SpeechDetector, transcriber: Transcriber, out: TextIO
) -> None:
    source = MicrophoneSource(settings.sample_rate, device=settings.audio_input_device or None)
    config = SegmenterConfig.from_settings(settings)
    print(texts.STT_LISTENING, file=out, flush=True)
    while True:
        turn = listen_for_turn(source, detector, config, settings.listen_timeout_s)
        if turn is None:
            print(texts.STT_NO_SPEECH, file=out, flush=True)
            continue
        print_transcription(turn, transcriber, out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m voxemo.stt", description=__doc__)
    parser.add_argument("--file", help="fichier audio à transcrire au lieu du micro")
    args = parser.parse_args(argv)

    settings = get_settings()
    detector = SileroSpeechDetector(settings.sample_rate)
    transcriber = WhisperTranscriber.from_settings(settings)
    try:
        if args.file:
            transcribe_file(args.file, settings, detector, transcriber, sys.stdout)
        else:
            transcribe_microphone(settings, detector, transcriber, sys.stdout)
    except AudioDeviceError as exc:
        print(exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(texts.CLI_GOODBYE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
