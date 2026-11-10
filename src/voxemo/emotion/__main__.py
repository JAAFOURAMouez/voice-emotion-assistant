"""Estimate vocal emotion cues turn by turn: ``python -m voxemo.emotion [--file audio.wav]``.

Each turn is compared with the previous ones (rolling baseline), as in the assistant.
Useful to tune the fusion rules. Nothing is saved to disk.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterable
from typing import TextIO

from voxemo import texts
from voxemo.audio.recorder import ArraySource, AudioDeviceError, MicrophoneSource
from voxemo.audio.vad import SegmenterConfig, SileroSpeechDetector, SpeechTurn, iter_turns
from voxemo.config import Settings, get_settings
from voxemo.emotion.classifier import Wav2Vec2EmotionClassifier
from voxemo.emotion.fusion import EmotionAnalyzer, FusionConfig, VoiceBaseline
from voxemo.emotion.types import EmotionEstimate


def report_turns(
    turns: Iterable[SpeechTurn], analyzer: EmotionAnalyzer, settings: Settings, out: TextIO
) -> list[EmotionEstimate]:
    history: list[EmotionEstimate] = []
    for turn in turns:
        started = time.perf_counter()
        baseline = VoiceBaseline.from_estimates(history, settings.baseline_window)
        estimate = analyzer.analyze(turn.audio, turn.sample_rate, baseline)
        f = estimate.features
        print(
            texts.EMOTION_RESULT.format(
                start=turn.start_s,
                label=estimate.label,
                confidence=estimate.confidence,
                arousal=estimate.arousal,
                scores=" ".join(f"{k}={v:.2f}" for k, v in estimate.scores.items()),
                pitch=f.pitch_mean_hz,
                pitch_std=f.pitch_std_hz,
                rate=f.speech_rate,
                pauses=f.pause_ratio,
                elapsed=time.perf_counter() - started,
            ),
            file=out,
            flush=True,
        )
        history.append(estimate)
    return history


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m voxemo.emotion", description=__doc__)
    parser.add_argument("--file", help="fichier audio à analyser au lieu du micro")
    args = parser.parse_args(argv)

    settings = get_settings()
    analyzer = EmotionAnalyzer(
        Wav2Vec2EmotionClassifier.from_settings(settings), FusionConfig.from_settings(settings)
    )
    if args.file:
        source = ArraySource.from_file(args.file, settings.sample_rate)
    else:
        source = MicrophoneSource(settings.sample_rate, device=settings.audio_input_device or None)
        print(texts.STT_LISTENING, flush=True)
    turns = iter_turns(
        source, SileroSpeechDetector(settings.sample_rate), SegmenterConfig.from_settings(settings)
    )
    try:
        report_turns(turns, analyzer, settings, sys.stdout)
    except AudioDeviceError as exc:
        print(exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(texts.CLI_GOODBYE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
