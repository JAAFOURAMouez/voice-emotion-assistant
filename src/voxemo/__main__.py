"""Voice assistant on the command line: ``python -m voxemo [--file audio.wav]``.

Each turn: speech → transcription and emotion estimate (in parallel) → reply adapted
to the vocal cues, streamed. Nothing is saved to disk yet.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Iterator
from typing import TextIO

from voxemo import texts
from voxemo.audio.recorder import ArraySource, AudioDeviceError, AudioSource, MicrophoneSource
from voxemo.audio.vad import (
    SegmenterConfig,
    SileroSpeechDetector,
    SpeechDetector,
    SpeechTurn,
    iter_turns,
    listen_for_turn,
)
from voxemo.config import Settings, get_settings
from voxemo.display import cue_detail, decimal
from voxemo.offline import prefer_local_models
from voxemo.pipeline import HistoryRecorder, TurnAnalysis, TurnResult, VoicePipeline
from voxemo.tracking.store import InteractionStore
from voxemo.tts.speaker import SpeechQueue, create_speech
from voxemo.wellbeing.guide import guide


def print_analysis(
    analysis: TurnAnalysis, out: TextIO, *, min_turns: int = 5, details: bool = False
) -> None:
    print(texts.PIPELINE_USER.format(text=analysis.transcription.text), file=out)
    estimate = analysis.estimate
    if estimate is None:
        print(texts.PIPELINE_CUES_UNAVAILABLE.format(stt=decimal(analysis.stt_s)), file=out)
    else:
        calibration = (
            texts.PIPELINE_CUE_CALIBRATION.format(n=analysis.baseline_turns, total=min_turns)
            if analysis.baseline_turns < min_turns
            else ""
        )
        print(
            texts.PIPELINE_CUES.format(
                label=estimate.label,
                detail=cue_detail(estimate),
                calibration=calibration,
                stt=decimal(analysis.stt_s),
                emotion=decimal(analysis.emotion_s),
            ),
            file=out,
        )
        if details:
            f = estimate.features
            print(
                texts.PIPELINE_DETAILS.format(
                    scores=" ".join(f"{k}={decimal(v, 2)}" for k, v in estimate.scores.items()),
                    model=" ".join(
                        f"{k}={decimal(v, 2)}" for k, v in estimate.model_scores.items()
                    ),
                    arousal=decimal(estimate.arousal, 2),
                    pitch=f.pitch_mean_hz,
                    pitch_std=f.pitch_std_hz,
                    rate=decimal(f.speech_rate),
                    pauses=f.pause_ratio,
                ),
                file=out,
            )
    print(texts.CLI_ASSISTANT, end="", file=out, flush=True)


def run_turn(
    pipeline: VoicePipeline,
    turn: SpeechTurn,
    out: TextIO,
    *,
    details: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    speech: SpeechQueue | None = None,
) -> TurnResult | None:
    min_turns = pipeline.fusion_config.baseline_min_turns

    def on_chunk(chunk: str) -> None:
        print(chunk, end="", file=out, flush=True)
        if speech is not None:
            speech.feed(chunk)

    try:
        result = pipeline.process(
            turn,
            on_analysis=lambda a: print_analysis(a, out, min_turns=min_turns, details=details),
            on_chunk=on_chunk,
        )
        if result is None:
            return None
        if speech is not None:
            speech.flush()
        _finish_turn(result, out, details=details, sleep=sleep, speech=speech)
    except KeyboardInterrupt:
        if speech is not None:
            speech.cancel()
        raise
    return result


def _finish_turn(
    result: TurnResult,
    out: TextIO,
    *,
    details: bool,
    sleep: Callable[[float], None],
    speech: SpeechQueue | None,
) -> None:
    if result.error:
        print(f"\n{result.error}", file=out)
    else:
        print(file=out)
        print(
            texts.PIPELINE_LATENCY.format(
                first=decimal(result.first_chunk_s or 0.0), total=decimal(result.total_s)
            ),
            file=out,
        )
    if details and result.audio_path:
        print(texts.PIPELINE_AUDIO_SAVED.format(path=result.audio_path), file=out)
    if result.exercise is not None:
        # The microphone is closed meanwhile: listening resumes after the exercise.
        if speech is not None:
            speech.wait()  # let the introduction be heard before the timer starts
        guide(
            result.exercise,
            out,
            sleep=sleep,
            interactive=out.isatty(),
            announce=speech.say if speech is not None else None,
        )
    if speech is not None:
        had_error = speech.error is not None
        speech.wait()  # never listen while the assistant is still speaking (echo)
        if speech.error and not had_error:
            print(speech.error, file=out)
    print(file=out, flush=True)


def microphone_turns(
    source: AudioSource, detector: SpeechDetector, settings: Settings
) -> Iterator[SpeechTurn]:
    """Listen turn after turn; the microphone is closed while the assistant answers."""
    config = SegmenterConfig.from_settings(settings)
    while True:
        turn = listen_for_turn(source, detector, config, settings.listen_timeout_s)
        if turn is not None:
            yield turn


def run(
    pipeline: VoicePipeline,
    source: AudioSource,
    detector: SpeechDetector,
    settings: Settings,
    out: TextIO,
    *,
    live: bool,
    warm_up: bool = True,
    details: bool = False,
    recorder: HistoryRecorder | None = None,
    sleep: Callable[[float], None] = time.sleep,
    speech: SpeechQueue | None = None,
) -> list[TurnResult]:
    if pipeline.audio_dir is not None:
        print(texts.PIPELINE_STORING_AUDIO.format(path=pipeline.audio_dir), file=out)
    if warm_up:
        print(texts.PIPELINE_LOADING, file=out, flush=True)
        started = time.perf_counter()
        for warning in pipeline.warm_up(settings.sample_rate):
            print(texts.PIPELINE_WARMUP_WARNING.format(detail=warning), file=out)
        if speech is not None:
            speech.warm_up()
            if speech.error:
                print(texts.PIPELINE_WARMUP_WARNING.format(detail=speech.error), file=out)
        print(texts.PIPELINE_READY.format(elapsed=time.perf_counter() - started), file=out)
    if live:
        print(texts.STT_LISTENING, file=out, flush=True)
        turns = microphone_turns(source, detector, settings)
    else:
        turns = iter_turns(source, detector, SegmenterConfig.from_settings(settings))
    results: list[TurnResult] = []
    for turn in turns:
        if result := run_turn(pipeline, turn, out, details=details, sleep=sleep, speech=speech):
            results.append(result)
            if recorder is not None:
                recorder.record(result)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m voxemo", description=__doc__)
    parser.add_argument("--file", help="fichier audio à utiliser au lieu du micro")
    parser.add_argument(
        "--no-warmup", action="store_true", help="ne pas précharger les modèles au démarrage"
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="afficher les scores, les sorties du modèle et les mesures de la voix",
    )
    parser.add_argument(
        "--no-history", action="store_true", help="ne rien enregistrer dans l'historique"
    )
    parser.add_argument(
        "--no-tts", action="store_true", help="ne pas lire les réponses à voix haute"
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    prefer_local_models(settings)
    source: AudioSource
    if args.file:
        source = ArraySource.from_file(args.file, settings.sample_rate)
    else:
        source = MicrophoneSource(settings.sample_rate, device=settings.audio_input_device or None)
    detector = SileroSpeechDetector(settings.sample_rate)
    store = None if args.no_history else InteractionStore(settings.db_path)
    history = store.recent_estimates(settings.baseline_window) if store else []
    recorder = HistoryRecorder(store) if store else None
    speech = create_speech(settings, enabled=not args.no_tts)
    if store is not None:
        print(texts.PIPELINE_HISTORY.format(path=store.path, n=len(history)))
    try:
        with VoicePipeline.from_settings(settings, history=history) as pipeline:
            run(
                pipeline,
                source,
                detector,
                settings,
                sys.stdout,
                live=not args.file,
                warm_up=not args.no_warmup,
                details=args.details,
                recorder=recorder,
                speech=speech,
            )
    except AudioDeviceError as exc:
        print(exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
    finally:
        if speech is not None:
            speech.close()
        if recorder is not None:
            recorder.close()
        if store is not None:
            store.close()
    print(texts.CLI_GOODBYE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
