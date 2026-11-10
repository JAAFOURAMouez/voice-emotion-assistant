"""One conversation turn: audio → (transcription ∥ emotion analysis) → adapted LLM reply.

The turn audio comes from the VAD. Transcription and emotion analysis run in
parallel (both release the GIL in native code), then the reply is streamed so the
first words arrive as early as possible. Emotion analysis is optional: if it fails,
the turn goes on without a vocal cue.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from voxemo import texts
from voxemo.audio.recorder import AudioArray, save_wav
from voxemo.audio.vad import SpeechTurn
from voxemo.config import Settings
from voxemo.emotion.classifier import Wav2Vec2EmotionClassifier
from voxemo.emotion.fusion import EmotionAnalyzer, FusionConfig, VoiceBaseline, apply_text_cue
from voxemo.emotion.text_cues import text_cues
from voxemo.emotion.types import EmotionEstimate
from voxemo.llm.client import Conversation, LLMUnavailableError, OllamaChatModel
from voxemo.stt.transcriber import Transcriber, Transcription, WhisperTranscriber
from voxemo.tracking.store import InteractionStore
from voxemo.wellbeing.exercises import Exercise, ExerciseAdvisor, Suggestion, classify_answer

logger = logging.getLogger(__name__)


class VoiceAnalyzer(Protocol):
    """Turn audio → estimated emotion, relative to the user's baseline."""

    def analyze(
        self, audio: AudioArray, sample_rate: int, baseline: VoiceBaseline | None = None
    ) -> EmotionEstimate: ...


@dataclass(frozen=True)
class TurnAnalysis:
    """What the user said and how (before the reply)."""

    turn: SpeechTurn | None
    """``None`` for a typed message."""
    transcription: Transcription
    estimate: EmotionEstimate | None
    """``None`` when the emotion analysis failed."""
    stt_s: float
    emotion_s: float
    baseline_turns: int = 0
    """Number of previous turns the voice was compared with."""

    @property
    def understood(self) -> bool:
        return not self.transcription.is_empty


@dataclass(frozen=True)
class TurnResult:
    analysis: TurnAnalysis
    reply: str
    timestamp: datetime
    first_chunk_s: float | None = None
    """Delay between the end of the user's turn and the first reply chunk."""
    total_s: float = 0.0
    error: str | None = None
    """User-facing message when the LLM could not answer."""
    timings: dict[str, float] = field(default_factory=dict)
    audio_path: Path | None = None
    """Where the turn audio was saved (only when STORE_RAW_AUDIO is enabled)."""
    offer: Suggestion | None = None
    """An exercise offered at the end of this reply (the user may decline)."""
    exercise: Exercise | None = None
    """An exercise the user just accepted: the interface should guide it now."""


def warm_up_signal(sample_rate: int, seconds: float = 1.0) -> AudioArray:
    """Short voiced-like signal that exercises every code path (incl. pitch tracking)."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    f0 = 150.0 + 10.0 * np.sin(2 * np.pi * 3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    signal = sum(np.sin(k * phase) / k for k in range(1, 4))
    return (0.1 * signal * np.hanning(len(t))).astype(np.float32)


class VoicePipeline:
    """Orchestrates a turn; keeps the rolling voice baseline of the session."""

    def __init__(
        self,
        transcriber: Transcriber,
        analyzer: VoiceAnalyzer,
        conversation: Conversation,
        *,
        baseline_window: int = 20,
        history: Iterable[EmotionEstimate] = (),
        fusion_config: FusionConfig | None = None,
        language: str = "fr",
        audio_dir: Path | None = None,
        advisor: ExerciseAdvisor | None = None,
        clock: Callable[[], float] = time.perf_counter,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.transcriber = transcriber
        self.analyzer = analyzer
        self.conversation = conversation
        self.baseline_window = baseline_window
        self.fusion_config = fusion_config or FusionConfig()
        self.language = language
        self.audio_dir = audio_dir
        """Raw audio is saved only when a folder is given (off by default)."""
        self.advisor = advisor
        """Exercise offers are disabled when ``None``."""
        self.pending_offer: Suggestion | None = None
        self._now = now
        self._estimates: deque[EmotionEstimate] = deque(history, maxlen=baseline_window)
        self._clock = clock
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="voxemo")

    @classmethod
    def from_settings(
        cls, settings: Settings, history: Iterable[EmotionEstimate] = ()
    ) -> VoicePipeline:
        fusion_config = FusionConfig.from_settings(settings)
        analyzer = EmotionAnalyzer(Wav2Vec2EmotionClassifier.from_settings(settings), fusion_config)
        return cls(
            WhisperTranscriber.from_settings(settings),
            analyzer,
            Conversation(OllamaChatModel.from_settings(settings), settings),
            baseline_window=settings.baseline_window,
            history=history,
            fusion_config=fusion_config,
            language=settings.language,
            audio_dir=settings.audio_dir if settings.store_raw_audio else None,
            advisor=ExerciseAdvisor.from_settings(settings),
        )

    # --- lifecycle ---

    def warm_up(self, sample_rate: int = 16_000) -> list[str]:
        """Load every model now rather than on the first turn.

        Speech and emotion models must load (errors propagate); an unreachable LLM
        only produces a warning, since it may be started later. The LLM goes first,
        alone: loaded while torch and Whisper start, it was measured several times
        slower (over a minute instead of about ten seconds).
        """
        warnings = self._warm_up_llm()
        signal = warm_up_signal(sample_rate)
        emotion = self._executor.submit(self.analyzer.analyze, signal, sample_rate)
        self.transcriber.transcribe(signal, sample_rate)
        emotion.result()
        return warnings

    def _warm_up_llm(self) -> list[str]:
        try:
            self.conversation.warm_up()
        except LLMUnavailableError as exc:
            return [str(exc)]
        return []

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> VoicePipeline:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # --- one turn ---

    @property
    def baseline(self) -> VoiceBaseline:
        return VoiceBaseline.from_estimates(list(self._estimates), self.baseline_window)

    @property
    def estimates(self) -> Sequence[EmotionEstimate]:
        return tuple(self._estimates)

    def analyze(self, turn: SpeechTurn) -> TurnAnalysis:
        """Transcribe and estimate the emotion of a turn, in parallel."""
        baseline = self.baseline
        emotion: Future[tuple[EmotionEstimate, float]] = self._executor.submit(
            self._timed_analysis, turn, baseline
        )
        started = self._clock()
        transcription = self.transcriber.transcribe(turn.audio, turn.sample_rate)
        stt_s = self._clock() - started
        try:
            estimate, emotion_s = emotion.result()
        except Exception:  # the cue is a nice-to-have: never lose the turn for it
            logger.exception("emotion analysis failed; continuing without a vocal cue")
            estimate, emotion_s = None, 0.0
        if estimate is not None and not transcription.is_empty:
            estimate = apply_text_cue(
                estimate,
                text_cues(transcription.text, self.language),
                baseline_turns=baseline.n_turns,
                duration_s=turn.duration_s,
                config=self.fusion_config,
            )
        analysis = TurnAnalysis(
            turn, transcription, estimate, stt_s, emotion_s, baseline_turns=baseline.n_turns
        )
        if analysis.understood and estimate is not None:
            # Unintelligible turns (noise, coughs) would pollute the user's baseline.
            self._estimates.append(estimate)
        return analysis

    def _timed_analysis(
        self, turn: SpeechTurn, baseline: VoiceBaseline
    ) -> tuple[EmotionEstimate, float]:
        started = self._clock()
        estimate = self.analyzer.analyze(turn.audio, turn.sample_rate, baseline)
        return estimate, self._clock() - started

    def process(
        self,
        turn: SpeechTurn,
        *,
        on_analysis: Callable[[TurnAnalysis], None] | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> TurnResult | None:
        """Run a full turn; returns ``None`` when nothing intelligible was said."""
        timestamp = self._now()
        started = self._clock()
        audio_path = self._save_audio(turn, timestamp)
        analysis = self.analyze(turn)
        if not analysis.understood:
            return None
        if on_analysis:
            on_analysis(analysis)
        return self._respond(analysis, timestamp, started, audio_path, on_chunk)

    def process_text(
        self, text: str, *, on_chunk: Callable[[str], None] | None = None
    ) -> TurnResult | None:
        """A typed message: same flow, without vocal cues (no voice to analyze)."""
        text = text.strip()
        if not text:
            return None
        analysis = TurnAnalysis(
            turn=None,
            transcription=Transcription(text, self.language, 1.0, 0.0),
            estimate=None,
            stt_s=0.0,
            emotion_s=0.0,
            baseline_turns=len(self._estimates),
        )
        return self._respond(analysis, self._now(), self._clock(), None, on_chunk)

    def answer_offer(self, accepted: bool) -> tuple[str, Exercise | None]:
        """Answer the pending exercise offer from a button; returns (reply, exercise)."""
        offer, self.pending_offer = self.pending_offer, None
        if offer is None:
            return "", None
        user_text = texts.EXERCISE_ACCEPT_BUTTON if accepted else texts.EXERCISE_DECLINE_BUTTON
        return self._settle_offer(offer, accepted, user_text)

    def _settle_offer(
        self, offer: Suggestion, accepted: bool, user_text: str
    ) -> tuple[str, Exercise | None]:
        reply = offer.exercise.introduction() if accepted else texts.EXERCISE_DECLINED
        note = texts.EXERCISE_HISTORY_NOTE.format(name=offer.exercise.name)
        self.conversation.add_exchange(user_text, f"{reply} {note}" if accepted else reply)
        return reply, offer.exercise if accepted else None

    def forget_voice_baseline(self) -> None:
        """Start learning the voice again (e.g. after erasing the history)."""
        self._estimates.clear()

    def _respond(
        self,
        analysis: TurnAnalysis,
        timestamp: datetime,
        started: float,
        audio_path: Path | None,
        on_chunk: Callable[[str], None] | None,
    ) -> TurnResult:
        text = analysis.transcription.text
        timings = {"stt_s": analysis.stt_s, "emotion_s": analysis.emotion_s}
        offer, self.pending_offer = self.pending_offer, None
        answer = classify_answer(text) if offer else None
        if offer and answer:
            # A short yes/no to the previous offer: no LLM call needed.
            reply, exercise = self._settle_offer(offer, answer == "accept", text)
            if on_chunk:
                on_chunk(reply)
            return TurnResult(
                analysis=analysis,
                reply=reply,
                timestamp=timestamp,
                first_chunk_s=self._clock() - started,
                total_s=self._clock() - started,
                timings=timings,
                audio_path=audio_path,
                exercise=exercise,
            )

        suggestion = (
            self.advisor.observe(analysis.estimate, text, timestamp) if self.advisor else None
        )
        parts: list[str] = []
        first_chunk_s: float | None = None
        error: str | None = None
        try:
            for chunk in self.conversation.reply_stream(
                text,
                analysis.estimate,
                note=texts.EXERCISE_OFFER_NOTE if suggestion else None,
                suffix=suggestion.exercise.offer(suggestion.label) if suggestion else "",
            ):
                if first_chunk_s is None:
                    first_chunk_s = self._clock() - started
                parts.append(chunk)
                if on_chunk:
                    on_chunk(chunk)
        except LLMUnavailableError as exc:
            error = str(exc)
            suggestion = None  # the offer was not delivered
        self.pending_offer = suggestion
        return TurnResult(
            analysis=analysis,
            reply="".join(parts).strip(),
            timestamp=timestamp,
            first_chunk_s=first_chunk_s,
            total_s=self._clock() - started,
            error=error,
            timings=timings,
            audio_path=audio_path,
            offer=suggestion,
        )

    def _save_audio(self, turn: SpeechTurn, timestamp: datetime) -> Path | None:
        if self.audio_dir is None:
            return None
        name = f"turn-{timestamp.astimezone():%Y%m%d-%H%M%S-%f}.wav"
        return save_wav(turn.audio, turn.sample_rate, self.audio_dir / name)


class HistoryRecorder:
    """Writes the turns of one conversation session to the local history.

    The session row is created with the first recorded turn, so that starting and
    quitting without speaking leaves no trace.
    """

    def __init__(self, store: InteractionStore) -> None:
        self.store = store
        self.session_id: int | None = None

    def record(self, result: TurnResult) -> int:
        if self.session_id is None:
            self.session_id = self.store.start_session(result.timestamp)
        return self.store.add_interaction(
            self.session_id,
            transcript=result.analysis.transcription.text,
            reply=result.reply,
            estimate=result.analysis.estimate,
            timestamp=result.timestamp,
            audio_path=result.audio_path,
        )

    def close(self) -> None:
        if self.session_id is not None:
            self.store.end_session(self.session_id)
