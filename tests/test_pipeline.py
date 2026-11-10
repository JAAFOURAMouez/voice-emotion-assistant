"""Tests for the turn orchestration and the voice CLI (all models are fakes)."""

from __future__ import annotations

import dataclasses
import io
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest

from conftest import SAMPLE_RATE, EnergyDetector, FakeChatModel, make_estimate, silence, tone
from voxemo import texts
from voxemo.__main__ import run, run_turn
from voxemo.audio.recorder import ArraySource, load_audio
from voxemo.audio.vad import SpeechTurn
from voxemo.config import Settings
from voxemo.emotion.fusion import VoiceBaseline
from voxemo.emotion.types import EmotionEstimate
from voxemo.llm.client import Conversation, LLMUnavailableError
from voxemo.llm.prompts import ChatMessage, system_prompt
from voxemo.pipeline import VoicePipeline, warm_up_signal
from voxemo.stt.transcriber import Transcription


class FakeTranscriber:
    def __init__(self, *texts_: str, barrier: threading.Barrier | None = None) -> None:
        self.texts = list(texts_) or ["Bonjour"]
        self.calls = 0
        self.barrier = barrier

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Transcription:
        if self.barrier:
            self.barrier.wait(timeout=5)  # fails unless the analyzer runs at the same time
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return Transcription(text, "fr", 0.99, len(audio) / sample_rate)


class FakeAnalyzer:
    def __init__(
        self,
        estimate: EmotionEstimate | None = None,
        *,
        error: Exception | None = None,
        barrier: threading.Barrier | None = None,
    ) -> None:
        self.estimate = estimate or make_estimate("neutre", 0.6)
        self.error = error
        self.barrier = barrier
        self.baselines: list[VoiceBaseline | None] = []

    def analyze(
        self, audio: np.ndarray, sample_rate: int, baseline: VoiceBaseline | None = None
    ) -> EmotionEstimate:
        self.baselines.append(baseline)
        if self.barrier:
            self.barrier.wait(timeout=5)
        if self.error:
            raise self.error
        return self.estimate


class FailingChatModel:
    def stream_chat(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        yield "Je "
        raise LLMUnavailableError(texts.OLLAMA_UNREACHABLE.format(host="http://localhost:11434"))


class WarmableChatModel(FakeChatModel):
    def __init__(self, error: Exception | None = None) -> None:
        super().__init__()
        self.warmed = False
        self.warm_up_messages: list[ChatMessage] = []
        self.error = error

    def warm_up(self, messages: Sequence[ChatMessage] = ()) -> None:
        self.warmed = True
        self.warm_up_messages = list(messages)
        if self.error:
            raise self.error


def _turn(seconds: float = 1.5) -> SpeechTurn:
    return SpeechTurn(tone(seconds), SAMPLE_RATE, 0.0)


def _pipeline(
    settings: Settings,
    transcriber: FakeTranscriber | None = None,
    analyzer: FakeAnalyzer | None = None,
    model: object | None = None,
) -> VoicePipeline:
    return VoicePipeline(
        transcriber or FakeTranscriber(),
        analyzer or FakeAnalyzer(),
        Conversation(model or FakeChatModel("Bien sûr."), settings),  # type: ignore[arg-type]
        baseline_window=settings.baseline_window,
    )


def test_transcription_and_emotion_run_in_parallel(settings: Settings) -> None:
    barrier = threading.Barrier(2)
    pipeline = _pipeline(settings, FakeTranscriber(barrier=barrier), FakeAnalyzer(barrier=barrier))
    with pipeline:
        analysis = pipeline.analyze(_turn())
    assert analysis.transcription.text == "Bonjour"
    assert analysis.estimate is not None


def test_process_streams_reply_with_emotion_hint(settings: Settings) -> None:
    model = FakeChatModel("Prenons une chose à la fois.")
    pipeline = _pipeline(settings, analyzer=FakeAnalyzer(make_estimate("stress", 0.8)), model=model)
    chunks: list[str] = []
    seen = []
    with pipeline:
        result = pipeline.process(_turn(), on_analysis=seen.append, on_chunk=chunks.append)
    assert result is not None
    assert result.reply == "Prenons une chose à la fois."
    assert "".join(chunks).strip() == result.reply
    assert seen == [result.analysis]
    assert result.first_chunk_s is not None and result.first_chunk_s <= result.total_s
    sent = "\n".join(m.content for m in model.calls[0])
    assert "la voix suggère" in sent
    assert model.calls[0][-1] == ChatMessage("user", "Bonjour")


def test_baseline_grows_turn_after_turn(settings: Settings) -> None:
    analyzer = FakeAnalyzer()
    with _pipeline(settings, analyzer=analyzer) as pipeline:
        for _ in range(3):
            pipeline.process(_turn())
    assert [b.n_turns for b in analyzer.baselines if b is not None] == [0, 1, 2]
    assert len(pipeline.estimates) == 3


def test_baseline_respects_window_and_initial_history(settings: Settings) -> None:
    history = [make_estimate("neutre", 0.6)] * 4
    pipeline = VoicePipeline(
        FakeTranscriber(),
        FakeAnalyzer(),
        Conversation(FakeChatModel(), settings),
        baseline_window=5,
        history=history,
    )
    with pipeline:
        assert pipeline.baseline.n_turns == 4
        pipeline.process(_turn())
        pipeline.process(_turn())
    assert len(pipeline.estimates) == 5


def test_unintelligible_turn_is_skipped(settings: Settings) -> None:
    model = FakeChatModel()
    with _pipeline(settings, FakeTranscriber(""), model=model) as pipeline:
        assert pipeline.process(_turn()) is None
        assert pipeline.estimates == ()  # noise must not pollute the voice baseline
    assert model.calls == []


def test_emotion_failure_does_not_lose_the_turn(settings: Settings) -> None:
    model = FakeChatModel("D'accord.")
    analyzer = FakeAnalyzer(error=RuntimeError("model crashed"))
    with _pipeline(settings, analyzer=analyzer, model=model) as pipeline:
        result = pipeline.process(_turn())
    assert result is not None
    assert result.analysis.estimate is None
    assert result.reply == "D'accord."
    assert "la voix suggère" not in "\n".join(m.content for m in model.calls[0])


def test_llm_failure_is_reported_not_raised(settings: Settings) -> None:
    with _pipeline(settings, model=FailingChatModel()) as pipeline:
        result = pipeline.process(_turn())
        assert result is not None
        assert result.error is not None and "Ollama" in result.error
        assert pipeline.conversation.history == ()
        assert len(pipeline.estimates) == 1  # the voice analysis is still useful


def test_distress_in_speech_points_to_helplines(settings: Settings) -> None:
    transcriber = FakeTranscriber("Je n'en peux plus, je veux mourir.")
    model = FakeChatModel("Essayons une respiration lente.")
    with _pipeline(
        settings, transcriber, FakeAnalyzer(make_estimate("tristesse", 0.9)), model
    ) as p:
        result = p.process(_turn())
    assert result is not None
    assert "3114" in result.reply


def test_warm_up_loads_every_model(settings: Settings) -> None:
    transcriber, analyzer, model = FakeTranscriber(), FakeAnalyzer(), WarmableChatModel()
    with _pipeline(settings, transcriber, analyzer, model) as pipeline:
        assert pipeline.warm_up(SAMPLE_RATE) == []
    assert transcriber.calls == 1
    assert len(analyzer.baselines) == 1
    assert model.warmed
    # The system prompt is pre-processed so the first real turn starts faster.
    assert [m.role for m in model.warm_up_messages] == ["system"]
    assert model.warm_up_messages[0].content == system_prompt(settings.language)
    assert pipeline.estimates == ()  # warm-up is not a user turn


def test_warm_up_reports_unreachable_llm(settings: Settings) -> None:
    model = WarmableChatModel(LLMUnavailableError("Ollama absent"))
    with _pipeline(settings, model=model) as pipeline:
        assert pipeline.warm_up(SAMPLE_RATE) == ["Ollama absent"]


def test_warm_up_signal_is_voiced_and_bounded() -> None:
    signal = warm_up_signal(SAMPLE_RATE)
    assert signal.dtype == np.float32
    assert len(signal) == SAMPLE_RATE
    assert 0.05 < np.abs(signal).max() <= 0.2


# --- command line ---


def test_run_turn_prints_transcript_cues_and_reply(settings: Settings) -> None:
    out = io.StringIO()
    analyzer = FakeAnalyzer(make_estimate("stress", 0.72))
    with _pipeline(settings, analyzer=analyzer, model=FakeChatModel("Faisons une pause.")) as p:
        run_turn(p, _turn(), out)
    text = out.getvalue()
    assert "Vous : Bonjour" in text
    assert "émotion estimée : stress, confiance 0,72" in text
    assert "Assistant : Faisons une pause." in text
    assert "début de réponse" in text


def test_run_turn_prints_llm_error(settings: Settings) -> None:
    out = io.StringIO()
    with _pipeline(settings, model=FailingChatModel()) as pipeline:
        run_turn(pipeline, _turn(), out)
    assert "Impossible de joindre Ollama" in out.getvalue()


def test_run_on_a_file_processes_each_turn(settings: Settings) -> None:
    audio = np.concatenate([silence(0.5), tone(1.0), silence(1.5), tone(1.0), silence(1.5)])
    model = WarmableChatModel()
    transcriber = FakeTranscriber("(préchargement)", "Premier tour.", "Second tour.")
    out = io.StringIO()
    with _pipeline(settings, transcriber, model=model) as pipeline:
        results = run(
            pipeline,
            ArraySource(audio, SAMPLE_RATE),
            EnergyDetector(),
            settings,
            out,
            live=False,
        )
    assert [r.analysis.transcription.text for r in results] == ["Premier tour.", "Second tour."]
    assert model.warmed
    assert "Prêt" in out.getvalue()
    assert len(model.calls) == 2
    # The second turn sees the first one in the conversation history.
    assert ChatMessage("user", "Premier tour.") in model.calls[1]


@pytest.mark.parametrize("raw", [0.5, 0.0, 1.0])
def test_confidence_shown_with_comma(raw: float, settings: Settings) -> None:
    out = io.StringIO()
    with _pipeline(settings, analyzer=FakeAnalyzer(make_estimate("joie", raw))) as pipeline:
        run_turn(pipeline, _turn(), out)
    assert f"confiance {raw:.2f}".replace(".", ",") in out.getvalue()


def test_words_refine_the_voice_estimate(settings: Settings) -> None:
    transcriber = FakeTranscriber("Je suis très énervé.")
    undecided = dataclasses.replace(
        make_estimate("neutre", 0.24),
        scores={"neutre": 0.4, "joie": 0.2, "tristesse": 0.2, "stress": 0.2},
    )
    with _pipeline(settings, transcriber, FakeAnalyzer(undecided)) as pipeline:
        result = pipeline.process(_turn())
    assert result is not None and result.analysis.estimate is not None
    assert result.analysis.estimate.label == "stress"
    assert result.analysis.estimate.text_terms == ("énervé",)
    assert result.analysis.baseline_turns == 0


def test_raw_audio_is_not_saved_by_default(settings: Settings, tmp_path: Path) -> None:
    with _pipeline(settings) as pipeline:
        result = pipeline.process(_turn())
    assert result is not None and result.audio_path is None
    assert pipeline.audio_dir is None


def test_raw_audio_saved_when_enabled(settings: Settings, tmp_path: Path) -> None:
    pipeline = VoicePipeline(
        FakeTranscriber(),
        FakeAnalyzer(),
        Conversation(FakeChatModel(), settings),
        audio_dir=tmp_path / "audio",
    )
    with pipeline:
        result = pipeline.process(_turn(1.5))
    assert result is not None and result.audio_path is not None
    saved = load_audio(result.audio_path, SAMPLE_RATE)
    assert len(saved) == int(1.5 * SAMPLE_RATE)


def test_from_settings_respects_store_raw_audio(tmp_path: Path) -> None:
    with VoicePipeline.from_settings(Settings(_env_file=None)) as off:
        assert off.audio_dir is None
    on_settings = Settings(_env_file=None, store_raw_audio=True, audio_dir=tmp_path)
    with VoicePipeline.from_settings(on_settings) as on:
        assert on.audio_dir == tmp_path
        assert on.fusion_config.text_gain == 5.0


def test_neutral_line_shows_closest_label_and_calibration(settings: Settings) -> None:
    out = io.StringIO()
    estimate = dataclasses.replace(
        make_estimate("neutre", 0.2),
        scores={"neutre": 0.35, "joie": 0.2, "tristesse": 0.15, "stress": 0.3},
    )
    with _pipeline(settings, analyzer=FakeAnalyzer(estimate)) as pipeline:
        run_turn(pipeline, _turn(), out)
    text = out.getvalue()
    assert "neutre, rien de marqué (le plus proche : stress 0,30)" in text
    assert "apprentissage de votre voix 0/5" in text


def test_details_show_scores_and_words(settings: Settings) -> None:
    out = io.StringIO()
    transcriber = FakeTranscriber("Je suis tellement triste")
    with _pipeline(settings, transcriber) as pipeline:
        run_turn(pipeline, _turn(), out, details=True)
    text = out.getvalue()
    assert "mots relevés : triste" in text
    assert "scores neutre=" in text
    assert "pitch 180±25 Hz" in text


def test_history_recorder_writes_one_session(settings: Settings, tmp_path: Path) -> None:
    from voxemo.pipeline import HistoryRecorder
    from voxemo.tracking.store import InteractionStore

    with InteractionStore(tmp_path / "h.db") as store:
        recorder = HistoryRecorder(store)
        recorder.close()
        assert store.sessions() == []  # nothing said: no empty session
        transcriber = FakeTranscriber("(préchargement)", "Premier tour.", "Second tour.")
        audio = np.concatenate([silence(0.5), tone(1.0), silence(1.5), tone(1.0), silence(1.5)])
        with _pipeline(settings, transcriber, model=FakeChatModel("Oui.")) as pipeline:
            run(
                pipeline,
                ArraySource(audio, SAMPLE_RATE),
                EnergyDetector(),
                settings,
                io.StringIO(),
                live=False,
                recorder=recorder,
            )
        recorder.close()
        [session] = store.sessions()
        assert session.interaction_count == 2 and session.ended_at is not None
        items = store.interactions()
        assert [i.transcript for i in items] == ["Premier tour.", "Second tour."]
        assert items[0].reply == "Oui."
        assert items[0].estimate is not None


def test_stored_history_seeds_the_next_session_baseline(settings: Settings) -> None:
    history = [make_estimate("neutre", 0.6)] * 6
    analyzer = FakeAnalyzer()
    pipeline = VoicePipeline(
        FakeTranscriber(), analyzer, Conversation(FakeChatModel(), settings), history=history
    )
    out = io.StringIO()
    with pipeline:
        run_turn(pipeline, _turn(), out)
    assert analyzer.baselines[0] is not None and analyzer.baselines[0].n_turns == 6
    assert "apprentissage de votre voix" not in out.getvalue()


# --- exercise offers ---


def _advised_pipeline(settings: Settings, *texts_: str, model: FakeChatModel | None = None):
    from datetime import UTC, datetime, timedelta

    from voxemo.wellbeing.exercises import ExerciseAdvisor

    times = iter(datetime(2026, 11, 10, 12, tzinfo=UTC) + timedelta(minutes=k) for k in range(50))
    return VoicePipeline(
        FakeTranscriber(*texts_),
        FakeAnalyzer(make_estimate("stress", 0.8)),
        Conversation(model or FakeChatModel("Je vous écoute."), settings),
        advisor=ExerciseAdvisor(),
        now=lambda: next(times),
    )


def test_offer_after_two_tense_turns_then_accept(settings: Settings) -> None:
    model = FakeChatModel("Je vous écoute.")
    with _advised_pipeline(
        settings, "J'ai trop de travail", "Et mon chef me presse", "Oui", model=model
    ) as pipeline:
        first = pipeline.process(_turn())
        second = pipeline.process(_turn())
        third = pipeline.process(_turn())
    assert first is not None and first.offer is None
    assert second is not None and second.offer is not None
    assert "Voulez-vous essayer ?" in second.reply
    # The LLM was told not to offer an exercise itself.
    assert any("N'en propose pas toi-même" in m.content for m in model.calls[1])
    assert third is not None and third.exercise == second.offer.exercise
    assert third.reply.startswith("Très bien.")
    assert len(model.calls) == 2  # a plain "oui" needs no LLM call
    assert pipeline.conversation.history[-1].content.endswith(
        f"(Exercice guidé : {second.offer.exercise.name}.)"
    )


def test_declined_offer_gets_a_short_acknowledgement(settings: Settings) -> None:
    with _advised_pipeline(settings, "a", "b", "Non merci") as pipeline:
        pipeline.process(_turn())
        pipeline.process(_turn())
        result = pipeline.process(_turn())
    assert result is not None and result.exercise is None
    assert "pas de souci" in result.reply


def test_unrelated_answer_goes_to_the_llm(settings: Settings) -> None:
    model = FakeChatModel("Je vous écoute.")
    with _advised_pipeline(
        settings, "a", "b", "Je préfère te parler de mon examen de demain", model=model
    ) as pipeline:
        for _ in range(3):
            result = pipeline.process(_turn())
    assert result is not None and result.exercise is None
    assert len(model.calls) == 3
    assert pipeline.pending_offer is None


def test_distress_after_an_offer_reaches_the_safety_path(settings: Settings) -> None:
    model = FakeChatModel("Je comprends.")
    with _advised_pipeline(settings, "a", "b", "Non, je veux mourir", model=model) as pipeline:
        for _ in range(3):
            result = pipeline.process(_turn())
    assert result is not None and result.exercise is None
    assert "3114" in result.reply
    assert len(model.calls) == 3


def test_accepted_exercise_is_guided_in_the_cli(settings: Settings) -> None:
    out, slept = io.StringIO(), []
    with _advised_pipeline(settings, "a", "b", "Oui") as pipeline:
        for _ in range(3):
            run_turn(pipeline, _turn(), out, sleep=slept.append)
    text = out.getvalue()
    assert "Voulez-vous essayer ?" in text
    assert "C'est terminé" in text
    assert slept  # the exercise was paced


# --- speech ---


class _RecordingSpeech:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.error: str | None = None

    def feed(self, chunk: str) -> None:
        self.events.append(f"feed:{chunk}")

    def flush(self) -> None:
        self.events.append("flush")

    def say(self, text: str) -> None:
        self.events.append(f"say:{text}")

    def wait(self, timeout: float | None = None) -> bool:
        self.events.append("wait")
        return True

    def cancel(self) -> None:
        self.events.append("cancel")


def test_reply_is_spoken_and_awaited_before_listening(settings: Settings) -> None:
    speech = _RecordingSpeech()
    with _pipeline(settings, model=FakeChatModel("Bonjour à vous.")) as pipeline:
        run_turn(pipeline, _turn(), io.StringIO(), speech=speech)  # type: ignore[arg-type]
    spoken = "".join(e[5:] for e in speech.events if e.startswith("feed:"))
    assert spoken.strip() == "Bonjour à vous."
    assert speech.events[-2:] == ["flush", "wait"]


def test_exercise_steps_are_announced(settings: Settings) -> None:
    speech = _RecordingSpeech()
    with _advised_pipeline(settings, "a", "b", "Oui") as pipeline:
        for _ in range(3):
            run_turn(pipeline, _turn(), io.StringIO(), sleep=lambda _: None, speech=speech)  # type: ignore[arg-type]
    said = [e[4:] for e in speech.events if e.startswith("say:")]
    assert "Inspirez" in said or "Inspirez par le nez" in said
    assert said[-1].startswith("C'est terminé")


def test_llm_is_warmed_up_first_and_alone(settings: Settings) -> None:
    order: list[str] = []

    class Model(WarmableChatModel):
        def warm_up(self, messages: Sequence[ChatMessage] = ()) -> None:
            order.append("llm")

    class Transcriber(FakeTranscriber):
        def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Transcription:
            order.append("stt")
            return super().transcribe(audio, sample_rate)

    with _pipeline(settings, Transcriber(), FakeAnalyzer(), Model()) as pipeline:
        pipeline.warm_up(SAMPLE_RATE)
    assert order == ["llm", "stt"]
