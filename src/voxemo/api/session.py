"""One WebSocket connection = one conversation.

Client → server
    {"type": "start", "sample_rate": 16000}   then binary frames of 16-bit LE mono PCM
    {"type": "end"}                           process the recorded turn
    {"type": "cancel"}                        drop the recording
    {"type": "text", "text": "..."}           typed message (no vocal cue)
    {"type": "offer", "accept": true|false}   answer to an exercise offer
    {"type": "voice", "enabled": true|false}  spoken replies on/off
    {"type": "reset"}                         new conversation

Server → client (JSON unless stated)
    status, recording, silence, transcript, emotion, reply_start, reply_chunk,
    speech (header, followed by one binary frame: a WAV file), reply_end, offer,
    exercise, not_understood, error, done
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from starlette.websockets import WebSocket, WebSocketDisconnect

from voxemo import texts
from voxemo.api.payloads import estimate_payload, exercise_payload, offer_payload
from voxemo.api.services import Services
from voxemo.pipeline import HistoryRecorder, TurnAnalysis, TurnResult, VoicePipeline
from voxemo.tts.speaker import split_sentences

logger = logging.getLogger(__name__)
_LOOPBACK = {"localhost", "127.0.0.1", "::1", "[::1]"}


def origin_allowed(origin: str | None, host: str | None, extra: str = "") -> bool:
    """Block cross-site WebSocket hijacking: only this app's own pages may connect.

    Browsers always send ``Origin``; other clients (tests, scripts) usually do not.
    """
    if not origin:
        return True
    parsed = urlparse(origin)
    if parsed.hostname in _LOOPBACK:
        return True
    if host and parsed.netloc == host:
        return True
    allowed = {o.strip().rstrip("/") for o in extra.split(",") if o.strip()}
    return origin.rstrip("/") in allowed


class ServiceHolder:
    """Models load in the background; connections can open meanwhile."""

    def __init__(self) -> None:
        self.services: Services | None = None
        self.error: str | None = None
        self.ready = threading.Event()
        self.sessions: set[ConversationSession] = set()

    @property
    def state(self) -> str:
        if self.services is not None:
            return "ready"
        return "error" if self.error else "loading"


class ConversationSession:
    def __init__(
        self,
        websocket: WebSocket,
        holder: ServiceHolder,
        *,
        sample_rate: int,
        max_turn_s: float,
        voice: bool,
    ) -> None:
        self.ws = websocket
        self.holder = holder
        self.sample_rate = sample_rate
        self.max_bytes = int((max_turn_s + 5) * sample_rate * 2)
        self.voice = voice
        self.pipeline: VoicePipeline | None = None
        self.recorder: HistoryRecorder | None = None
        self._recording: bytearray | None = None
        self._send_lock = asyncio.Lock()

    # --- sending ---

    async def send(self, kind: str, **payload: Any) -> None:
        async with self._send_lock:
            await self.ws.send_text(json.dumps({"type": kind, **payload}, ensure_ascii=False))

    async def send_speech(self, index: int, wav: bytes) -> None:
        async with self._send_lock:  # header and audio must stay together
            header = {"type": "speech", "index": index, "bytes": len(wav)}
            await self.ws.send_text(json.dumps(header))
            await self.ws.send_bytes(wav)

    async def send_status(self) -> None:
        holder = self.holder
        services = holder.services
        message = None
        if holder.state == "loading":
            message = texts.API_LOADING
        elif holder.state == "error":
            message = holder.error
        llm_problem = await asyncio.to_thread(services.llm_problem) if services else None
        await self.send(
            "status",
            state=holder.state,
            message=message,
            llm={"ok": services is not None and llm_problem is None, "message": llm_problem},
            voice=bool(services and services.speaker is not None),
            warnings=list(dict.fromkeys(services.warnings)) if services else [],
        )

    # --- lifecycle ---

    async def run(self) -> None:
        await self.send_status()
        waiter = asyncio.create_task(self._announce_when_ready())
        try:
            while True:
                message = await self.ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    await self._on_audio(message["bytes"])
                elif message.get("text") is not None:
                    await self._on_message(message["text"])
        except WebSocketDisconnect:
            pass
        finally:
            waiter.cancel()
            self.close()

    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.close()
        if self.pipeline is not None:
            self.pipeline.close()
        self.holder.sessions.discard(self)

    async def _announce_when_ready(self) -> None:
        if self.holder.state != "loading":
            return
        await asyncio.to_thread(self.holder.ready.wait)
        await self.send_status()

    def _ensure_pipeline(self) -> VoicePipeline | None:
        services = self.holder.services
        if services is None:
            return None
        if self.pipeline is None:
            self.pipeline = services.new_pipeline()
            self.recorder = HistoryRecorder(services.store)
        return self.pipeline

    # --- incoming ---

    async def _on_audio(self, data: bytes) -> None:
        if self._recording is None:
            return  # audio outside start/end is ignored
        if len(self._recording) + len(data) > self.max_bytes:
            self._recording = None
            max_s = self.max_bytes / (2 * self.sample_rate) - 5
            await self.send(
                "error", code="too_long", message=texts.API_TOO_LONG.format(max_s=max_s)
            )
            return
        self._recording.extend(data)

    async def _on_message(self, raw: str) -> None:
        try:
            message = json.loads(raw)
            kind = message["type"]
        except (ValueError, KeyError, TypeError):
            await self.send("error", code="bad_message", message=texts.API_BAD_MESSAGE)
            return
        if kind == "start":
            rate = int(message.get("sample_rate", self.sample_rate))
            if rate != self.sample_rate:
                await self.send(
                    "error", code="bad_rate", message=f"sample_rate must be {self.sample_rate}"
                )
                return
            self._recording = bytearray()
            await self.send("recording")
        elif kind == "cancel":
            self._recording = None
        elif kind == "end":
            pcm, self._recording = self._recording, None
            if pcm is not None:
                await self._voice_turn(bytes(pcm))
        elif kind == "text":
            await self._text_turn(str(message.get("text", "")))
        elif kind == "offer":
            await self._answer_offer(bool(message.get("accept")))
        elif kind == "voice":
            self.voice = bool(message.get("enabled", True))
        elif kind == "reset":
            if self.pipeline is not None:
                self.pipeline.conversation.reset()
                self.pipeline.pending_offer = None
            await self.send("done")
        else:
            await self.send("error", code="bad_message", message=texts.API_BAD_MESSAGE)

    # --- turns ---

    async def _not_ready(self) -> bool:
        if self._ensure_pipeline() is None:
            await self.send(
                "error",
                code=self.holder.state,
                message=texts.API_LOADING if self.holder.state == "loading" else self.holder.error,
            )
            return True
        return False

    async def _voice_turn(self, pcm: bytes) -> None:
        if await self._not_ready():
            return
        services = self.holder.services
        assert services is not None and self.pipeline is not None
        turn = await asyncio.to_thread(services.turn_from_pcm, pcm)
        if turn is None:
            await self.send("silence")
            await self.send("done")
            return
        pipeline = self.pipeline
        await self._stream(
            lambda on_analysis, on_chunk: pipeline.process(
                turn, on_analysis=on_analysis, on_chunk=on_chunk
            )
        )

    async def _text_turn(self, text: str) -> None:
        if not text.strip() or await self._not_ready():
            return
        pipeline = self.pipeline
        assert pipeline is not None
        await self.send("transcript", text=text.strip(), typed=True)
        await self._stream(
            lambda _on_analysis, on_chunk: pipeline.process_text(text, on_chunk=on_chunk)
        )

    async def _stream(
        self, run: Callable[[Callable[[TurnAnalysis], None], Callable[[str], None]], Any]
    ) -> None:
        """Run a blocking pipeline call in a thread and relay its events in order."""
        loop = asyncio.get_running_loop()
        events: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        def push(kind: str) -> Callable[[Any], None]:
            return lambda value: loop.call_soon_threadsafe(events.put_nowait, (kind, value))

        def work() -> None:
            try:
                result = run(push("analysis"), push("chunk"))
                push("result")(result)
            except Exception as exc:  # never lose an error in a thread
                logger.exception("turn failed")
                push("failure")(exc)

        threading.Thread(target=work, name="voxemo-turn", daemon=True).start()
        speech = _SpeechStream(self)
        started_reply = False
        while True:
            kind, value = await events.get()
            if kind == "analysis":
                await self._send_analysis(value)
            elif kind == "chunk":
                if not started_reply:
                    started_reply = True
                    await self.send("reply_start")
                await self.send("reply_chunk", text=value)
                speech.feed(value)
            elif kind == "failure":
                await speech.finish()
                await self.send("error", code="internal", message=str(value))
                await self.send("done")
                return
            else:  # result
                await speech.finish()
                await self._send_result(value, started_reply)
                return

    async def _send_analysis(self, analysis: TurnAnalysis) -> None:
        assert self.pipeline is not None
        await self.send(
            "transcript",
            text=analysis.transcription.text,
            typed=False,
            duration_s=round(analysis.transcription.duration_s, 2),
        )
        if analysis.estimate is not None:
            await self.send(
                "emotion",
                **estimate_payload(
                    analysis.estimate,
                    baseline_turns=analysis.baseline_turns,
                    min_turns=self.pipeline.fusion_config.baseline_min_turns,
                ),
            )

    async def _send_result(self, result: TurnResult | None, started_reply: bool) -> None:
        if result is None:
            await self.send("not_understood", message=texts.UI_NOT_UNDERSTOOD)
            await self.send("done")
            return
        if self.recorder is not None:
            try:
                await asyncio.to_thread(self.recorder.record, result)
            except Exception:  # the conversation goes on even if the history cannot be saved
                logger.exception("could not save the interaction")
        if not started_reply:
            await self.send("reply_start")
        await self.send("reply_end", text=result.reply, error=result.error)
        if result.error:
            await self.send("error", code="llm", message=result.error)
            await self.send_status()  # the page shows the LLM as unavailable
        if result.offer is not None:
            await self.send("offer", **offer_payload(result.offer))
        if result.exercise is not None:
            await self.send("exercise", **exercise_payload(result.exercise))
        await self.send(
            "done",
            timings={
                "stt_s": round(result.analysis.stt_s, 2),
                "emotion_s": round(result.analysis.emotion_s, 2),
                "first_chunk_s": round(result.first_chunk_s or 0.0, 2),
                "total_s": round(result.total_s, 2),
            },
        )

    async def _answer_offer(self, accept: bool) -> None:
        if await self._not_ready():
            return
        assert self.pipeline is not None
        reply, exercise = self.pipeline.answer_offer(accept)
        if not reply:
            await self.send("done")
            return
        speech = _SpeechStream(self)
        await self.send("reply_start")
        await self.send("reply_chunk", text=reply)
        speech.feed(reply)
        await speech.finish()
        await self.send("reply_end", text=reply, error=None)
        if exercise is not None:
            await self.send("exercise", **exercise_payload(exercise))
        await self.send("done")


class _SpeechStream:
    """Synthesizes complete sentences in order while the reply streams."""

    def __init__(self, session: ConversationSession) -> None:
        self.session = session
        self.buffer = ""
        self.index = 0
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.enabled = session.voice and bool(
            session.holder.services and session.holder.services.speaker is not None
        )
        self.task = asyncio.create_task(self._worker()) if self.enabled else None

    def feed(self, chunk: str) -> None:
        if not self.enabled:
            return
        self.buffer += chunk
        sentences, self.buffer = split_sentences(self.buffer)
        for sentence in sentences:
            self.queue.put_nowait(sentence)

    async def finish(self) -> None:
        if self.task is None:
            return
        if self.buffer.strip():
            self.queue.put_nowait(self.buffer)
        self.buffer = ""
        self.queue.put_nowait(None)
        await self.task

    async def _worker(self) -> None:
        services = self.session.holder.services
        assert services is not None
        while (sentence := await self.queue.get()) is not None:
            wav = await asyncio.to_thread(services.speech_wav, sentence)
            if wav:
                await self.session.send_speech(self.index, wav)
                self.index += 1
