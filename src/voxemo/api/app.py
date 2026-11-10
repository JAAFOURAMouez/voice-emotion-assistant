"""FastAPI application: ``python -m voxemo.api`` (or ``uvicorn voxemo.api.app:app``)."""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from voxemo import texts
from voxemo.api.payloads import (
    exercise_payload,
    history_payload,
    simulated_interactions,
)
from voxemo.api.services import Services, build_services, warm_up
from voxemo.api.session import ConversationSession, ServiceHolder, origin_allowed
from voxemo.config import Settings, get_settings
from voxemo.offline import prefer_local_models
from voxemo.wellbeing.exercises import CATALOGUE, Exercise, get_exercise

logger = logging.getLogger(__name__)

_NOT_BUILT = """<!doctype html><html lang="fr"><meta charset="utf-8"><title>voxemo</title>
<p>L'interface n'est pas encore construite : <code>cd web &amp;&amp; npm install &amp;&amp;
npm run build</code>, ou en développement <code>npm run dev</code>.</p></html>"""


def spoken_texts(exercise: Exercise) -> list[str]:
    """Texts an exercise may speak, without duplicates (index used by the audio route)."""
    seen = dict.fromkeys(s.instruction for s in exercise.steps)
    seen[texts.EXERCISE_DONE] = None
    return list(seen)


def create_app(
    settings: Settings | None = None,
    *,
    services_factory: Callable[[Settings], Services] = build_services,
    warm: Callable[[Services], None] = warm_up,
    background: bool = True,
) -> FastAPI:
    settings = settings or get_settings()
    prefer_local_models(settings)
    holder = ServiceHolder()

    def load() -> None:
        try:
            services = services_factory(settings)
            warm(services)
            holder.services = services
        except Exception as exc:  # shown to the user, the server keeps running
            logger.exception("model loading failed")
            holder.error = texts.API_LOAD_FAILED.format(detail=exc)
        finally:
            holder.ready.set()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if background:
            threading.Thread(target=load, name="voxemo-load", daemon=True).start()
        else:
            load()
        yield
        if holder.services is not None:
            holder.services.store.close()

    # No /docs: Swagger UI would load scripts from a CDN (everything must stay local).
    app = FastAPI(title="voxemo", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.holder = holder

    def services_or_503() -> Services:
        if holder.services is None:
            detail = texts.API_LOADING if holder.state == "loading" else holder.error
            raise HTTPException(503, detail=detail)
        return holder.services

    @app.get("/api/status")
    def status() -> dict:
        services = holder.services
        llm_problem = services.llm_problem() if services else None
        return {
            "state": holder.state,
            "message": texts.API_LOADING if holder.state == "loading" else holder.error,
            "llm": {"ok": services is not None and llm_problem is None, "message": llm_problem},
            "voice": bool(services and services.speaker is not None),
            "warnings": list(dict.fromkeys(services.warnings)) if services else [],
            "sample_rate": settings.sample_rate,
            "max_turn_s": settings.max_turn_s,
            "language": settings.language,
            "local_only": True,
            "notice": texts.UI_PRIVACY,
        }

    @app.get("/api/history")
    def history(demo: bool = Query(False, description="simulated week, clearly marked")) -> dict:
        if demo:
            return history_payload(simulated_interactions(), simulated=True)
        return history_payload(services_or_503().store.interactions())

    @app.delete("/api/history")
    def clear_history() -> dict:
        services = services_or_503()
        deleted = services.store.clear_history()
        for session in list(holder.sessions):
            if session.pipeline is not None:
                session.pipeline.forget_voice_baseline()
        return {"deleted": deleted}

    @app.get("/api/exercises")
    def exercises() -> list[dict]:
        return [exercise_payload(e) | {"spoken": spoken_texts(e)} for e in CATALOGUE]

    def find_exercise(exercise_id: str) -> Exercise:
        try:
            return get_exercise(exercise_id)
        except KeyError:
            raise HTTPException(404, detail=f"unknown exercise {exercise_id!r}") from None

    @app.get("/api/exercises/{exercise_id}")
    def exercise(exercise_id: str) -> dict:
        found = find_exercise(exercise_id)
        return exercise_payload(found) | {"spoken": spoken_texts(found)}

    @lru_cache(maxsize=64)
    def cached_speech(text: str) -> bytes | None:
        return services_or_503().speech_wav(text)

    @app.get("/api/exercises/{exercise_id}/speech/{index}")
    def exercise_speech(exercise_id: str, index: int) -> Response:
        spoken = spoken_texts(find_exercise(exercise_id))
        if not 0 <= index < len(spoken):
            raise HTTPException(404, detail="no such text")
        wav = cached_speech(spoken[index])
        if wav is None:
            raise HTTPException(503, detail=texts.TTS_UNAVAILABLE.format(detail="voix absente"))
        return Response(wav, media_type="audio/wav", headers={"Cache-Control": "max-age=86400"})

    @app.websocket("/api/turn")
    async def turn(websocket: WebSocket) -> None:
        if not origin_allowed(
            websocket.headers.get("origin"), websocket.headers.get("host"), settings.allowed_origins
        ):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        session = ConversationSession(
            websocket,
            holder,
            sample_rate=settings.sample_rate,
            max_turn_s=settings.max_turn_s,
            voice=settings.tts_enabled,
        )
        holder.sessions.add(session)
        await session.run()

    dist = Path(settings.web_dist)
    if (dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    else:

        @app.get("/", response_class=HTMLResponse)
        def not_built() -> str:
            return _NOT_BUILT

    return app


def __getattr__(name: str) -> FastAPI:
    """``uvicorn voxemo.api.app:app`` builds the app on first access."""
    if name == "app":
        return create_app()
    raise AttributeError(name)
