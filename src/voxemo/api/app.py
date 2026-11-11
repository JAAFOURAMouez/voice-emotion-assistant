"""FastAPI application: ``python -m voxemo.api`` (or ``uvicorn voxemo.api.app:app``)."""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from voxemo import texts
from voxemo.api.access import (
    SUBPROTOCOL,
    WS_ACCESS_DENIED,
    bearer,
    subprotocols,
    token_matches,
)
from voxemo.api.payloads import (
    exercise_payload,
    history_payload,
    simulated_interactions,
)
from voxemo.api.services import Services, build_services, warm_up
from voxemo.api.session import ConversationSession, ServiceHolder, origin_allowed
from voxemo.config import Settings, get_settings
from voxemo.llm.client import LLMUnavailableError
from voxemo.llm.prompts import ChatMessage, system_prompt
from voxemo.llm.selection import LlmSelector, privacy_notice
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


class LlmRequest(BaseModel):
    provider: str = Field(max_length=40)
    model: str = Field("", max_length=200)
    api_key: str | None = Field(None, max_length=512)
    base_url: str | None = Field(None, max_length=500)


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
    if settings.allowed_origin_list and not settings.access_token:
        logger.warning(
            "ALLOWED_ORIGINS is set but ACCESS_TOKEN is empty: anyone reaching this "
            "server can read and erase the history. Set ACCESS_TOKEN."
        )

    @app.middleware("http")
    async def require_access_code(request: Request, call_next: Callable) -> Response:
        if (
            request.url.path.startswith("/api/")
            and request.method != "OPTIONS"  # CORS preflight carries no credentials
            and not token_matches(
                bearer(request.headers.get("authorization")), settings.access_token
            )
        ):
            return JSONResponse({"detail": texts.API_ACCESS_REQUIRED}, status_code=401)
        return await call_next(request)

    if settings.allowed_origin_list:  # pages served elsewhere (e.g. a static host)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.allowed_origin_list,
            allow_methods=["GET", "PUT", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
            max_age=600,
        )

    def services_or_503() -> Services:
        if holder.services is None:
            detail = texts.API_LOADING if holder.state == "loading" else holder.error
            raise HTTPException(503, detail=detail)
        return holder.services

    @app.get("/api/status")
    def status() -> dict:
        services = holder.services
        llm_problem = services.llm_problem() if services else None
        llm = {"ok": services is not None and llm_problem is None, "message": llm_problem}
        return {
            "state": holder.state,
            "message": texts.API_LOADING if holder.state == "loading" else holder.error,
            "llm": llm | (services.llm_state() if services else {}),
            "voice": bool(services and services.speaker is not None),
            "warnings": list(dict.fromkeys(services.warnings)) if services else [],
            "sample_rate": settings.sample_rate,
            "max_turn_s": settings.max_turn_s,
            "language": settings.language,
            "local_only": services is None or services.llm_state()["local"],
            "notice": privacy_notice(services.selector if services else None),
        }

    def selector_or_404() -> LlmSelector:
        selector = services_or_503().selector
        if selector is None:
            raise HTTPException(404, detail="model choice unavailable")
        return selector

    @app.get("/api/llm")
    def llm_settings() -> dict:
        return selector_or_404().describe()

    @app.post("/api/llm/models")
    def llm_models(request: LlmRequest) -> dict:
        """Models a provider offers, to fill the list in the settings panel."""
        selector = selector_or_404()
        try:
            models = selector.remote_models(
                request.provider, api_key=request.api_key, base_url=request.base_url
            )
        except (ValueError, LLMUnavailableError) as exc:
            raise HTTPException(400, detail=str(exc)) from None
        return {"models": models}

    @app.put("/api/llm")
    def choose_llm(request: LlmRequest) -> dict:
        selector = selector_or_404()
        try:
            choice = selector.select(
                request.provider, request.model, api_key=request.api_key, base_url=request.base_url
            )
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from None
        if choice.local:  # load it now rather than on the next turn
            prompt = [ChatMessage("system", system_prompt(settings.language))]

            def warm() -> None:
                try:
                    selector.warm_up(prompt)
                except LLMUnavailableError as exc:
                    logger.warning("warm-up of %s failed: %s", choice.model, exc)

            threading.Thread(target=warm, name="voxemo-llm-warm", daemon=True).start()
        return selector.describe()

    @app.delete("/api/llm/keys/{provider}")
    def forget_key(provider: str) -> dict:
        selector = selector_or_404()
        selector.forget_key(provider)
        return selector.describe()

    @app.get("/api/history")
    def history(
        demo: bool = Query(False, description="simulated week, clearly marked"),
        days: int | None = Query(None, ge=1, le=366, description="only the last N days"),
    ) -> dict:
        if demo:
            return history_payload(simulated_interactions(), simulated=True)
        since = datetime.now(UTC) - timedelta(days=days) if days else None
        return history_payload(services_or_503().store.interactions(since=since))

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
        wants_subprotocol, code = subprotocols(websocket.headers.get("sec-websocket-protocol"))
        await websocket.accept(subprotocol=SUBPROTOCOL if wants_subprotocol else None)
        if not token_matches(code, settings.access_token):
            # Accepted first so that the page can tell "wrong code" from "server down".
            await websocket.close(code=WS_ACCESS_DENIED, reason="access code")
            return
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
