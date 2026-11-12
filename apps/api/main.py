"""FastAPI application factory."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from starlette.exceptions import HTTPException as StarletteHTTPException

from aegis.config import Settings, get_settings
from aegis.observability.logging import configure_logging, get_logger
from aegis.observability.tracing import setup_tracing
from aegis.services.core import AegisService
from aegis.services.errors import ServiceError
from aegis.storage.db import Database
from apps.api.deps import parse_tokens
from apps.api.routes import data, evaluation, health, incidents, observability, remediation

log = get_logger("aegis.api")


def _error(
    request: Request, status: int, code: str, message: str, details: dict | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": details or {},
                "request_id": getattr(request.state, "request_id", None),
            }
        },
    )


def create_app(settings: Settings | None = None, service: AegisService | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)
    setup_tracing(settings.otlp_endpoint)
    tokens = parse_tokens(settings.api_tokens)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owns = service is None
        app.state.service = service or AegisService(settings, Database(settings.database_url))
        if not tokens:
            log.warning(
                "no_api_tokens_configured", detail="every authenticated endpoint will return 401"
            )
        yield
        if owns:
            app.state.service.shutdown()
            app.state.service.db.dispose()

    app = FastAPI(
        title="AEGIS - Autonomous ML Incident Commander",
        version="0.1.0",
        lifespan=lifespan,
        description="Detection, investigation, diagnosis and controlled remediation of ML incidents.",
    )
    app.state.settings, app.state.tokens = settings, tokens
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:8080"],
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = (
            request.headers.get("x-request-id", "")[:64] or uuid.uuid4().hex[:16]
        )
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > settings.max_upload_bytes + 65536:
            return _error(
                request,
                413,
                "payload_too_large",
                f"request body exceeds {settings.max_upload_bytes} bytes",
            )
        response = await call_next(request)
        response.headers["x-request-id"] = request.state.request_id
        return response

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        return _error(request, exc.status, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        problems = [
            {"field": ".".join(str(p) for p in e["loc"]), "problem": e["msg"]}
            for e in exc.errors()[:10]
        ]
        return _error(
            request, 422, "invalid_request", "request validation failed", {"problems": problems}
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed", 401: "unauthenticated"}.get(
            exc.status_code, "http_error"
        )
        return _error(request, exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        from sqlalchemy.exc import OperationalError

        if isinstance(exc, OperationalError):
            log.error("database_unavailable", path=request.url.path)
            return _error(
                request,
                503,
                "database_unavailable",
                "the database is not reachable; no state was changed",
            )
        log.error("unhandled_error", path=request.url.path, error=repr(exc)[:500])
        return _error(request, 500, "internal_error", "an unexpected error occurred")

    for module in (health, data, incidents, remediation, observability, evaluation):
        app.include_router(module.router)
    FastAPIInstrumentor.instrument_app(app, excluded_urls="health,metrics")
    return app


def app_factory() -> FastAPI:  # entry point for `uvicorn apps.api.main:app_factory --factory`
    return create_app()
