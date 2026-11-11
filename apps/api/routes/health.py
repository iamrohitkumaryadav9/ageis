"""Liveness and readiness. Liveness says the process answers; readiness says it can do its job.
The LLM is reported but never makes the service 'not ready': it is optional by design."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from sqlalchemy import select

from aegis import __doc__ as about
from aegis.services.core import AegisService
from aegis.storage import models as m
from apps.api.deps import get_service
from apps.api.schemas import ComponentStatus, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "aegis", "about": (about or "").strip()}


@router.get("/ready", response_model=ReadyResponse)
def ready(response: Response, service: AegisService = Depends(get_service)) -> ReadyResponse:
    components: dict[str, ComponentStatus] = {}
    db_ok = service.db.ping()
    components["database"] = ComponentStatus(status="ok" if db_ok else "unavailable", required=True)
    # Models: every active deployment must point at a registered model version.
    problems: list[str] = []
    deployed = 0
    if db_ok:
        with service.db.session() as session:
            active = session.scalars(
                select(m.DeploymentRevision).where(m.DeploymentRevision.active)
            ).all()
        deployed = len(active)
        for row in active:
            try:
                service.registry.get(row.dataset, row.state["model_version"])
            except (KeyError, ValueError):
                problems.append(
                    f"{row.dataset}: {row.state.get('model_version')} is not registered"
                )
    if db_ok and deployed == 0:
        problems.append("no active deployment; run `make bootstrap`")
    components["models"] = ComponentStatus(
        status="ok" if db_ok and not problems else "unavailable",
        required=True,
        detail="; ".join(problems)
        if problems
        else (f"{deployed} active deployment(s)" if db_ok else "unknown"),
    )
    components["runbooks"] = ComponentStatus(
        status="ok",
        required=False,
        detail=f"{len(service.retriever.passages)} passages, corpus {service.retriever.corpus_version}",
    )
    if service.llm is None:
        components["llm"] = ComponentStatus(
            status="disabled", required=False, detail="LLM disabled by configuration"
        )
    else:
        llm = service.llm.health()
        components["llm"] = ComponentStatus(
            status="ok" if llm.available else "unavailable",
            required=False,
            detail=(
                f"{llm.active_model} via Ollama {llm.version}"
                + (f" ({llm.detail})" if llm.detail else "")
            )
            if llm.available
            else (llm.detail or "Ollama unreachable")
            + "; deterministic diagnosis remains available",
        )
    is_ready = all(c.status == "ok" for c in components.values() if c.required)
    if not is_ready:
        response.status_code = 503
    return ReadyResponse(status="ready" if is_ready else "not_ready", components=components)
