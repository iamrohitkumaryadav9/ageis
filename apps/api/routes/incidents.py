"""Incidents and investigations."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import func, select

from aegis.services.core import AegisService
from aegis.services.errors import NotFoundError, Principal
from aegis.storage import models as m
from apps.api.deps import get_service, require
from apps.api.schemas import (
    EvidenceOut,
    HypothesisOut,
    IncidentDetail,
    IncidentOut,
    InvestigateRequest,
    InvestigationDetail,
    InvestigationSummary,
    Page,
    RemediationOut,
    ToolTraceOut,
)

router = APIRouter(tags=["incidents"])


@router.get("/incidents", response_model=Page[IncidentOut])
def list_incidents(
    limit: int = Query(25, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None, max_length=24),
    severity: str | None = Query(None, max_length=12),
    dataset: str | None = Query(None, max_length=40),
    service: AegisService = Depends(get_service),
    _: Principal = Depends(require("viewer")),
) -> Page[IncidentOut]:
    filters = []
    if status:
        filters.append(m.Incident.status == status)
    if severity:
        filters.append(m.Incident.severity == severity)
    if dataset:
        filters.append(m.Incident.dataset == dataset)
    with service.db.session() as session:
        total = session.scalar(select(func.count()).select_from(m.Incident).where(*filters)) or 0
        rows = session.scalars(
            select(m.Incident)
            .where(*filters)
            .order_by(m.Incident.detected_at.desc())
            .limit(limit)
            .offset(offset)
        ).all()
        return Page[IncidentOut](
            items=[IncidentOut.model_validate(r) for r in rows],
            total=total,
            limit=limit,
            offset=offset,
        )


@router.get("/incidents/{incident_id}", response_model=IncidentDetail)
def get_incident(
    incident_id: str,
    service: AegisService = Depends(get_service),
    _: Principal = Depends(require("viewer")),
) -> IncidentDetail:
    with service.db.session() as session:
        incident = session.get(m.Incident, incident_id)
        if incident is None:
            raise NotFoundError(f"incident {incident_id!r} not found")
        run = session.get(m.MonitoringRun, incident.monitoring_run_id)
        evidence = session.scalars(
            select(m.EvidenceRecord)
            .where(m.EvidenceRecord.incident_id == incident_id)
            .order_by(m.EvidenceRecord.id)
        ).all()
        investigations = session.scalars(
            select(m.Investigation)
            .where(m.Investigation.incident_id == incident_id)
            .order_by(m.Investigation.started_at.desc())
        ).all()
        remediations = session.scalars(
            select(m.Remediation)
            .where(m.Remediation.incident_id == incident_id)
            .order_by(m.Remediation.created_at.desc())
        ).all()
        findings = [f for f in run.report["findings"] if f["triggered"]]
        return IncidentDetail(
            **IncidentOut.model_validate(incident).model_dump(),
            findings=findings,
            skipped_checks=run.report.get("skipped", []),
            performance=run.performance,
            evidence=[EvidenceOut.model_validate(e) for e in evidence],
            investigations=[InvestigationSummary.model_validate(i) for i in investigations],
            remediations=[RemediationOut.model_validate(r) for r in remediations],
        )


@router.post(
    "/incidents/{incident_id}/investigate", response_model=InvestigationSummary, status_code=202
)
def investigate(
    incident_id: str,
    response: Response,
    body: InvestigateRequest | None = None,
    service: AegisService = Depends(get_service),
    principal: Principal = Depends(require("operator")),
) -> InvestigationSummary:
    body = body or InvestigateRequest()
    row = service.start_investigation(principal, incident_id, use_llm=body.use_llm, wait=body.wait)
    if row.status != "running":
        response.status_code = 200
    return InvestigationSummary.model_validate(row)


def _detail(session, row: m.Investigation) -> InvestigationDetail:
    traces = session.scalars(
        select(m.ToolTraceRecord)
        .where(m.ToolTraceRecord.investigation_id == row.id)
        .order_by(m.ToolTraceRecord.seq)
    ).all()
    hypotheses = session.scalars(
        select(m.HypothesisRecord)
        .where(m.HypothesisRecord.investigation_id == row.id)
        .order_by(m.HypothesisRecord.rank)
    ).all()
    return InvestigationDetail(
        **{
            k: getattr(row, k)
            for k in InvestigationDetail.model_fields
            if k not in ("tool_traces", "hypotheses")
        },
        tool_traces=[ToolTraceOut.model_validate(t) for t in traces],
        hypotheses=[HypothesisOut.model_validate(h) for h in hypotheses],
    )


@router.get("/investigations/{investigation_id}", response_model=InvestigationDetail)
def get_investigation(
    investigation_id: str,
    service: AegisService = Depends(get_service),
    _: Principal = Depends(require("viewer")),
) -> InvestigationDetail:
    with service.db.session() as session:
        row = session.get(m.Investigation, investigation_id)
        if row is None:
            raise NotFoundError(f"investigation {investigation_id!r} not found")
        return _detail(session, row)


@router.post("/investigations/{investigation_id}/cancel", response_model=InvestigationSummary)
def cancel_investigation(
    investigation_id: str,
    service: AegisService = Depends(get_service),
    principal: Principal = Depends(require("operator")),
) -> InvestigationSummary:
    return InvestigationSummary.model_validate(
        service.cancel_investigation(principal, investigation_id)
    )
