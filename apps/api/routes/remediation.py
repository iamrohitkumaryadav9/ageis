"""Remediation lifecycle. Approval, rejection and rollback require the approver role, checked
server-side from the bearer token. The request body cannot express 'approved' or name a reviewer."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Response
from sqlalchemy import select

from aegis.remediation.lifecycle import allowed_events
from aegis.schemas.remediation import RemediationState
from aegis.services.core import AegisService
from aegis.services.errors import NotFoundError, Principal
from aegis.storage import models as m
from apps.api.deps import get_principal, get_service, require
from apps.api.schemas import (
    ApprovalOut,
    DecisionRequest,
    RemediationDetail,
    RemediationOut,
    RollbackRequest,
)

router = APIRouter(tags=["remediation"])


def _detail(service: AegisService, remediation_id: str) -> RemediationDetail:
    with service.db.session() as session:
        row = session.get(m.Remediation, remediation_id)
        if row is None:
            raise NotFoundError(f"remediation {remediation_id!r} not found")
        approval = session.scalars(
            select(m.Approval).where(m.Approval.remediation_id == remediation_id)
        ).first()
        return RemediationDetail(
            **RemediationOut.model_validate(row).model_dump(),
            approval=ApprovalOut.model_validate(approval) if approval else None,
            allowed_events=[e.value for e in allowed_events(RemediationState(row.state))],
            executed=row.executed_revision is not None,
        )


@router.post(
    "/incidents/{incident_id}/remediation", response_model=RemediationDetail, status_code=201
)
def create_remediation(
    incident_id: str,
    response: Response,
    service: AegisService = Depends(get_service),
    principal: Principal = Depends(require("operator")),
) -> RemediationDetail:
    row, created = service.create_remediation(principal, incident_id)
    if not created:
        response.status_code = 200
    return _detail(service, row.id)


@router.get("/remediation/{remediation_id}", response_model=RemediationDetail)
def get_remediation(
    remediation_id: str,
    service: AegisService = Depends(get_service),
    _: Principal = Depends(require("viewer")),
) -> RemediationDetail:
    return _detail(service, remediation_id)


@router.post("/remediation/{remediation_id}/validate", response_model=RemediationDetail)
def validate(
    remediation_id: str,
    service: AegisService = Depends(get_service),
    principal: Principal = Depends(require("operator")),
) -> RemediationDetail:
    service.validate_remediation(principal, remediation_id)
    return _detail(service, remediation_id)


@router.post("/remediation/{remediation_id}/approve", response_model=RemediationDetail)
def approve(
    remediation_id: str,
    body: DecisionRequest,
    service: AegisService = Depends(get_service),
    principal: Principal = Depends(get_principal),
    idempotency_key: str | None = Header(default=None, max_length=80),
) -> RemediationDetail:
    service.decide(
        principal,
        remediation_id,
        "approved",
        expected_version=body.expected_version,
        comment=body.comment,
        idempotency_key=idempotency_key,
    )
    return _detail(service, remediation_id)


@router.post("/remediation/{remediation_id}/reject", response_model=RemediationDetail)
def reject(
    remediation_id: str,
    body: DecisionRequest,
    service: AegisService = Depends(get_service),
    principal: Principal = Depends(get_principal),
    idempotency_key: str | None = Header(default=None, max_length=80),
) -> RemediationDetail:
    service.decide(
        principal,
        remediation_id,
        "rejected",
        expected_version=body.expected_version,
        comment=body.comment,
        idempotency_key=idempotency_key,
    )
    return _detail(service, remediation_id)


@router.post("/remediation/{remediation_id}/rollback", response_model=RemediationDetail)
def rollback(
    remediation_id: str,
    body: RollbackRequest,
    service: AegisService = Depends(get_service),
    principal: Principal = Depends(get_principal),
) -> RemediationDetail:
    service.rollback(principal, remediation_id, body.reason)
    return _detail(service, remediation_id)
