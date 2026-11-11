"""Typed API responses and request bodies."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Orm(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class Page[T](BaseModel):
    items: list[T]
    total: int
    limit: int
    offset: int


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class BatchOut(Orm):
    id: str
    dataset: str
    role: str
    rows: int
    columns: list[str]
    sha256: str
    source: str
    active: bool
    created_by: str
    created_at: datetime


class MonitorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str = Field(max_length=40)
    batch_id: str | None = Field(default=None, max_length=40)


class MonitoringRunOut(Orm):
    id: str
    dataset: str
    batch_id: str
    deployment_revision: int
    model_version: str
    reference_version: str
    incident_detected: bool
    max_severity: str
    triggered_count: int
    duration_ms: float
    created_at: datetime


class IncidentOut(Orm):
    id: str
    dataset: str
    incident_type: str
    severity: str
    status: str
    model_version: str
    reference_version: str
    summary: str
    detected_at: datetime
    updated_at: datetime
    monitoring_run_id: str
    batch_id: str


class MonitorResponse(BaseModel):
    run: MonitoringRunOut
    incident: IncidentOut | None
    replayed: bool


class EvidenceOut(Orm):
    evidence_id: str
    source: str
    level: str
    severity: str
    feature: str | None
    statement: str
    finding_id: str | None
    investigation_id: str | None
    data: dict[str, Any]


class InvestigationSummary(Orm):
    id: str
    incident_id: str
    status: str
    requested_by: str
    use_llm: bool
    llm_status: str | None
    llm_model: str | None
    started_at: datetime
    finished_at: datetime | None
    duration_ms: float | None


class RemediationOut(Orm):
    id: str
    incident_id: str
    investigation_id: str | None
    dataset: str
    state: str
    action: str
    risk_level: str
    proposal: dict[str, Any]
    validation: dict[str, Any] | None
    policy_decision: dict[str, Any] | None
    verification: dict[str, Any] | None
    base_revision: int
    executed_revision: int | None
    rollback_revision: int | None
    version: int
    created_by: str
    created_at: datetime
    updated_at: datetime


class ApprovalOut(Orm):
    decision: str
    reviewer: str
    reviewer_role: str
    comment: str
    remediation_version: int
    decided_at: datetime


class RemediationDetail(RemediationOut):
    approval: ApprovalOut | None = None
    allowed_events: list[str] = Field(default_factory=list)
    #: True only once a deployment revision was actually written for this remediation.
    executed: bool = False


class IncidentDetail(IncidentOut):
    findings: list[dict[str, Any]]
    skipped_checks: list[dict[str, Any]]
    performance: dict[str, Any]
    evidence: list[EvidenceOut]
    investigations: list[InvestigationSummary]
    remediations: list[RemediationOut]


class ToolTraceOut(Orm):
    id: str
    seq: int
    tool: str
    requested_by: str
    status: str
    requested_args: dict[str, Any]
    validated_args: dict[str, Any] | None
    output: dict[str, Any] | None
    error: str | None
    evidence_ids: list[str]
    duration_ms: float
    started_at: datetime


class HypothesisOut(Orm):
    rank: int
    hypothesis_id: str
    root_cause: str
    confidence: float
    ranked_by: str
    supporting_evidence_ids: list[str]
    contradicting_evidence_ids: list[str]
    payload: dict[str, Any]


class InvestigationDetail(InvestigationSummary):
    cancel_requested: bool
    model_version: str
    reference_version: str
    deployment_revision: int
    diagnosis: dict[str, Any] | None
    rule_diagnosis: dict[str, Any] | None
    llm_ranking_stats: dict[str, Any] | None
    recommendation: dict[str, Any] | None
    policy_decision: dict[str, Any] | None
    explanation: dict[str, Any] | None
    runbook_refs: list[dict[str, Any]]
    llm_calls: list[dict[str, Any]]
    events: list[dict[str, Any]]
    errors: list[str]
    trace_id: str | None
    tool_traces: list[ToolTraceOut]
    hypotheses: list[HypothesisOut]


class InvestigateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    use_llm: bool = True
    wait: bool = False


class DecisionRequest(BaseModel):
    """Note what is *absent*: there is no 'approved' or 'reviewer' field. The decision is the
    endpoint that was called; the reviewer is the authenticated caller."""

    model_config = ConfigDict(extra="forbid")
    expected_version: int = Field(ge=1, description="remediation.version the reviewer looked at")
    comment: str = Field(default="", max_length=2000)


class RollbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(min_length=3, max_length=500)


class AuditEventOut(Orm):
    id: int
    at: datetime
    actor: str
    actor_role: str
    action: str
    entity_type: str
    entity_id: str
    incident_id: str | None
    outcome: str
    detail: dict[str, Any]


class ComponentStatus(BaseModel):
    status: Literal["ok", "unavailable", "disabled", "degraded"]
    required: bool
    detail: str = ""


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    components: dict[str, ComponentStatus]
