"""Relational schema.

Structured, queryable facts are columns; rich nested payloads (reports,
proposals, validation results) are JSONB documents next to them. The
``audit_events`` table is append-only by convention *and* by trigger (see the
initial migration): rows cannot be updated or deleted.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

JsonType = JSON().with_variant(JSONB(), "postgresql")


def now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JsonType, list[Any]: JsonType}  # noqa: RUF012


class DataBatch(Base):
    __tablename__ = "data_batches"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    dataset: Mapped[str] = mapped_column(String(40), index=True)
    role: Mapped[str] = mapped_column(String(12))  # reference | current
    rows: Mapped[int] = mapped_column(Integer)
    columns: Mapped[list[Any]] = mapped_column(JsonType)
    sha256: Mapped[str] = mapped_column(String(64))
    path: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(80), default="upload")
    active: Mapped[bool] = mapped_column(Boolean, default=True)  # for references: the one in use
    created_by: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class DeploymentRevision(Base):
    __tablename__ = "deployment_revisions"
    __table_args__ = (UniqueConstraint("dataset", "revision"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset: Mapped[str] = mapped_column(String(40), index=True)
    revision: Mapped[int] = mapped_column(Integer)
    state: Mapped[dict[str, Any]] = mapped_column(JsonType)
    active: Mapped[bool] = mapped_column(Boolean, default=False)
    reason: Mapped[str] = mapped_column(Text, default="")
    remediation_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    created_by: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class MonitoringRun(Base):
    __tablename__ = "monitoring_runs"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)  # report id
    dataset: Mapped[str] = mapped_column(String(40), index=True)
    batch_id: Mapped[str] = mapped_column(ForeignKey("data_batches.id"))
    reference_batch_id: Mapped[str] = mapped_column(ForeignKey("data_batches.id"))
    deployment_revision: Mapped[int] = mapped_column(Integer)
    model_version: Mapped[str] = mapped_column(String(80))
    reference_version: Mapped[str] = mapped_column(String(40))
    incident_detected: Mapped[bool] = mapped_column(Boolean)
    max_severity: Mapped[str] = mapped_column(String(12))
    triggered_count: Mapped[int] = mapped_column(Integer)
    report: Mapped[dict[str, Any]] = mapped_column(JsonType)
    performance: Mapped[dict[str, Any]] = mapped_column(JsonType)
    duration_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)


class Incident(Base):
    __tablename__ = "incidents"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    dataset: Mapped[str] = mapped_column(String(40), index=True)
    monitoring_run_id: Mapped[str] = mapped_column(ForeignKey("monitoring_runs.id"), unique=True)
    batch_id: Mapped[str] = mapped_column(ForeignKey("data_batches.id"))
    incident_type: Mapped[str] = mapped_column(String(40))  # dominant monitor family, not a cause
    severity: Mapped[str] = mapped_column(String(12), index=True)
    status: Mapped[str] = mapped_column(String(24), index=True)
    model_version: Mapped[str] = mapped_column(String(80))
    reference_version: Mapped[str] = mapped_column(String(40))
    summary: Mapped[str] = mapped_column(Text)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)


class EvidenceRecord(Base):
    __tablename__ = "evidence"
    __table_args__ = (UniqueConstraint("incident_id", "evidence_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.id"), index=True)
    investigation_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    evidence_id: Mapped[str] = mapped_column(String(60))
    source: Mapped[str] = mapped_column(String(80))
    level: Mapped[str] = mapped_column(String(24))
    severity: Mapped[str] = mapped_column(String(12))
    feature: Mapped[str | None] = mapped_column(String(80), nullable=True)
    statement: Mapped[str] = mapped_column(Text)
    finding_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    data: Mapped[dict[str, Any]] = mapped_column(JsonType)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Investigation(Base):
    __tablename__ = "investigations"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.id"), index=True)
    status: Mapped[str] = mapped_column(String(24), index=True)
    requested_by: Mapped[str] = mapped_column(String(80))
    use_llm: Mapped[bool] = mapped_column(Boolean, default=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    llm_status: Mapped[str | None] = mapped_column(String(24), nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String(80), nullable=True)
    model_version: Mapped[str] = mapped_column(String(80))
    reference_version: Mapped[str] = mapped_column(String(40))
    deployment_revision: Mapped[int] = mapped_column(Integer)
    diagnosis: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)
    rule_diagnosis: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)
    llm_ranking_stats: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)
    recommendation: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)
    policy_decision: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)
    explanation: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)
    runbook_refs: Mapped[list[Any]] = mapped_column(JsonType, default=list)
    llm_calls: Mapped[list[Any]] = mapped_column(JsonType, default=list)
    events: Mapped[list[Any]] = mapped_column(JsonType, default=list)
    errors: Mapped[list[Any]] = mapped_column(JsonType, default=list)
    trace_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ToolTraceRecord(Base):
    __tablename__ = "tool_traces"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)  # call id
    investigation_id: Mapped[str] = mapped_column(ForeignKey("investigations.id"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    tool: Mapped[str] = mapped_column(String(80))
    requested_by: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(12))
    requested_args: Mapped[dict[str, Any]] = mapped_column(JsonType)
    validated_args: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)
    output: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_ids: Mapped[list[Any]] = mapped_column(JsonType, default=list)
    duration_ms: Mapped[float] = mapped_column(Float)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class HypothesisRecord(Base):
    __tablename__ = "hypotheses"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    investigation_id: Mapped[str] = mapped_column(ForeignKey("investigations.id"), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    hypothesis_id: Mapped[str] = mapped_column(String(60))
    root_cause: Mapped[str] = mapped_column(String(40))
    confidence: Mapped[float] = mapped_column(Float)
    ranked_by: Mapped[str] = mapped_column(String(24))
    supporting_evidence_ids: Mapped[list[Any]] = mapped_column(JsonType)
    contradicting_evidence_ids: Mapped[list[Any]] = mapped_column(JsonType)
    payload: Mapped[dict[str, Any]] = mapped_column(JsonType)


class Remediation(Base):
    __tablename__ = "remediations"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.id"), index=True)
    investigation_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    dataset: Mapped[str] = mapped_column(String(40))
    state: Mapped[str] = mapped_column(String(24), index=True)
    action: Mapped[str] = mapped_column(String(40))
    risk_level: Mapped[str] = mapped_column(String(12))
    proposal: Mapped[dict[str, Any]] = mapped_column(JsonType)
    validation: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)
    policy_decision: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)
    verification: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)
    base_revision: Mapped[int] = mapped_column(
        Integer
    )  # deployment revision it was validated against
    executed_revision: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rollback_revision: Mapped[int | None] = mapped_column(Integer, nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)  # bumped on every state change
    created_by: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)


class Approval(Base):
    __tablename__ = "approvals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # UNIQUE: a remediation can be decided exactly once, enforced by the database itself.
    remediation_id: Mapped[str] = mapped_column(ForeignKey("remediations.id"), unique=True)
    decision: Mapped[str] = mapped_column(String(12))  # approved | rejected
    reviewer: Mapped[str] = mapped_column(String(80))
    reviewer_role: Mapped[str] = mapped_column(String(24))
    comment: Mapped[str] = mapped_column(Text, default="")
    remediation_version: Mapped[int] = mapped_column(Integer)
    idempotency_key: Mapped[str | None] = mapped_column(String(80), nullable=True)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_entity", "entity_type", "entity_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)
    actor: Mapped[str] = mapped_column(String(80))
    actor_role: Mapped[str] = mapped_column(String(24))
    action: Mapped[str] = mapped_column(String(60), index=True)
    entity_type: Mapped[str] = mapped_column(String(24))
    entity_id: Mapped[str] = mapped_column(String(60))
    incident_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    outcome: Mapped[str] = mapped_column(String(16), default="ok")  # ok | denied | failed
    detail: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
