"""The AEGIS application service.

Every state change in the system happens in a method of this class, inside one
database transaction, with the affected row locked (``SELECT ... FOR UPDATE``)
and an audit event written in the same transaction. Authorisation for
approval, rejection and rollback is checked here, on the server, against the
authenticated principal; nothing in a request body, the UI or an LLM output can
stand in for it.
"""

from __future__ import annotations

import hashlib
import io
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from aegis.agents.graph import AgentRuntime, run_investigation
from aegis.agents.ollama import OllamaClient
from aegis.agents.tools import ToolContext, ToolExecutor
from aegis.config import Settings
from aegis.diagnosis.evidence import build_evidence
from aegis.ingestion.contracts import DataContract, derive_contract
from aegis.ingestion.datasets import DATASETS, TARGET, DatasetSpec
from aegis.ingestion.splits import read_split
from aegis.models.registry import ModelRegistry
from aegis.monitoring.engine import MonitoringInput, MonitoringResult, run_monitoring
from aegis.observability.logging import get_logger
from aegis.observability.metrics import (
    DENIED_ACTIONS,
    INCIDENTS,
    MONITORING_SECONDS,
    REMEDIATION_TRANSITIONS,
)
from aegis.observability.tracing import current_trace_id, span
from aegis.rag.retriever import RunbookRetriever
from aegis.remediation.executor import StaleProposalError, state_to_execute
from aegis.remediation.lifecycle import Event, IllegalTransitionError, next_state
from aegis.remediation.policy import PolicyEngine
from aegis.remediation.validation import validate_proposal
from aegis.schemas.deployment import DeploymentState
from aegis.schemas.diagnosis import Diagnosis, Evidence
from aegis.schemas.findings import Severity
from aegis.schemas.remediation import (
    STATE_CHANGING,
    ActionType,
    PolicyOutcome,
    RemediationProposal,
    RemediationState,
    ValidationResult,
)
from aegis.services.errors import (
    ConflictError,
    ForbiddenError,
    InvalidInputError,
    NotFoundError,
    Principal,
)
from aegis.storage import models as m
from aegis.storage.db import Database

log = get_logger(__name__)
S = RemediationState
OPEN_REMEDIATION_STATES = {
    S.PROPOSED,
    S.VALIDATING,
    S.VALIDATED,
    S.PENDING_APPROVAL,
    S.APPROVED,
    S.EXECUTED,
}


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.now(UTC)


class AegisService:
    def __init__(self, settings: Settings, db: Database, llm: OllamaClient | None = None) -> None:
        self.settings = settings
        self.db = db
        self.registry = ModelRegistry(settings.model_dir)
        self.llm = (
            llm if llm is not None else (OllamaClient(settings) if settings.llm_enabled else None)
        )
        self.retriever = RunbookRetriever(
            settings.runbook_dir,
            embedder=self.llm if settings.rag_method != "lexical" else None,
            cache_dir=settings.artifacts_dir / "rag",
            default_method=settings.rag_method,
        )
        self.policy = PolicyEngine()
        self._frames: dict[str, pd.DataFrame] = {}
        self._contracts: dict[str, DataContract] = {}
        self._lock = threading.Lock()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aegis-investigation")

    # ------------------------------------------------------------------ audit
    def audit(
        self,
        session: Session,
        principal: Principal,
        action: str,
        entity_type: str,
        entity_id: str,
        *,
        incident_id: str | None = None,
        outcome: str = "ok",
        **detail: Any,
    ) -> None:
        session.add(
            m.AuditEvent(
                actor=principal.identity,
                actor_role=principal.role,
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                incident_id=incident_id,
                outcome=outcome,
                detail=detail,
            )
        )

    def audit_denied(
        self, principal: Principal, action: str, entity_type: str, entity_id: str, reason: str
    ) -> None:
        """Record a refused privileged action in its own transaction (the caller's is rolling back)."""
        DENIED_ACTIONS.labels(reason.split(":")[0][:40]).inc()
        with self.db.session() as session:
            incident_id = None
            if entity_type == "remediation":
                row = session.get(m.Remediation, entity_id)
                incident_id = row.incident_id if row else None
            self.audit(
                session,
                principal,
                action,
                entity_type,
                entity_id,
                incident_id=incident_id,
                outcome="denied",
                reason=reason,
            )

    # ------------------------------------------------------------------- data
    def _spec(self, dataset: str) -> DatasetSpec:
        if dataset not in DATASETS:
            raise InvalidInputError(f"unknown dataset {dataset!r}", known=sorted(DATASETS))
        return DATASETS[dataset]

    def _batch_dir(self) -> Path:
        path = self.settings.artifacts_dir / "batches"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def register_batch(
        self,
        principal: Principal,
        dataset: str,
        role: str,
        frame: pd.DataFrame,
        source: str = "upload",
    ) -> m.DataBatch:
        spec = self._spec(dataset)
        if role not in ("reference", "current"):
            raise InvalidInputError("role must be 'reference' or 'current'")
        if len(frame) == 0:
            raise InvalidInputError("the batch has no rows")
        if len(frame) > self.settings.max_upload_rows:
            raise InvalidInputError(
                f"the batch has {len(frame)} rows; the limit is {self.settings.max_upload_rows}"
            )
        if len(frame.columns) > 200:
            raise InvalidInputError("the batch has more than 200 columns")
        frame = frame.copy()
        frame.columns = [str(c)[:120] for c in frame.columns]
        if len(set(frame.columns)) != len(frame.columns):
            raise InvalidInputError("duplicate column names")
        if role == "reference":
            missing = [c for c in [*spec.features, TARGET] if c not in frame.columns]
            if missing:
                raise InvalidInputError(
                    "a reference window must contain every contracted feature and the target",
                    missing=missing,
                )
            if frame[TARGET].isna().any() or not set(frame[TARGET].unique()) <= {0, 1}:
                raise InvalidInputError("reference target must be fully labelled with 0/1")
        for column in frame.columns:  # mixed-type object columns cannot be stored reliably
            if frame[column].dtype == object:
                frame[column] = frame[column].where(frame[column].isna(), frame[column].astype(str))
        batch_id = _id("BAT")
        path = self._batch_dir() / f"{batch_id}.parquet"
        frame.reset_index(drop=True).to_parquet(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.db.session() as session:
            if role == "reference":
                for old in session.scalars(
                    select(m.DataBatch).where(
                        m.DataBatch.dataset == dataset,
                        m.DataBatch.role == "reference",
                        m.DataBatch.active,
                    )
                ):
                    old.active = False
            batch = m.DataBatch(
                id=batch_id,
                dataset=dataset,
                role=role,
                rows=len(frame),
                columns=list(frame.columns),
                sha256=digest,
                path=str(path),
                source=source,
                active=True,
                created_by=principal.identity,
            )
            session.add(batch)
            self.audit(
                session,
                principal,
                f"data.{role}.registered",
                "batch",
                batch_id,
                dataset=dataset,
                rows=len(frame),
                sha256=digest,
                source=source,
            )
        return batch

    def parse_csv(self, payload: bytes) -> pd.DataFrame:
        if len(payload) > self.settings.max_upload_bytes:
            raise InvalidInputError(f"upload exceeds {self.settings.max_upload_bytes} bytes")
        try:
            return pd.read_csv(io.BytesIO(payload), nrows=self.settings.max_upload_rows + 1)
        except Exception as exc:
            raise InvalidInputError(f"could not parse CSV: {type(exc).__name__}") from None

    def _frame(self, batch: m.DataBatch) -> pd.DataFrame:
        with self._lock:
            if batch.id not in self._frames:
                path = Path(batch.path).resolve()
                if self._batch_dir().resolve() not in path.parents:
                    raise InvalidInputError("batch path is outside the batch store")
                if hashlib.sha256(path.read_bytes()).hexdigest() != batch.sha256:
                    raise ConflictError(
                        "stored batch does not match its recorded checksum", batch_id=batch.id
                    )
                if len(self._frames) > 16:
                    self._frames.pop(next(iter(self._frames)))
                self._frames[batch.id] = pd.read_parquet(path)
            return self._frames[batch.id]

    def _reference(self, session: Session, dataset: str) -> m.DataBatch:
        batch = session.scalars(
            select(m.DataBatch)
            .where(
                m.DataBatch.dataset == dataset, m.DataBatch.role == "reference", m.DataBatch.active
            )
            .order_by(m.DataBatch.created_at.desc())
        ).first()
        if batch is None:
            raise ConflictError(
                f"no reference window registered for {dataset}; run bootstrap or POST /data/reference"
            )
        return batch

    def _contract(self, spec: DatasetSpec, reference: m.DataBatch) -> DataContract:
        if reference.id not in self._contracts:
            self._contracts[reference.id] = derive_contract(spec, self._frame(reference))
        return self._contracts[reference.id]

    # ------------------------------------------------------------- deployment
    def active_deployment(
        self, session: Session, dataset: str, *, lock: bool = False
    ) -> m.DeploymentRevision:
        query = select(m.DeploymentRevision).where(
            m.DeploymentRevision.dataset == dataset, m.DeploymentRevision.active
        )
        if lock:
            query = query.with_for_update()
        row = session.scalars(query).first()
        if row is None:
            raise ConflictError(f"no active deployment for {dataset}; run bootstrap first")
        return row

    def _activate(
        self,
        session: Session,
        principal: Principal,
        dataset: str,
        state: DeploymentState,
        reason: str,
        remediation_id: str | None = None,
        incident_id: str | None = None,
    ) -> m.DeploymentRevision:
        latest = (
            session.scalar(
                select(func.max(m.DeploymentRevision.revision)).where(
                    m.DeploymentRevision.dataset == dataset
                )
            )
            or 0
        )
        for row in session.scalars(
            select(m.DeploymentRevision)
            .where(m.DeploymentRevision.dataset == dataset, m.DeploymentRevision.active)
            .with_for_update()
        ):
            row.active = False
        session.flush()
        revision = m.DeploymentRevision(
            dataset=dataset,
            revision=latest + 1,
            state=state.model_dump(mode="json"),
            active=True,
            reason=reason,
            remediation_id=remediation_id,
            created_by=principal.identity,
        )
        session.add(revision)
        session.flush()
        self.audit(
            session,
            principal,
            "deployment.revision.activated",
            "deployment",
            f"{dataset}#{revision.revision}",
            incident_id=incident_id,
            reason=reason,
            remediation_id=remediation_id,
            state=revision.state,
        )
        return revision

    def bootstrap(self, principal: Principal) -> dict[str, Any]:
        """Idempotently register the default reference window and initial deployment per dataset."""
        out: dict[str, Any] = {}
        for name, _spec in DATASETS.items():
            try:
                baseline = self.registry.baseline(name)
                reference = read_split(name, "validation", self.settings.processed_dir)
            except (KeyError, FileNotFoundError) as exc:
                out[name] = {"status": "skipped", "reason": str(exc)}
                continue
            with self.db.session() as session:
                has_ref = session.scalar(
                    select(func.count())
                    .select_from(m.DataBatch)
                    .where(
                        m.DataBatch.dataset == name,
                        m.DataBatch.role == "reference",
                        m.DataBatch.active,
                    )
                )
                has_dep = session.scalar(
                    select(func.count())
                    .select_from(m.DeploymentRevision)
                    .where(m.DeploymentRevision.dataset == name)
                )
            if not has_ref:
                self.register_batch(
                    principal, name, "reference", reference, source="bootstrap:validation-split"
                )
            if not has_dep:
                with self.db.session() as session:
                    self._activate(
                        session,
                        principal,
                        name,
                        DeploymentState(dataset=name, model_version=baseline.version),
                        "bootstrap",
                    )
            out[name] = {
                "status": "ready",
                "model_version": baseline.version,
                "created_reference": not has_ref,
                "created_deployment": not has_dep,
            }
        return out

    def set_deployment(
        self, principal: Principal, dataset: str, state: DeploymentState, reason: str
    ) -> int:
        """Deployment tooling entry point (used by scripts and tests, not exposed over HTTP)."""
        with self.db.session() as session:
            return self._activate(session, principal, dataset, state, reason).revision

    # ------------------------------------------------------------- monitoring
    def _input(
        self, session: Session, dataset: str, batch: m.DataBatch, state: DeploymentState
    ) -> tuple[MonitoringInput, m.DataBatch]:
        spec = self._spec(dataset)
        reference = self._reference(session, dataset)
        artifact = None
        if state.serving_artifact:
            artifact = (self.settings.artifacts_dir / state.serving_artifact).resolve()
            if self.settings.artifacts_dir.resolve() not in artifact.parents:
                raise InvalidInputError("serving artifact is outside the artifacts directory")
        return MonitoringInput(
            spec,
            self._contract(spec, reference),
            self._frame(reference),
            self._frame(batch),
            self.registry,
            state.model_version,
            artifact,
            state.ingestion,
        ), reference

    def run_monitoring(
        self, principal: Principal, dataset: str, batch_id: str | None = None
    ) -> dict[str, Any]:
        with span("service.run_monitoring", dataset=dataset), self.db.session() as session:
            batch = self._current_batch(session, dataset, batch_id)
            deployment = self.active_deployment(session, dataset)
            existing = session.scalars(
                select(m.MonitoringRun).where(
                    m.MonitoringRun.batch_id == batch.id,
                    m.MonitoringRun.deployment_revision == deployment.revision,
                )
            ).first()
            if existing is not None:  # same batch, same deployment: nothing new to learn
                incident = session.scalars(
                    select(m.Incident).where(m.Incident.monitoring_run_id == existing.id)
                ).first()
                return {"run": existing, "incident": incident, "replayed": True}
            state = DeploymentState.model_validate(deployment.state)
            inp, reference = self._input(session, dataset, batch, state)
            result = run_monitoring(inp, report_id=_id("REP"))
            report = result.report
            MONITORING_SECONDS.labels(dataset).observe(report.duration_ms / 1000)
            run = m.MonitoringRun(
                id=report.report_id,
                dataset=dataset,
                batch_id=batch.id,
                reference_batch_id=reference.id,
                deployment_revision=deployment.revision,
                model_version=report.model_version,
                reference_version=report.reference_version,
                incident_detected=report.incident_detected,
                max_severity=report.max_severity.value,
                triggered_count=len(report.triggered),
                report=report.model_dump(mode="json"),
                performance=_json(result.performance),
                duration_ms=report.duration_ms,
            )
            session.add(run)
            session.flush()
            # The audit trail reads in causal order: the run first, then the incident it opened.
            incident_id = _id("INC") if report.incident_detected else None
            self.audit(
                session,
                principal,
                "monitoring.run",
                "monitoring_run",
                run.id,
                incident_id=incident_id,
                dataset=dataset,
                batch_id=batch.id,
                incident_detected=report.incident_detected,
                max_severity=report.max_severity.value,
                duration_ms=report.duration_ms,
                deployment_revision=deployment.revision,
            )
            incident = None
            if incident_id is not None:
                incident = self._open_incident(session, principal, run, result, incident_id)
            return {"run": run, "incident": incident, "replayed": False}

    def _current_batch(self, session: Session, dataset: str, batch_id: str | None) -> m.DataBatch:
        self._spec(dataset)
        if batch_id:
            batch = session.get(m.DataBatch, batch_id)
            if batch is None or batch.dataset != dataset or batch.role != "current":
                raise NotFoundError(f"current batch {batch_id!r} not found for dataset {dataset}")
            return batch
        batch = session.scalars(
            select(m.DataBatch)
            .where(m.DataBatch.dataset == dataset, m.DataBatch.role == "current")
            .order_by(m.DataBatch.created_at.desc())
        ).first()
        if batch is None:
            raise ConflictError(f"no current batch uploaded for {dataset}")
        return batch

    def _open_incident(
        self,
        session: Session,
        principal: Principal,
        run: m.MonitoringRun,
        result: MonitoringResult,
        incident_id: str,
    ) -> m.Incident:
        report = result.report
        worst = sorted(report.triggered, key=lambda f: -f.severity.rank)
        families = [f.monitor for f in worst if f.severity.rank >= Severity.MEDIUM.rank]
        incident = m.Incident(
            id=incident_id,
            dataset=run.dataset,
            monitoring_run_id=run.id,
            batch_id=run.batch_id,
            incident_type=families[0] if families else "unknown",
            severity=report.max_severity.value,
            status="open",
            model_version=run.model_version,
            reference_version=run.reference_version,
            summary="; ".join(f.interpretation for f in worst[:2])[:600],
        )
        session.add(incident)
        session.flush()
        self._store_evidence(session, incident.id, None, build_evidence(report))
        INCIDENTS.labels(run.dataset, incident.severity).inc()
        self.audit(
            session,
            principal,
            "incident.opened",
            "incident",
            incident.id,
            incident_id=incident.id,
            severity=incident.severity,
            incident_type=incident.incident_type,
            monitoring_run_id=run.id,
            model_version=run.model_version,
            reference_version=run.reference_version,
        )
        return incident

    def _store_evidence(
        self,
        session: Session,
        incident_id: str,
        investigation_id: str | None,
        evidence: list[Evidence],
    ) -> None:
        known = set(
            session.scalars(
                select(m.EvidenceRecord.evidence_id).where(
                    m.EvidenceRecord.incident_id == incident_id
                )
            )
        )
        for item in evidence:
            if item.evidence_id in known:
                continue
            known.add(item.evidence_id)
            session.add(
                m.EvidenceRecord(
                    incident_id=incident_id,
                    investigation_id=investigation_id,
                    evidence_id=item.evidence_id,
                    source=item.source,
                    level=item.level.value,
                    severity=item.severity.value,
                    feature=item.feature,
                    statement=item.statement,
                    finding_id=item.finding_id,
                    data=_json(item.data),
                )
            )

    # ---------------------------------------------------------- investigation
    def start_investigation(
        self, principal: Principal, incident_id: str, *, use_llm: bool = True, wait: bool = False
    ) -> m.Investigation:
        with self.db.session() as session:
            incident = session.get(m.Incident, incident_id, with_for_update=True)
            if incident is None:
                raise NotFoundError(f"incident {incident_id!r} not found")
            running = session.scalars(
                select(m.Investigation).where(
                    m.Investigation.incident_id == incident_id, m.Investigation.status == "running"
                )
            ).first()
            if running is not None:
                return running  # idempotent: one investigation at a time per incident
            deployment = self.active_deployment(session, incident.dataset)
            investigation = m.Investigation(
                id=_id("INV"),
                incident_id=incident_id,
                status="running",
                requested_by=principal.identity,
                use_llm=use_llm and self.llm is not None,
                model_version=incident.model_version,
                reference_version=incident.reference_version,
                deployment_revision=deployment.revision,
            )
            session.add(investigation)
            incident.status = "investigating"
            self.audit(
                session,
                principal,
                "investigation.started",
                "investigation",
                investigation.id,
                incident_id=incident_id,
                use_llm=investigation.use_llm,
                model_version=incident.model_version,
                reference_version=incident.reference_version,
            )
        if wait:
            self._run_investigation(investigation.id, principal)
        else:
            self._worker.submit(self._run_investigation_safely, investigation.id, principal)
        with self.db.session() as session:
            return session.get(m.Investigation, investigation.id)

    def _run_investigation_safely(self, investigation_id: str, principal: Principal) -> None:
        try:
            self._run_investigation(investigation_id, principal)
        except Exception as exc:  # a background failure must be visible, not silent
            log.error(
                "investigation_failed", investigation_id=investigation_id, error=repr(exc)[:400]
            )
            with self.db.session() as session:
                row = session.get(m.Investigation, investigation_id)
                if row is not None and row.status == "running":
                    row.status, row.finished_at = "failed", _now()
                    row.errors = [*row.errors, f"{type(exc).__name__}: {str(exc)[:300]}"]
                    incident = session.get(m.Incident, row.incident_id)
                    if incident is not None and incident.status == "investigating":
                        incident.status = "open"
                    self.audit(
                        session,
                        principal,
                        "investigation.failed",
                        "investigation",
                        investigation_id,
                        incident_id=row.incident_id,
                        outcome="failed",
                        error=type(exc).__name__,
                    )

    def _cancelled(self, investigation_id: str) -> bool:
        with self.db.session() as session:
            return bool(
                session.scalar(
                    select(m.Investigation.cancel_requested).where(
                        m.Investigation.id == investigation_id
                    )
                )
            )

    def _run_investigation(self, investigation_id: str, principal: Principal) -> None:
        with self.db.session() as session:
            investigation = session.get(m.Investigation, investigation_id)
            incident = session.get(m.Incident, investigation.incident_id)
            run = session.get(m.MonitoringRun, incident.monitoring_run_id)
            batch = session.get(m.DataBatch, incident.batch_id)
            deployment = self.active_deployment(session, incident.dataset)
            state = DeploymentState.model_validate(deployment.state)
            inp, _ = self._input(session, incident.dataset, batch, state)
            use_llm, revision, report_id, incident_id = (
                investigation.use_llm,
                deployment.revision,
                run.id,
                incident.id,
            )
        # Re-derive the monitoring result under the *current* deployment. Finding ids are a
        # function of the report id, so evidence ids match what was stored at detection time.
        result = run_monitoring(inp, report_id=report_id)
        tools = ToolExecutor(
            ToolContext(inp, result, self.retriever, state),
            max_calls=self.settings.max_tool_calls,
            timeout_s=self.settings.tool_timeout_s,
        )
        runtime = AgentRuntime(
            tools=tools,
            policy=self.policy,
            llm=self.llm if use_llm else None,
            declared_version=state.model_version,
            is_cancelled=lambda: self._cancelled(investigation_id),
        )
        with span(
            "service.investigation", investigation_id=investigation_id, incident_id=incident_id
        ):
            trace_id = current_trace_id()
            final = run_investigation(runtime, investigation_id, incident_id)
        self._persist_investigation(investigation_id, principal, final, tools, revision, trace_id)

    def _persist_investigation(
        self,
        investigation_id: str,
        principal: Principal,
        final: dict[str, Any],
        tools: ToolExecutor,
        revision: int,
        trace_id: str | None,
    ) -> None:
        with self.db.session() as session:
            row = session.get(m.Investigation, investigation_id, with_for_update=True)
            incident = session.get(m.Incident, row.incident_id, with_for_update=True)
            evidence = [Evidence.model_validate(e) for e in final.get("evidence", [])]
            self._store_evidence(session, incident.id, investigation_id, evidence)
            for seq, trace in enumerate(tools.traces):
                session.add(
                    m.ToolTraceRecord(
                        id=trace.call_id,
                        investigation_id=investigation_id,
                        seq=seq,
                        tool=trace.tool,
                        requested_by=trace.requested_by,
                        status=trace.status,
                        requested_args=trace.requested_args,
                        validated_args=trace.validated_args,
                        output=trace.output,
                        error=trace.error,
                        evidence_ids=trace.evidence_ids,
                        duration_ms=trace.duration_ms,
                        started_at=trace.started_at,
                    )
                )
            diagnosis = final.get("diagnosis") or {"hypotheses": []}
            for rank, hyp in enumerate(diagnosis.get("hypotheses", []), start=1):
                session.add(
                    m.HypothesisRecord(
                        investigation_id=investigation_id,
                        rank=rank,
                        hypothesis_id=hyp["hypothesis_id"],
                        root_cause=hyp["root_cause"],
                        confidence=hyp["confidence"],
                        ranked_by=hyp["ranked_by"],
                        supporting_evidence_ids=hyp["supporting_evidence_ids"],
                        contradicting_evidence_ids=hyp["contradicting_evidence_ids"],
                        payload=hyp,
                    )
                )
            cancelled = final.get("status") == "cancelled"
            row.status = "cancelled" if cancelled else "completed"
            row.llm_status, row.llm_model = final.get("llm_status"), final.get("llm_model")
            row.diagnosis, row.rule_diagnosis = final.get("diagnosis"), final.get("rule_diagnosis")
            row.llm_ranking_stats = final.get("llm_ranking_stats")
            row.recommendation, row.policy_decision = (
                final.get("proposal"),
                final.get("policy_decision"),
            )
            row.explanation = final.get("explanation")
            row.runbook_refs = [
                {
                    k: p[k]
                    for k in (
                        "passage_id",
                        "doc_id",
                        "doc_title",
                        "doc_version",
                        "section",
                        "score",
                        "method",
                        "suspicious",
                        "text",
                    )
                }
                for p in final.get("runbook_passages", [])
            ]
            row.llm_calls, row.events = final.get("llm_calls", []), final.get("events", [])
            row.errors = final.get("errors", [])
            row.trace_id, row.duration_ms, row.finished_at = (
                trace_id,
                final.get("duration_ms"),
                _now(),
            )
            row.deployment_revision = revision

            remediation = None
            proposal = final.get("proposal")
            if not cancelled and proposal and ActionType(proposal["action"]) in STATE_CHANGING:
                remediation = self._record_remediation(
                    session, principal, incident, investigation_id, final, revision
                )
            incident.status = (
                "open"
                if cancelled
                else "awaiting_approval"
                if remediation is not None and remediation.state == S.PENDING_APPROVAL.value
                else "investigated"
            )
            top = diagnosis.get("hypotheses", [{}])[0] if diagnosis.get("hypotheses") else {}
            self.audit(
                session,
                principal,
                "investigation.cancelled" if cancelled else "investigation.completed",
                "investigation",
                investigation_id,
                incident_id=incident.id,
                llm_status=row.llm_status,
                llm_model=row.llm_model,
                tool_calls=len(tools.traces),
                rejected_tool_calls=sum(t.status == "rejected" for t in tools.traces),
                top_hypothesis=top.get("root_cause"),
                top_confidence=top.get("confidence"),
                supporting_evidence_ids=top.get("supporting_evidence_ids", []),
                recommended_action=(proposal or {}).get("action"),
                policy_outcome=(final.get("policy_decision") or {}).get("outcome"),
                remediation_id=remediation.id if remediation else None,
                duration_ms=final.get("duration_ms"),
                trace_id=trace_id,
            )

    # ------------------------------------------------------------ remediation
    def _transition(
        self,
        session: Session,
        principal: Principal,
        row: m.Remediation,
        event: Event,
        **detail: Any,
    ) -> None:
        try:
            new_state = next_state(S(row.state), event)
        except IllegalTransitionError as exc:
            raise ConflictError(str(exc), state=row.state, event=event.value) from None
        previous, row.state, row.version = row.state, new_state.value, row.version + 1
        REMEDIATION_TRANSITIONS.labels(new_state.value).inc()
        self.audit(
            session,
            principal,
            f"remediation.{event.value}",
            "remediation",
            row.id,
            incident_id=row.incident_id,
            from_state=previous,
            to_state=new_state.value,
            **detail,
        )

    def _new_remediation(
        self,
        session: Session,
        principal: Principal,
        incident: m.Incident,
        investigation_id: str | None,
        proposal: dict[str, Any],
        revision: int,
    ) -> m.Remediation:
        row = m.Remediation(
            id=_id("REM"),
            incident_id=incident.id,
            investigation_id=investigation_id,
            dataset=incident.dataset,
            state=S.PROPOSED.value,
            action=proposal["action"],
            risk_level=proposal["risk_level"],
            proposal=proposal,
            base_revision=revision,
            created_by=principal.identity,
        )
        session.add(row)
        session.flush()
        REMEDIATION_TRANSITIONS.labels(S.PROPOSED.value).inc()
        self.audit(
            session,
            principal,
            "remediation.proposed",
            "remediation",
            row.id,
            incident_id=incident.id,
            remediation_action=row.action,
            risk_level=row.risk_level,
            parameters=proposal.get("parameters", {}),
            evidence_ids=proposal.get("evidence_ids", []),
            investigation_id=investigation_id,
        )
        return row

    def _apply_validation(
        self,
        session: Session,
        principal: Principal,
        row: m.Remediation,
        validation: dict[str, Any] | None,
        decision: dict[str, Any] | None,
    ) -> None:
        """Advance a PROPOSED remediation according to a validation result and policy decision."""
        row.policy_decision = decision
        outcome = (decision or {}).get("outcome")
        if validation is None:
            if outcome in (PolicyOutcome.REJECT.value, PolicyOutcome.INVESTIGATE_MORE.value):
                self._transition(
                    session,
                    principal,
                    row,
                    Event.REJECT,
                    by="policy",
                    reasons=(decision or {}).get("reasons", []),
                    unmet=(decision or {}).get("unmet_requirements", []),
                )
            return
        self._transition(session, principal, row, Event.START_VALIDATION)
        row.validation = validation
        if not validation["passed"]:
            self._transition(
                session,
                principal,
                row,
                Event.VALIDATION_FAILED,
                summary=validation["summary"],
                failed_checks=[c["name"] for c in validation["checks"] if not c["passed"]],
            )
            return
        self._transition(
            session,
            principal,
            row,
            Event.VALIDATION_PASSED,
            summary=validation["summary"],
            checks=[c["name"] for c in validation["checks"]],
        )
        if outcome == PolicyOutcome.REQUEST_HUMAN_APPROVAL.value:
            self._transition(
                session,
                principal,
                row,
                Event.REQUEST_APPROVAL,
                policy_version=decision["policy_version"],
                score=decision["score"],
            )
        elif outcome == PolicyOutcome.REJECT.value:
            self._transition(
                session, principal, row, Event.REJECT, by="policy", reasons=decision["reasons"]
            )

    def _record_remediation(
        self,
        session: Session,
        principal: Principal,
        incident: m.Incident,
        investigation_id: str,
        final: dict[str, Any],
        revision: int,
    ) -> m.Remediation:
        row = self._new_remediation(
            session, principal, incident, investigation_id, final["proposal"], revision
        )
        self._apply_validation(
            session, principal, row, final.get("validation"), final.get("policy_decision")
        )
        return row

    def create_remediation(
        self, principal: Principal, incident_id: str
    ) -> tuple[m.Remediation, bool]:
        """Create a remediation from the latest investigation's recommendation. Idempotent."""
        with self.db.session() as session:
            incident = session.get(m.Incident, incident_id, with_for_update=True)
            if incident is None:
                raise NotFoundError(f"incident {incident_id!r} not found")
            existing = session.scalars(
                select(m.Remediation).where(
                    m.Remediation.incident_id == incident_id,
                    m.Remediation.state.in_([s.value for s in OPEN_REMEDIATION_STATES]),
                )
            ).first()
            if existing is not None:
                return existing, False
            investigation = session.scalars(
                select(m.Investigation)
                .where(
                    m.Investigation.incident_id == incident_id,
                    m.Investigation.status == "completed",
                )
                .order_by(m.Investigation.finished_at.desc())
            ).first()
            if investigation is None or not investigation.recommendation:
                raise ConflictError(
                    "the incident has no completed investigation to base a remediation on"
                )
            proposal = investigation.recommendation
            if ActionType(proposal["action"]) not in STATE_CHANGING:
                raise ConflictError(
                    f"the investigation recommends '{proposal['action']}', which changes no state; "
                    "there is nothing to remediate",
                    rationale=proposal.get("rationale"),
                )
            deployment = self.active_deployment(session, incident.dataset)
            return self._new_remediation(
                session, principal, incident, investigation.id, proposal, deployment.revision
            ), True

    def _remediation_context(
        self, session: Session, row: m.Remediation, deployment: m.DeploymentRevision
    ):
        incident = session.get(m.Incident, row.incident_id)
        run = session.get(m.MonitoringRun, incident.monitoring_run_id)
        batch = session.get(m.DataBatch, incident.batch_id)
        state = DeploymentState.model_validate(deployment.state)
        inp, _ = self._input(session, row.dataset, batch, state)
        return incident, run, inp, state

    def validate_remediation(
        self, principal: Principal, remediation_id: str
    ) -> tuple[m.Remediation, bool]:
        with (
            span("service.validate_remediation", remediation_id=remediation_id),
            self.db.session() as session,
        ):
            row = session.get(m.Remediation, remediation_id, with_for_update=True)
            if row is None:
                raise NotFoundError(f"remediation {remediation_id!r} not found")
            if row.state != S.PROPOSED.value:
                if row.validation is not None:
                    return row, False  # already validated: replay, do not validate twice
                raise ConflictError(
                    f"remediation is {row.state}; only a PROPOSED remediation can be validated"
                )
            deployment = self.active_deployment(session, row.dataset)
            incident, run, inp, state = self._remediation_context(session, row, deployment)
            result = run_monitoring(inp, report_id=run.id)
            proposal = RemediationProposal.model_validate(row.proposal)
            investigation = (
                session.get(m.Investigation, row.investigation_id) if row.investigation_id else None
            )
            evidence = [
                Evidence(
                    evidence_id=e.evidence_id,
                    source=e.source,
                    level=e.level,
                    statement=e.statement,
                    feature=e.feature,
                    severity=e.severity,
                    data=e.data,
                    finding_id=e.finding_id,
                )
                for e in session.scalars(
                    select(m.EvidenceRecord).where(m.EvidenceRecord.incident_id == row.incident_id)
                )
            ]
            if investigation is None or not investigation.diagnosis:
                raise ConflictError("the remediation has no diagnosis to evaluate policy against")
            diagnosis = Diagnosis.model_validate(investigation.diagnosis)
            rules = (
                Diagnosis.model_validate(investigation.rule_diagnosis)
                if investigation.rule_diagnosis
                else None
            )
            pre = self.policy.evaluate(proposal, diagnosis, evidence, rule_diagnosis=rules)
            validation = None
            decision = pre
            if pre.outcome is PolicyOutcome.PROPOSE_ACTION:
                validation = validate_proposal(proposal, inp, result, state)
                decision = self.policy.evaluate(
                    proposal, diagnosis, evidence, rule_diagnosis=rules, validation=validation
                )
            row.base_revision = deployment.revision
            self._apply_validation(
                session,
                principal,
                row,
                validation.model_dump(mode="json") if validation else None,
                decision.model_dump(mode="json"),
            )
            if row.state == S.PENDING_APPROVAL.value:
                incident.status = "awaiting_approval"
            return row, True

    def decide(
        self,
        principal: Principal,
        remediation_id: str,
        decision: str,
        *,
        expected_version: int,
        comment: str = "",
        idempotency_key: str | None = None,
    ) -> tuple[m.Remediation, bool]:
        """Approve (and execute) or reject. Returns (remediation, newly_decided)."""
        action = f"remediation.{'approve' if decision == 'approved' else 'reject'}"
        if not principal.at_least("approver"):
            self.audit_denied(
                principal, action, "remediation", remediation_id, "role: approver required"
            )
            raise ForbiddenError(
                "the approver role is required to decide a remediation", role=principal.role
            )
        try:
            with (
                span("service.decide", remediation_id=remediation_id, decision=decision),
                self.db.session() as session,
            ):
                row = session.get(m.Remediation, remediation_id, with_for_update=True)
                if row is None:
                    raise NotFoundError(f"remediation {remediation_id!r} not found")
                prior = session.scalars(
                    select(m.Approval).where(m.Approval.remediation_id == remediation_id)
                ).first()
                if prior is not None:
                    same = (
                        idempotency_key is not None
                        and prior.idempotency_key == idempotency_key
                        and prior.decision == decision
                        and prior.reviewer == principal.identity
                    )
                    if same:
                        return row, False  # replay of the same request: return the recorded outcome
                    raise ConflictError(
                        f"remediation was already {prior.decision} by {prior.reviewer}",
                        decided_at=str(prior.decided_at),
                    )
                if row.state != S.PENDING_APPROVAL.value:
                    raise ConflictError(
                        f"remediation is {row.state}; only PENDING_APPROVAL can be decided"
                    )
                if expected_version != row.version:
                    raise ConflictError(
                        "stale decision: the remediation changed after it was reviewed",
                        expected_version=expected_version,
                        current_version=row.version,
                    )
                if self.settings.require_distinct_approver and principal.identity == row.created_by:
                    raise ForbiddenError(
                        "separation of duties: the approver must differ from the requester"
                    )
                approval = m.Approval(
                    remediation_id=row.id,
                    decision=decision,
                    reviewer=principal.identity,
                    reviewer_role=principal.role,
                    comment=comment[:2000],
                    remediation_version=row.version,
                    idempotency_key=idempotency_key,
                )
                session.add(approval)
                session.flush()
                incident = session.get(m.Incident, row.incident_id, with_for_update=True)
                if decision == "rejected":
                    self._transition(
                        session,
                        principal,
                        row,
                        Event.REJECT,
                        by="human",
                        reviewer=principal.identity,
                        comment=comment[:500],
                    )
                    incident.status = "investigated"
                    return row, True
                self._transition(
                    session,
                    principal,
                    row,
                    Event.APPROVE,
                    reviewer=principal.identity,
                    reviewer_role=principal.role,
                    comment=comment[:500],
                    approved_version=approval.remediation_version,
                )
                self._execute(session, principal, row, incident)
                return row, True
        except IntegrityError:
            raise ConflictError("remediation was decided concurrently by another request") from None
        except (ConflictError, ForbiddenError) as exc:
            self.audit_denied(
                principal, action, "remediation", remediation_id, f"{exc.code}: {exc.message}"
            )
            raise

    def _execute(
        self, session: Session, principal: Principal, row: m.Remediation, incident: m.Incident
    ) -> None:
        deployment = self.active_deployment(session, row.dataset, lock=True)
        if deployment.revision != row.base_revision:
            raise ConflictError(
                "stale remediation: the deployment changed since validation; reject it and re-validate a new one",
                validated_against=row.base_revision,
                active_revision=deployment.revision,
            )
        current = DeploymentState.model_validate(deployment.state)
        try:
            new_state = state_to_execute(
                current,
                RemediationProposal.model_validate(row.proposal),
                ValidationResult.model_validate(row.validation),
            )
        except (StaleProposalError, ValueError) as exc:
            raise ConflictError(f"cannot execute: {exc}") from None
        revision = self._activate(
            session,
            principal,
            row.dataset,
            new_state,
            f"remediation {row.id}: {row.action}",
            row.id,
            row.incident_id,
        )
        row.executed_revision = revision.revision
        self._transition(
            session, principal, row, Event.EXECUTE, deployment_revision=revision.revision
        )
        # Post-execution verification: does the incident batch look healthy under the new state?
        _, _, inp, _ = self._remediation_context(session, row, revision)
        verification = self._verify(row, inp, new_state)
        row.verification = verification
        if verification["passed"]:
            self._transition(
                session, principal, row, Event.VERIFY, verification=verification["summary"]
            )
            incident.status = "mitigated"
        else:
            incident.status = "executed_unverified"
            self.audit(
                session,
                principal,
                "remediation.verification_failed",
                "remediation",
                row.id,
                incident_id=row.incident_id,
                outcome="failed",
                verification=verification["summary"],
            )

    def _verify(
        self, row: m.Remediation, inp: MonitoringInput, state: DeploymentState
    ) -> dict[str, Any]:
        if row.action == ActionType.QUARANTINE_BATCH.value:
            return {
                "passed": state.quarantined,
                "summary": "batch quarantined; model and ingestion unchanged",
                "remaining_findings": [],
            }
        result = run_monitoring(
            replace(inp, model_version=state.model_version, ingestion=state.ingestion)
        )
        remaining = sorted(
            {
                f"{f.check}:{f.feature or '-'}"
                for f in result.report.triggered
                if f.severity.rank >= Severity.MEDIUM.rank
            }
        )
        return {
            "passed": not result.report.incident_detected,
            "summary": "incident batch is healthy under the new deployment state"
            if not remaining
            else f"findings remain after execution: {remaining}",
            "remaining_findings": remaining,
        }

    def rollback(
        self, principal: Principal, remediation_id: str, reason: str
    ) -> tuple[m.Remediation, bool]:
        if not principal.at_least("approver"):
            self.audit_denied(
                principal,
                "remediation.rollback",
                "remediation",
                remediation_id,
                "role: approver required",
            )
            raise ForbiddenError(
                "the approver role is required to roll back a remediation", role=principal.role
            )
        try:
            with (
                span("service.rollback", remediation_id=remediation_id),
                self.db.session() as session,
            ):
                row = session.get(m.Remediation, remediation_id, with_for_update=True)
                if row is None:
                    raise NotFoundError(f"remediation {remediation_id!r} not found")
                if row.state == S.ROLLED_BACK.value:
                    return row, False  # idempotent
                if row.state not in (S.EXECUTED.value, S.COMPLETED.value):
                    raise ConflictError(
                        f"remediation is {row.state}; only an executed remediation can be rolled back"
                    )
                deployment = self.active_deployment(session, row.dataset, lock=True)
                if deployment.revision != row.executed_revision:
                    raise ConflictError(
                        "the deployment has moved on since this remediation was executed; "
                        "rolling it back would discard later changes",
                        executed_revision=row.executed_revision,
                        active_revision=deployment.revision,
                    )
                base = session.scalars(
                    select(m.DeploymentRevision).where(
                        m.DeploymentRevision.dataset == row.dataset,
                        m.DeploymentRevision.revision == row.base_revision,
                    )
                ).one()
                restored = self._activate(
                    session,
                    principal,
                    row.dataset,
                    DeploymentState.model_validate(base.state),
                    f"rollback of {row.id}: {reason[:200]}",
                    row.id,
                    row.incident_id,
                )
                row.rollback_revision = restored.revision
                self._transition(
                    session,
                    principal,
                    row,
                    Event.ROLLBACK,
                    reason=reason[:500],
                    restored_state_of_revision=row.base_revision,
                    deployment_revision=restored.revision,
                )
                incident = session.get(m.Incident, row.incident_id, with_for_update=True)
                incident.status = "investigated"
                return row, True
        except ConflictError as exc:
            self.audit_denied(
                principal,
                "remediation.rollback",
                "remediation",
                remediation_id,
                f"{exc.code}: {exc.message}",
            )
            raise

    def cancel_investigation(self, principal: Principal, investigation_id: str) -> m.Investigation:
        with self.db.session() as session:
            row = session.get(m.Investigation, investigation_id, with_for_update=True)
            if row is None:
                raise NotFoundError(f"investigation {investigation_id!r} not found")
            if row.status == "running" and not row.cancel_requested:
                row.cancel_requested = True
                self.audit(
                    session,
                    principal,
                    "investigation.cancel_requested",
                    "investigation",
                    row.id,
                    incident_id=row.incident_id,
                )
            return row

    def shutdown(self) -> None:
        self._worker.shutdown(wait=False, cancel_futures=True)
        if self.llm is not None:
            self.llm.close()


def _json(value: Any) -> Any:
    """Make numpy scalars and non-finite floats JSON-safe."""
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
