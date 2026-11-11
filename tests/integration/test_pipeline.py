"""Integration of ingestion, monitoring, incidents, investigation and remediation on PostgreSQL."""

import pandas as pd
import pytest
from sqlalchemy import select, text

from aegis.schemas.deployment import DeploymentState
from aegis.services.errors import ConflictError, InvalidInputError, NotFoundError
from aegis.storage import models as m
from tests.conftest import open_incident

pytestmark = pytest.mark.postgres


def rows(service, model, **filters):
    with service.db.session() as session:
        return list(session.scalars(select(model).filter_by(**filters)))


def actions(service, incident_id):
    return [e.action for e in rows(service, m.AuditEvent, incident_id=incident_id)]


# ----------------------------------------------------------- ingestion -> monitoring
def test_migration_created_the_schema_with_an_append_only_audit_table(service):
    with service.db.session() as session:
        tables = set(
            session.scalars(text("SELECT tablename FROM pg_tables WHERE schemaname='public'"))
        )
        version = session.scalar(text("SELECT version_num FROM alembic_version"))
    assert {
        "incidents",
        "evidence",
        "investigations",
        "tool_traces",
        "hypotheses",
        "remediations",
        "approvals",
        "audit_events",
        "deployment_revisions",
        "monitoring_runs",
        "data_batches",
    } <= tables
    assert version == "0001"
    for statement in ("UPDATE audit_events SET actor='mallory'", "DELETE FROM audit_events"):
        with pytest.raises(Exception, match="append-only"), service.db.session() as session:
            session.execute(text(statement))
    assert rows(service, m.AuditEvent)  # seed events are still there


def test_ingested_batch_is_stored_with_checksum_and_provenance(service, ctx, principals):
    batch = service.register_batch(
        principals["operator"], "adult", "current", ctx.clean().current, source="unit"
    )
    (stored,) = rows(service, m.DataBatch, id=batch.id)
    assert stored.rows == 2000 and len(stored.sha256) == 64 and stored.created_by == "bob"
    stored_frame = pd.read_parquet(stored.path)
    original = ctx.clean().current
    assert list(stored_frame.columns) == list(original.columns)
    assert stored_frame["age"].tolist() == original["age"].tolist()
    assert stored_frame["occupation"].isna().sum() == original["occupation"].isna().sum()
    assert "data.current.registered" in [
        e.action for e in rows(service, m.AuditEvent, entity_id=batch.id)
    ]


@pytest.mark.parametrize(
    "frame,match",
    [
        (pd.DataFrame(), "no rows"),
        (pd.DataFrame({"a": [1], "b": [2]}).rename(columns={"b": "a"}), "duplicate column"),
    ],
)
def test_invalid_batches_are_refused(service, principals, frame, match):
    with pytest.raises(InvalidInputError, match=match):
        service.register_batch(principals["operator"], "adult", "current", frame)


def test_reference_must_be_complete_and_labelled(service, ctx, principals):
    with pytest.raises(InvalidInputError, match="every contracted feature"):
        service.register_batch(
            principals["operator"], "adult", "reference", ctx.reference.drop(columns=["age"])
        )
    with pytest.raises(InvalidInputError, match="unknown dataset"):
        service.register_batch(principals["operator"], "mnist", "current", ctx.reference)


def test_tampered_stored_batch_is_detected(service, ctx, principals):
    batch = service.register_batch(principals["operator"], "adult", "current", ctx.clean().current)
    with open(batch.path, "ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ConflictError, match="checksum"):
        service.run_monitoring(principals["operator"], "adult", batch.id)


def test_clean_batch_creates_a_run_but_no_incident(service, ctx, principals):
    incident, _ = open_incident(service, ctx, principals, fault="CLEAN", severity="none")
    assert incident is None
    (run,) = rows(service, m.MonitoringRun)
    assert (
        not run.incident_detected
        and run.triggered_count == 0
        and run.performance["current"]["roc_auc"] > 0.85
    )
    assert rows(service, m.Incident) == [] and rows(service, m.EvidenceRecord) == []


# ----------------------------------------------------- monitoring -> incident -> evidence
def test_fault_creates_a_persistent_incident_with_traceable_evidence(service, ctx, principals):
    incident, _ = open_incident(service, ctx, principals, investigate=False)
    (stored,) = rows(service, m.Incident)
    (run,) = rows(service, m.MonitoringRun)
    assert (
        stored.id == incident.id
        and stored.status == "open"
        and stored.severity in ("high", "critical")
    )
    assert (
        stored.model_version == ctx.record.version
        and stored.reference_version == ctx.contract.version
    )
    evidence = rows(service, m.EvidenceRecord, incident_id=incident.id)
    finding_ids = {f["finding_id"] for f in run.report["findings"] if f["triggered"]}
    assert {
        e.finding_id for e in evidence if e.finding_id
    } == finding_ids  # every finding is evidence
    assert {e.level for e in evidence} <= {"observed_fact", "statistical_anomaly"}
    assert actions(service, incident.id) == ["monitoring.run", "incident.opened"] or set(
        actions(service, incident.id)
    ) == {"monitoring.run", "incident.opened"}


def test_monitoring_the_same_batch_twice_is_idempotent(service, ctx, principals):
    batch = service.register_batch(
        principals["operator"], "adult", "current", ctx.scenario("F06").current
    )
    first = service.run_monitoring(principals["operator"], "adult", batch.id)
    second = service.run_monitoring(principals["operator"], "adult", batch.id)
    assert (
        not first["replayed"]
        and second["replayed"]
        and second["incident"].id == first["incident"].id
    )
    assert len(rows(service, m.MonitoringRun)) == 1 and len(rows(service, m.Incident)) == 1


def test_monitoring_unknown_batch_or_dataset_fails_cleanly(service, principals):
    with pytest.raises(NotFoundError):
        service.run_monitoring(principals["operator"], "adult", "BAT-missing")
    with pytest.raises(ConflictError, match="no current batch"):
        service.run_monitoring(principals["operator"], "adult")


# --------------------------------------------- evidence -> diagnosis -> runbook -> proposal
def test_investigation_persists_traces_hypotheses_runbooks_and_recommendation(
    service, ctx, principals
):
    incident, investigation = open_incident(service, ctx, principals)
    assert investigation.status == "completed" and investigation.llm_status == "disabled"
    assert investigation.model_version == ctx.record.version and investigation.duration_ms > 0
    traces = rows(service, m.ToolTraceRecord, investigation_id=investigation.id)
    assert [t.seq for t in sorted(traces, key=lambda t: t.seq)] == list(range(len(traces)))
    assert {"inspect_feature_distribution", "retrieve_runbook", "run_candidate_validation"} <= {
        t.tool for t in traces
    }
    for trace in traces:
        assert (
            trace.status == "ok"
            and trace.validated_args is not None
            and trace.output
            and trace.duration_ms >= 0
        )
    hypotheses = sorted(
        rows(service, m.HypothesisRecord, investigation_id=investigation.id), key=lambda h: h.rank
    )
    assert hypotheses[0].root_cause == "scaling_error" and hypotheses[0].rank == 1
    evidence_ids = {e.evidence_id for e in rows(service, m.EvidenceRecord, incident_id=incident.id)}
    for h in hypotheses:
        assert set(h.supporting_evidence_ids) | set(h.contradicting_evidence_ids) <= evidence_ids
    assert any(
        e.source == "tool:inspect_feature_distribution" for e in rows(service, m.EvidenceRecord)
    )
    assert investigation.runbook_refs and investigation.runbook_refs[0]["doc_id"].startswith("RB-")
    assert investigation.recommendation["action"] == "fix_ingestion_scale"
    assert investigation.policy_decision["outcome"] == "REQUEST_HUMAN_APPROVAL"
    assert investigation.events[0]["node"] == "collect_incident"


def test_investigation_creates_a_remediation_waiting_for_a_human(service, ctx, principals):
    incident, investigation = open_incident(service, ctx, principals)
    (remediation,) = rows(service, m.Remediation)
    assert (
        remediation.state == "PENDING_APPROVAL" and remediation.investigation_id == investigation.id
    )
    assert remediation.validation["passed"] and remediation.executed_revision is None
    assert rows(service, m.Approval) == []
    assert rows(service, m.Incident)[0].status == "awaiting_approval"
    assert [r.revision for r in rows(service, m.DeploymentRevision)] == [1]  # nothing was deployed
    trail = actions(service, incident.id)
    order = [
        "remediation.proposed",
        "remediation.start_validation",
        "remediation.validation_passed",
        "remediation.request_approval",
        "investigation.completed",
    ]
    assert [a for a in trail if a in order] == order


def test_failed_sandbox_validation_ends_the_remediation_without_approval_request(
    service, ctx, principals
):
    open_incident(service, ctx, principals, fault="F05")
    (remediation,) = rows(service, m.Remediation)
    assert remediation.state == "VALIDATION_FAILED" and remediation.validation["passed"] is False
    assert rows(service, m.Incident)[0].status == "investigated" and rows(service, m.Approval) == []


def test_non_actionable_recommendation_creates_no_remediation(service, ctx, principals):
    incident, investigation = open_incident(service, ctx, principals, fault="F09")
    assert (
        investigation.recommendation["action"] == "no_action" and rows(service, m.Remediation) == []
    )
    with pytest.raises(ConflictError, match="changes no state"):
        service.create_remediation(principals["operator"], incident.id)


def test_only_one_investigation_runs_per_incident_and_reruns_do_not_duplicate_evidence(
    service, ctx, principals
):
    incident, first = open_incident(service, ctx, principals, fault="F09")
    before = len(rows(service, m.EvidenceRecord, incident_id=incident.id))
    second = service.start_investigation(
        principals["operator"], incident.id, use_llm=False, wait=True
    )
    assert second.id != first.id and len(rows(service, m.Investigation)) == 2
    assert len(rows(service, m.EvidenceRecord, incident_id=incident.id)) == before
    with pytest.raises(NotFoundError):
        service.start_investigation(principals["operator"], "INC-missing", wait=True)


def test_investigation_can_be_cancelled(service, ctx, principals, monkeypatch):
    incident, _ = open_incident(service, ctx, principals, investigate=False)
    monkeypatch.setattr(type(service), "_cancelled", lambda self, investigation_id: True)
    investigation = service.start_investigation(
        principals["operator"], incident.id, use_llm=False, wait=True
    )
    assert investigation.status == "cancelled" and rows(service, m.Remediation) == []
    assert rows(service, m.Incident)[0].status == "open"
    assert "investigation.cancelled" in actions(service, incident.id)


def test_background_investigation_failure_is_recorded_not_lost(
    service, ctx, principals, monkeypatch
):
    incident, _ = open_incident(service, ctx, principals, investigate=False)

    def explode(self, investigation_id, principal):
        raise RuntimeError("worker crashed")

    monkeypatch.setattr(type(service), "_run_investigation", explode)
    investigation = service.start_investigation(principals["operator"], incident.id, use_llm=False)
    service._worker.shutdown(wait=True)
    (stored,) = rows(service, m.Investigation, id=investigation.id)
    assert stored.status == "failed" and "worker crashed" in stored.errors[0]
    assert rows(service, m.Incident)[0].status == "open"


# ------------------------------------------- approval -> state transition -> rollback -> audit
def test_approval_executes_verifies_and_writes_a_new_deployment_revision(service, ctx, principals):
    incident, _ = open_incident(service, ctx, principals)
    (remediation,) = rows(service, m.Remediation)
    decided, fresh = service.decide(
        principals["approver"],
        remediation.id,
        "approved",
        expected_version=remediation.version,
        comment="unit confirmed with data owner",
    )
    assert fresh and decided.state == "COMPLETED" and decided.executed_revision == 2
    assert decided.verification["passed"] and decided.verification["remaining_findings"] == []
    revisions = {r.revision: r for r in rows(service, m.DeploymentRevision)}
    assert (
        revisions[2].active
        and not revisions[1].active
        and revisions[2].remediation_id == remediation.id
    )
    feature = remediation.proposal["parameters"]["feature"]
    assert revisions[2].state["ingestion"]["unit_scale"][feature] == pytest.approx(
        remediation.proposal["parameters"]["factor"]
    )
    (approval,) = rows(service, m.Approval)
    assert (approval.reviewer, approval.reviewer_role, approval.decision) == (
        "alice",
        "approver",
        "approved",
    )
    assert approval.decided_at is not None and rows(service, m.Incident)[0].status == "mitigated"
    # The same batch, monitored again under the new deployment, is healthy.
    rerun = service.run_monitoring(principals["operator"], "adult", incident.batch_id)
    assert (
        not rerun["replayed"] and rerun["incident"] is None and not rerun["run"].incident_detected
    )


def test_rejection_closes_the_remediation_and_changes_nothing(service, ctx, principals):
    open_incident(service, ctx, principals)
    (remediation,) = rows(service, m.Remediation)
    decided, _ = service.decide(
        principals["approver"],
        remediation.id,
        "rejected",
        expected_version=remediation.version,
        comment="not convinced",
    )
    assert decided.state == "REJECTED" and decided.executed_revision is None
    assert [r.revision for r in rows(service, m.DeploymentRevision)] == [1]
    assert (
        rows(service, m.Approval)[0].decision == "rejected"
        and rows(service, m.Incident)[0].status == "investigated"
    )


def test_rollback_restores_the_previous_state_and_is_audited(service, ctx, principals):
    incident, _ = open_incident(service, ctx, principals)
    (remediation,) = rows(service, m.Remediation)
    service.decide(
        principals["approver"], remediation.id, "approved", expected_version=remediation.version
    )
    rolled, fresh = service.rollback(
        principals["approver2"], remediation.id, "upstream fixed the unit"
    )
    assert fresh and rolled.state == "ROLLED_BACK" and rolled.rollback_revision == 3
    revisions = {r.revision: r for r in rows(service, m.DeploymentRevision)}
    assert (
        revisions[3].active and revisions[3].state == revisions[1].state and not revisions[2].active
    )
    events = rows(service, m.AuditEvent, incident_id=incident.id)
    rollback_event = next(e for e in events if e.action == "remediation.rollback")
    assert (
        rollback_event.actor == "carol"
        and rollback_event.detail["reason"] == "upstream fixed the unit"
    )
    assert (
        rollback_event.detail["from_state"] == "COMPLETED"
        and rollback_event.detail["deployment_revision"] == 3
    )
    again, fresh = service.rollback(principals["approver"], remediation.id, "again")
    assert (
        not fresh and again.state == "ROLLED_BACK" and len(rows(service, m.DeploymentRevision)) == 3
    )


def test_complete_audit_trail_links_every_required_fact(service, ctx, principals):
    incident, investigation = open_incident(service, ctx, principals)
    (remediation,) = rows(service, m.Remediation)
    service.decide(
        principals["approver"], remediation.id, "approved", expected_version=remediation.version
    )
    events = rows(service, m.AuditEvent, incident_id=incident.id)
    by_action = {e.action: e for e in events}
    assert [e.id for e in events] == sorted(e.id for e in events)
    assert by_action["incident.opened"].detail["model_version"] == ctx.record.version
    assert by_action["incident.opened"].detail["reference_version"] == ctx.contract.version
    done = by_action["investigation.completed"].detail
    assert (
        done["top_hypothesis"] == "scaling_error"
        and done["supporting_evidence_ids"]
        and done["tool_calls"] >= 4
    )
    assert (
        done["llm_status"] == "disabled"
        and done["remediation_id"] == remediation.id
        and done["trace_id"]
    )
    assert by_action["remediation.proposed"].detail["parameters"]["feature"]
    assert by_action["remediation.validation_passed"].detail["checks"]
    approve = by_action["remediation.approve"]
    assert approve.actor == "alice" and approve.actor_role == "approver" and approve.at is not None
    assert by_action["remediation.execute"].detail["deployment_revision"] == 2
    assert by_action["remediation.verify"].detail["to_state"] == "COMPLETED"
    assert investigation.id == by_action["investigation.completed"].entity_id


def test_manual_remediation_path_create_then_validate(service, ctx, principals):
    """POST /incidents/{id}/remediation + /validate: a second remediation after a rejection."""
    incident, _ = open_incident(service, ctx, principals)
    (first,) = rows(service, m.Remediation)
    same, created = service.create_remediation(principals["operator"], incident.id)
    assert not created and same.id == first.id  # idempotent while one is open
    service.decide(principals["approver"], first.id, "rejected", expected_version=first.version)
    second, created = service.create_remediation(principals["operator"], incident.id)
    assert created and second.state == "PROPOSED" and second.id != first.id
    validated, ran = service.validate_remediation(principals["operator"], second.id)
    assert ran and validated.state == "PENDING_APPROVAL" and validated.validation["passed"]
    replay, ran = service.validate_remediation(principals["operator"], second.id)
    assert not ran and replay.version == validated.version  # validating twice does nothing
    decided, _ = service.decide(
        principals["approver"], second.id, "approved", expected_version=validated.version
    )
    assert decided.state == "COMPLETED"


@pytest.mark.parametrize(
    "fault,action",
    [
        ("F06", "enable_deduplication"),
        ("F08", "quarantine_batch"),
        ("F10", "restore_model_version"),
    ],
)
def test_other_remediations_execute_and_verify(service, ctx, principals, fault, action):
    open_incident(service, ctx, principals, fault=fault, severity="medium")
    (remediation,) = rows(service, m.Remediation)
    assert remediation.action == action and remediation.state == "PENDING_APPROVAL"
    decided, _ = service.decide(
        principals["approver"], remediation.id, "approved", expected_version=remediation.version
    )
    assert decided.state == "COMPLETED", decided.verification
    active = DeploymentState.model_validate(
        next(r for r in rows(service, m.DeploymentRevision) if r.active).state
    )
    if fault == "F06":
        assert active.ingestion.deduplicate
    elif fault == "F08":
        assert active.quarantined and active.retraining_blocked
    else:
        assert active.serving_artifact is None and active.model_version == ctx.record.version
