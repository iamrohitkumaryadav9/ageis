"""Attempts to get a remediation executed without a valid, single, current human approval."""

import threading

import pytest
from sqlalchemy import select

from aegis.schemas.deployment import DeploymentState, IngestionConfig
from aegis.services.errors import ConflictError, ForbiddenError, Principal
from aegis.storage import models as m
from tests.conftest import auth, open_incident

pytestmark = pytest.mark.postgres


def rows(service, model, **filters):
    with service.db.session() as session:
        return list(session.scalars(select(model).filter_by(**filters)))


def pending(service, ctx, principals, fault="F04"):
    open_incident(service, ctx, principals, fault=fault)
    (remediation,) = rows(service, m.Remediation)
    assert remediation.state == "PENDING_APPROVAL"
    return remediation


def untouched(service):
    return [r.revision for r in rows(service, m.DeploymentRevision)] == [1] and rows(
        service, m.Approval
    ) == []


def denied(service, remediation_id):
    return [
        e for e in rows(service, m.AuditEvent, entity_id=remediation_id) if e.outcome == "denied"
    ]


# ------------------------------------------------------------ who may approve
@pytest.mark.parametrize("role", ["operator", "viewer"])
def test_unauthorised_reviewer_is_refused_and_recorded(service, ctx, principals, role):
    remediation = pending(service, ctx, principals)
    for decision in ("approved", "rejected"):
        with pytest.raises(ForbiddenError):
            service.decide(
                principals[role], remediation.id, decision, expected_version=remediation.version
            )
    with pytest.raises(ForbiddenError):
        service.rollback(principals[role], remediation.id, "let me")
    assert untouched(service) and rows(service, m.Remediation)[0].state == "PENDING_APPROVAL"
    attempts = denied(service, remediation.id)
    assert len(attempts) == 3 and {a.actor_role for a in attempts} == {role}
    assert {a.action for a in attempts} == {
        "remediation.approve",
        "remediation.reject",
        "remediation.rollback",
    }


def test_made_up_roles_and_agent_identities_cannot_approve(service, ctx, principals):
    remediation = pending(service, ctx, principals)
    for principal in (
        Principal("aegis-agent", "agent"),
        Principal("llm", "system"),
        Principal("root", "admin"),
        Principal("alice", "APPROVER"),
        Principal("", ""),
    ):
        with pytest.raises(ForbiddenError):
            service.decide(
                principal, remediation.id, "approved", expected_version=remediation.version
            )
    assert untouched(service)


def test_approval_cannot_be_asserted_through_the_request_body(api, service, ctx, principals):
    remediation = pending(service, ctx, principals)
    url = f"/remediation/{remediation.id}/approve"
    for body in (
        {"expected_version": remediation.version, "approved": True},
        {"expected_version": remediation.version, "reviewer": "alice", "role": "approver"},
        {"expected_version": remediation.version, "state": "EXECUTED"},
    ):
        response = api.post(url, json=body, headers=auth("operator"))
        assert response.status_code in (403, 422)
        assert (
            api.post(url, json=body, headers=auth("approver")).status_code == 422
        )  # unknown fields rejected
    assert api.post(url, json={"expected_version": remediation.version}).status_code == 401
    assert untouched(service)


def test_no_endpoint_or_tool_lets_the_agent_approve(api):
    from aegis.agents.tools import TOOLS

    paths = set(api.app.openapi()["paths"])
    deciding = sorted(p for p in paths if any(w in p for w in ("approve", "reject", "rollback")))
    assert deciding == [
        "/remediation/{remediation_id}/approve",
        "/remediation/{remediation_id}/reject",
        "/remediation/{remediation_id}/rollback",
    ]
    assert not [name for name in TOOLS if "approve" in name and name != "request_human_approval"]
    assert not [p for p in paths if "execute" in p or "deploy" in p or "promote" in p]


def test_separation_of_duties_when_enabled(service, ctx, principals):
    service.settings.require_distinct_approver = True
    try:
        open_incident(service, ctx, principals, fault="F04", investigate=False)
        incident = rows(service, m.Incident)[0]
        service.start_investigation(
            principals["approver"], incident.id, use_llm=False, wait=True
        )  # alice requests
        (remediation,) = rows(service, m.Remediation)
        with pytest.raises(ForbiddenError, match="separation of duties"):
            service.decide(
                principals["approver"],
                remediation.id,
                "approved",
                expected_version=remediation.version,
            )
        assert untouched(service)
        decided, _ = service.decide(
            principals["approver2"],
            remediation.id,
            "approved",
            expected_version=remediation.version,
        )
        assert decided.state == "COMPLETED"
    finally:
        service.settings.require_distinct_approver = False


# --------------------------------------------------------- duplicate / replay
def test_duplicate_approval_is_refused_and_executes_once(service, ctx, principals):
    remediation = pending(service, ctx, principals)
    first, fresh = service.decide(
        principals["approver"], remediation.id, "approved", expected_version=remediation.version
    )
    assert fresh and first.state == "COMPLETED"
    for principal, version in (
        (principals["approver"], remediation.version),
        (principals["approver2"], first.version),
    ):
        with pytest.raises(ConflictError, match="already approved by alice"):
            service.decide(principal, remediation.id, "approved", expected_version=version)
    with pytest.raises(ConflictError, match="already approved"):
        service.decide(
            principals["approver2"], remediation.id, "rejected", expected_version=first.version
        )
    assert len(rows(service, m.Approval)) == 1 and len(rows(service, m.DeploymentRevision)) == 2
    executes = [
        e
        for e in rows(service, m.AuditEvent, entity_id=remediation.id)
        if e.action == "remediation.execute"
    ]
    assert len(executes) == 1 and len(denied(service, remediation.id)) == 3


def test_idempotent_replay_returns_the_recorded_outcome_without_reexecuting(
    api, service, ctx, principals
):
    remediation = pending(service, ctx, principals)
    url, body = f"/remediation/{remediation.id}/approve", {"expected_version": remediation.version}
    headers = {**auth("approver"), "Idempotency-Key": "key-abc-123"}
    first = api.post(url, json=body, headers=headers)
    replay = api.post(url, json=body, headers=headers)
    assert first.status_code == replay.status_code == 200
    assert (
        first.json()["state"] == replay.json()["state"] == "COMPLETED"
        and first.json()["executed"] is True
    )
    assert (
        first.json()["version"] == replay.json()["version"]
        and len(rows(service, m.DeploymentRevision)) == 2
    )
    assert (
        api.post(
            url, json=body, headers={**auth("approver"), "Idempotency-Key": "other-key"}
        ).status_code
        == 409
    )
    assert (
        api.post(
            url, json=body, headers={**auth("approver2"), "Idempotency-Key": "key-abc-123"}
        ).status_code
        == 409
    )
    assert api.post(url, json=body, headers=auth("approver")).status_code == 409


def test_concurrent_approvals_result_in_exactly_one_execution(service, ctx, principals):
    remediation = pending(service, ctx, principals)
    outcomes, barrier = [], threading.Barrier(6)

    def attempt(principal):
        barrier.wait()
        try:
            service.decide(
                principal, remediation.id, "approved", expected_version=remediation.version
            )
            outcomes.append("ok")
        except ConflictError:
            outcomes.append("conflict")
        except Exception as exc:  # anything else is a bug
            outcomes.append(repr(exc))

    threads = [
        threading.Thread(target=attempt, args=(principals["approver" if i % 2 else "approver2"],))
        for i in range(6)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert sorted(outcomes) == ["conflict"] * 5 + ["ok"]
    assert len(rows(service, m.Approval)) == 1 and len(rows(service, m.DeploymentRevision)) == 2
    assert sum(r.active for r in rows(service, m.DeploymentRevision)) == 1


# ------------------------------------------------------------------- staleness
def test_stale_approval_of_an_outdated_version_is_refused(service, ctx, principals):
    remediation = pending(service, ctx, principals)
    with pytest.raises(ConflictError, match="stale decision"):
        service.decide(
            principals["approver"],
            remediation.id,
            "approved",
            expected_version=remediation.version - 1,
        )
    with pytest.raises(ConflictError, match="stale decision"):
        service.decide(
            principals["approver"],
            remediation.id,
            "approved",
            expected_version=remediation.version + 5,
        )
    assert untouched(service) and len(denied(service, remediation.id)) == 2


def test_approval_is_refused_when_the_deployment_changed_after_validation(service, ctx, principals):
    remediation = pending(service, ctx, principals)
    service.set_deployment(
        principals["operator"],
        "adult",
        DeploymentState(
            dataset="adult",
            model_version=ctx.record.version,
            ingestion=IngestionConfig(deduplicate=True),
        ),
        "unrelated change",
    )
    with pytest.raises(ConflictError, match="stale remediation"):
        service.decide(
            principals["approver"], remediation.id, "approved", expected_version=remediation.version
        )
    # The whole decision rolled back: no approval row, no execution, still pending.
    assert (
        rows(service, m.Approval) == []
        and rows(service, m.Remediation)[0].state == "PENDING_APPROVAL"
    )
    assert [r.revision for r in rows(service, m.DeploymentRevision)] == [1, 2]
    assert "stale remediation" in denied(service, remediation.id)[0].detail["reason"]


@pytest.mark.parametrize("fault,state", [("F05", "VALIDATION_FAILED")])
def test_unvalidated_or_failed_remediation_cannot_be_approved(
    service, ctx, principals, fault, state
):
    open_incident(service, ctx, principals, fault=fault)
    (remediation,) = rows(service, m.Remediation)
    assert remediation.state == state
    with pytest.raises(ConflictError, match="only PENDING_APPROVAL"):
        service.decide(
            principals["approver"], remediation.id, "approved", expected_version=remediation.version
        )
    assert untouched(service)


def test_proposed_remediation_cannot_skip_validation(service, ctx, principals):
    incident, _ = open_incident(service, ctx, principals)
    (first,) = rows(service, m.Remediation)
    service.decide(principals["approver"], first.id, "rejected", expected_version=first.version)
    proposed, _ = service.create_remediation(principals["operator"], incident.id)
    assert proposed.state == "PROPOSED" and proposed.validation is None
    with pytest.raises(ConflictError, match="only PENDING_APPROVAL"):
        service.decide(
            principals["approver"], proposed.id, "approved", expected_version=proposed.version
        )
    with pytest.raises(ConflictError, match="only an executed remediation"):
        service.rollback(principals["approver"], proposed.id, "nothing to undo")
    assert len(rows(service, m.DeploymentRevision)) == 1


def test_rejected_remediation_stays_rejected(service, ctx, principals):
    remediation = pending(service, ctx, principals)
    rejected, _ = service.decide(
        principals["approver"], remediation.id, "rejected", expected_version=remediation.version
    )
    with pytest.raises(ConflictError):
        service.decide(
            principals["approver2"], remediation.id, "approved", expected_version=rejected.version
        )
    assert (
        rows(service, m.Remediation)[0].state == "REJECTED"
        and len(rows(service, m.DeploymentRevision)) == 1
    )


# ----------------------------------------------------- repeated execution/rollback
def test_repeated_rollback_and_rollback_after_later_changes(service, ctx, principals):
    remediation = pending(service, ctx, principals)
    service.decide(
        principals["approver"], remediation.id, "approved", expected_version=remediation.version
    )
    service.set_deployment(
        principals["operator"],
        "adult",
        DeploymentState(dataset="adult", model_version=ctx.record.version),
        "later change",
    )
    with pytest.raises(ConflictError, match="moved on"):
        service.rollback(principals["approver"], remediation.id, "too late")
    assert (
        rows(service, m.Remediation)[0].state == "COMPLETED"
        and len(rows(service, m.DeploymentRevision)) == 3
    )


def test_tampered_validation_record_cannot_be_executed(service, ctx, principals):
    """Even with direct database access to the proposal, execution re-derives the state and refuses."""
    remediation = pending(service, ctx, principals)
    with service.db.session() as session:
        row = session.get(m.Remediation, remediation.id)
        row.proposal = {
            **row.proposal,
            "parameters": {**row.proposal["parameters"], "factor": 12345.0},
        }
    with pytest.raises(ConflictError, match="cannot execute"):
        service.decide(
            principals["approver"], remediation.id, "approved", expected_version=remediation.version
        )
    assert untouched(service)


def test_database_failure_during_execution_leaves_no_partial_state(
    service, ctx, principals, monkeypatch
):
    remediation = pending(service, ctx, principals)

    def fail(*args, **kwargs):
        raise RuntimeError("connection lost mid-transaction")

    monkeypatch.setattr(type(service), "_verify", fail)
    with pytest.raises(RuntimeError):
        service.decide(
            principals["approver"], remediation.id, "approved", expected_version=remediation.version
        )
    monkeypatch.undo()
    # Nothing committed: not the approval, not the new revision, not the state change.
    assert untouched(service) and rows(service, m.Remediation)[0].state == "PENDING_APPROVAL"
    decided, _ = service.decide(
        principals["approver"], remediation.id, "approved", expected_version=remediation.version
    )
    assert decided.state == "COMPLETED"
