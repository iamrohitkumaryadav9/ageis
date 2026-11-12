"""End-to-end: a complete incident lifecycle driven only through the public HTTP API.

dataset ingestion -> baseline model -> fault injection -> monitoring -> incident -> investigation
-> hypotheses -> runbook retrieval -> remediation proposal -> validation -> human approval
-> execution (or rejection) -> audit record -> rollback
"""

import io

import pytest

from tests.conftest import auth

pytestmark = pytest.mark.postgres


def upload(api, frame, role="current"):
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    response = api.post(
        f"/data/{role}",
        data={"dataset": "adult"},
        files={"file": ("batch.csv", buffer.getvalue().encode(), "text/csv")},
        headers=auth(),
    )
    assert response.status_code == 201, response.text
    return response.json()


def run_to_pending_approval(api, ctx, fault="F04", use_llm=False):
    scenario = ctx.scenario(fault, "high", seed=4)
    batch = upload(api, scenario.current)
    monitored = api.post(
        "/monitor/run", json={"dataset": "adult", "batch_id": batch["id"]}, headers=auth()
    ).json()
    incident = monitored["incident"]
    assert monitored["run"]["incident_detected"] and incident["status"] == "open"
    investigation = api.post(
        f"/incidents/{incident['id']}/investigate",
        json={"use_llm": use_llm, "wait": True},
        headers=auth(),
    )
    assert investigation.status_code == 200, investigation.text
    detail = api.get(f"/investigations/{investigation.json()['id']}", headers=auth("viewer")).json()
    return scenario, batch, incident, detail


def test_full_lifecycle_approve_execute_audit_rollback(api, ctx):
    # A clean batch first: the system must stay quiet.
    clean = upload(api, ctx.clean(seed=11).current)
    quiet = api.post(
        "/monitor/run", json={"dataset": "adult", "batch_id": clean["id"]}, headers=auth()
    ).json()
    assert quiet["incident"] is None and not quiet["run"]["incident_detected"]

    scenario, batch, incident, investigation = run_to_pending_approval(api, ctx)
    feature = scenario.truth.affected_features[0]

    # Hypotheses are evidence-grounded and worded as hypotheses.
    top = investigation["hypotheses"][0]
    assert top["root_cause"] == "scaling_error" and feature in top["payload"]["affected_features"]
    incident_detail = api.get(f"/incidents/{incident['id']}", headers=auth("viewer")).json()
    evidence_ids = {e["evidence_id"] for e in incident_detail["evidence"]}
    assert top["supporting_evidence_ids"] and set(top["supporting_evidence_ids"]) <= evidence_ids
    assert "not established causation" in investigation["diagnosis"]["conclusion"]
    assert any(ref["doc_id"] == "RB-004" for ref in investigation["runbook_refs"])

    # A validated proposal is waiting; nothing has been executed.
    (remediation,) = incident_detail["remediations"]
    assert (
        remediation["state"] == "PENDING_APPROVAL"
        and remediation["action"] == "fix_ingestion_scale"
    )
    assert remediation["validation"]["passed"] and remediation["executed_revision"] is None
    assert incident_detail["status"] == "awaiting_approval"
    overview = api.get("/overview", headers=auth("viewer")).json()
    assert overview["pending_approvals"] == 1
    assert (
        next(d for d in overview["datasets"] if d["dataset"] == "adult")["deployment"]["revision"]
        == 1
    )

    # The operator who ran the investigation cannot approve it.
    url = f"/remediation/{remediation['id']}"
    assert (
        api.post(
            f"{url}/approve", json={"expected_version": remediation["version"]}, headers=auth()
        ).status_code
        == 403
    )

    # A human approver does; the action executes and is verified.
    approved = api.post(
        f"{url}/approve",
        json={"expected_version": remediation["version"], "comment": "confirmed"},
        headers=auth("approver"),
    )
    assert approved.status_code == 200, approved.text
    body = approved.json()
    assert body["state"] == "COMPLETED" and body["executed"] and body["executed_revision"] == 2
    assert body["approval"]["reviewer"] == "alice" and body["approval"]["decision"] == "approved"
    assert body["verification"]["passed"]
    assert (
        api.get(f"/incidents/{incident['id']}", headers=auth("viewer")).json()["status"]
        == "mitigated"
    )

    # The same data is healthy under the new deployment.
    healthy = api.post(
        "/monitor/run", json={"dataset": "adult", "batch_id": batch["id"]}, headers=auth()
    ).json()
    assert healthy["incident"] is None and healthy["run"]["deployment_revision"] == 2

    # The audit trail tells the whole story, in order, with identities.
    trail = api.get(
        f"/audit?incident_id={incident['id']}&limit=200", headers=auth("viewer")
    ).json()["items"]
    sequence = [e["action"] for e in trail if e["outcome"] == "ok"]
    expected = [
        "monitoring.run",
        "incident.opened",
        "investigation.started",
        "remediation.proposed",
        "remediation.start_validation",
        "remediation.validation_passed",
        "remediation.request_approval",
        "investigation.completed",
        "remediation.approve",
        "deployment.revision.activated",
        "remediation.execute",
        "remediation.verify",
    ]
    assert [a for a in sequence if a in expected] == expected
    denied = [e for e in trail if e["outcome"] == "denied"]
    assert [(e["actor"], e["action"]) for e in denied] == [("bob", "remediation.approve")]
    granted = next(
        e for e in trail if e["action"] == "remediation.approve" and e["outcome"] == "ok"
    )
    assert granted["actor"] == "alice" and granted["actor_role"] == "approver"

    # Rollback restores the previous deployment state and is itself audited.
    rolled = api.post(
        f"{url}/rollback",
        json={"reason": "upstream reverted the unit change"},
        headers=auth("approver2"),
    )
    assert (
        rolled.status_code == 200
        and rolled.json()["state"] == "ROLLED_BACK"
        and rolled.json()["rollback_revision"] == 3
    )
    deployment = next(
        d
        for d in api.get("/overview", headers=auth("viewer")).json()["datasets"]
        if d["dataset"] == "adult"
    )["deployment"]
    assert deployment["revision"] == 3 and deployment["ingestion"]["unit_scale"] == {}
    broken_again = api.post(
        "/monitor/run", json={"dataset": "adult", "batch_id": batch["id"]}, headers=auth()
    ).json()
    assert broken_again["run"][
        "incident_detected"
    ]  # the fault is visible again once the fix is removed

    # Every remediation record in the audit trail is accounted for by the reconciler.
    report = api.get(
        f"/remediation-lifecycle/reconciliation?incident_id={incident['id']}",
        headers=auth("viewer"),
    ).json()
    assert report["reconciled"], report["failures"]
    assert report["proposals"] == 1
    assert report["proposal_statuses"] == {remediation["id"]: "rolled_back"}
    assert report["denied_attempts_excluded"] >= 1  # the operator's refused approval
    assert report["stage_counts"]["EXECUTED"] == 1 and report["stage_counts"]["ROLLED_BACK"] == 1
    assert report["rates"]["rollback_success_rate"]["denominator"] == 1
    assert api.get("/remediation-lifecycle/reconciliation").status_code == 401


def test_full_lifecycle_rejection_path(api, ctx):
    _, _, incident, _ = run_to_pending_approval(api, ctx, fault="F06")
    (remediation,) = api.get(f"/incidents/{incident['id']}", headers=auth("viewer")).json()[
        "remediations"
    ]
    rejected = api.post(
        f"/remediation/{remediation['id']}/reject",
        json={
            "expected_version": remediation["version"],
            "comment": "duplicates are expected this week",
        },
        headers=auth("approver"),
    )
    assert (
        rejected.status_code == 200
        and rejected.json()["state"] == "REJECTED"
        and not rejected.json()["executed"]
    )
    assert rejected.json()["approval"]["comment"] == "duplicates are expected this week"
    assert (
        api.get("/overview", headers=auth("viewer")).json()["datasets"][0]["deployment"]["revision"]
        == 1
    )
    trail = [
        e["action"]
        for e in api.get(f"/audit?incident_id={incident['id']}", headers=auth("viewer")).json()[
            "items"
        ]
    ]
    assert trail[-1] == "remediation.reject" and "remediation.execute" not in trail
    report = api.get("/remediation-lifecycle/reconciliation", headers=auth("viewer")).json()
    assert report["reconciled"], report["failures"]
    assert report["proposal_statuses"][remediation["id"]] == "rejected_by_approver"
    assert report["stage_counts"]["EXECUTED"] == 0


def test_lifecycle_with_llm_offline_still_completes_and_says_so(api, ctx, service):
    from tests.helpers import FakeOllama

    service.llm = FakeOllama(reachable=False).client()
    _, _, incident, investigation = run_to_pending_approval(api, ctx, use_llm=True)
    assert investigation["status"] == "completed" and investigation["llm_status"] == "unavailable"
    assert any("LLM investigation unavailable" in e for e in investigation["errors"])
    assert investigation["hypotheses"][0]["root_cause"] == "scaling_error"
    assert investigation["hypotheses"][0]["ranked_by"] == "rules"
    assert (
        api.get(f"/incidents/{incident['id']}", headers=auth("viewer")).json()["status"]
        == "awaiting_approval"
    )


@pytest.mark.ollama
def test_lifecycle_with_the_real_local_model(api, ctx, service, settings):
    """Runs only when a local Ollama with the configured model is actually reachable."""
    from aegis.agents.ollama import OllamaClient

    client = OllamaClient(settings)
    if not client.health().available:
        pytest.skip("local Ollama model not available (optional component)")
    service.llm = client
    _, _, incident, investigation = run_to_pending_approval(api, ctx, use_llm=True)
    assert investigation["status"] == "completed"
    assert investigation["llm_status"] in ("available", "degraded", "unavailable")
    assert investigation["llm_calls"] and all(
        "duration_ms" in c for c in investigation["llm_calls"]
    )
    known = {
        e["evidence_id"]
        for e in api.get(f"/incidents/{incident['id']}", headers=auth("viewer")).json()["evidence"]
    }
    for hypothesis in investigation[
        "hypotheses"
    ]:  # whatever the model said, only real evidence is cited
        assert set(hypothesis["supporting_evidence_ids"]) <= known
    by_llm = [t for t in investigation["tool_traces"] if t["requested_by"] == "llm"]
    assert all(
        t["tool"]
        in (
            "get_data_quality_report",
            "get_data_drift_report",
            "get_model_metrics",
            "inspect_feature_distribution",
            "compare_model_versions",
        )
        or t["status"] == "rejected"
        for t in by_llm
    )
    assert investigation["policy_decision"]["requires_approval"] in (True, False)
