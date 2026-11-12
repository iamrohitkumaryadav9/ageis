"""Remediation lifecycle reconciliation: every record is accounted for or reported."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aegis.remediation.reconciliation import (
    LifecycleEvent,
    Stage,
    benchmark_events,
    events_from_audit,
    events_from_legacy_row,
    reconcile,
)

S = Stage


def lifecycle(key="a", *, upto=S.ROLLED_BACK, action="fix_ingestion_scale", **overrides):
    """A complete, valid lifecycle for one proposal, truncated after ``upto``."""
    steps = [
        (S.PROPOSED, "actionable", {}),
        (
            S.POLICY_DECIDED,
            "PROPOSE_ACTION",
            {"policy_decision_id": f"POL-{key}-pre", "phase": "pre_validation"},
        ),
        (S.VALIDATION, "passed", {"validation_id": f"VAL-{key}"}),
        (
            S.POLICY_DECIDED,
            "REQUEST_HUMAN_APPROVAL",
            {"policy_decision_id": f"POL-{key}-post", "phase": "post_validation"},
        ),
        (
            S.APPROVAL_REQUESTED,
            "requested",
            {"approval_id": f"APR-{key}", "validation_id": f"VAL-{key}"},
        ),
        (S.APPROVAL_DECIDED, "approved", {"approval_id": f"APR-{key}"}),
        (S.EXECUTED, "succeeded", {"execution_id": f"EXE-{key}", "approval_id": f"APR-{key}"}),
        (S.VERIFIED, "resolved", {"execution_id": f"EXE-{key}"}),
        (S.ROLLED_BACK, "succeeded", {"rollback_id": f"RBK-{key}", "execution_id": f"EXE-{key}"}),
    ]
    events = []
    for index, (stage, outcome, fields) in enumerate(steps):
        outcome = overrides.get(stage.value, outcome)
        events.append(
            LifecycleEvent(
                event_id=f"E-{key}-{index}",
                stage=stage,
                sequence=index,
                outcome=outcome,
                proposal_id=f"PRP-{key}",
                incident_id=f"INC-{key}",
                scenario_id=f"SCN-{key}",
                action=action,
                **fields,
            )
        )
        if stage is upto and (stage is not S.POLICY_DECIDED or index == 3 or upto_first(overrides)):
            break
    return events


def upto_first(overrides):
    return overrides.get("_stop_at_first_policy", False)


def codes(report):
    return sorted(report["failure_counts"])


def status(report, key="a"):
    return report["proposal_statuses"][f"PRP-{key}"]


def deferred(key):
    return [
        LifecycleEvent(
            event_id=f"E-{key}-0",
            stage=S.PROPOSED,
            sequence=0,
            outcome="no_action",
            proposal_id=f"PRP-{key}",
            incident_id=f"INC-{key}",
            action="no_action",
        ),
        LifecycleEvent(
            event_id=f"E-{key}-1",
            stage=S.POLICY_DECIDED,
            sequence=1,
            outcome="DEFER",
            proposal_id=f"PRP-{key}",
            incident_id=f"INC-{key}",
            policy_decision_id=f"POL-{key}-pre",
            phase="pre_validation",
        ),
    ]


def rejected_before_validation(key):
    events = lifecycle(key)[:2]
    events[1] = events[1].model_copy(update={"outcome": "REJECT"})
    return events


def failed_validation(key):
    events = lifecycle(key)[:4]
    events[2] = events[2].model_copy(update={"outcome": "failed"})
    events[3] = events[3].model_copy(update={"outcome": "REJECT"})
    return events


# --------------------------------------------------------------- happy paths
def test_complete_lifecycle_reconciles():
    report = reconcile(lifecycle(), expected_proposals=1, require_closed=True)
    assert report["reconciled"] and report["failures"] == []
    assert status(report) == "rolled_back"
    assert all(i["holds"] for i in report["accounting_identities"])
    assert report["schema_version"] == "lifecycle-1.0"


def test_every_legitimate_stopping_point_has_a_named_status():
    approver_rejects = lifecycle("r")[:6]
    approver_rejects[5] = approver_rejects[5].model_copy(update={"outcome": "rejected"})
    unresolved = lifecycle("u", upto=S.VERIFIED, VERIFIED="unresolved")
    events = [
        *deferred("d"),
        *rejected_before_validation("p"),
        *failed_validation("v"),
        *approver_rejects,
        *unresolved,
        *lifecycle("ok", upto=S.VERIFIED),
        *lifecycle("rb"),
    ]
    report = reconcile(events, expected_proposals=7, require_closed=True)
    assert report["reconciled"], report["failures"]
    assert report["terminal_status_counts"] == {
        "deferred_no_action": 1,
        "rejected_by_policy_before_validation": 1,
        "rejected_validation_failed": 1,
        "rejected_by_approver": 1,
        "executed_resolved": 1,
        "executed_unresolved": 1,
        "rolled_back": 1,
    }


def test_rates_carry_numerator_denominator_and_definition():
    events = [
        *lifecycle("a"),
        *failed_validation("b"),
        *rejected_before_validation("c"),
        *deferred("d"),
    ]
    rates = reconcile(events)["rates"]
    v = rates["validation_success_rate"]
    assert (v["numerator"], v["denominator"], v["value"]) == (1, 2, 0.5)
    assert v["definition"] == "validation attempts that passed / all validation attempts"
    # a proposal that never reached validation is not in the validation denominator
    assert rates["execution_resolution_rate"]["denominator"] == 1
    assert rates["rollback_success_rate"]["definition"].endswith("/ rollback attempts")


def test_rate_with_empty_denominator_is_none_not_zero_or_one():
    rates = reconcile(deferred("d"))["rates"]
    assert rates["validation_success_rate"]["value"] is None
    assert rates["rollback_success_rate"] == {
        "numerator": 0,
        "denominator": 0,
        "value": None,
        "definition": "rollbacks that restored the previous state / rollback attempts",
    }


def test_rollback_denominator_is_attempts_not_executions():
    # two executions, one rolled back: the rate is 1/1, and execution resolution is over 2
    events = [*lifecycle("a"), *lifecycle("b", upto=S.VERIFIED)]
    rates = reconcile(events)["rates"]
    assert (
        rates["rollback_success_rate"]["numerator"],
        rates["rollback_success_rate"]["denominator"],
    ) == (1, 1)
    assert rates["execution_resolution_rate"]["denominator"] == 2


# ------------------------------------------------------------ failure modes
def test_duplicate_event_id_is_reported():
    events = lifecycle()
    events.append(events[2].model_copy())
    report = reconcile(events)
    assert not report["reconciled"]
    assert "duplicate_event_id" in codes(report)


def test_duplicate_stage_is_reported():
    events = lifecycle()
    events.append(events[2].model_copy(update={"event_id": "E-dup", "sequence": 9}))
    report = reconcile(events)
    assert "duplicate_stage" in codes(report)


def test_two_policy_decisions_in_the_same_phase_are_a_duplicate():
    events = lifecycle()
    events[3] = events[3].model_copy(update={"phase": "pre_validation"})
    assert "duplicate_stage" in codes(reconcile(events))


def test_validation_without_a_proposal_is_reported_not_dropped():
    orphan = [e for e in lifecycle() if e.stage is not S.PROPOSED]
    report = reconcile(orphan, expected_proposals=1)
    assert "missing_link" in codes(report)
    assert report["stage_counts"]["VALIDATION"] == 1  # still counted, and flagged


def test_validation_without_policy_permission_is_reported():
    events = [
        e for e in lifecycle() if not (e.stage is S.POLICY_DECIDED and e.phase == "pre_validation")
    ]
    assert "missing_link" in codes(reconcile(events))


def test_approval_request_after_failed_validation_is_reported():
    events = lifecycle(upto=S.APPROVAL_REQUESTED, VALIDATION="failed")
    report = reconcile(events)
    assert "missing_link" in codes(report)
    assert any("without a passed validation" in f["message"] for f in report["failures"])


def test_execution_without_approval_is_reported():
    events = [e for e in lifecycle() if e.stage is not S.APPROVAL_DECIDED]
    report = reconcile(events)
    assert any(
        f["code"] == "missing_link" and "EXECUTED" in f["message"] for f in report["failures"]
    )


def test_execution_after_a_rejected_approval_is_reported():
    events = lifecycle(APPROVAL_DECIDED="rejected")
    report = reconcile(events)
    assert any("EXECUTED recorded without an approval" in f["message"] for f in report["failures"])


def test_rollback_without_execution_is_reported():
    events = [e for e in lifecycle() if e.stage not in (S.EXECUTED, S.VERIFIED)]
    assert any("ROLLED_BACK" in f["message"] for f in reconcile(events)["failures"])


def test_out_of_order_events_are_reported():
    events = lifecycle()
    events[6] = events[6].model_copy(update={"sequence": 0})  # execution "before" everything
    assert "out_of_order" in codes(reconcile(events))


def test_dangling_reference_is_reported():
    events = lifecycle()
    events[6] = events[6].model_copy(update={"approval_id": "APR-somebody-else"})
    assert "dangling_reference" in codes(reconcile(events))


def test_missing_stage_identifier_is_reported():
    events = lifecycle()
    events[2] = events[2].model_copy(update={"validation_id": None})
    assert "missing_stage_id" in codes(reconcile(events))


def test_invalid_outcome_is_reported():
    events = lifecycle()
    events[2] = events[2].model_copy(update={"outcome": "sort-of"})
    assert "invalid_outcome" in codes(reconcile(events))


def test_malformed_record_is_reported_and_the_rest_still_reconciles():
    events = [e.model_dump() for e in lifecycle()] + [{"event_id": "bad", "proposal_id": "PRP-x"}]
    report = reconcile(events)
    assert report["failure_counts"] == {"malformed_event": 1}
    assert report["proposals"] == 1


@pytest.mark.parametrize(
    ("upto", "message"),
    [
        (S.PROPOSED, "never decided by policy"),
        (S.VALIDATION, "validation result was never decided"),
        (S.APPROVAL_DECIDED, "approved but never executed"),
        (S.EXECUTED, "non-terminal state (executed_unverified)"),
    ],
)
def test_incomplete_lifecycle_is_a_failure(upto, message):
    report = reconcile(lifecycle(upto=upto, _stop_at_first_policy=True))
    assert "incomplete_lifecycle" in codes(report), report["failures"]
    assert any(message in f["message"] for f in report["failures"])


def test_pending_approval_is_open_in_a_live_system_and_a_failure_in_a_closed_run():
    events = lifecycle(upto=S.APPROVAL_REQUESTED)
    live = reconcile(events)
    assert live["reconciled"] and status(live) == "pending_approval"
    closed = reconcile(events, require_closed=True)
    assert codes(closed) == ["incomplete_lifecycle"]


def test_execution_failure_and_rollback_failure_are_outcomes_not_lost_records():
    exec_failed = lifecycle("x", upto=S.EXECUTED, EXECUTED="failed")
    rollback_failed = lifecycle("y", ROLLED_BACK="failed")
    report = reconcile([*exec_failed, *rollback_failed], expected_proposals=2)
    assert report["reconciled"], report["failures"]
    assert status(report, "x") == "execution_failed" and status(report, "y") == "rollback_failed"
    assert report["rates"]["execution_success_rate"]["numerator"] == 1
    assert report["rates"]["execution_success_rate"]["denominator"] == 2
    assert report["rates"]["rollback_success_rate"]["value"] == 0.0


def test_verification_after_a_failed_execution_is_reported():
    events = lifecycle(upto=S.VERIFIED, EXECUTED="failed")
    assert any(
        "VERIFIED recorded without a successful execution" in f["message"]
        for f in reconcile(events)["failures"]
    )


def test_proposal_count_mismatch_is_reported():
    report = reconcile(lifecycle(), expected_proposals=3)
    assert codes(report) == ["proposal_count_mismatch"]


def test_events_of_one_proposal_naming_two_incidents_are_reported():
    events = lifecycle()
    events[4] = events[4].model_copy(update={"incident_id": "INC-other"})
    assert "inconsistent_incident" in codes(reconcile(events))


def test_two_proposals_with_colliding_events_do_not_mask_each_other():
    report = reconcile([*lifecycle("a"), *lifecycle("b")[:3]], expected_proposals=2)
    assert status(report, "a") == "rolled_back"
    assert "incomplete_lifecycle" in codes(report)  # b stopped after validation


# ----------------------------------------------------------- benchmark rows
def row(**kw):
    base = {
        "action": "fix_ingestion_scale",
        "policy_outcome": "REQUEST_HUMAN_APPROVAL",
        "validated": True,
        "approval_requested": True,
        "executed_resolves_incident": True,
        "rollback_restores_state": True,
    }
    return {**base, **kw}


def test_legacy_rows_become_derived_events_and_say_what_is_inferred():
    events = events_from_legacy_row("SCN-1", "B", row())
    assert all(e.derived for e in events)
    assert events[1].detail["inferred"].startswith("validation only runs after PROPOSE_ACTION")
    report = reconcile(events, expected_proposals=1, require_closed=True)
    assert report["reconciled"] and report["derived_events"] == len(events)


def test_crashed_and_remediation_less_investigations_are_surfaced():
    records = [
        {"scenario_id": "SCN-1", "configs": {"B": {"remediation": row()}}},
        {"scenario_id": "SCN-2", "configs": {"B": {"error": "RuntimeError: boom"}}},
        {"scenario_id": "SCN-3", "configs": {"B": {"remediation": None}}},
        {"scenario_id": "SCN-4", "configs": {}},  # undetected: not investigated
    ]
    events, coverage = benchmark_events(records, "B")
    assert coverage["investigated"] == 3
    assert coverage["crashed"] == ["SCN-2"] and coverage["without_remediation"] == ["SCN-3"]
    report = reconcile(events, expected_proposals=coverage["investigated"])
    assert codes(report) == ["proposal_count_mismatch"]  # 3 investigated, 1 lifecycle


def test_historical_run_main_reconciles_and_explains_87_versus_85():
    path = Path("results/aegis-ir/main/scenarios.jsonl")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    events, coverage = benchmark_events(records, "B")
    report = reconcile(events, expected_proposals=coverage["investigated"], require_closed=True)
    assert report["reconciled"], report["failures"]
    assert coverage == {
        "config": "B",
        "investigated": 119,
        "recorded": 0,
        "derived_from_legacy_fields": 119,
        "crashed": [],
        "without_remediation": [],
    }
    assert report["terminal_status_counts"] == {
        "deferred_no_action": 32,
        "rejected_by_policy_before_validation": 2,
        "rejected_validation_failed": 17,
        "rolled_back": 68,
    }
    actionable = report["outcome_counts"]["PROPOSED:actionable"]
    assert actionable == 87 and report["stage_counts"]["VALIDATION"] == 85
    assert (
        actionable - report["stage_counts"]["VALIDATION"]
        == 2
        == report["terminal_status_counts"]["rejected_by_policy_before_validation"]
    )
    rates = report["rates"]
    assert (
        rates["validation_success_rate"]["numerator"],
        rates["validation_success_rate"]["denominator"],
    ) == (68, 85)
    assert (
        rates["execution_resolution_rate"]["numerator"],
        rates["execution_resolution_rate"]["denominator"],
    ) == (67, 68)
    assert (
        rates["rollback_success_rate"]["numerator"],
        rates["rollback_success_rate"]["denominator"],
    ) == (68, 68)


# -------------------------------------------------------------- audit trail
def audit(i, action, rid="REM-1", outcome="ok", **detail):
    return {
        "id": i,
        "action": action,
        "entity_id": rid,
        "incident_id": "INC-1",
        "outcome": outcome,
        "actor": "alice",
        "detail": detail,
    }


def test_audit_trail_is_converted_and_denied_attempts_are_not_lifecycle_stages():
    rows = [
        audit(1, "remediation.proposed", remediation_action="fix_ingestion_scale"),
        audit(2, "remediation.start_validation"),
        audit(3, "remediation.validation_passed"),
        audit(4, "remediation.request_approval"),
        audit(5, "remediation.approve", outcome="denied"),
        audit(6, "remediation.approve"),
        audit(7, "remediation.execute", deployment_revision=2),
        audit(8, "remediation.verify"),
        audit(9, "remediation.rollback"),
    ]
    events = events_from_audit(rows)
    assert [e.stage for e in events] == [
        S.PROPOSED,
        S.VALIDATION,
        S.APPROVAL_REQUESTED,
        S.APPROVAL_DECIDED,
        S.EXECUTED,
        S.VERIFIED,
        S.ROLLED_BACK,
    ]
    assert events[4].detail["deployment_revision"] == 2
    report = reconcile(events, require_policy_events=False)
    assert report["reconciled"], report["failures"]
    assert report["proposal_statuses"] == {"REM-1": "rolled_back"}


def test_audit_rejection_before_any_request_is_a_policy_decision_not_an_approver_decision():
    rows = [
        audit(1, "remediation.proposed"),
        audit(2, "remediation.start_validation"),
        audit(3, "remediation.validation_failed"),
    ]
    assert reconcile(events_from_audit(rows), require_policy_events=False)["proposal_statuses"] == {
        "REM-1": "rejected_validation_failed"
    }
    rows = [audit(1, "remediation.proposed"), audit(2, "remediation.reject")]
    events = events_from_audit(rows)
    assert events[1].stage is S.POLICY_DECIDED and events[1].outcome == "REJECT"


def test_audit_trail_with_a_missing_row_fails_reconciliation():
    rows = [audit(1, "remediation.proposed"), audit(7, "remediation.execute")]
    report = reconcile(events_from_audit(rows), require_policy_events=False)
    assert "missing_link" in codes(report)
