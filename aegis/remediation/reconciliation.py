"""Canonical remediation lifecycle events and their reconciliation.

One remediation proposal moves through at most these stages::

    PROPOSED -> POLICY_DECIDED(pre) -> VALIDATION -> POLICY_DECIDED(post)
             -> APPROVAL_REQUESTED -> APPROVAL_DECIDED -> EXECUTED -> VERIFIED -> ROLLED_BACK

Most proposals stop early, and every stopping point is a legitimate, named outcome (deferred,
rejected by policy, failed validation, rejected by the approver, ...). What is *not* legitimate
is a record that cannot be placed: a validation with no proposal, an execution with no approval,
the same stage recorded twice, a lifecycle that just stops in the middle. :func:`reconcile`
reports each of those as a failure. Nothing is dropped from a count because it looked odd.

Identifiers. Each stage has its own id, because they are different things: a proposal can be
decided by policy twice, and an approval is not an execution. Events are linked by
``proposal_id`` (always), ``incident_id``/``scenario_id`` (context), and by the explicit
``links`` an event declares to earlier stages (an execution names the approval it ran under).

Two sources produce these events: the benchmark (``aegis.evaluation.benchmark``), and the
database audit trail (:func:`events_from_audit`).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "lifecycle-1.0"


class Stage(StrEnum):
    PROPOSED = "PROPOSED"
    POLICY_DECIDED = "POLICY_DECIDED"
    VALIDATION = "VALIDATION"
    APPROVAL_REQUESTED = "APPROVAL_REQUESTED"
    APPROVAL_DECIDED = "APPROVAL_DECIDED"
    EXECUTED = "EXECUTED"
    VERIFIED = "VERIFIED"
    ROLLED_BACK = "ROLLED_BACK"
    CANCELLED = "CANCELLED"


#: The id field that names each stage's own record.
STAGE_ID_FIELD: dict[Stage, str | None] = {
    Stage.PROPOSED: "proposal_id",
    Stage.POLICY_DECIDED: "policy_decision_id",
    Stage.VALIDATION: "validation_id",
    Stage.APPROVAL_REQUESTED: "approval_id",
    Stage.APPROVAL_DECIDED: "approval_id",
    Stage.EXECUTED: "execution_id",
    Stage.VERIFIED: "execution_id",
    Stage.ROLLED_BACK: "rollback_id",
    Stage.CANCELLED: None,
}
#: Outcomes each stage may carry.
OUTCOMES: dict[Stage, frozenset[str]] = {
    Stage.PROPOSED: frozenset({"actionable", "no_action", "investigate_further"}),
    Stage.POLICY_DECIDED: frozenset(
        {"PROPOSE_ACTION", "REQUEST_HUMAN_APPROVAL", "REJECT", "DEFER", "INVESTIGATE_MORE"}
    ),
    Stage.VALIDATION: frozenset({"passed", "failed"}),
    Stage.APPROVAL_REQUESTED: frozenset({"requested"}),
    Stage.APPROVAL_DECIDED: frozenset({"approved", "rejected"}),
    Stage.EXECUTED: frozenset({"succeeded", "failed"}),
    Stage.VERIFIED: frozenset({"resolved", "unresolved"}),
    Stage.ROLLED_BACK: frozenset({"succeeded", "failed"}),
    Stage.CANCELLED: frozenset({"cancelled"}),
}


class LifecycleEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    stage: Stage
    sequence: int = Field(ge=0)  # order within one proposal's lifecycle
    outcome: str
    proposal_id: str = Field(min_length=1)
    incident_id: str = Field(min_length=1)
    scenario_id: str | None = None
    investigation_id: str | None = None
    validation_id: str | None = None
    policy_decision_id: str | None = None
    approval_id: str | None = None
    execution_id: str | None = None
    rollback_id: str | None = None
    phase: str | None = None  # POLICY_DECIDED only: "pre_validation" | "post_validation"
    action: str | None = None
    simulated: bool = False  # True when no human/API/database was involved (benchmark)
    derived: bool = False  # True when reconstructed from summary fields, not recorded live
    detail: dict[str, Any] = Field(default_factory=dict)


class Failure(BaseModel):
    code: str
    proposal_id: str | None
    event_id: str | None = None
    message: str


#: Terminal classifications. Every proposal ends in exactly one.
TERMINAL_STATUSES = (
    "deferred_no_action",
    "investigate_more",
    "rejected_by_policy_before_validation",
    "rejected_validation_failed",
    "rejected_by_policy_after_validation",
    "pending_approval",
    "rejected_by_approver",
    "execution_failed",
    "executed_unverified",
    "executed_resolved",
    "executed_unresolved",
    "rolled_back",
    "rollback_failed",
    "cancelled",
    "unclassifiable",
)
_OPEN = frozenset({"pending_approval"})
_INCOMPLETE = frozenset({"executed_unverified", "unclassifiable"})


def _rate(numerator: int, denominator: int, definition: str) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": round(numerator / denominator, 4) if denominator else None,
        "definition": definition,
    }


def _classify(events: list[LifecycleEvent]) -> str:
    """Terminal status of one proposal from its (ordered, already link-checked) events."""
    by = defaultdict(list)
    for event in events:
        by[event.stage].append(event)
    proposed = by[Stage.PROPOSED][0] if by[Stage.PROPOSED] else None
    if by[Stage.CANCELLED]:
        return "cancelled"
    if by[Stage.ROLLED_BACK]:
        return (
            "rolled_back" if by[Stage.ROLLED_BACK][-1].outcome == "succeeded" else "rollback_failed"
        )
    if by[Stage.EXECUTED]:
        if by[Stage.EXECUTED][-1].outcome == "failed":
            return "execution_failed"
        if not by[Stage.VERIFIED]:
            return "executed_unverified"
        return (
            "executed_resolved"
            if by[Stage.VERIFIED][-1].outcome == "resolved"
            else "executed_unresolved"
        )
    if by[Stage.APPROVAL_DECIDED]:
        # approved-but-never-executed is caught as incomplete by the caller
        return (
            "rejected_by_approver"
            if by[Stage.APPROVAL_DECIDED][-1].outcome == "rejected"
            else "unclassifiable"
        )
    if by[Stage.APPROVAL_REQUESTED]:
        return "pending_approval"
    decisions = by[Stage.POLICY_DECIDED]
    last_policy = decisions[-1].outcome if decisions else None
    if by[Stage.VALIDATION]:
        if by[Stage.VALIDATION][-1].outcome == "failed":
            return "rejected_validation_failed"
        if last_policy == "REJECT" and decisions[-1].phase == "post_validation":
            return "rejected_by_policy_after_validation"
        return "unclassifiable"  # passed validation and then nothing
    if last_policy == "DEFER" or (proposed and proposed.outcome == "no_action" and not decisions):
        return "deferred_no_action"
    if last_policy == "INVESTIGATE_MORE":
        return "investigate_more"
    if last_policy == "REJECT":
        return "rejected_by_policy_before_validation"
    return "unclassifiable"


def reconcile(
    events: list[LifecycleEvent] | list[dict],
    *,
    expected_proposals: int | None = None,
    require_closed: bool = False,
    require_policy_events: bool = True,
) -> dict[str, Any]:
    """Account for every lifecycle record.

    ``expected_proposals``: how many proposals an outside count says there should be (for the
    benchmark: investigated scenarios). A mismatch is a failure.
    ``require_closed``: treat a proposal still waiting for approval as a failure. The benchmark
    sets it (its approvals are simulated at once); a live system does not.
    ``require_policy_events``: require a pre-validation policy decision before a validation.
    Sources that do not record policy decisions as events (the audit trail) turn it off.

    Returns a JSON-serialisable report. ``reconciled`` is True only with zero failures.
    """
    failures: list[Failure] = []
    parsed: list[LifecycleEvent] = []
    for raw in events:
        try:
            parsed.append(
                raw if isinstance(raw, LifecycleEvent) else LifecycleEvent.model_validate(raw)
            )
        except Exception as exc:  # a malformed record is a finding, not a crash
            failures.append(
                Failure(
                    code="malformed_event",
                    proposal_id=(raw.get("proposal_id") if isinstance(raw, dict) else None),
                    event_id=(raw.get("event_id") if isinstance(raw, dict) else None),
                    message=f"{type(exc).__name__}: {str(exc)[:200]}",
                )
            )

    seen_ids: Counter[str] = Counter(e.event_id for e in parsed)
    for event_id, n in seen_ids.items():
        if n > 1:
            first = next(e for e in parsed if e.event_id == event_id)
            failures.append(
                Failure(
                    code="duplicate_event_id",
                    proposal_id=first.proposal_id,
                    event_id=event_id,
                    message=f"event id appears {n} times",
                )
            )

    by_proposal: dict[str, list[LifecycleEvent]] = defaultdict(list)
    for event in parsed:
        by_proposal[event.proposal_id].append(event)

    statuses: dict[str, str] = {}
    stage_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    for proposal_id, items in sorted(by_proposal.items()):
        items.sort(key=lambda e: (e.sequence, e.event_id))

        def fail(
            code: str, message: str, event: LifecycleEvent | None = None, pid=proposal_id
        ) -> None:
            failures.append(
                Failure(
                    code=code,
                    proposal_id=pid,
                    event_id=event.event_id if event else None,
                    message=message,
                )
            )

        by: dict[Stage, list[LifecycleEvent]] = defaultdict(list)
        for event in items:
            by[event.stage].append(event)
            stage_counts[event.stage.value] += 1
            outcome_counts[f"{event.stage.value}:{event.outcome}"] += 1
            if event.outcome not in OUTCOMES[event.stage]:
                fail(
                    "invalid_outcome",
                    f"{event.stage.value} cannot have outcome '{event.outcome}'",
                    event,
                )
            id_field = STAGE_ID_FIELD[event.stage]
            if id_field and not getattr(event, id_field):
                fail("missing_stage_id", f"{event.stage.value} event has no {id_field}", event)

        if len({e.incident_id for e in items}) > 1:
            fail("inconsistent_incident", "events of one proposal name different incidents")
        if len({e.scenario_id for e in items}) > 1:
            fail("inconsistent_scenario", "events of one proposal name different scenarios")

        for stage, allowed in (
            (Stage.PROPOSED, 1),
            (Stage.VALIDATION, 1),
            (Stage.APPROVAL_REQUESTED, 1),
            (Stage.APPROVAL_DECIDED, 1),
            (Stage.EXECUTED, 1),
            (Stage.VERIFIED, 1),
            (Stage.ROLLED_BACK, 1),
            (Stage.POLICY_DECIDED, 2),
        ):
            if len(by[stage]) > allowed:
                fail(
                    "duplicate_stage",
                    f"{len(by[stage])} {stage.value} events (at most {allowed} allowed)",
                    by[stage][allowed],
                )
        phases = [e.phase for e in by[Stage.POLICY_DECIDED]]
        if len(phases) != len(set(phases)):
            fail("duplicate_stage", f"policy decided twice in the same phase: {phases}")

        def first_seq(stage: Stage, by=by) -> int | None:
            return by[stage][0].sequence if by[stage] else None

        def need(
            later: Stage,
            earlier: Stage,
            outcome: str | None,
            what: str,
            by=by,
            fail=fail,
            first_seq=first_seq,
        ) -> None:
            if not by[later]:
                return
            event = by[later][0]
            prior = [e for e in by[earlier] if outcome is None or e.outcome == outcome]
            if not prior:
                fail("missing_link", f"{later.value} recorded without {what}", event)
            elif min(e.sequence for e in prior) > event.sequence:
                fail("out_of_order", f"{later.value} recorded before {what}", event)

        if not by[Stage.PROPOSED]:
            fail("missing_link", "events exist for a proposal that was never recorded as PROPOSED")
        elif by[Stage.PROPOSED][0].sequence != min(e.sequence for e in items):
            fail("out_of_order", "PROPOSED is not the first event", by[Stage.PROPOSED][0])
        if require_policy_events:
            pre = [e for e in by[Stage.POLICY_DECIDED] if e.phase == "pre_validation"]
            if by[Stage.VALIDATION] and not any(e.outcome == "PROPOSE_ACTION" for e in pre):
                fail(
                    "missing_link",
                    "VALIDATION recorded without a pre-validation policy decision of PROPOSE_ACTION",
                    by[Stage.VALIDATION][0],
                )
            if by[Stage.PROPOSED] and not by[Stage.POLICY_DECIDED]:
                fail("incomplete_lifecycle", "proposal was never decided by policy")
            if by[Stage.VALIDATION] and not any(
                e.phase == "post_validation" for e in by[Stage.POLICY_DECIDED]
            ):
                fail("incomplete_lifecycle", "validation result was never decided by policy")
        need(Stage.APPROVAL_REQUESTED, Stage.VALIDATION, "passed", "a passed validation")
        need(Stage.APPROVAL_DECIDED, Stage.APPROVAL_REQUESTED, None, "an approval request")
        need(Stage.EXECUTED, Stage.APPROVAL_DECIDED, "approved", "an approval")
        need(Stage.VERIFIED, Stage.EXECUTED, "succeeded", "a successful execution")
        need(Stage.ROLLED_BACK, Stage.EXECUTED, None, "an execution")

        # Declared cross-references must point at records that exist in this lifecycle.
        ids = {
            "validation_id": {e.validation_id for e in by[Stage.VALIDATION]},
            "approval_id": {e.approval_id for e in by[Stage.APPROVAL_REQUESTED]},
            "execution_id": {e.execution_id for e in by[Stage.EXECUTED]},
        }
        for event in items:
            own = STAGE_ID_FIELD[event.stage]
            for field, known in ids.items():
                value = getattr(event, field)
                if value and field != own and value not in known:
                    fail(
                        "dangling_reference",
                        f"{event.stage.value} references unknown {field} {value}",
                        event,
                    )
        decided, asked = by[Stage.APPROVAL_DECIDED], by[Stage.APPROVAL_REQUESTED]
        if decided and asked and decided[0].approval_id != asked[0].approval_id:
            fail("dangling_reference", "approval decision does not match the approval request")

        status = _classify(items)
        approved = any(e.outcome == "approved" for e in by[Stage.APPROVAL_DECIDED])
        if approved and not by[Stage.EXECUTED]:
            status = "unclassifiable"
            fail("incomplete_lifecycle", "approved but never executed")
        if status in _INCOMPLETE:
            fail("incomplete_lifecycle", f"lifecycle stops in a non-terminal state ({status})")
        if status in _OPEN and require_closed:
            fail("incomplete_lifecycle", "approval was requested and never decided")
        statuses[proposal_id] = status

    if expected_proposals is not None and expected_proposals != len(by_proposal):
        failures.append(
            Failure(
                code="proposal_count_mismatch",
                proposal_id=None,
                message=f"{expected_proposals} proposals expected, {len(by_proposal)} have events",
            )
        )

    status_counts = Counter(statuses.values())
    o = outcome_counts
    proposals = len(by_proposal)
    actionable = o["PROPOSED:actionable"]
    validations = stage_counts["VALIDATION"]
    executed = o["EXECUTED:succeeded"]
    execution_attempts = stage_counts["EXECUTED"]
    rollbacks = stage_counts["ROLLED_BACK"]
    rejected_pre = status_counts["rejected_by_policy_before_validation"]
    identities = [
        _identity(
            "every proposal has exactly one terminal status",
            proposals,
            sum(status_counts.values()),
        ),
        _identity(
            "proposals = actionable + no_action + investigate_further",
            proposals,
            actionable + o["PROPOSED:no_action"] + o["PROPOSED:investigate_further"],
        ),
        _identity(
            "validation attempts = passed + failed",
            validations,
            o["VALIDATION:passed"] + o["VALIDATION:failed"],
        ),
        _identity(
            "approval requests = approved + rejected + still pending",
            stage_counts["APPROVAL_REQUESTED"],
            o["APPROVAL_DECIDED:approved"]
            + o["APPROVAL_DECIDED:rejected"]
            + status_counts["pending_approval"],
        ),
        _identity(
            "execution attempts = approvals granted",
            execution_attempts,
            o["APPROVAL_DECIDED:approved"],
        ),
    ]
    if require_policy_events:
        identities.append(
            _identity(
                "actionable proposals = rejected before validation + investigate-more "
                "+ cancelled before validation + validation attempts",
                actionable,
                rejected_pre
                + sum(
                    1
                    for pid, s in statuses.items()
                    if s in ("investigate_more", "cancelled", "unclassifiable")
                    and any(
                        e.outcome == "actionable"
                        for e in by_proposal[pid]
                        if e.stage is Stage.PROPOSED
                    )
                    and not any(e.stage is Stage.VALIDATION for e in by_proposal[pid])
                )
                + validations,
            )
        )
    for identity in identities:
        if not identity["holds"]:
            failures.append(
                Failure(
                    code="accounting_identity_violated",
                    proposal_id=None,
                    message=f"{identity['name']}: {identity['left']} != {identity['right']}",
                )
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "reconciled": not failures,
        "events": len(parsed),
        "simulated_events": sum(e.simulated for e in parsed),
        "derived_events": sum(e.derived for e in parsed),
        "proposals": proposals,
        "stage_counts": {s.value: stage_counts[s.value] for s in Stage},
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "terminal_status_counts": {
            s: status_counts[s] for s in TERMINAL_STATUSES if status_counts[s]
        },
        "accounting_identities": identities,
        "rates": {
            "validation_success_rate": _rate(
                o["VALIDATION:passed"],
                validations,
                "validation attempts that passed / all validation attempts",
            ),
            "approval_grant_rate": _rate(
                o["APPROVAL_DECIDED:approved"],
                stage_counts["APPROVAL_DECIDED"],
                "approval decisions that approved / all approval decisions",
            ),
            "execution_success_rate": _rate(
                executed, execution_attempts, "executions that succeeded / execution attempts"
            ),
            "execution_resolution_rate": _rate(
                o["VERIFIED:resolved"],
                executed,
                "executed fixes after which the incident batch was healthy / executed fixes",
            ),
            "rollback_success_rate": _rate(
                o["ROLLED_BACK:succeeded"],
                rollbacks,
                "rollbacks that restored the previous state / rollback attempts",
            ),
        },
        "failures": [f.model_dump() for f in failures],
        "failure_counts": dict(Counter(f.code for f in failures)),
        "proposal_statuses": statuses,
    }


def _identity(name: str, left: int, right: int) -> dict[str, Any]:
    return {"name": name, "left": left, "right": right, "holds": left == right}


# --------------------------------------------------------------------- audit trail
_AUDIT_STAGE: dict[str, tuple[Stage, str]] = {
    "remediation.proposed": (Stage.PROPOSED, "actionable"),
    "remediation.validation_passed": (Stage.VALIDATION, "passed"),
    "remediation.validation_failed": (Stage.VALIDATION, "failed"),
    "remediation.request_approval": (Stage.APPROVAL_REQUESTED, "requested"),
    "remediation.approve": (Stage.APPROVAL_DECIDED, "approved"),
    "remediation.reject": (Stage.APPROVAL_DECIDED, "rejected"),
    "remediation.execute": (Stage.EXECUTED, "succeeded"),
    "remediation.verify": (Stage.VERIFIED, "resolved"),
    "remediation.verification_failed": (Stage.VERIFIED, "unresolved"),
    "remediation.rollback": (Stage.ROLLED_BACK, "succeeded"),
}


def events_from_audit(audit_rows: list[dict[str, Any]]) -> list[LifecycleEvent]:
    """Build lifecycle events from audit-trail rows of entity type ``remediation``.

    Each row needs ``id``, ``action``, ``entity_id``, ``incident_id``, ``outcome`` and ``detail``.
    Denied attempts (``outcome == "denied"``) are access-control records, not lifecycle stages,
    and are skipped. ``remediation.start_validation`` marks the start of a stage that the
    passed/failed event completes, so it does not become an event of its own.

    The audit trail does not record policy decisions as separate rows, so reconcile the result
    with ``require_policy_events=False``. A rejection that happens before any approval request
    is the policy's (or validation's) rejection and is recorded as a policy decision here.
    """
    events: list[LifecycleEvent] = []
    requested: set[str] = set()
    for row in sorted(audit_rows, key=lambda r: r["id"]):
        if row.get("outcome") == "denied" or row["action"] not in _AUDIT_STAGE:
            continue
        stage, outcome = _AUDIT_STAGE[row["action"]]
        rid = str(row["entity_id"])
        detail = dict(row.get("detail") or {})
        phase = None
        if stage is Stage.PROPOSED and detail.get("remediation_action") in ("no_action",):
            outcome = "no_action"
        if stage is Stage.PROPOSED and detail.get("remediation_action") == "investigate_further":
            outcome = "investigate_further"
        if stage is Stage.APPROVAL_REQUESTED:
            requested.add(rid)
        if stage is Stage.APPROVAL_DECIDED and outcome == "rejected" and rid not in requested:
            stage, outcome, phase = Stage.POLICY_DECIDED, "REJECT", "system"
        if stage is Stage.ROLLED_BACK and row.get("outcome") == "failed":
            outcome = "failed"
        if stage is Stage.EXECUTED and row.get("outcome") == "failed":
            outcome = "failed"
        events.append(
            LifecycleEvent(
                event_id=f"AUD-{row['id']}",
                stage=stage,
                sequence=int(row["id"]),
                outcome=outcome,
                proposal_id=rid,
                incident_id=str(row.get("incident_id") or "unknown"),
                validation_id=f"VAL-{rid}" if stage in (Stage.VALIDATION,) else None,
                policy_decision_id=f"POL-{rid}-{row['id']}"
                if stage is Stage.POLICY_DECIDED
                else None,
                approval_id=f"APR-{rid}"
                if stage in (Stage.APPROVAL_REQUESTED, Stage.APPROVAL_DECIDED)
                else None,
                execution_id=f"EXE-{rid}" if stage in (Stage.EXECUTED, Stage.VERIFIED) else None,
                rollback_id=f"RBK-{rid}" if stage is Stage.ROLLED_BACK else None,
                phase=phase,
                action=detail.get("remediation_action"),
                detail={
                    "actor": row.get("actor"),
                    "audit_action": row["action"],
                    **(
                        {"deployment_revision": detail["deployment_revision"]}
                        if "deployment_revision" in detail
                        else {}
                    ),
                },
            )
        )
    return events


# ------------------------------------------------------------- historical benchmark rows
def events_from_legacy_row(
    scenario_id: str, config: str, row: dict[str, Any]
) -> list[LifecycleEvent]:
    """Re-express a pre-lifecycle benchmark row (run ``main``) as events.

    Those rows recorded only summary fields: the action, the final policy outcome, whether
    validation passed, and the simulated execution results. Every event built here is marked
    ``derived``: it restates recorded fields, it was not captured when the run happened. One
    thing is inferred rather than restated, and is marked in ``detail``: that a validation was
    preceded by a policy decision of PROPOSE_ACTION, which the code path guarantees but the row
    does not record.
    """
    key = f"{scenario_id.removeprefix('SCN-')}-{config}"
    base = {
        "proposal_id": f"PRP-{key}",
        "incident_id": f"INC-{key}",
        "scenario_id": scenario_id,
        "action": row["action"],
        "derived": True,
    }
    events: list[LifecycleEvent] = []

    def emit(stage: Stage, outcome: str, **fields: Any) -> None:
        events.append(
            LifecycleEvent(
                event_id=f"LEG-{key}-{len(events)}",
                stage=stage,
                sequence=len(events),
                outcome=outcome,
                **base,
                **fields,
            )
        )

    kind = (
        "no_action"
        if row["action"] == "no_action"
        else ("investigate_further" if row["action"] == "investigate_further" else "actionable")
    )
    emit(Stage.PROPOSED, kind)
    validated = row.get("validated")
    if validated is None:
        emit(
            Stage.POLICY_DECIDED,
            row["policy_outcome"],
            policy_decision_id=f"POL-{key}-pre",
            phase="pre_validation",
        )
        return events
    emit(
        Stage.POLICY_DECIDED,
        "PROPOSE_ACTION",
        policy_decision_id=f"POL-{key}-pre",
        phase="pre_validation",
        detail={"inferred": "validation only runs after PROPOSE_ACTION; not recorded in the row"},
    )
    emit(Stage.VALIDATION, "passed" if validated else "failed", validation_id=f"VAL-{key}")
    emit(
        Stage.POLICY_DECIDED,
        row["policy_outcome"],
        policy_decision_id=f"POL-{key}-post",
        phase="post_validation",
    )
    if row.get("approval_requested"):
        emit(Stage.APPROVAL_REQUESTED, "requested", approval_id=f"APR-{key}")
    if row.get("executed_resolves_incident") is not None:
        emit(Stage.APPROVAL_DECIDED, "approved", approval_id=f"APR-{key}", simulated=True)
        emit(Stage.EXECUTED, "succeeded", execution_id=f"EXE-{key}", simulated=True)
        emit(
            Stage.VERIFIED,
            "resolved" if row["executed_resolves_incident"] else "unresolved",
            execution_id=f"EXE-{key}",
            simulated=True,
        )
        emit(
            Stage.ROLLED_BACK,
            "succeeded" if row.get("rollback_restores_state") else "failed",
            rollback_id=f"RBK-{key}",
            simulated=True,
        )
    return events


def benchmark_events(
    records: list[dict[str, Any]], config: str
) -> tuple[list[LifecycleEvent], dict]:
    """Collect one configuration's lifecycle events from benchmark records.

    Returns the events and a coverage dict saying how many investigated scenarios there were,
    how many carried recorded events, how many had to be derived from legacy fields, and which
    crashed or carry no remediation at all (those are reconciliation failures for the caller).
    """
    events: list[LifecycleEvent] = []
    coverage: dict[str, Any] = {
        "config": config,
        "investigated": 0,
        "recorded": 0,
        "derived_from_legacy_fields": 0,
        "crashed": [],
        "without_remediation": [],
    }
    for record in records:
        cfg = record.get("configs", {}).get(config)
        if cfg is None:
            continue
        coverage["investigated"] += 1
        if "error" in cfg:
            coverage["crashed"].append(record["scenario_id"])
            continue
        row = cfg.get("remediation")
        if not row:
            coverage["without_remediation"].append(record["scenario_id"])
            continue
        if row.get("lifecycle"):
            coverage["recorded"] += 1
            events += [LifecycleEvent.model_validate(e) for e in row["lifecycle"]]
        else:
            coverage["derived_from_legacy_fields"] += 1
            events += events_from_legacy_row(record["scenario_id"], config, row)
    return events, coverage
