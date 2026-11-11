"""The remediation state machine, as data.

Every transition anywhere in the system goes through :func:`next_state`. An
event that is not listed for the current state is illegal, full stop: there is
no code path from PROPOSED to EXECUTED that does not pass through VALIDATED and
APPROVED, because no such edge exists in the table.
"""

from __future__ import annotations

from enum import StrEnum

from aegis.schemas.remediation import RemediationState as S


class Event(StrEnum):
    START_VALIDATION = "start_validation"
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"
    REQUEST_APPROVAL = "request_approval"
    APPROVE = "approve"
    REJECT = "reject"
    EXECUTE = "execute"
    VERIFY = "verify"
    ROLLBACK = "rollback"


TRANSITIONS: dict[tuple[S, Event], S] = {
    (S.PROPOSED, Event.START_VALIDATION): S.VALIDATING,
    (S.VALIDATING, Event.VALIDATION_PASSED): S.VALIDATED,
    (S.VALIDATING, Event.VALIDATION_FAILED): S.VALIDATION_FAILED,
    (S.VALIDATED, Event.REQUEST_APPROVAL): S.PENDING_APPROVAL,
    (S.PENDING_APPROVAL, Event.APPROVE): S.APPROVED,
    (S.PENDING_APPROVAL, Event.REJECT): S.REJECTED,
    (S.PROPOSED, Event.REJECT): S.REJECTED,
    (S.VALIDATED, Event.REJECT): S.REJECTED,
    (S.APPROVED, Event.EXECUTE): S.EXECUTED,
    (S.EXECUTED, Event.VERIFY): S.COMPLETED,
    (S.EXECUTED, Event.ROLLBACK): S.ROLLED_BACK,
    (S.COMPLETED, Event.ROLLBACK): S.ROLLED_BACK,
}

TERMINAL = frozenset({S.VALIDATION_FAILED, S.REJECTED, S.ROLLED_BACK})


class IllegalTransitionError(Exception):
    def __init__(self, state: S, event: Event) -> None:
        super().__init__(f"event '{event.value}' is not allowed in state {state.value}")
        self.state, self.event = state, event


def next_state(state: S, event: Event) -> S:
    try:
        return TRANSITIONS[(S(state), Event(event))]
    except KeyError:
        raise IllegalTransitionError(S(state), Event(event)) from None


def allowed_events(state: S) -> list[Event]:
    return [event for (source, event) in TRANSITIONS if source == state]
