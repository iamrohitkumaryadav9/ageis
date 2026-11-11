"""Execution and rollback of approved remediations, as pure state functions.

The executor does not decide anything. It receives the *validated candidate
state* and the *current* state, refuses if the world moved since validation,
and returns the state to activate. Persistence and authorisation live in the
service layer, which calls this inside one transaction.
"""

from __future__ import annotations

from aegis.remediation.proposals import candidate_state
from aegis.schemas.deployment import DeploymentState
from aegis.schemas.remediation import RemediationProposal, ValidationResult


class StaleProposalError(Exception):
    """The deployment state changed after the proposal was validated."""


def state_to_execute(
    current: DeploymentState, proposal: RemediationProposal, validation: ValidationResult
) -> DeploymentState:
    if not validation.passed or validation.candidate_state is None:
        raise ValueError("cannot execute: the proposal has no passing validation")
    validated = DeploymentState.model_validate(validation.candidate_state)
    params = dict(proposal.parameters)
    if validation.candidate_model_version:
        params["version"] = validation.candidate_model_version
    expected = candidate_state(current, proposal.model_copy(update={"parameters": params}))
    if expected != validated:
        raise StaleProposalError(
            "deployment state changed since validation; re-validate against the current state"
        )
    return validated
