"""Remediation contracts: actions, proposals, validation results, policy decisions."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ActionType(StrEnum):
    FIX_INGESTION_SCALE = "fix_ingestion_scale"  # revert an incorrect preprocessing/unit config
    ENABLE_DEDUPLICATION = "enable_deduplication"
    RESTORE_MODEL_VERSION = "restore_model_version"  # redeploy a registered, hash-verified version
    QUARANTINE_BATCH = "quarantine_batch"  # reject incompatible/untrusted data
    RETRAIN_CANDIDATE = "retrain_candidate"  # retrain on recent labelled data, promote if approved
    INVESTIGATE_FURTHER = "investigate_further"  # no state change
    NO_ACTION = "no_action"  # monitor only


class RiskLevel(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


#: Actions that change the deployment state and therefore need validation + human approval.
STATE_CHANGING = frozenset(
    {
        ActionType.FIX_INGESTION_SCALE,
        ActionType.ENABLE_DEDUPLICATION,
        ActionType.RESTORE_MODEL_VERSION,
        ActionType.QUARANTINE_BATCH,
        ActionType.RETRAIN_CANDIDATE,
    }
)


class RemediationState(StrEnum):
    PROPOSED = "PROPOSED"
    VALIDATING = "VALIDATING"
    VALIDATED = "VALIDATED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXECUTED = "EXECUTED"
    COMPLETED = "COMPLETED"
    ROLLED_BACK = "ROLLED_BACK"


class RemediationProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ActionType
    parameters: dict[str, Any] = Field(default_factory=dict)
    target_root_cause: str | None = None
    hypothesis_id: str | None = None
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)
    risk_level: RiskLevel
    expected_effect: str
    rollback_plan: str
    runbook_refs: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class ValidationCheck(BaseModel):
    name: str
    passed: bool
    detail: str
    category: str = "effect"  # effect | regression | contract | integrity | precondition


class ValidationResult(BaseModel):
    passed: bool
    checks: list[ValidationCheck]
    metrics_before: dict[str, Any] = Field(default_factory=dict)
    metrics_after: dict[str, Any] = Field(default_factory=dict)
    candidate_state: dict[str, Any] | None = None  # the DeploymentState that would be applied
    candidate_model_version: str | None = None
    duration_ms: float = 0.0
    summary: str = ""

    @property
    def regression_failed(self) -> bool:
        return any(c.category == "regression" and not c.passed for c in self.checks)


class PolicyOutcome(StrEnum):
    INVESTIGATE_MORE = "INVESTIGATE_MORE"
    PROPOSE_ACTION = "PROPOSE_ACTION"
    DEFER = "DEFER"
    REJECT = "REJECT"
    REQUEST_HUMAN_APPROVAL = "REQUEST_HUMAN_APPROVAL"


class PolicyDecision(BaseModel):
    outcome: PolicyOutcome
    policy_version: str
    action: ActionType
    risk_level: RiskLevel
    score: float
    expected_benefit: float
    failure_risk: float
    execution_cost: float
    requires_approval: bool
    reasons: list[str]
    unmet_requirements: list[str] = Field(default_factory=list)
