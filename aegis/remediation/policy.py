"""Deterministic, versioned decision policy.

    Score(action) = ExpectedBenefit(action) - FailureRisk(action) - ExecutionCost(action)

This is a transparent heuristic, not a learned policy and not reinforcement
learning. Its inputs are the evidence-fit of the leading hypothesis, fixed
per-action priors, and (after validation) the validation result. The LLM has no
input here: an LLM-ranked hypothesis only counts to the extent the rule engine
independently supports the same cause.

Outcomes
--------
INVESTIGATE_MORE        evidence is too thin, or a required piece is missing
DEFER                   the right move is to wait/monitor (no state change)
REJECT                  the action's risk and cost outweigh its benefit, or validation failed
PROPOSE_ACTION          the action may proceed to sandbox validation
REQUEST_HUMAN_APPROVAL  validated; a human must decide (every state-changing action)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from aegis.schemas.diagnosis import Diagnosis, Evidence
from aegis.schemas.remediation import (
    STATE_CHANGING,
    ActionType,
    PolicyDecision,
    PolicyOutcome,
    RemediationProposal,
    RiskLevel,
    ValidationResult,
)

A, R = ActionType, RiskLevel


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = "policy-1.0"
    min_confidence_to_act: float = 0.60  # evidence fit of the targeted hypothesis
    min_rule_support: float = 0.50  # deterministic support required for an LLM-ranked cause
    min_margin_over_runner_up: float = 0.05
    min_score: float = 0.0
    #: P(action resolves the incident | hypothesis is right): engineering priors, not measurements.
    efficacy: dict[ActionType, float] = {
        A.FIX_INGESTION_SCALE: 0.90,
        A.ENABLE_DEDUPLICATION: 0.90,
        A.RESTORE_MODEL_VERSION: 0.90,
        A.QUARANTINE_BATCH: 0.70,
        A.RETRAIN_CANDIDATE: 0.55,
        A.INVESTIGATE_FURTHER: 0.0,
        A.NO_ACTION: 0.0,
    }
    #: Harm if the action is wrong, by risk class.
    risk_weight: dict[RiskLevel, float] = {R.NONE: 0.0, R.LOW: 0.10, R.MEDIUM: 0.30, R.HIGH: 0.60}
    execution_cost: dict[ActionType, float] = {
        A.FIX_INGESTION_SCALE: 0.05,
        A.ENABLE_DEDUPLICATION: 0.03,
        A.RESTORE_MODEL_VERSION: 0.08,
        A.QUARANTINE_BATCH: 0.05,
        A.RETRAIN_CANDIDATE: 0.20,
        A.INVESTIGATE_FURTHER: 0.0,
        A.NO_ACTION: 0.0,
    }
    #: Evidence that must exist (as anomaly evidence) before an action is even considered.
    required_evidence: dict[ActionType, list[list[str]]] = {
        # each inner list is an OR-group; all groups must be satisfied
        A.FIX_INGESTION_SCALE: [["feature_pattern"], ["numeric_drift", "range_violation_rate"]],
        A.ENABLE_DEDUPLICATION: [["duplicate_rate"]],
        A.RESTORE_MODEL_VERSION: [["artifact_hash"]],
        A.QUARANTINE_BATCH: [
            [
                "schema_missing_columns",
                "schema_type_mismatch",
                "missing_rate",
                "label_association_gain",
            ]
        ],
        A.RETRAIN_CANDIDATE: [
            ["accuracy_drop", "precision_drop", "recall_drop", "f1_drop", "roc_auc_drop"]
        ],
    }


class PolicyEngine:
    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    def evaluate(
        self,
        proposal: RemediationProposal,
        diagnosis: Diagnosis,
        evidence: list[Evidence],
        *,
        rule_diagnosis: Diagnosis | None = None,
        validation: ValidationResult | None = None,
    ) -> PolicyDecision:
        cfg = self.config
        action, risk = proposal.action, proposal.risk_level
        target = next(
            (h for h in diagnosis.hypotheses if h.root_cause.value == proposal.target_root_cause),
            None,
        )
        confidence = target.confidence if target else 0.0
        reasons: list[str] = []
        unmet: list[str] = []

        # The LLM cannot raise confidence above what deterministic rules support for the same cause.
        if rule_diagnosis is not None and target is not None:
            rule_conf = next(
                (
                    h.confidence
                    for h in rule_diagnosis.hypotheses
                    if h.root_cause == target.root_cause
                ),
                0.0,
            )
            if rule_conf < cfg.min_rule_support:
                unmet.append(
                    f"rule engine support for {target.root_cause.value} is {rule_conf:.2f} "
                    f"(< {cfg.min_rule_support}); an LLM ranking alone cannot justify action"
                )
            confidence = min(confidence, rule_conf)

        benefit = confidence * cfg.efficacy[action]
        failure_risk = (1.0 - confidence) * cfg.risk_weight[risk]
        cost = cfg.execution_cost[action]
        if validation is not None:
            if validation.passed:
                failure_risk *= 0.5  # a passed sandbox validation halves, never removes, the risk
            else:
                benefit = 0.0
        score = round(benefit - failure_risk - cost, 4)

        def decide(outcome: PolicyOutcome) -> PolicyDecision:
            return PolicyDecision(
                outcome=outcome,
                policy_version=cfg.version,
                action=action,
                risk_level=risk,
                score=score,
                expected_benefit=round(benefit, 4),
                failure_risk=round(failure_risk, 4),
                execution_cost=cost,
                requires_approval=action in STATE_CHANGING,
                reasons=reasons,
                unmet_requirements=unmet,
            )

        if action is A.NO_ACTION:
            reasons.append("no state change is warranted; continue monitoring")
            return decide(PolicyOutcome.DEFER)
        if action is A.INVESTIGATE_FURTHER:
            reasons.append(proposal.rationale)
            return decide(PolicyOutcome.INVESTIGATE_MORE)

        if not diagnosis.sufficient_evidence or target is None:
            unmet.append("no hypothesis with sufficient evidence targets this action")
        if confidence < cfg.min_confidence_to_act:
            unmet.append(
                f"evidence fit {confidence:.2f} is below the action threshold {cfg.min_confidence_to_act}"
            )
        runner_up = next(
            (h for h in diagnosis.hypotheses if target and h.hypothesis_id != target.hypothesis_id),
            None,
        )
        if (
            target
            and diagnosis.hypotheses
            and diagnosis.hypotheses[0].hypothesis_id != target.hypothesis_id
        ):
            unmet.append("the action does not target the leading hypothesis")
        elif (
            target
            and runner_up
            and target.confidence - runner_up.confidence < cfg.min_margin_over_runner_up
        ):
            unmet.append(
                f"leading hypothesis is not separated from {runner_up.root_cause.value} "
                f"(margin < {cfg.min_margin_over_runner_up})"
            )
        present = {e.data.get("check") for e in evidence if e.data.get("polarity") == "anomaly"}
        cited = {e.evidence_id for e in evidence} & set(proposal.evidence_ids)
        for group in cfg.required_evidence.get(action, []):
            if not present & set(group):
                unmet.append(f"required evidence missing: one of {group}")
        if not cited:
            unmet.append("the proposal cites no existing evidence")
        if unmet:
            reasons.append("evidence requirements are not met")
            return decide(PolicyOutcome.INVESTIGATE_MORE)

        if validation is not None and not validation.passed:
            reasons.append(f"sandbox validation failed: {validation.summary}")
            return decide(PolicyOutcome.REJECT)
        if score <= cfg.min_score:
            reasons.append(
                f"score {score:.2f} = benefit {benefit:.2f} - risk {failure_risk:.2f} - cost {cost:.2f} "
                "is not positive"
            )
            return decide(PolicyOutcome.REJECT)

        reasons.append(
            f"score {score:.2f} = benefit {benefit:.2f} - risk {failure_risk:.2f} - cost {cost:.2f}"
        )
        if validation is None:
            reasons.append("eligible for sandbox validation")
            return decide(PolicyOutcome.PROPOSE_ACTION)
        reasons.append(
            f"validated in sandbox; {risk.value}-risk state change requires human approval"
        )
        return decide(PolicyOutcome.REQUEST_HUMAN_APPROVAL)
