"""Map a diagnosis to one candidate remediation.

The mapping is deterministic. It encodes two restraints: drift without
measured performance loss leads to *no action*, and anything the evidence
cannot parameterise (e.g. a scale fix without a measured factor) leads to
*further investigation* rather than a guess.
"""

from __future__ import annotations

from aegis.diagnosis.rules import PERF_CHECKS
from aegis.schemas.deployment import DeploymentState
from aegis.schemas.diagnosis import Diagnosis, Evidence, Hypothesis, RootCause
from aegis.schemas.remediation import ActionType, RemediationProposal, RiskLevel

A, R, C = ActionType, RiskLevel, RootCause


def candidate_state(state: DeploymentState, proposal: RemediationProposal) -> DeploymentState:
    """The deployment state that executing ``proposal`` would produce. Pure."""
    new = state.model_copy(deep=True)
    p = proposal.parameters
    if proposal.action is A.FIX_INGESTION_SCALE:
        feature = p["feature"]
        new.ingestion.unit_scale[feature] = new.ingestion.unit_scale.get(feature, 1.0) * float(
            p["factor"]
        )
    elif proposal.action is A.ENABLE_DEDUPLICATION:
        new.ingestion.deduplicate = True
    elif proposal.action in (A.RESTORE_MODEL_VERSION, A.RETRAIN_CANDIDATE):
        if p.get("version"):
            new.model_version = p["version"]
    elif proposal.action is A.QUARANTINE_BATCH:
        new.quarantined = True
        new.retraining_blocked = bool(p.get("block_retraining", False))
    return new


def _anomalies(evidence: list[Evidence], *checks: str) -> list[Evidence]:
    return [
        e for e in evidence if e.data.get("check") in checks and e.data.get("polarity") == "anomaly"
    ]


def _investigate(reason: str, hypothesis: Hypothesis | None = None) -> RemediationProposal:
    return RemediationProposal(
        action=A.INVESTIGATE_FURTHER,
        rationale=reason,
        risk_level=R.NONE,
        target_root_cause=hypothesis.root_cause.value if hypothesis else None,
        hypothesis_id=hypothesis.hypothesis_id if hypothesis else None,
        evidence_ids=hypothesis.supporting_evidence_ids if hypothesis else [],
        expected_effect="none: gathers evidence only",
        rollback_plan="not applicable (no state change)",
    )


def propose(
    diagnosis: Diagnosis, evidence: list[Evidence], state: DeploymentState, declared_version: str
) -> RemediationProposal:
    top = diagnosis.top
    if top is None or not diagnosis.sufficient_evidence:
        return _investigate(
            "Insufficient evidence to determine the root cause; no action is justified.", top
        )

    def make(action, risk, rationale, effect, rollback, *, parameters=None, caveats=()):
        return RemediationProposal(
            action=action,
            parameters=parameters or {},
            target_root_cause=top.root_cause.value,
            hypothesis_id=top.hypothesis_id,
            rationale=rationale,
            evidence_ids=top.supporting_evidence_ids,
            risk_level=risk,
            expected_effect=effect,
            rollback_plan=rollback,
            caveats=list(caveats),
        )

    cause = top.root_cause
    perf_loss = _anomalies(evidence, *PERF_CHECKS)

    if cause is C.SCALING_ERROR:
        patterns = [
            e
            for e in evidence
            if e.data.get("check") == "feature_pattern"
            and e.data.get("pattern") == "scale"
            and e.data.get("scale_factor")
        ]
        if not patterns:
            return _investigate(
                "A scaling error is suspected but no scale factor has been measured.", top
            )
        pattern = patterns[0]
        observed = float(pattern.data["scale_factor"])
        return make(
            A.FIX_INGESTION_SCALE,
            R.MEDIUM,
            f"'{pattern.feature}' arrives multiplied by ~{observed:.4g} relative to the reference; "
            "apply the inverse factor in ingestion.",
            "feature returns to its reference distribution; model inputs are in the trained units",
            "re-activate the previous deployment revision (removes the correction factor)",
            parameters={
                "feature": pattern.feature,
                "factor": 1.0 / observed,
                "observed_factor": observed,
            },
            caveats=[
                "The correction is conditional on upstream continuing to send the changed unit; "
                "remove it when the source is fixed."
            ],
        )
    if cause is C.DUPLICATE_RECORDS:
        return make(
            A.ENABLE_DEDUPLICATION,
            R.LOW,
            "Exact duplicate rows are far above the expected rate.",
            "duplicates are dropped at ingestion before monitoring and scoring",
            "re-activate the previous deployment revision (disables deduplication)",
        )
    if cause is C.ARTIFACT_MISMATCH:
        return make(
            A.RESTORE_MODEL_VERSION,
            R.MEDIUM,
            f"The serving artifact does not match the registry record for {declared_version}; "
            "redeploy the registered, hash-verified artifact.",
            "serving artifact hash equals the registry record",
            "re-activate the previous deployment revision",
            parameters={"version": declared_version},
            caveats=[
                "Predictions made while the wrong artifact was live are not corrected by this action."
            ],
        )
    if cause is C.SCHEMA_CHANGE:
        return make(
            A.QUARANTINE_BATCH,
            R.LOW,
            "The batch violates the data contract; reject it rather than coerce it.",
            "the incompatible batch is held out of decisions and training",
            "lift the quarantine by re-activating the previous deployment revision",
            parameters={"block_retraining": True},
        )
    if cause is C.FEATURE_LEAKAGE:
        return make(
            A.QUARANTINE_BATCH,
            R.LOW,
            "A feature appears to carry label information; training on this batch would bake the leak in.",
            "the batch is held and retraining on it is blocked; the deployed model is untouched",
            "lift the quarantine by re-activating the previous deployment revision",
            parameters={"block_retraining": True},
            caveats=["The deployed model is not at fault and is not changed."],
        )
    if cause is C.MISSING_VALUES:
        return make(
            A.QUARANTINE_BATCH,
            R.LOW,
            "A feature is no longer being populated; hold the batch until upstream backfills it.",
            "the degraded batch is held out of decisions and training",
            "lift the quarantine by re-activating the previous deployment revision",
            parameters={"block_retraining": True},
        )
    if cause is C.LABEL_DELAY:
        return make(
            A.NO_ACTION,
            R.NONE,
            "Labels are late. Missing labels say nothing about model quality; wait and re-monitor.",
            "none",
            "not applicable (no state change)",
        )
    if cause in (C.CONCEPT_DRIFT, C.NUMERICAL_SHIFT, C.CATEGORICAL_SHIFT):
        if not perf_loss:
            return make(
                A.NO_ACTION,
                R.NONE,
                "Inputs drifted but no performance loss has been measured. Drift alone does not "
                "mean the model is defective; keep monitoring.",
                "none",
                "not applicable (no state change)",
            )
        proposal = make(
            A.RETRAIN_CANDIDATE,
            R.HIGH,
            "Performance degraded on labelled recent data; evaluate whether a model retrained on "
            "recent data actually does better before considering promotion.",
            "a candidate model that outperforms the deployed model on held-out recent data",
            "re-activate the previous deployment revision (previous model version)",
            caveats=[
                "Promotion replaces the serving model and always requires human approval.",
                "If labels for the period are noisy, retraining will not help.",
            ],
        )
        proposal.evidence_ids = list(
            dict.fromkeys(proposal.evidence_ids + [e.evidence_id for e in perf_loss])
        )
        return proposal
    return _investigate(f"No remediation is defined for {cause.value}.", top)
