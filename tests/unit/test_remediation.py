import itertools

import numpy as np
import pytest

from aegis.diagnosis.pipeline import deterministic_diagnosis
from aegis.monitoring.engine import MonitoringInput, run_monitoring
from aegis.remediation.executor import StaleProposalError, state_to_execute
from aegis.remediation.lifecycle import (
    TERMINAL,
    TRANSITIONS,
    Event,
    IllegalTransitionError,
    allowed_events,
    next_state,
)
from aegis.remediation.policy import PolicyConfig, PolicyEngine
from aegis.remediation.proposals import candidate_state, propose
from aegis.remediation.validation import validate_proposal
from aegis.schemas.deployment import DeploymentState, IngestionConfig
from aegis.schemas.diagnosis import Diagnosis, Evidence, Hypothesis, RootCause
from aegis.schemas.findings import EpistemicLevel, Severity
from aegis.schemas.remediation import (
    ActionType,
    PolicyOutcome,
    RemediationProposal,
    RiskLevel,
    ValidationCheck,
    ValidationResult,
)
from aegis.schemas.remediation import RemediationState as S

A, PO = ActionType, PolicyOutcome


def pipeline(ctx, fault, severity="high", seed=4, n_rows=2000):
    scenario = ctx.scenario(fault, severity, seed, n_rows)
    inp = MonitoringInput(
        ctx.spec,
        ctx.contract,
        ctx.reference,
        scenario.current,
        ctx.registry,
        ctx.record.version,
        scenario.artifact_path,
    )
    result = run_monitoring(inp)
    evidence, diagnosis = deterministic_diagnosis(inp, result)
    state = DeploymentState(dataset="adult", model_version=ctx.record.version)
    return (
        scenario,
        inp,
        result,
        evidence,
        diagnosis,
        state,
        propose(diagnosis, evidence, state, ctx.record.version),
    )


# ----------------------------------------------------------------- lifecycle
def test_happy_path_and_rollback():
    state = S.PROPOSED
    for event in (
        Event.START_VALIDATION,
        Event.VALIDATION_PASSED,
        Event.REQUEST_APPROVAL,
        Event.APPROVE,
        Event.EXECUTE,
        Event.VERIFY,
        Event.ROLLBACK,
    ):
        state = next_state(state, event)
    assert state is S.ROLLED_BACK


def test_no_path_reaches_executed_without_validation_and_approval():
    """Exhaustively walk the transition table from PROPOSED."""
    paths, frontier = [], [(S.PROPOSED, [])]
    while frontier:
        state, events = frontier.pop()
        if state is S.EXECUTED:
            paths.append(events)
            continue
        for (source, event), target in TRANSITIONS.items():
            if source is state and len(events) < 12:
                frontier.append((target, [*events, event]))
    assert paths
    for events in paths:
        assert (
            events.index(Event.VALIDATION_PASSED)
            < events.index(Event.APPROVE)
            < events.index(Event.EXECUTE)
        )


@pytest.mark.parametrize(
    "state,event",
    [
        (S.PROPOSED, Event.EXECUTE),
        (S.PROPOSED, Event.APPROVE),
        (S.VALIDATING, Event.APPROVE),
        (S.VALIDATED, Event.EXECUTE),
        (S.VALIDATED, Event.APPROVE),
        (S.PENDING_APPROVAL, Event.EXECUTE),
        (S.VALIDATION_FAILED, Event.REQUEST_APPROVAL),
        (S.VALIDATION_FAILED, Event.APPROVE),
        (S.REJECTED, Event.APPROVE),
        (S.REJECTED, Event.EXECUTE),
        (S.APPROVED, Event.APPROVE),
        (S.EXECUTED, Event.EXECUTE),
        (S.COMPLETED, Event.EXECUTE),
        (S.ROLLED_BACK, Event.ROLLBACK),
        (S.ROLLED_BACK, Event.EXECUTE),
        (S.APPROVED, Event.ROLLBACK),
    ],
)
def test_illegal_transitions_raise(state, event):
    with pytest.raises(IllegalTransitionError):
        next_state(state, event)


def test_terminal_states_have_no_exits_and_every_pair_is_decided():
    for state in TERMINAL:
        assert allowed_events(state) == []
    for state, event in itertools.product(S, Event):
        if (state, event) not in TRANSITIONS:
            with pytest.raises(IllegalTransitionError):
                next_state(state, event)


# -------------------------------------------------------------------- policy
def hyp(cause, confidence, ids=("E1",)):
    return Hypothesis(
        hypothesis_id=f"H-{cause.value}",
        root_cause=cause,
        confidence=confidence,
        supporting_evidence_ids=list(ids),
        reasoning="r",
    )


def anomaly(eid, check, **data):
    return Evidence(
        evidence_id=eid,
        source="t",
        level=EpistemicLevel.STATISTICAL_ANOMALY,
        statement=check,
        severity=Severity.HIGH,
        data={"check": check, "polarity": "anomaly", **data},
    )


def diag(*hyps, sufficient=True):
    return Diagnosis(hypotheses=list(hyps), conclusion="c", sufficient_evidence=sufficient)


def prop(action, cause, risk=RiskLevel.MEDIUM, ids=("E1",), **params):
    return RemediationProposal(
        action=action,
        parameters=params,
        target_root_cause=cause.value,
        rationale="r",
        evidence_ids=list(ids),
        risk_level=risk,
        expected_effect="e",
        rollback_plan="rb",
    )


SCALE_EVIDENCE = [anomaly("E1", "numeric_drift"), anomaly("E2", "feature_pattern", pattern="scale")]
PASSED = ValidationResult(passed=True, checks=[ValidationCheck(name="x", passed=True, detail="ok")])
FAILED = ValidationResult(
    passed=False,
    summary="failed: ['x']",
    checks=[ValidationCheck(name="x", passed=False, detail="bad")],
)


def test_policy_score_is_benefit_minus_risk_minus_cost():
    d = diag(hyp(RootCause.SCALING_ERROR, 0.9))
    decision = PolicyEngine().evaluate(
        prop(A.FIX_INGESTION_SCALE, RootCause.SCALING_ERROR), d, SCALE_EVIDENCE
    )
    assert decision.outcome is PO.PROPOSE_ACTION and decision.policy_version == "policy-1.0"
    assert decision.expected_benefit == pytest.approx(0.9 * 0.9)
    assert decision.failure_risk == pytest.approx(0.1 * 0.30)
    assert decision.execution_cost == 0.05
    assert decision.score == pytest.approx(0.81 - 0.03 - 0.05)
    assert decision.requires_approval


def test_validated_state_change_always_goes_to_a_human():
    d = diag(hyp(RootCause.SCALING_ERROR, 0.9))
    for risk in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH):
        decision = PolicyEngine().evaluate(
            prop(A.FIX_INGESTION_SCALE, RootCause.SCALING_ERROR, risk),
            d,
            SCALE_EVIDENCE,
            validation=PASSED,
        )
        assert decision.outcome is PO.REQUEST_HUMAN_APPROVAL and decision.requires_approval
    # No configuration of the policy has an "auto-execute" outcome.
    assert {o.value for o in PO} == {
        "INVESTIGATE_MORE",
        "PROPOSE_ACTION",
        "DEFER",
        "REJECT",
        "REQUEST_HUMAN_APPROVAL",
    }


def test_failed_validation_is_rejected():
    d = diag(hyp(RootCause.SCALING_ERROR, 0.9))
    decision = PolicyEngine().evaluate(
        prop(A.FIX_INGESTION_SCALE, RootCause.SCALING_ERROR), d, SCALE_EVIDENCE, validation=FAILED
    )
    assert (
        decision.outcome is PO.REJECT
        and decision.expected_benefit == 0
        and "validation failed" in decision.reasons[-1]
    )


def test_low_confidence_or_missing_evidence_means_investigate_more():
    engine = PolicyEngine()
    weak = engine.evaluate(
        prop(A.FIX_INGESTION_SCALE, RootCause.SCALING_ERROR),
        diag(hyp(RootCause.SCALING_ERROR, 0.55)),
        SCALE_EVIDENCE,
    )
    assert weak.outcome is PO.INVESTIGATE_MORE and any(
        "below the action threshold" in u for u in weak.unmet_requirements
    )
    no_pattern = engine.evaluate(
        prop(A.FIX_INGESTION_SCALE, RootCause.SCALING_ERROR),
        diag(hyp(RootCause.SCALING_ERROR, 0.9)),
        [anomaly("E1", "numeric_drift")],
    )
    assert no_pattern.outcome is PO.INVESTIGATE_MORE and any(
        "feature_pattern" in u for u in no_pattern.unmet_requirements
    )
    uncited = engine.evaluate(
        prop(A.FIX_INGESTION_SCALE, RootCause.SCALING_ERROR, ids=("EV-made-up",)),
        diag(hyp(RootCause.SCALING_ERROR, 0.9)),
        SCALE_EVIDENCE,
    )
    assert uncited.outcome is PO.INVESTIGATE_MORE and "cites no existing evidence" in " ".join(
        uncited.unmet_requirements
    )
    insufficient = engine.evaluate(
        prop(A.FIX_INGESTION_SCALE, RootCause.SCALING_ERROR),
        diag(hyp(RootCause.SCALING_ERROR, 0.9), sufficient=False),
        SCALE_EVIDENCE,
    )
    assert insufficient.outcome is PO.INVESTIGATE_MORE


def test_ambiguous_or_non_leading_hypothesis_blocks_action():
    engine = PolicyEngine()
    tie = diag(hyp(RootCause.SCALING_ERROR, 0.80), hyp(RootCause.NUMERICAL_SHIFT, 0.78))
    assert (
        engine.evaluate(
            prop(A.FIX_INGESTION_SCALE, RootCause.SCALING_ERROR), tie, SCALE_EVIDENCE
        ).outcome
        is PO.INVESTIGATE_MORE
    )
    second = diag(hyp(RootCause.NUMERICAL_SHIFT, 0.9), hyp(RootCause.SCALING_ERROR, 0.7))
    decision = engine.evaluate(
        prop(A.FIX_INGESTION_SCALE, RootCause.SCALING_ERROR), second, SCALE_EVIDENCE
    )
    assert decision.outcome is PO.INVESTIGATE_MORE and "leading hypothesis" in " ".join(
        decision.unmet_requirements
    )


def test_risky_action_with_modest_confidence_is_rejected_on_score():
    evidence = [anomaly("E1", "f1_drop")]
    d = diag(hyp(RootCause.CONCEPT_DRIFT, 0.62))
    decision = PolicyEngine().evaluate(
        prop(A.RETRAIN_CANDIDATE, RootCause.CONCEPT_DRIFT, RiskLevel.HIGH), d, evidence
    )
    assert (
        decision.outcome is PO.REJECT
        and decision.score < 0
        and "not positive" in decision.reasons[-1]
    )
    confident = PolicyEngine().evaluate(
        prop(A.RETRAIN_CANDIDATE, RootCause.CONCEPT_DRIFT, RiskLevel.HIGH),
        diag(hyp(RootCause.CONCEPT_DRIFT, 0.95)),
        evidence,
    )
    assert confident.outcome is PO.PROPOSE_ACTION


def test_retraining_requires_measured_performance_loss_not_just_drift():
    d = diag(hyp(RootCause.NUMERICAL_SHIFT, 0.9))
    decision = PolicyEngine().evaluate(
        prop(A.RETRAIN_CANDIDATE, RootCause.NUMERICAL_SHIFT, RiskLevel.HIGH),
        d,
        [anomaly("E1", "numeric_drift")],
    )
    assert decision.outcome is PO.INVESTIGATE_MORE


def test_llm_ranking_cannot_exceed_rule_engine_support():
    llm_view = diag(hyp(RootCause.ARTIFACT_MISMATCH, 0.99))
    rule_view = diag(hyp(RootCause.NUMERICAL_SHIFT, 0.88))
    decision = PolicyEngine().evaluate(
        prop(A.RESTORE_MODEL_VERSION, RootCause.ARTIFACT_MISMATCH, version="v0"),
        llm_view,
        [anomaly("E1", "artifact_hash")],
        rule_diagnosis=rule_view,
    )
    assert decision.outcome is PO.INVESTIGATE_MORE and decision.expected_benefit == 0
    assert "LLM ranking alone cannot justify action" in " ".join(decision.unmet_requirements)


def test_no_action_defers_and_investigate_investigates():
    d = diag(hyp(RootCause.LABEL_DELAY, 0.9))
    assert (
        PolicyEngine()
        .evaluate(prop(A.NO_ACTION, RootCause.LABEL_DELAY, RiskLevel.NONE), d, [])
        .outcome
        is PO.DEFER
    )
    decision = PolicyEngine().evaluate(
        prop(A.INVESTIGATE_FURTHER, RootCause.LABEL_DELAY, RiskLevel.NONE), d, []
    )
    assert decision.outcome is PO.INVESTIGATE_MORE and not decision.requires_approval


def test_policy_is_versioned_and_configurable_but_strict_about_unknown_keys():
    strict = PolicyEngine(PolicyConfig(version="policy-test", min_confidence_to_act=0.95))
    decision = strict.evaluate(
        prop(A.FIX_INGESTION_SCALE, RootCause.SCALING_ERROR),
        diag(hyp(RootCause.SCALING_ERROR, 0.9)),
        SCALE_EVIDENCE,
    )
    assert decision.policy_version == "policy-test" and decision.outcome is PO.INVESTIGATE_MORE
    with pytest.raises(ValueError):
        PolicyConfig(auto_approve=True)


# ----------------------------------------------------- proposals + validation
@pytest.mark.parametrize(
    "fault,action",
    [
        ("F04", A.FIX_INGESTION_SCALE),
        ("F06", A.ENABLE_DEDUPLICATION),
        ("F08", A.QUARANTINE_BATCH),
        ("F10", A.RESTORE_MODEL_VERSION),
        ("F03", A.QUARANTINE_BATCH),
        ("F09", A.NO_ACTION),
        ("F05", A.RETRAIN_CANDIDATE),
    ],
)
def test_each_cause_maps_to_its_documented_action(ctx, fault, action):
    *_, evidence, diagnosis, _state, proposal = pipeline(ctx, fault)
    assert proposal.action is action and proposal.rollback_plan and proposal.rationale
    assert proposal.target_root_cause == diagnosis.top.root_cause.value
    assert set(proposal.evidence_ids) <= {e.evidence_id for e in evidence}


def test_drift_without_performance_loss_proposes_no_action(ctx):
    for seed in range(30):
        _, _, result, _, diagnosis, _, proposal = pipeline(ctx, "F02", "medium", seed)
        if not [f for f in result.report.triggered if f.check.endswith("_drop")]:
            break
    assert diagnosis.top.root_cause is RootCause.CATEGORICAL_SHIFT
    assert proposal.action is A.NO_ACTION and "Drift alone does not" in proposal.rationale


def test_insufficient_evidence_proposes_investigation(ctx):
    state = DeploymentState(dataset="adult", model_version="v")
    proposal = propose(
        Diagnosis(hypotheses=[], conclusion="x", sufficient_evidence=False), [], state, "v"
    )
    assert proposal.action is A.INVESTIGATE_FURTHER and proposal.risk_level is RiskLevel.NONE


def test_scale_fix_validates_on_held_out_half_and_restores_performance(ctx):
    scenario, inp, result, _, _, state, proposal = pipeline(ctx, "F04")
    assert proposal.parameters["factor"] == pytest.approx(
        1 / scenario.truth.params["factor"], rel=0.1
    )
    validation = validate_proposal(proposal, inp, result, state)
    assert validation.passed, validation.summary
    names = {c.name: c for c in validation.checks}
    assert {
        "factor_reproducible_on_independent_half",
        "targeted_findings_cleared",
        "no_new_findings",
        "data_contract_valid",
        "performance_not_degraded",
    } <= set(names)
    assert {c.category for c in validation.checks} >= {
        "effect",
        "regression",
        "contract",
        "precondition",
    }
    feature = proposal.parameters["feature"]
    assert validation.candidate_state["ingestion"]["unit_scale"][feature] == pytest.approx(
        proposal.parameters["factor"]
    )
    assert state.ingestion == IngestionConfig()  # the live state object was not touched


def test_wrong_scale_factor_fails_validation(ctx):
    _, inp, result, _, _, state, proposal = pipeline(ctx, "F04")
    wrong = proposal.model_copy(
        update={"parameters": {**proposal.parameters, "factor": proposal.parameters["factor"] * 7}}
    )
    validation = validate_proposal(wrong, inp, result, state)
    assert not validation.passed and validation.candidate_state is None
    failed = {c.name for c in validation.checks if not c.passed}
    assert {"factor_reproducible_on_independent_half", "targeted_findings_cleared"} <= failed


@pytest.mark.parametrize(
    "params",
    [
        {"feature": "sex", "factor": 2.0},
        {"feature": "age", "factor": -1},
        {"feature": "age", "factor": float("nan")},
        {"feature": "nope", "factor": 2.0},
        {"feature": "age"},
        {"feature": "age", "factor": 1.0},
    ],
)
def test_scale_fix_rejects_nonsense_parameters(ctx, params):
    _, inp, result, _, _, state, proposal = pipeline(ctx, "F04")
    validation = validate_proposal(
        proposal.model_copy(update={"parameters": params}), inp, result, state
    )
    assert not validation.passed and validation.checks[0].category == "precondition"


def test_deduplication_validation(ctx):
    _, inp, result, _, _, state, proposal = pipeline(ctx, "F06")
    validation = validate_proposal(proposal, inp, result, state)
    assert validation.passed and validation.candidate_state["ingestion"]["deduplicate"] is True
    # On clean data there is nothing to fix, so the same proposal must not validate.
    clean = ctx.clean(seed=9)
    clean_inp = MonitoringInput(
        ctx.spec, ctx.contract, ctx.reference, clean.current, ctx.registry, ctx.record.version
    )
    assert not validate_proposal(proposal, clean_inp, run_monitoring(clean_inp), state).passed


def test_restore_validation_checks_hash_and_recorded_performance(ctx):
    _, inp, result, _, _, state, proposal = pipeline(ctx, "F10", "medium")
    validation = validate_proposal(proposal, inp, result, state)
    assert validation.passed and {c.category for c in validation.checks} >= {"integrity", "effect"}
    bogus = proposal.model_copy(update={"parameters": {"version": "never-registered"}})
    rejected = validate_proposal(bogus, inp, result, state)
    assert not rejected.passed and rejected.checks[0].name == "target_version_registered"


def test_restore_refuses_a_target_whose_registry_artifact_is_corrupt(
    ctx, adult_splits, adult_contract, tmp_path
):
    from aegis.models.registry import ModelRegistry
    from tests.conftest import MonitorContext, _register

    registry = ModelRegistry(tmp_path / "models")
    record = _register(ctx.spec, adult_splits, adult_contract, registry, "v1")
    _register(ctx.spec, adult_splits, adult_contract, registry, "v0", C=0.001)
    local = MonitorContext(
        ctx.spec, adult_splits, adult_contract, registry, record, tmp_path / "scn"
    )
    _, inp, result, _, _, state, proposal = pipeline(local, "F10", "medium")
    path = registry.artifact_path("adult", "v1")
    path.write_bytes(path.read_bytes()[:1000])  # the "good" copy is itself damaged
    validation = validate_proposal(proposal, inp, result, state)
    assert not validation.passed and validation.checks[0].name == "target_artifact_hash_verified"


def test_quarantine_needs_real_grounds(ctx):
    _, inp, result, _, _, state, proposal = pipeline(ctx, "F08")
    validation = validate_proposal(proposal, inp, result, state)
    assert validation.passed and validation.candidate_state["quarantined"] is True
    clean = ctx.clean(seed=9)
    clean_inp = MonitoringInput(
        ctx.spec, ctx.contract, ctx.reference, clean.current, ctx.registry, ctx.record.version
    )
    refused = validate_proposal(proposal, clean_inp, run_monitoring(clean_inp), state)
    assert not refused.passed and refused.checks[0].name == "grounds_for_quarantine_present"


def test_retraining_on_flipped_labels_is_not_validated(ctx):
    """Retraining 'works' on the corrupted batch but regresses badly on the reference set."""
    _, inp, result, _, _, state, proposal = pipeline(ctx, "F05", "high")
    before = {r.version for r in ctx.registry.list("adult")}
    validation = validate_proposal(proposal, inp, result, state)
    assert not validation.passed and validation.regression_failed
    assert validation.candidate_model_version is None and validation.candidate_state is None
    assert {r.version for r in ctx.registry.list("adult")} == before  # nothing registered


def test_retraining_needs_enough_labels(ctx):
    _, inp, result, _, _, state, _ = pipeline(
        ctx, "F09", "medium", n_rows=1000
    )  # 200 labelled rows
    proposal = prop(A.RETRAIN_CANDIDATE, RootCause.CONCEPT_DRIFT, RiskLevel.HIGH)
    validation = validate_proposal(proposal, inp, result, state)
    assert not validation.passed and validation.checks[0].name == "enough_labelled_recent_data"


def test_validation_crash_counts_as_failure(ctx, monkeypatch):
    from aegis.remediation import validation as module

    _, inp, result, _, _, state, proposal = pipeline(ctx, "F06")
    monkeypatch.setitem(
        module._VALIDATORS,
        A.ENABLE_DEDUPLICATION,
        lambda *a: (_ for _ in ()).throw(RuntimeError("sandbox exploded")),
    )
    validation = validate_proposal(proposal, inp, result, state)
    assert not validation.passed and "sandbox exploded" in validation.checks[0].detail


def test_non_state_changing_actions_have_nothing_to_validate(ctx):
    _, inp, result, _, _, state, proposal = pipeline(ctx, "F09")
    assert not validate_proposal(proposal, inp, result, state).passed


# ------------------------------------------------------------------ executor
def test_executor_applies_exactly_the_validated_state(ctx):
    _, inp, result, _, _, state, proposal = pipeline(ctx, "F04")
    validation = validate_proposal(proposal, inp, result, state)
    new_state = state_to_execute(state, proposal, validation)
    assert new_state == candidate_state(state, proposal) and new_state != state
    monitored = run_monitoring(
        MonitoringInput(
            ctx.spec,
            ctx.contract,
            ctx.reference,
            inp.current,
            ctx.registry,
            new_state.model_version,
            None,
            new_state.ingestion,
        )
    )
    assert not monitored.report.incident_detected  # the executed fix actually resolves the incident


def test_executor_refuses_unvalidated_and_stale_proposals(ctx):
    _, inp, result, _, _, state, proposal = pipeline(ctx, "F04")
    with pytest.raises(ValueError):
        state_to_execute(state, proposal, FAILED)
    with pytest.raises(ValueError):
        state_to_execute(state, proposal, PASSED)  # passed, but carries no candidate state
    validation = validate_proposal(proposal, inp, result, state)
    moved = state.model_copy(update={"ingestion": IngestionConfig(deduplicate=True)})
    with pytest.raises(StaleProposalError):
        state_to_execute(moved, proposal, validation)


def test_candidate_state_composes_with_existing_scale_and_is_pure():
    state = DeploymentState(
        dataset="adult", model_version="v1", ingestion=IngestionConfig(unit_scale={"age": 2.0})
    )
    proposal = prop(A.FIX_INGESTION_SCALE, RootCause.SCALING_ERROR, feature="age", factor=0.5)
    new = candidate_state(state, proposal)
    assert (
        new.ingestion.unit_scale["age"] == pytest.approx(1.0)
        and state.ingestion.unit_scale["age"] == 2.0
    )
    assert np.isclose(candidate_state(new, proposal).ingestion.unit_scale["age"], 0.5)
