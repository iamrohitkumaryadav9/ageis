"""Deterministic claim validation: language classification and the evidence-strength policy."""

from __future__ import annotations

import pytest

from aegis.agents import claims as C
from aegis.agents.claims import (
    ClaimCategory,
    ClaimLanguage,
    ClaimVerdict,
    Intervention,
    classify_language,
    evaluate_claim,
)
from aegis.schemas.diagnosis import Diagnosis, Evidence, Hypothesis, RootCause
from aegis.schemas.findings import EpistemicLevel, Severity
from aegis.schemas.remediation import (
    ActionType,
    RemediationProposal,
    RiskLevel,
    ValidationCheck,
    ValidationResult,
)


def ev(eid, check, *, feature=None, polarity="anomaly", source="monitoring:drift"):
    return Evidence(
        evidence_id=eid,
        source=source,
        level=EpistemicLevel.STATISTICAL_ANOMALY,
        statement=f"{check} on {feature}",
        feature=feature,
        severity=Severity.MEDIUM,
        data={"check": check, "polarity": polarity},
    )


DRIFT = ev("EV-drift", "numeric_drift", feature="hours_per_week")
RANGE = ev(
    "EV-range", "range_violation_rate", feature="hours_per_week", source="monitoring:quality"
)
PATTERN = ev(
    "EV-pattern",
    "feature_pattern",
    feature="hours_per_week",
    source="tool:inspect_feature_distribution",
)
PERF = ev("EV-perf", "roc_auc_drop", source="monitoring:performance")
DUP = ev("EV-dup", "duplicate_rate", source="monitoring:quality")
NO_DRIFT = ev("EV-ok-drift", "no_covariate_drift", polarity="normal", source="monitoring:summary")
RUNBOOK = ev(
    "EV-runbook", "numeric_drift", feature="hours_per_week", source="tool:retrieve_runbook"
)
ALL = [DRIFT, RANGE, PATTERN, PERF, DUP, NO_DRIFT]


def claim(text, *, cited=None, causes=None, evidence=None, **kw):
    return evaluate_claim(
        claim_id="T-1",
        origin="test",
        text=text,
        evidence=ALL if evidence is None else evidence,
        cited_ids=cited,
        causes=causes,
        **kw,
    )


def passed_intervention(cause="scaling_error", *, passed=True, effect=True):
    return Intervention(
        target_root_cause=cause,
        features=["hours_per_week"],
        passed=passed,
        effect_checks_passed=effect,
    )


# ------------------------------------------------------------------ language
@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("The feature distribution shift caused the model performance degradation.", "causal"),
        ("Performance dropped due to the shift in hours_per_week.", "causal"),
        ("The unit change led to more errors.", "causal"),
        ("Errors rose because of the scaling error.", "causal"),
        ("The scaling error is responsible for the range violations.", "causal"),
        ("The root cause is a scaling error.", "causal"),
        ("Duplicate rows resulted in an inflated batch.", "causal"),
        ("This explains why precision fell.", "causal"),
        ("The drop stems from the missing values.", "causal"),
        ("The shift degraded the model.", "causal"),
        ("The shift hurt the model.", "assertive"),  # outside the lexicon: a documented limit
        ("The shift degraded the model's accuracy, consequently recall fell.", "causal"),
        ("This proves a unit error.", "certain"),
        ("It is definitely concept drift.", "certain"),
        ("The evidence conclusively shows leakage.", "certain"),
        ("The evidence is consistent with a scaling error.", "hedged"),
        ("The shift may have caused the degradation.", "hedged"),
        ("A unit error is the likely explanation.", "hedged"),
        ("Leading hypothesis: scaling_error (evidence fit 0.88).", "hedged"),
        (
            "A feature distribution shift was detected alongside model performance degradation.",
            "associative",
        ),
        ("Drift in age is correlated with the accuracy drop.", "associative"),
        ("The available evidence does not establish causality.", "negated_causal"),
        ("There is no evidence that the shift caused the drop.", "negated_causal"),
        ("Causation is not established.", "negated_causal"),
        ("hours_per_week values are about 60 times larger than the reference.", "assertive"),
        ("3 of 14 features drifted.", "assertive"),
        ("The most consistent root cause candidate is scaling error.", "hedged"),
        ("Root cause ranking: scaling error first.", "assertive"),
    ],
)
def test_language_classification(sentence, expected):
    assert classify_language(sentence) == ClaimLanguage(expected)


def test_negation_does_not_excuse_a_second_causal_assertion():
    text = "The shift caused the drop, although this does not prove anything."
    assert classify_language(text) is ClaimLanguage.CAUSAL


def test_double_negative_is_not_treated_as_a_denial():
    assert classify_language("It is not unlikely that the shift caused the drop.") in (
        ClaimLanguage.CAUSAL,
        ClaimLanguage.HEDGED,
    )


def test_certainty_overrides_a_hedge():
    assert classify_language("This possibly proves the leak.") is ClaimLanguage.CERTAIN


def test_hedge_in_one_sentence_does_not_soften_a_causal_sentence():
    text = "Evidence is consistent with a unit error. The unit error caused the errors."
    assert C.strongest_language(text) is ClaimLanguage.CAUSAL


def test_the_noun_phrase_root_cause_alone_is_not_a_causal_assertion():
    assert classify_language("Root cause candidates were ranked.") is ClaimLanguage.HEDGED
    assert classify_language("Two root causes were considered.") is ClaimLanguage.ASSERTIVE


def test_text_is_bounded():
    long = "The shift caused the drop. " * 500
    assert len(C.split_sentences(long)) <= C.MAX_SENTENCES


# ------------------------------------------------------------ evidence policy
def test_drift_alert_alone_cannot_justify_a_causal_claim():
    c = claim(
        "The feature distribution shift caused the model performance degradation.",
        cited=["EV-drift", "EV-perf"],
        causes=[RootCause.NUMERICAL_SHIFT],
    )
    assert c.category is ClaimCategory.CAUSAL_CLAIM
    assert c.verdict is ClaimVerdict.REJECTED
    assert c.evidence_strength == "OBSERVATION" and c.required_strength == "INTERVENTION"
    assert any("association, not causation" in r for r in c.reasons)


def test_the_same_evidence_supports_the_associative_wording():
    c = claim(
        "A feature distribution shift was detected alongside model performance degradation. ",
        cited=["EV-drift", "EV-perf"],
        causes=[RootCause.NUMERICAL_SHIFT],
    )
    assert c.category is ClaimCategory.STATISTICAL_ASSOCIATION
    assert c.verdict is ClaimVerdict.SUPPORTED


def test_association_with_performance_needs_a_measured_performance_change():
    c = claim(
        "Drift in hours_per_week was detected alongside a performance degradation.",
        cited=["EV-drift"],
        causes=[RootCause.NUMERICAL_SHIFT],
        evidence=[DRIFT, RANGE],
    )
    assert c.category is ClaimCategory.UNSUPPORTED_CLAIM and c.verdict is ClaimVerdict.REJECTED


def test_mechanism_evidence_escalates_instead_of_supporting():
    c = claim(
        "The unit error caused the range violations in hours_per_week.",
        cited=["EV-pattern", "EV-range"],
        causes=[RootCause.SCALING_ERROR],
    )
    assert c.category is ClaimCategory.CAUSAL_CLAIM
    assert c.verdict is ClaimVerdict.ESCALATED
    assert c.evidence_strength == "MECHANISM"


def test_passed_intervention_for_the_same_cause_supports_a_causal_claim():
    c = claim(
        "The scaling error caused the range violations in hours_per_week.",
        cited=["EV-pattern", "EV-range"],
        causes=[RootCause.SCALING_ERROR],
        intervention=passed_intervention(),
    )
    assert c.verdict is ClaimVerdict.SUPPORTED and c.evidence_strength == "INTERVENTION"


@pytest.mark.parametrize(
    "intervention",
    [
        passed_intervention("duplicate_records"),  # an intervention for a different cause
        passed_intervention(passed=False),  # validation failed
        passed_intervention(effect=False),  # the fix did not clear the targeted findings
    ],
)
def test_intervention_must_pass_and_must_be_for_the_same_cause(intervention):
    c = claim(
        "The scaling error caused the range violations in hours_per_week.",
        cited=["EV-range"],
        causes=[RootCause.SCALING_ERROR],
        intervention=intervention,
    )
    assert c.verdict is not ClaimVerdict.SUPPORTED


def test_intervention_is_built_from_a_real_validation_result():
    proposal = RemediationProposal(
        action=ActionType.FIX_INGESTION_SCALE,
        parameters={"feature": "hours_per_week", "factor": 1 / 60},
        target_root_cause="scaling_error",
        rationale="r",
        risk_level=RiskLevel.MEDIUM,
        expected_effect="e",
        rollback_plan="p",
    )
    good = ValidationResult(
        passed=True,
        checks=[
            ValidationCheck(
                name="targeted_findings_cleared", passed=True, detail="", category="effect"
            ),
            ValidationCheck(name="no_new_findings", passed=True, detail="", category="regression"),
        ],
    )
    built = Intervention.from_validation(proposal, good)
    assert built is not None and built.supports("scaling_error")
    assert not built.supports("numerical_shift")
    no_effect = ValidationResult(
        passed=True,
        checks=[ValidationCheck(name="x", passed=True, detail="", category="regression")],
    )
    assert not Intervention.from_validation(proposal, no_effect).supports("scaling_error")
    assert Intervention.from_validation(None, good) is None
    assert Intervention.from_validation(proposal.model_dump(), good.model_dump()).supports(
        "scaling_error"
    )


def test_fabricated_ids_are_recorded_and_do_not_count_as_support():
    c = claim(
        "Evidence is consistent with a scaling error.",
        cited=["EV-nope", "EV-also-nope"],
        causes=[RootCause.SCALING_ERROR],
    )
    assert c.fabricated_evidence_ids == ["EV-nope", "EV-also-nope"]
    assert c.category is ClaimCategory.UNSUPPORTED_CLAIM and c.verdict is ClaimVerdict.REJECTED


def test_irrelevant_evidence_is_stripped_and_cannot_carry_the_claim():
    c = claim(
        "Evidence is consistent with a scaling error.",
        cited=["EV-dup"],
        causes=[RootCause.SCALING_ERROR],
    )
    assert c.irrelevant_evidence_ids == ["EV-dup"]
    assert c.category is ClaimCategory.UNSUPPORTED_CLAIM


def test_relevant_evidence_supports_a_hypothesis():
    c = claim(
        "Evidence is consistent with a scaling error.",
        cited=["EV-drift", "EV-dup"],
        causes=[RootCause.SCALING_ERROR],
    )
    assert c.category is ClaimCategory.HYPOTHESIS and c.verdict is ClaimVerdict.SUPPORTED
    assert c.valid_evidence_ids == ["EV-drift"] and c.irrelevant_evidence_ids == ["EV-dup"]


def test_evidence_without_deterministic_provenance_is_not_evidence():
    c = claim(
        "Evidence is consistent with a scaling error.",
        cited=["EV-runbook"],
        causes=[RootCause.SCALING_ERROR],
        evidence=[*ALL, RUNBOOK],
    )
    assert c.untrusted_evidence_ids == ["EV-runbook"]
    assert c.category is ClaimCategory.UNSUPPORTED_CLAIM
    assert C.strength_of(RUNBOOK) is C.EvidenceStrength.NONE


def test_normal_evidence_alone_does_not_support_a_claim():
    c = claim(
        "Evidence is consistent with concept drift.",
        cited=["EV-ok-drift"],
        causes=[RootCause.CONCEPT_DRIFT],
    )
    assert c.category is ClaimCategory.UNSUPPORTED_CLAIM


def rules_with_contradiction():
    return Diagnosis(
        hypotheses=[
            Hypothesis(
                hypothesis_id="HYP-1",
                root_cause=RootCause.CONCEPT_DRIFT,
                supporting_evidence_ids=["EV-perf"],
                contradicting_evidence_ids=["EV-drift"],
                confidence=0.4,
                reasoning="r",
            )
        ],
        engine="rules",
        sufficient_evidence=False,
        conclusion="c",
    )


def test_citing_contradicting_evidence_as_support_is_a_contradictory_claim():
    c = claim(
        "Evidence is consistent with concept drift.",
        cited=["EV-drift", "EV-perf"],
        causes=[RootCause.CONCEPT_DRIFT],
        rule_diagnosis=rules_with_contradiction(),
    )
    assert c.category is ClaimCategory.CONTRADICTORY_CLAIM and c.verdict is ClaimVerdict.REJECTED


def test_denying_a_recorded_anomaly_is_a_contradictory_claim():
    c = claim("hours_per_week shows no drift.", cited=None)
    assert c.category is ClaimCategory.CONTRADICTORY_CLAIM
    assert claim("fnlwgt shows no drift.", cited=None).category is ClaimCategory.OBSERVED_FACT


def test_observed_fact_about_a_recorded_anomaly_is_supported():
    c = claim("hours_per_week values are far above the reference range.", cited=None)
    assert c.category is ClaimCategory.OBSERVED_FACT and c.verdict is ClaimVerdict.SUPPORTED
    assert c.subject_features == ["hours_per_week"]


def test_negated_causal_statement_is_accepted_without_evidence():
    c = claim("The available evidence does not establish causality.", cited=None, evidence=[])
    assert c.verdict is ClaimVerdict.SUPPORTED and c.category is ClaimCategory.HYPOTHESIS


# ------------------------------------------------------------------ counters
def test_counters_separate_generated_supported_rejected_escalated():
    claims = [
        claim("The shift caused the drop.", cited=["EV-drift"], causes=[RootCause.NUMERICAL_SHIFT]),
        claim(
            "The unit error caused the violations.",
            cited=["EV-pattern"],
            causes=[RootCause.SCALING_ERROR],
        ),
        claim(
            "The scaling error caused the violations.",
            cited=["EV-range"],
            causes=[RootCause.SCALING_ERROR],
            intervention=passed_intervention(),
        ),
        claim("Consistent with a scaling error.", cited=["EV-x"], causes=[RootCause.SCALING_ERROR]),
        claim(
            "Consistent with a scaling error.", cited=["EV-drift"], causes=[RootCause.SCALING_ERROR]
        ),
    ]
    counters = C.count(claims)
    assert counters["claims_evaluated"] == 5
    assert counters["causal_claims_generated"] == 3
    assert counters["causal_claims_rejected"] == 1
    assert counters["causal_claims_escalated"] == 1
    assert counters["causal_claims_supported"] == 1
    assert (
        counters["causal_claims_generated"]
        == counters["causal_claims_supported"]
        + counters["causal_claims_rejected"]
        + counters["causal_claims_escalated"]
    )
    assert counters["unsupported_claims_generated"] == counters["unsupported_claims_rejected"] == 1
    assert counters["fabricated_evidence_ids_generated"] == 1
    assert counters["fabricated_evidence_ids_stripped"] == 1


def test_counters_add_up_and_ignore_unknown_keys():
    a, b = C.count([claim("x caused y.", cited=[])]), C.count([])
    total = C.add_counters(a, b, None, {"nonsense": 5, "causal_claims_escaped": 2})
    assert total["causal_claims_generated"] == 1 and total["causal_claims_escaped"] == 2
    assert "nonsense" not in total


def test_hedged_rewrite_never_contains_causal_language():
    text = C.hedged_rewrite(["scaling_error"], ["hours_per_week"])
    assert all(
        C.classify_language(s) in (ClaimLanguage.HEDGED, ClaimLanguage.NEGATED_CAUSAL)
        for s in C.split_sentences(text)
    )


# --------------------------------------------------------------- final audit
def test_audit_detects_what_validation_should_have_removed():
    leaked = Diagnosis(
        hypotheses=[
            Hypothesis(
                hypothesis_id="H",
                root_cause=RootCause.SCALING_ERROR,
                supporting_evidence_ids=["EV-ghost"],
                confidence=0.8,
                reasoning="The unit error caused the drop.",
                ranked_by="llm",
            )
        ],
        engine="llm+rules",
        sufficient_evidence=True,
        conclusion="c",
    )
    audit = C.audit_final_output(
        diagnosis=leaked,
        explanation={
            "source": "llm:qwen3:4b",
            "summary": "Drift led to errors.",
            "operator_checks": [],
        },
        evidence=ALL,
    )
    assert audit == {
        "causal_claims_escaped": 2,
        "unsupported_claims_escaped": 1,
        "fabricated_evidence_ids_escaped": 1,
    }


def test_audit_ignores_deterministic_text_and_passes_clean_output():
    clean = Diagnosis(
        hypotheses=[
            Hypothesis(
                hypothesis_id="H",
                root_cause=RootCause.SCALING_ERROR,
                supporting_evidence_ids=["EV-drift"],
                confidence=0.8,
                reasoning="Evidence is consistent with a unit error.",
                ranked_by="llm",
            )
        ],
        engine="llm+rules",
        sufficient_evidence=True,
        conclusion="c",
    )
    audit = C.audit_final_output(
        diagnosis=clean,
        explanation={"source": "deterministic", "summary": "X caused Y."},  # not LLM text
        evidence=ALL,
    )
    assert audit == dict.fromkeys(C.ESCAPE_COUNTERS, 0)


def test_prose_that_cites_nothing_is_not_charged_with_irrelevant_evidence():
    c = claim("hours_per_week values are far above the reference range.", cited=None)
    assert c.irrelevant_evidence_ids == [] and c.untrusted_evidence_ids == []
    assert C.count([c])["irrelevant_evidence_ids_stripped"] == 0
    cited = claim(
        "Evidence is consistent with a scaling error.",
        cited=["EV-drift", "EV-dup"],
        causes=[RootCause.SCALING_ERROR],
    )
    assert C.count([cited])["irrelevant_evidence_ids_stripped"] == 1


def test_ranking_item_that_denies_its_own_cause_is_unsupported_not_accepted():
    # Seen in a real run: the model ranked "scaling_error" with the reasoning below and a
    # fabricated id. A denial is not support; the item must be dropped, not kept as a hypothesis.
    c = claim(
        "No evidence of scale change in features",
        cited=["EV-ok-data_quality_report-7a5753"],
        causes=[RootCause.SCALING_ERROR],
    )
    assert c.language is ClaimLanguage.NEGATED_CAUSAL
    assert c.category is ClaimCategory.UNSUPPORTED_CLAIM and c.verdict is ClaimVerdict.REJECTED
    assert c.fabricated_evidence_ids == ["EV-ok-data_quality_report-7a5753"]
    # the same words as free prose, citing nothing, remain an acceptable statement
    assert claim("No evidence of scale change in features", cited=None, evidence=[]).verdict is (
        ClaimVerdict.SUPPORTED
    )
