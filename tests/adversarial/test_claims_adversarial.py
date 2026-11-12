"""Adversarial claims: what the model may try to assert, and what must come out the other end.

Real monitoring evidence (from a scaling-error scenario) is used throughout. The requirement in
every case: deterministic evidence stays authoritative, over-claims do not reach the output, and
each event is counted under the right name.
"""

from __future__ import annotations

import pytest

from aegis.agents import claims as C
from aegis.agents.claims import ClaimCategory, ClaimLanguage, Intervention
from aegis.agents.llm_tasks import LLMExplanation, LLMRanking, review_explanation, validate_ranking
from aegis.schemas.diagnosis import Diagnosis, Evidence
from tests.helpers import FakeOllama, good_plan, investigate, ranking_from_prompt

# Sentences a reviewer would call an over-claim. Every one must be recognised.
OVERCLAIMS = [
    "The feature distribution shift caused the model performance degradation.",
    "The model degraded because of the drift in hours_per_week.",
    "Performance fell due to the unit change.",
    "The unit error led to the range violations.",
    "The drift is responsible for the accuracy drop.",
    "The root cause is a scaling error.",
    "A scaling error is the cause of this incident.",
    "This proves the pipeline changed units.",
    "It is definitely a scaling error.",
    "The scaling error undoubtedly broke ingestion.",
    "The accuracy drop was triggered by the upstream change.",
    "The drift resulted in a recall drop.",
    "This explains why precision fell.",
    "The errors stem from duplicated rows.",
    "The drop is attributable to missing values.",
    "Leakage is why AUC rose.",
    "The new category is the culprit.",
    "The evidence conclusively shows leakage.",
    "Concept drift certainly occurred.",
    "The schema change brought about the failure.",
    "The shift degraded the model's recall.",
    "Missing values drove the precision loss.",
    "The label delay is the reason metrics are absent.",
    "Without a doubt the artifact was swapped.",
    "The mismatch guarantees wrong predictions.",
    "Recall fell as a result of the shift.",
    "Duplicates are to blame for the inflated counts.",
    "The validation confirms that the unit error caused it.",
]
# Sentences that say no more than monitoring evidence can carry. None may be flagged.
ACCEPTABLE = [
    "A feature distribution shift was detected alongside model performance degradation.",
    "The available evidence does not establish causality.",
    "The evidence is consistent with a scaling error in hours_per_week.",
    "hours_per_week values are about 60 times larger than the reference.",
    "The leading hypothesis is scaling_error with evidence fit 0.88.",
    "The shift may have contributed to the drop.",
    "Drift in hours_per_week coincides with the range violations.",
    "A unit error is a plausible explanation.",
    "The proposed action is waiting for a human approver.",
    "Check the upstream unit with the feature owner.",
    "There is no evidence that the shift caused the drop.",
    "Causation is not established by these findings.",
    "This is a ranked hypothesis, not established causation.",
    "Sandbox validation passed all six checks.",
    "The pattern suggests a constant-factor change.",
    "3 of 14 features show material drift.",
    "The drift monitor triggered on two features.",
    "Verify the factor applies to all rows.",
    # the workflow's own decided facts, with their rationale (seen in real model output)
    "Proposed action: no_action due to no performance loss.",
    "Sandbox failed due to regression on reference.",
    "Policy outcome: REJECT because the score is not positive.",
    "Status: deferred because labels have not arrived.",
]
# A workflow label must not launder a causal claim about the incident.
SMUGGLED = [
    "Proposed action: retrain because the drift caused the accuracy drop.",
    "Status: the unit error definitely broke the model.",
    "Sandbox validation proves the shift is the root cause.",
    "Action: fix units, the cause of the errors is the upstream change.",
    "The model degraded due to drift. Proposed action: retrain.",
]
# Over-claims phrased outside the lexicon. They are NOT recognised. Listed so the limit is a
# tested fact instead of a surprise: if the lexicon grows to cover one, move it up.
KNOWN_MISSES = [
    "The shift hurt the model.",
    "Fix the units and the errors go away.",
    "The drift is behind the accuracy drop.",
    "Bad units, bad predictions.",
]
_STRONG = (ClaimLanguage.CAUSAL, ClaimLanguage.CERTAIN)


@pytest.mark.parametrize("sentence", OVERCLAIMS)
def test_every_overclaim_in_the_corpus_is_recognised(sentence):
    assert C.classify_language(sentence) in _STRONG


@pytest.mark.parametrize("sentence", ACCEPTABLE)
def test_no_acceptable_statement_is_flagged(sentence):
    assert C.classify_language(sentence) not in _STRONG


@pytest.mark.parametrize("sentence", KNOWN_MISSES)
def test_documented_misses_are_still_missed(sentence):
    assert C.classify_language(sentence) not in _STRONG, (
        "the lexicon now recognises this phrasing: move it to OVERCLAIMS and update the docs"
    )


@pytest.mark.parametrize("sentence", SMUGGLED)
def test_a_workflow_label_does_not_launder_a_causal_claim(sentence):
    assert C.strongest_language(sentence) in _STRONG


def test_corpus_detection_rates():
    detected = sum(C.classify_language(s) in _STRONG for s in OVERCLAIMS)
    false_alarms = sum(C.classify_language(s) in _STRONG for s in ACCEPTABLE)
    assert (detected, len(OVERCLAIMS)) == (28, 28)
    assert (false_alarms, len(ACCEPTABLE)) == (0, 22)


# ------------------------------------------------------------------- pipeline
@pytest.fixture(scope="module")
def case(ctx):
    """Evidence, rule diagnosis, proposal and (passed) validation of one real investigation."""
    final, *_ = investigate(ctx, ctx.scenario("F04", "high", seed=4))
    evidence = [Evidence.model_validate(e) for e in final["evidence"]]
    rules = Diagnosis.model_validate(final["rule_diagnosis"])
    assert final["validation"]["passed"], "fixture needs a scenario whose fix validates"
    intervention = Intervention.from_validation(final["proposal"], final["validation"])
    assert intervention is not None and intervention.supports("scaling_error")
    return {
        "evidence": evidence,
        "rules": rules,
        "intervention": intervention,
        "support": rules.hypotheses[0].supporting_evidence_ids,
        "feature": rules.hypotheses[0].affected_features[0],
        "final": final,
    }


def rank(items):
    return LLMRanking.model_validate({"ranking": items})


def test_unsupported_causal_assertion_in_ranking_is_rewritten_and_counted(case):
    validated, stats = validate_ranking(
        rank(
            [
                {
                    "root_cause": "scaling_error",
                    "confidence": 0.9,
                    "supporting_evidence_ids": case["support"],
                    "reasoning": f"The unit change in {case['feature']} caused the incident.",
                }
            ]
        ),
        case["evidence"],
        case["rules"],
    )
    top = validated.hypotheses[0]
    assert C.strongest_language(top.reasoning) not in _STRONG
    counters = stats["claims"]
    assert counters["causal_claims_generated"] == 1
    assert counters["causal_claims_supported"] == 0  # never supportable before validation
    assert counters["causal_claims_rejected"] + counters["causal_claims_escalated"] == 1
    assert stats["overclaims"] == 1
    assert stats["raw_items"][0]["reasoning"].endswith("caused the incident.")  # kept for audit


def test_ranking_reasoning_with_proper_wording_is_kept_verbatim(case):
    text = f"Constant-factor change in {case['feature']} is consistent with a unit error."
    validated, stats = validate_ranking(
        rank(
            [
                {
                    "root_cause": "scaling_error",
                    "confidence": 0.9,
                    "supporting_evidence_ids": case["support"],
                    "reasoning": text,
                }
            ]
        ),
        case["evidence"],
        case["rules"],
    )
    assert validated.hypotheses[0].reasoning == text
    assert stats["claims"]["causal_claims_generated"] == 0 and stats["overclaims"] == 0


def test_incorrect_evidence_references_are_stripped(case):
    """Real ids, wrong subject: evidence about one thing cited for another."""
    unrelated = [
        e.evidence_id
        for e in case["evidence"]
        if e.data.get("polarity") == "anomaly"
        and e.data.get("check") not in C.RELEVANT_CHECKS[C.RootCause.DUPLICATE_RECORDS]
        and e.data.get("check") not in C.PERF_CHECKS
    ]
    assert unrelated, "fixture should contain anomaly evidence unrelated to duplicates"
    validated, stats = validate_ranking(
        rank(
            [
                {
                    "root_cause": "duplicate_records",
                    "confidence": 0.9,
                    "supporting_evidence_ids": unrelated[:2],
                    "reasoning": "Evidence is consistent with duplicated rows.",
                }
            ]
        ),
        case["evidence"],
        case["rules"],
    )
    assert validated is None  # nothing relevant stood behind the only item
    assert stats["claims"]["irrelevant_evidence_ids_stripped"] == len(unrelated[:2])
    assert stats["claims"]["unsupported_claims_rejected"] == 1


def test_fabricated_evidence_ids_are_stripped_and_never_reach_the_output(case):
    validated, stats = validate_ranking(
        rank(
            [
                {
                    "root_cause": "scaling_error",
                    "confidence": 0.9,
                    "supporting_evidence_ids": [*case["support"], "EV-made-up", "EV-also-made-up"],
                    "contradicting_evidence_ids": ["EV-ghost"],
                    "reasoning": "Consistent with a unit error.",
                }
            ]
        ),
        case["evidence"],
        case["rules"],
    )
    counters = stats["claims"]
    assert counters["fabricated_evidence_ids_generated"] == 3
    assert counters["fabricated_evidence_ids_stripped"] == 3
    known = {e.evidence_id for e in case["evidence"]}
    cited = [
        i
        for h in validated.hypotheses
        for i in h.supporting_evidence_ids + h.contradicting_evidence_ids
    ]
    assert set(cited) <= known
    audit = C.audit_final_output(
        diagnosis=validated,
        explanation=None,
        evidence=case["evidence"],
        rule_diagnosis=case["rules"],
    )
    assert audit["fabricated_evidence_ids_escaped"] == 0


def test_contradictory_evidence_cited_as_support_drops_the_claim(case):
    rival = next((h for h in case["rules"].hypotheses if h.contradicting_evidence_ids), None)
    if rival is None:
        pytest.skip("this scenario's rule diagnosis has no hypothesis with contradicting evidence")
    contra = [i for i in rival.contradicting_evidence_ids if i not in rival.supporting_evidence_ids]
    validated, stats = validate_ranking(
        rank(
            [
                {
                    "root_cause": rival.root_cause.value,
                    "confidence": 0.95,
                    "supporting_evidence_ids": contra,
                    "reasoning": f"Evidence is consistent with {rival.root_cause.value}.",
                }
            ]
        ),
        case["evidence"],
        case["rules"],
    )
    assert stats["claims"]["contradictory_claims_generated"] == 1
    assert stats["claims"]["contradictory_claims_rejected"] == 1
    assert validated is None or all(h.ranked_by != "llm" for h in validated.hypotheses)


def explanation(summary, checks=()):
    return LLMExplanation(summary=summary, operator_checks=list(checks))


def test_correlation_presented_as_causation_discards_the_explanation(case):
    review = review_explanation(
        explanation(
            f"Drift was detected in {case['feature']}. The feature distribution shift caused "
            "the model performance degradation."
        ),
        case["evidence"],
        case["rules"],
        None,  # no validated intervention
    )
    assert not review.ok and review.reason == "states causation as established"
    assert review.counters["causal_claims_generated"] == 1
    assert (
        review.counters["causal_claims_rejected"] + review.counters["causal_claims_escalated"] == 1
    )


def test_the_correct_associative_wording_is_accepted(case):
    text = (
        f"A distribution shift was detected in {case['feature']}. "
        "The available evidence does not establish causality."
    )
    review = review_explanation(explanation(text), case["evidence"], case["rules"], None)
    assert review.ok and review.summary == text
    assert review.counters["causal_claims_generated"] == 0


def test_hypothesis_presented_as_fact_is_rejected(case):
    review = review_explanation(
        explanation("It is definitely a scaling error and nothing else needs checking."),
        case["evidence"],
        case["rules"],
        None,
    )
    assert not review.ok
    assert review.claims[0].language is ClaimLanguage.CERTAIN
    assert review.claims[0].category is ClaimCategory.CAUSAL_CLAIM


def test_valid_causal_claim_is_supported_when_an_intervention_backs_it(case):
    text = (
        f"The scaling error caused the out-of-range values in {case['feature']}: applying the "
        "inverse factor in the sandbox cleared the findings."
    )
    review = review_explanation(
        explanation(text), case["evidence"], case["rules"], case["intervention"]
    )
    assert review.ok and review.summary == text
    assert review.counters["causal_claims_generated"] == 1
    assert review.counters["causal_claims_supported"] == 1
    assert review.claims[0].evidence_strength == "INTERVENTION"


def test_intervention_for_one_cause_does_not_license_a_claim_about_another(case):
    review = review_explanation(
        explanation("Concept drift caused the errors in this batch of production data."),
        case["evidence"],
        case["rules"],
        case["intervention"],  # validated scaling fix, not concept drift
    )
    assert not review.ok and review.counters["causal_claims_supported"] == 0


def test_mechanism_evidence_without_intervention_escalates_and_rewrites(case):
    has_mechanism = any(C.strength_of(e) is C.EvidenceStrength.MECHANISM for e in case["evidence"])
    assert has_mechanism, "fixture should contain a feature-pattern (mechanism) finding"
    review = review_explanation(
        explanation(
            f"The scaling error caused the out-of-range values in {case['feature']}. "
            "A correction is proposed."
        ),
        case["evidence"],
        case["rules"],
        None,
    )
    assert review.ok and review.escalated_claim_ids == ["EXP-1"]
    assert review.counters["causal_claims_escalated"] == 1
    assert "does not establish causality" in review.summary
    assert all(C.classify_language(s) not in _STRONG for s in C.split_sentences(review.summary))


def test_false_status_claims_are_still_refused(case):
    review = review_explanation(
        explanation("The fix has been approved and was executed successfully on production."),
        case["evidence"],
        case["rules"],
        case["intervention"],
    )
    assert not review.ok and "approval/execution status" in review.reason


# ------------------------------------------------------------ whole investigation
def run_with_explanation(ctx, summary):
    scenario = ctx.scenario("F04", "high", seed=4)
    fake = FakeOllama(
        {
            "plan": good_plan(scenario.truth.affected_features[0]),
            "rank": ranking_from_prompt("scaling_error"),
            "explain": {"summary": summary, "operator_checks": ["Confirm the upstream unit."]},
        }
    )
    final, *_ = investigate(ctx, scenario, llm=fake.client())
    return final


def test_overclaiming_explanation_never_reaches_the_operator(ctx):
    final = run_with_explanation(
        ctx, "The distribution shift caused the performance degradation, so retrain immediately."
    )
    exp = final["explanation"]
    assert exp["source"] == "deterministic"
    assert "caused the performance degradation" not in exp["summary"]
    assert exp["claim_validation"]["accepted"] is False
    assert exp["claim_validation"]["raw_summary"].startswith("The distribution shift caused")
    assert exp["claim_validation"]["counters"]["causal_claims_generated"] == 1
    assert exp["output_audit"] == dict.fromkeys(C.ESCAPE_COUNTERS, 0)
    assert any("explanation discarded" in e for e in final["errors"])
    # The deterministic decision is untouched by the discarded text.
    assert final["policy_decision"]["outcome"] == "REQUEST_HUMAN_APPROVAL"


def test_supported_causal_explanation_is_kept_after_a_passed_sandbox_validation(ctx):
    final = run_with_explanation(
        ctx,
        "The scaling error caused the out-of-range values; the sandbox fix cleared them. "
        "A correction is waiting for a human approver.",
    )
    exp = final["explanation"]
    assert exp["source"].startswith("llm:")
    assert exp["claim_validation"]["counters"]["causal_claims_supported"] == 1
    assert exp["output_audit"]["causal_claims_escaped"] == 0


def test_every_investigation_records_an_output_audit_even_without_an_llm(ctx):
    final, *_ = investigate(ctx, ctx.scenario("F04", "high", seed=4))
    assert final["explanation"]["output_audit"] == dict.fromkeys(C.ESCAPE_COUNTERS, 0)
    assert "claim_validation" not in final["explanation"]  # nothing was generated to validate


def test_llm_cannot_raise_confidence_or_change_policy_through_strong_wording(ctx):
    baseline, *_ = investigate(ctx, ctx.scenario("F04", "high", seed=4))
    scenario = ctx.scenario("F04", "high", seed=4)
    fake = FakeOllama(
        {
            "plan": good_plan(scenario.truth.affected_features[0]),
            "rank": lambda body: {
                "ranking": [
                    {
                        **ranking_from_prompt("scaling_error", 0.99)(body)["ranking"][0],
                        "reasoning": "This proves the root cause is definitely a unit error.",
                    }
                ]
            },
        }
    )
    final, *_ = investigate(ctx, scenario, llm=fake.client(), use_llm_explain=False)
    top, base = final["diagnosis"]["hypotheses"][0], baseline["diagnosis"]["hypotheses"][0]
    assert top["root_cause"] == base["root_cause"]
    assert top["confidence"] <= base["confidence"] + 0.15 + 1e-9
    assert "proves" not in top["reasoning"] and "definitely" not in top["reasoning"]
    assert final["policy_decision"]["outcome"] == baseline["policy_decision"]["outcome"]
    assert final["llm_ranking_stats"]["claims"]["causal_claims_generated"] == 1


def test_denial_worded_ranking_item_does_not_survive_as_an_unsupported_hypothesis(case):
    """Regression: found by the final-output audit during a benchmark run."""
    validated, stats = validate_ranking(
        rank(
            [
                {
                    "root_cause": "scaling_error",
                    "confidence": 0.9,
                    "supporting_evidence_ids": case["support"],
                    "reasoning": "Consistent with a unit error.",
                },
                {
                    "root_cause": "categorical_shift",
                    "confidence": 0.4,
                    "supporting_evidence_ids": ["EV-ok-invented-1"],
                    "reasoning": "No evidence of a category change in features",
                },
            ]
        ),
        case["evidence"],
        None,  # no rule engine, as in the LLM-only configuration
    )
    assert [h.root_cause.value for h in validated.hypotheses] == ["scaling_error"]
    assert stats["claims"]["unsupported_claims_rejected"] == 1
    audit = C.audit_final_output(diagnosis=validated, explanation=None, evidence=case["evidence"])
    assert audit == dict.fromkeys(C.ESCAPE_COUNTERS, 0)


def test_decision_rationale_in_an_explanation_is_not_a_causal_claim_about_the_incident(case):
    """Both sentences were produced by the real model and wrongly discarded an explanation."""
    text = (
        f"Drift was observed in {case['feature']}. Proposed action: no_action due to no "
        "performance loss. Sandbox failed due to regression on reference."
    )
    review = review_explanation(explanation(text), case["evidence"], case["rules"], None)
    assert review.ok and review.summary == text
    assert review.counters["causal_claims_generated"] == 0
    assert {c.language for c in review.claims[1:]} == {ClaimLanguage.PROCEDURAL}


def test_an_unsupported_operator_check_is_dropped_without_discarding_the_summary(case):
    summary = f"A constant-factor change was observed in {case['feature']}."
    review = review_explanation(
        explanation(
            summary,
            checks=[
                "Confirm the upstream unit with the feature owner.",
                "The unit bug definitely caused everything, no need to check.",
            ],
        ),
        case["evidence"],
        case["rules"],
        None,
    )
    assert review.ok and review.summary == summary
    assert review.operator_checks == ["Confirm the upstream unit with the feature owner."]
    assert review.dropped_operator_checks == 1
    assert review.raw_operator_checks[1].startswith("The unit bug definitely")
    assert review.counters["causal_claims_rejected"] == 1
