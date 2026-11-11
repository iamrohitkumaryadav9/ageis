import numpy as np
import pandas as pd
import pytest

from aegis.diagnosis.evidence import build_evidence, evidence_id_for
from aegis.diagnosis.features import _fit, inspect_feature
from aegis.diagnosis.importance import permutation_importances, shap_importances
from aegis.diagnosis.naive import naive_diagnosis
from aegis.diagnosis.performance import compare_versions, error_slices, evaluate_version
from aegis.diagnosis.pipeline import deterministic_diagnosis, features_worth_inspecting
from aegis.diagnosis.rules import MIN_CONFIDENCE, rank_hypotheses
from aegis.ingestion.datasets import ADULT, TARGET
from aegis.monitoring.engine import MonitoringInput
from aegis.schemas.diagnosis import INSUFFICIENT_EVIDENCE, Diagnosis, Evidence, RootCause
from aegis.schemas.findings import EpistemicLevel, Severity


def _diagnose(ctx, fault_id, severity="high", seed=4, n_rows=2000):
    scenario, result = ctx.run(fault_id, severity, seed, n_rows)
    inp = MonitoringInput(
        ctx.spec,
        ctx.contract,
        ctx.reference,
        scenario.current,
        ctx.registry,
        ctx.record.version,
        scenario.artifact_path,
    )
    evidence, diagnosis = deterministic_diagnosis(inp, result)
    return scenario, result, evidence, diagnosis


def ev(eid, check, polarity="anomaly", feature=None, **data):
    return Evidence(
        evidence_id=eid,
        source="test",
        level=EpistemicLevel.STATISTICAL_ANOMALY,
        statement=check,
        feature=feature,
        severity=Severity.MEDIUM,
        data={"check": check, "polarity": polarity, **data},
    )


# ------------------------------------------------------------------ evidence
def test_every_triggered_finding_becomes_traceable_evidence(ctx):
    _, result = ctx.run("F01", "high", seed=4)
    evidence = build_evidence(result.report)
    by_finding = {e.finding_id: e for e in evidence if e.finding_id}
    assert set(by_finding) == {f.finding_id for f in result.report.triggered}
    for finding in result.report.triggered:
        record = by_finding[finding.finding_id]
        assert record.evidence_id == evidence_id_for(finding.finding_id)
        assert record.statement == finding.interpretation and record.level == finding.level
    assert len({e.evidence_id for e in evidence}) == len(evidence)


def test_normal_results_are_kept_as_evidence_of_absence(ctx):
    _, result = ctx.run("F05", "high", seed=4)
    normal = {
        e.data["check"] for e in build_evidence(result.report) if e.data["polarity"] == "normal"
    }
    assert {"no_covariate_drift", "no_quality_violation", "no_artifact_mismatch"} <= normal
    assert "no_performance_drop" not in normal


# --------------------------------------------------------- feature inspection
def test_quantile_fit_separates_scale_from_shift():
    rng = np.random.default_rng(0)
    ref = rng.gamma(4, 10, 6000)
    cur = rng.gamma(4, 10, 2000)
    assert _fit(ref, cur)["pattern"] == "no_change"
    scaled = _fit(ref, cur * 60)
    assert scaled["pattern"] == "scale" and scaled["scale_factor"] == pytest.approx(60, rel=0.1)
    shifted = _fit(ref, cur + 25)
    assert shifted["pattern"] == "shift" and shifted["shift_offset"] == pytest.approx(25, rel=0.15)
    assert _fit(ref, cur * 0.01)["pattern"] == "scale"
    assert _fit(ref, np.abs(rng.normal(40, 40, 2000)))["pattern"] in ("shape_change", "shift")


def test_scale_fit_handles_zero_inflated_feature():
    rng = np.random.default_rng(1)
    ref = np.where(rng.random(8000) < 0.6, 0.0, rng.gamma(2, 500, 8000))
    cur = np.where(rng.random(3000) < 0.6, 0.0, rng.gamma(2, 500, 3000))
    assert _fit(ref, cur * 12)["pattern"] == "scale"
    assert _fit(ref, cur + 400)["pattern"] == "shift"  # zeros moved too: not a rescale


def test_inspect_feature_reports_category_movers(ctx):
    scenario, result = ctx.run("F02", "high", seed=4)
    (feature,) = scenario.truth.affected_features
    inspection = inspect_feature(
        feature, "categorical", ctx.reference, result.current, ADULT.numeric_features
    )
    assert inspection.pattern == "category_shift"
    top = max(inspection.category_shifts, key=inspection.category_shifts.get)
    assert top == scenario.truth.params["to_category"]


def test_rank_correlation_change_exposes_row_level_tampering(ctx):
    """Leakage permutes values: marginal intact, relationships to other columns disturbed."""
    for seed in range(20):
        scenario, result = ctx.run("F07", "high", seed=seed)
        if scenario.truth.affected_features[0] == "age":
            break
    leaked = inspect_feature(
        "age", "numeric", ctx.reference, result.current, ADULT.numeric_features
    )
    clean = inspect_feature(
        "age", "numeric", ctx.reference, ctx.clean().current, ADULT.numeric_features
    )
    assert leaked.pattern == "no_change"
    assert leaked.max_correlation_change > 3 * clean.max_correlation_change


# ---------------------------------------------------------------- importance
def test_permutation_importance_ranks_known_strong_features(ctx):
    model = ctx.registry.load("adult", ctx.record.version)
    out = permutation_importances(model, ctx.reference, ADULT.features, max_rows=800)
    top = list(out["importances"])[:5]
    assert out["available"] and out["rows"] == 800
    assert {"marital_status", "education_num", "capital_gain"} & set(top)
    assert out["importances"]["race"] < out["importances"][top[0]]


def test_permutation_importance_needs_labels(ctx):
    unlabelled = ctx.reference.head(300).assign(**{TARGET: np.nan})
    model = ctx.registry.load("adult", ctx.record.version)
    assert permutation_importances(model, unlabelled, ADULT.features)["available"] is False


def test_shap_is_bounded_and_aggregates_one_hot_columns(ctx):
    model = ctx.registry.load("adult", ctx.record.version)
    out = shap_importances(model, ctx.reference, ADULT.features, max_rows=150)
    shares = out["mean_abs_shap_share"]
    assert out["rows"] == 150 and set(shares) == set(ADULT.features)
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-3)
    assert out["seconds"] < 30


def test_shap_supports_the_tree_model_used_in_production(settings):
    from aegis.models.registry import ModelRegistry

    registry = ModelRegistry(settings.model_dir)
    try:
        record = registry.baseline("adult")
    except KeyError:
        pytest.skip("baseline not trained in this checkout (run scripts.train_baseline)")
    from aegis.ingestion.splits import read_split

    frame = read_split("adult", "validation", settings.processed_dir)
    out = shap_importances(
        registry.load("adult", record.version), frame, ADULT.features, max_rows=100
    )
    assert out["explainer"] == (
        "tree" if record.algorithm == "hist_gradient_boosting" else "linear"
    )
    assert max(out["mean_abs_shap_share"].values()) > 0.1


# --------------------------------------------------------------- performance
def test_compare_versions_shows_weaker_model_is_weaker(ctx, adult_model_v0):
    out = compare_versions(
        ctx.registry,
        ADULT,
        ctx.record.version,
        adult_model_v0.version,
        ctx.reference,
        ctx.clean().current,
    )
    for name in ("reference", "current"):
        assert out["datasets"][name]["available"]
        assert out["datasets"][name]["delta_b_minus_a"]["roc_auc"] < 0
        assert out["datasets"][name]["delta_b_minus_a"]["brier"] > 0


def test_compare_versions_reports_unavailable_without_labels(ctx, adult_model_v0):
    unlabelled = ctx.clean().current.assign(**{TARGET: np.nan})
    out = compare_versions(
        ctx.registry, ADULT, ctx.record.version, adult_model_v0.version, ctx.reference, unlabelled
    )
    assert out["datasets"]["current"]["available"] is False
    assert evaluate_version(ctx.registry, ADULT, ctx.record.version, unlabelled) is None


def test_error_slices_localise_a_segment_specific_failure(ctx):
    current = ctx.clean(n_rows=3000).current
    ref_err = np.zeros(len(ctx.reference))
    cur_err = (current["sex"] == "Female").to_numpy().astype(float)
    top = error_slices(ADULT, ctx.reference, ref_err, current, cur_err)[0]
    assert (top["feature"], top["segment"], top["rise"]) == ("sex", "Female", 1.0)


# --------------------------------------------------------------------- rules
@pytest.mark.parametrize(
    "fault_id,cause",
    [
        ("F01", RootCause.NUMERICAL_SHIFT),
        ("F02", RootCause.CATEGORICAL_SHIFT),
        ("F03", RootCause.MISSING_VALUES),
        ("F04", RootCause.SCALING_ERROR),
        ("F05", RootCause.CONCEPT_DRIFT),
        ("F06", RootCause.DUPLICATE_RECORDS),
        ("F08", RootCause.SCHEMA_CHANGE),
        ("F09", RootCause.LABEL_DELAY),
        ("F10", RootCause.ARTIFACT_MISMATCH),
    ],
)
def test_rules_rank_the_injected_cause_first(ctx, fault_id, cause):
    scenario, _, evidence, diagnosis = _diagnose(ctx, fault_id)
    assert diagnosis.sufficient_evidence and diagnosis.top.root_cause is cause
    known = {e.evidence_id for e in evidence}
    for h in diagnosis.hypotheses:
        assert h.supporting_evidence_ids and set(h.supporting_evidence_ids) <= known
        assert set(h.contradicting_evidence_ids) <= known
        assert not set(h.supporting_evidence_ids) & set(h.contradicting_evidence_ids)
        assert h.reasoning and h.next_diagnostic_action
    if scenario.truth.affected_features and fault_id != "F08":
        assert set(scenario.truth.affected_features) <= set(diagnosis.top.affected_features)
    assert "not established causation" in diagnosis.conclusion
    assert "caused by" not in diagnosis.model_dump_json().lower()


def test_rules_identify_leakage(ctx):
    for seed in range(20):
        scenario = ctx.scenario("F07", "high", seed=seed)
        if scenario.truth.affected_features[0] in ("age", "hours_per_week", "education_num"):
            break
    _, _, _, diagnosis = _diagnose(ctx, "F07", seed=seed)
    assert diagnosis.top.root_cause is RootCause.FEATURE_LEAKAGE
    assert diagnosis.top.affected_features == scenario.truth.affected_features


def test_scaling_and_shift_contradict_each_other(ctx):
    _, _, evidence, diagnosis = _diagnose(ctx, "F04")
    by_cause = {h.root_cause: h for h in diagnosis.hypotheses}
    scale, shift = by_cause[RootCause.SCALING_ERROR], by_cause[RootCause.NUMERICAL_SHIFT]
    assert scale.confidence > 0.8 > 0.3 > shift.confidence
    pattern_ids = {e.evidence_id for e in evidence if e.data.get("check") == "feature_pattern"}
    assert pattern_ids & set(scale.supporting_evidence_ids)
    assert pattern_ids & set(shift.contradicting_evidence_ids)


def test_concept_drift_is_discounted_when_inputs_also_changed():
    perf = [ev("p1", "f1_drop"), ev("p2", "roc_auc_drop"), ev("a1", "adwin")]
    normal_inputs = [
        ev("n1", "no_covariate_drift", "normal"),
        ev("n2", "no_quality_violation", "normal"),
    ]
    alone = rank_hypotheses(perf + normal_inputs)
    assert alone.top.root_cause is RootCause.CONCEPT_DRIFT and alone.top.confidence >= 0.8
    assert {"n1", "n2"} <= set(alone.top.supporting_evidence_ids)

    drift = ev("d1", "categorical_drift", feature="occupation", observed=0.3)
    confounded = rank_hypotheses([*perf, drift])
    concept = next(h for h in confounded.hypotheses if h.root_cause is RootCause.CONCEPT_DRIFT)
    assert confounded.top.root_cause is RootCause.CATEGORICAL_SHIFT
    assert concept.confidence < MIN_CONFIDENCE and "d1" in concept.contradicting_evidence_ids


def test_wrong_model_explains_away_performance_loss():
    evidence = [
        ev("i1", "artifact_hash", matches_registered_version="v0"),
        ev("p1", "f1_drop"),
        ev("a1", "adwin"),
        ev("s1", "prediction_distribution"),
    ]
    diagnosis = rank_hypotheses(evidence)
    assert diagnosis.top.root_cause is RootCause.ARTIFACT_MISMATCH
    concept = next(h for h in diagnosis.hypotheses if h.root_cause is RootCause.CONCEPT_DRIFT)
    assert concept.confidence < 0.3 and "i1" in concept.contradicting_evidence_ids


def test_uninspected_drift_asks_for_the_missing_diagnostic():
    diagnosis = rank_hypotheses([ev("d1", "numeric_drift", feature="age", observed=0.4)])
    top = diagnosis.top
    assert top.root_cause is RootCause.NUMERICAL_SHIFT and top.confidence < 0.6
    assert any("additive-vs-multiplicative" in m for m in top.missing_evidence)
    assert top.next_diagnostic_action == "inspect_feature_distribution"


def test_insufficient_evidence_is_a_valid_answer():
    empty = rank_hypotheses([])
    assert empty.hypotheses == [] and not empty.sufficient_evidence
    assert empty.conclusion == INSUFFICIENT_EVIDENCE

    weak = rank_hypotheses(
        [
            ev("r1", "label_association_reversal", feature="age"),
            ev("d1", "numeric_drift", feature="age"),
            ev("p1", "feature_pattern", feature="age", pattern="shape_change"),
            ev("q1", "missing_rate", "normal"),
        ]
    )
    only_ambiguous = rank_hypotheses([ev("s1", "prediction_distribution")])
    assert only_ambiguous.conclusion == INSUFFICIENT_EVIDENCE
    assert weak.sufficient_evidence  # shape change still supports a (weak) shift hypothesis
    assert weak.top.confidence <= 0.6


def test_near_tie_is_called_out_in_the_conclusion():
    diagnosis = rank_hypotheses(
        [
            ev("d1", "numeric_drift", feature="age"),
            ev("p1", "feature_pattern", feature="age", pattern="ambiguous"),
        ]
    )
    assert {h.root_cause for h in diagnosis.hypotheses[:2]} == {
        RootCause.SCALING_ERROR,
        RootCause.NUMERICAL_SHIFT,
    }
    assert "not ruled out" in diagnosis.conclusion


def test_inspection_budget_prefers_strongest_drift():
    evidence = [
        ev(f"d{i}", "numeric_drift", feature=f"f{i}", observed=psi)
        for i, psi in enumerate([0.1, 0.9, 0.3, 0.5, 0.2])
    ]
    assert features_worth_inspecting(evidence) == ["f1", "f3", "f2"]


def test_diagnosis_round_trips_through_json(ctx):
    _, _, _, diagnosis = _diagnose(ctx, "F04")
    assert Diagnosis.model_validate_json(diagnosis.model_dump_json()) == diagnosis


# --------------------------------------------------------------------- naive
def test_naive_baseline_confuses_scaling_with_shift_but_rules_do_not(ctx):
    for seed in range(20):
        scenario = ctx.scenario("F04", "medium", seed=seed)
        if scenario.truth.params["factor"] < 1:  # scaled *down*: stays inside the reference range
            break
    _, result, _, rules = _diagnose(ctx, "F04", "medium", seed=seed)
    naive = naive_diagnosis(result.report)
    assert naive.engine == "monitoring-only" and naive.top.root_cause is not RootCause.SCALING_ERROR
    assert rules.top.root_cause is RootCause.SCALING_ERROR


def test_naive_baseline_on_clean_data_has_nothing_to_say(ctx):
    report = ctx.monitor(ctx.clean().current).report
    assert naive_diagnosis(report).conclusion == INSUFFICIENT_EVIDENCE


def test_diagnosis_does_not_mutate_inputs(ctx):
    scenario, result = ctx.run("F01", "high", seed=4)
    before = result.current.copy()
    inp = MonitoringInput(
        ctx.spec, ctx.contract, ctx.reference, scenario.current, ctx.registry, ctx.record.version
    )
    deterministic_diagnosis(inp, result)
    pd.testing.assert_frame_equal(result.current, before)


def test_scale_fit_on_mostly_zero_feature_uses_nonzero_values():
    """92% zeros: every central quantile is 0, so the factor must come from the non-zero tail."""
    rng = np.random.default_rng(3)
    ref = np.where(rng.random(8000) < 0.92, 0.0, rng.gamma(2, 4000, 8000))
    cur = np.where(rng.random(2000) < 0.92, 0.0, rng.gamma(2, 4000, 2000))
    scaled = _fit(ref, cur * 100)
    assert scaled["pattern"] == "scale" and scaled["scale_factor"] == pytest.approx(100, rel=0.25)
    assert _fit(ref, cur * 0.01)["pattern"] == "scale"
    assert _fit(ref, cur + 3000)["pattern"] == "shift"
