import numpy as np
import pandas as pd
import pytest

from aegis.faults.injectors import FAULTS, SEVERITIES
from aegis.faults.scenarios import (
    CLEAN,
    GroundTruth,
    ScenarioSpec,
    difficulty_of,
    scenario_grid,
    scenario_id,
)
from aegis.ingestion.datasets import ADULT, TARGET
from aegis.schemas.diagnosis import RootCause


def test_ten_fault_categories_cover_ten_distinct_root_causes():
    assert sorted(FAULTS) == [f"F{i:02d}" for i in range(1, 11)]
    assert {f.root_cause for f in FAULTS.values()} == set(RootCause)


@pytest.mark.parametrize("fault_id", sorted(FAULTS))
def test_scenarios_are_deterministic(ctx, fault_id):
    a = ctx.scenario(fault_id, "medium", seed=7, n_rows=500)
    b = ctx.scenario(fault_id, "medium", seed=7, n_rows=500)
    c = ctx.scenario(fault_id, "medium", seed=8, n_rows=500)
    pd.testing.assert_frame_equal(a.current, b.current)
    assert a.truth == b.truth and a.spec == b.spec
    assert c.spec.scenario_id != a.spec.scenario_id


def test_public_spec_reveals_nothing_about_the_fault(ctx):
    scenario = ctx.scenario("F04", "high", seed=3)
    public = scenario.spec.model_dump_json()
    for secret in ("F04", "scaling", "high", scenario.truth.affected_features[0]):
        assert secret not in public
    assert set(ScenarioSpec.model_fields) & set(GroundTruth.model_fields) == {"scenario_id"}
    # The id is a hash: same inputs give the same id, and it embeds no readable label.
    assert scenario_id("adult", "F04", "high", 3, 2000) == scenario.spec.scenario_id


def test_clean_control_is_an_untouched_production_sample(ctx):
    clean = ctx.clean(seed=5, n_rows=800)
    assert clean.truth.fault_id == CLEAN and clean.truth.root_cause is None
    merged = clean.current[ADULT.features].merge(ctx.production[ADULT.features].drop_duplicates())
    assert len(merged) >= len(clean.current.drop_duplicates(subset=ADULT.features))


def _clean_and(ctx, fault_id, severity, seed=11, n_rows=2000):
    faulty = ctx.scenario(fault_id, severity, seed, n_rows)
    # Rebuild the identical pre-fault sample: same RNG stream as build_scenario.
    from aegis.faults.scenarios import _seed_int

    rng = np.random.default_rng(_seed_int("adult", fault_id, severity, seed, n_rows))
    rows = rng.choice(len(ctx.production), size=n_rows, replace=False)
    return ctx.production.iloc[rows].reset_index(drop=True), faulty


@pytest.mark.parametrize("severity,k", [("low", 0.35), ("medium", 0.7), ("high", 1.5)])
def test_f01_shifts_one_feature_by_k_standard_deviations(ctx, severity, k):
    base, faulty = _clean_and(ctx, "F01", severity)
    (feature,) = faulty.truth.affected_features
    delta = (faulty.current[feature] - base[feature]).abs().mean()
    assert delta == pytest.approx(k * base[feature].std(), rel=0.05, abs=0.5)
    untouched = [f for f in ADULT.features if f != feature]
    pd.testing.assert_frame_equal(faulty.current[untouched], base[untouched])


def test_f02_moves_mass_to_one_category(ctx):
    base, faulty = _clean_and(ctx, "F02", "high")
    (feature,) = faulty.truth.affected_features
    target = faulty.truth.params["to_category"]
    gain = (faulty.current[feature] == target).mean() - (base[feature] == target).mean()
    assert gain > 0.3


def test_f03_blanks_requested_fraction(ctx):
    base, faulty = _clean_and(ctx, "F03", "medium")
    (feature,) = faulty.truth.affected_features
    added = faulty.current[feature].isna().mean() - base[feature].isna().mean()
    assert added == pytest.approx(0.30 * (1 - base[feature].isna().mean()), abs=0.04)


def test_f04_is_a_pure_rescaling(ctx):
    base, faulty = _clean_and(ctx, "F04", "high")
    (feature,) = faulty.truth.affected_features
    np.testing.assert_allclose(
        faulty.current[feature], base[feature] * faulty.truth.params["factor"]
    )


def test_f05_flips_labels_but_not_features(ctx):
    base, faulty = _clean_and(ctx, "F05", "medium")
    pd.testing.assert_frame_equal(faulty.current[ADULT.features], base[ADULT.features])
    assert (faulty.current[TARGET] != base[TARGET]).mean() == pytest.approx(0.55, abs=0.04)


def test_f06_adds_exact_duplicates_without_changing_batch_size(ctx):
    base, faulty = _clean_and(ctx, "F06", "medium")
    assert len(faulty.current) == len(base)
    assert faulty.current.duplicated().mean() - base.duplicated().mean() > 0.15


def test_f07_preserves_the_marginal_but_encodes_the_label(ctx):
    from sklearn.metrics import roc_auc_score

    for seed in range(11, 30):  # find a draw that targets a feature with few ties
        base, faulty = _clean_and(ctx, "F07", "high", seed=seed)
        if faulty.truth.affected_features[0] in ("age", "hours_per_week"):
            break
    (feature,) = faulty.truth.affected_features
    assert sorted(faulty.current[feature]) == sorted(base[feature])  # identical marginal
    assert roc_auc_score(faulty.current[TARGET], faulty.current[feature]) > 0.95
    assert roc_auc_score(base[TARGET], base[feature]) < 0.75


@pytest.mark.parametrize(
    "severity,change",
    [("low", "retyped_numeric_to_text"), ("medium", "renamed"), ("high", "dropped")],
)
def test_f08_breaks_the_contract_in_three_ways(ctx, severity, change):
    faulty = ctx.scenario("F08", severity, seed=2)
    assert faulty.truth.params["change"] == change
    for feature in faulty.truth.affected_features:
        column = faulty.current.get(feature)
        assert column is None or not pd.api.types.is_numeric_dtype(column)


def test_f09_removes_most_recent_labels_only(ctx):
    base, faulty = _clean_and(ctx, "F09", "medium")
    missing = faulty.current[TARGET].isna()
    assert missing.mean() == pytest.approx(0.8)
    assert not missing.iloc[: int(len(base) * 0.2) - 1].any() and missing.iloc[-1]
    pd.testing.assert_frame_equal(faulty.current[ADULT.features], base[ADULT.features])


@pytest.mark.parametrize("severity", SEVERITIES)
def test_f10_swaps_the_artifact_and_leaves_data_alone(ctx, severity):
    base, faulty = _clean_and(ctx, "F10", severity)
    pd.testing.assert_frame_equal(faulty.current[ADULT.features], base[ADULT.features])
    ok, _ = ctx.registry.verify("adult", ctx.record.version, faulty.artifact_path)
    assert faulty.artifact_path is not None and not ok


def test_grid_is_balanced_and_includes_controls():
    grid = scenario_grid(["adult", "bank"], seeds=[1, 2], clean_per_dataset=10)
    faults = [g for g in grid if g["fault_id"] != CLEAN]
    assert len(faults) == 2 * 10 * 3 * 2 and len(grid) - len(faults) == 20
    assert {g["n_rows"] for g in grid} == {1000, 3000}
    assert difficulty_of("high", 3000) == "easy" and difficulty_of("low", 1000) == "hard"


def test_unknown_fault_is_rejected(ctx):
    with pytest.raises(ValueError):
        ctx.scenario("F99")
