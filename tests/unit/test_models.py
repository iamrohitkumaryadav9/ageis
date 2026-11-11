import numpy as np
import pytest

from aegis.ingestion.datasets import ADULT, TARGET
from aegis.models.metrics import (
    best_f1_threshold,
    classification_metrics,
    expected_calibration_error,
)
from aegis.models.registry import ArtifactIntegrityError, ModelRegistry
from aegis.models.training import predict_proba, select, train_baseline


def test_classification_metrics_match_hand_computation():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.2, 0.3])
    m = classification_metrics(y, p, threshold=0.5)
    assert m["confusion_matrix"] == {"tn": 4, "fp": 1, "fn": 1, "tp": 2}
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(6 / 8)
    # 15 pos/neg pairs: 12 concordant + 1 tie (0.2 vs 0.2) counted as half.
    assert m["roc_auc"] == pytest.approx(12.5 / 15)


def test_roc_auc_is_none_when_one_class_present():
    m = classification_metrics(np.zeros(10), np.linspace(0, 1, 10), 0.5)
    assert m["roc_auc"] is None


def test_ece_is_zero_for_perfectly_calibrated_bins():
    rng = np.random.default_rng(0)
    proba = np.repeat([0.25, 0.75], 20000)
    y = (rng.random(proba.size) < proba).astype(int)
    assert expected_calibration_error(y, proba) < 0.01
    assert expected_calibration_error(1 - y, proba) > 0.4


def test_best_f1_threshold_finds_separating_point():
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([np.full(50, 0.3), np.full(50, 0.6)])
    assert 0.3 < best_f1_threshold(y, p) <= 0.6


def test_selection_prefers_simpler_model_within_margin():
    cv = {
        "logistic_regression": {"roc_auc_mean": 0.900},
        "hist_gradient_boosting": {"roc_auc_mean": 0.903},
    }
    assert select(cv)[0] == "logistic_regression"
    cv["hist_gradient_boosting"]["roc_auc_mean"] = 0.92
    assert select(cv)[0] == "hist_gradient_boosting"


def test_baseline_model_is_clearly_better_than_chance(adult_model, adult_splits, registry):
    model = registry.load("adult", adult_model.version)
    test = adult_splits["test"]
    m = classification_metrics(
        test[TARGET].to_numpy(), predict_proba(model, test, ADULT.features), adult_model.threshold
    )
    assert m["roc_auc"] > 0.88
    assert m["f1"] > 0.6
    # Accuracy alone is misleading here: predicting the majority class already gets ~0.76.
    assert m["accuracy"] > 1 - test[TARGET].mean() + 0.05


def test_pipeline_tolerates_unseen_categories_and_missing_values(
    adult_model, adult_splits, registry
):
    model = registry.load("adult", adult_model.version)
    rows = adult_splits["test"].head(20).copy()
    rows["occupation"] = "Quantum-plumber"
    rows.loc[rows.index[:5], "age"] = np.nan
    proba = predict_proba(model, rows, ADULT.features)
    assert np.isfinite(proba).all() and ((proba >= 0) & (proba <= 1)).all()


def test_registry_detects_tampered_artifact(adult_model, registry, tmp_path):
    ok, sha = registry.verify("adult", adult_model.version)
    assert ok and sha == adult_model.artifact_sha256
    tampered = tmp_path / "model.joblib"
    tampered.write_bytes(registry.artifact_path("adult", adult_model.version).read_bytes() + b"x")
    ok, sha = registry.verify("adult", adult_model.version, tampered)
    assert not ok and sha != adult_model.artifact_sha256


def test_registry_refuses_to_load_mismatching_artifact(adult_splits, adult_contract, tmp_path):
    from tests.conftest import _register

    reg = ModelRegistry(tmp_path)
    record = _register(ADULT, adult_splits, adult_contract, reg, "v1")
    path = reg.artifact_path("adult", "v1")
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(ArtifactIntegrityError):
        reg.load("adult", record.version)


def test_registry_rejects_path_traversal_and_unknown_versions(registry):
    with pytest.raises(ValueError):
        registry.artifact_path("adult", "../../etc/passwd")
    with pytest.raises(KeyError):
        registry.get("adult", "does-not-exist")


def test_train_baseline_registers_both_candidates_and_is_deterministic(
    adult_splits, adult_contract, tmp_path
):
    small = {
        "train": adult_splits["train"].head(3000),
        "validation": adult_splits["validation"].head(1500),
        "test": adult_splits["test"].head(1500),
    }
    reg = ModelRegistry(tmp_path)
    first = train_baseline(ADULT, small, adult_contract, reg, seed=5)
    second = train_baseline(ADULT, small, adult_contract, reg, seed=5)
    assert set(first.records) == {"logistic_regression", "hist_gradient_boosting"}
    assert first.selected.version == second.selected.version == reg.baseline("adult").version
    for name in first.records:
        assert (
            first.records[name].metrics["test"]["f1"] == second.records[name].metrics["test"]["f1"]
        )
        assert first.records[name].metrics["latency"]["single_row_p95_ms"] > 0
