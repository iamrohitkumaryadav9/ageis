"""Shared fixtures.

Tests run against the real public datasets (downloaded and checksum-verified
on first use) with a small, quickly trained model, so the behaviour under test
is the behaviour that ships rather than the behaviour of a mock.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from aegis.config import Settings, get_settings
from aegis.ingestion.contracts import DataContract, derive_contract
from aegis.ingestion.datasets import ADULT, BANK, TARGET, DatasetSpec, download, load_raw
from aegis.ingestion.splits import make_splits
from aegis.models.metrics import best_f1_threshold
from aegis.models.registry import ModelRecord, ModelRegistry
from aegis.models.training import build_pipeline, predict_proba

SEED = 1234


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


def _splits(spec: DatasetSpec, settings: Settings) -> dict[str, pd.DataFrame]:
    download(spec, settings.raw_dir)
    return make_splits(load_raw(spec, settings.raw_dir), SEED)


@pytest.fixture(scope="session")
def adult_splits(settings: Settings) -> dict[str, pd.DataFrame]:
    return _splits(ADULT, settings)


@pytest.fixture(scope="session")
def bank_splits(settings: Settings) -> dict[str, pd.DataFrame]:
    return _splits(BANK, settings)


@pytest.fixture(scope="session")
def adult_contract(adult_splits) -> DataContract:
    return derive_contract(ADULT, adult_splits["validation"])


@pytest.fixture(scope="session")
def registry(tmp_path_factory) -> ModelRegistry:
    return ModelRegistry(Path(tmp_path_factory.mktemp("models")))


def _register(spec, splits, contract, registry, version, **lr_kwargs) -> ModelRecord:
    pipeline = build_pipeline(spec, LogisticRegression(max_iter=500, **lr_kwargs))
    train = splits["train"]
    pipeline.fit(train[spec.features], train[TARGET])
    val = splits["validation"]
    threshold = best_f1_threshold(
        val[TARGET].to_numpy(), predict_proba(pipeline, val, spec.features)
    )
    return registry.register(
        {
            "dataset": spec.name,
            "version": version,
            "algorithm": "logistic_regression",
            "threshold": threshold,
            "features": spec.features,
            "contract_version": contract.version,
            "train_fingerprint": "test",
            "seed": SEED,
        },
        pipeline,
    )


@pytest.fixture(scope="session")
def adult_model(adult_splits, adult_contract, registry) -> ModelRecord:
    record = _register(ADULT, adult_splits, adult_contract, registry, "adult-test-v1")
    registry.set_baseline("adult", record.version)
    return record


@pytest.fixture(scope="session")
def adult_model_v0(adult_splits, adult_contract, registry) -> ModelRecord:
    """An older, weaker (heavily regularised) version, for version comparison and rollback."""
    return _register(ADULT, adult_splits, adult_contract, registry, "adult-test-v0", C=0.001)
