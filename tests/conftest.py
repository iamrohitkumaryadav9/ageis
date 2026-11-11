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


class MonitorContext:
    """Bundles everything needed to build scenarios and monitor them for one dataset."""

    def __init__(self, spec, splits, contract, registry, record, workdir):
        self.spec, self.contract, self.registry, self.record = spec, contract, registry, record
        self.reference = splits["validation"]
        self.production = splits["production"]
        self.workdir = workdir

    def scenario(self, fault_id, severity="high", seed=1, n_rows=2000):
        from aegis.faults.scenarios import build_scenario

        return build_scenario(
            self.spec,
            self.production,
            self.registry,
            self.record.version,
            fault_id=fault_id,
            severity=severity,
            seed=seed,
            n_rows=n_rows,
            workdir=self.workdir,
        )

    def clean(self, seed=1, n_rows=2000):
        return self.scenario("CLEAN", "none", seed, n_rows)

    def monitor(self, current, artifact_path=None, ingestion=None, cfg=None):
        from aegis.monitoring.engine import MonitoringInput, run_monitoring
        from aegis.schemas.deployment import IngestionConfig

        return run_monitoring(
            MonitoringInput(
                self.spec,
                self.contract,
                self.reference,
                current,
                self.registry,
                self.record.version,
                artifact_path,
                ingestion or IngestionConfig(),
            ),
            cfg,
        )

    def run(self, fault_id, severity="high", seed=1, n_rows=2000):
        scenario = self.scenario(fault_id, severity, seed, n_rows)
        return scenario, self.monitor(scenario.current, scenario.artifact_path)


@pytest.fixture(scope="session")
def ctx(adult_splits, adult_contract, registry, adult_model, adult_model_v0, tmp_path_factory):
    return MonitorContext(
        ADULT, adult_splits, adult_contract, registry, adult_model, tmp_path_factory.mktemp("scn")
    )


def triggered(result, check=None, monitor=None, feature=None):
    """Triggered findings of a monitoring result, optionally filtered."""
    return [
        f
        for f in result.report.triggered
        if (check is None or f.check == check)
        and (monitor is None or f.monitor == monitor)
        and (feature is None or f.feature == feature)
    ]


# --------------------------------------------------------------------------- database / API
TEST_DB_URL_ENV = "AEGIS_TEST_DATABASE_URL"
DEFAULT_TEST_DB = "postgresql+psycopg://aegis:aegis@localhost:55432/aegis_test"
TOKENS = {
    "approver": "test-token-approver-0001",
    "approver2": "test-token-approver-0002",
    "operator": "test-token-operator-0001",
    "viewer": "test-token-viewer-000001",
}
API_TOKENS = (
    f"{TOKENS['approver']}:alice:approver,{TOKENS['approver2']}:carol:approver,"
    f"{TOKENS['operator']}:bob:operator,{TOKENS['viewer']}:vera:viewer"
)


@pytest.fixture(scope="session")
def pg_url() -> str:
    import os

    from sqlalchemy import create_engine, text

    url = os.environ.get(TEST_DB_URL_ENV, DEFAULT_TEST_DB)
    try:
        engine = create_engine(url)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        engine.dispose()
    except Exception as exc:
        pytest.fail(
            f"PostgreSQL is required for these tests but {url} is unreachable ({type(exc).__name__}). "
            f"Start it with `make test-db-up` or set {TEST_DB_URL_ENV}.",
            pytrace=False,
        )
    return url


@pytest.fixture(scope="session")
def migrated_db(pg_url):
    """Build the schema once with the real Alembic migration (so migrations are tested too)."""
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, text

    engine = create_engine(pg_url)
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    engine.dispose()
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", pg_url)
    command.upgrade(config, "head")
    return pg_url


@pytest.fixture
def db(migrated_db):
    from sqlalchemy import text

    from aegis.storage.db import Database
    from aegis.storage.models import Base

    database = Database(migrated_db)
    with database.engine.begin() as connection:
        tables = ", ".join(t.name for t in Base.metadata.sorted_tables)
        connection.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    yield database
    database.dispose()


@pytest.fixture
def service(db, ctx, settings, tmp_path):
    """An AegisService on real PostgreSQL, wired to the test registry, seeded and LLM-free."""
    from aegis.schemas.deployment import DeploymentState
    from aegis.services.core import AegisService
    from aegis.services.errors import SYSTEM

    test_settings = settings.model_copy(
        update={
            "artifacts_dir": tmp_path / "artifacts",
            "llm_enabled": False,
            "api_tokens": API_TOKENS,
            "database_url": db.engine.url.render_as_string(hide_password=False),
            "mlflow_enabled": False,
        }
    )
    svc = AegisService(test_settings, db)
    svc.registry = ctx.registry
    svc.register_batch(SYSTEM, "adult", "reference", ctx.reference, source="test")
    svc.set_deployment(
        SYSTEM,
        "adult",
        DeploymentState(dataset="adult", model_version=ctx.record.version),
        "test seed",
    )
    yield svc
    svc.shutdown()


@pytest.fixture
def api(service):
    from fastapi.testclient import TestClient

    from apps.api.main import create_app

    with TestClient(create_app(service.settings, service), raise_server_exceptions=False) as client:
        yield client


def auth(role: str = "operator") -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKENS[role]}"}


@pytest.fixture
def principals():
    from aegis.services.errors import Principal

    return {
        "approver": Principal("alice", "approver"),
        "approver2": Principal("carol", "approver"),
        "operator": Principal("bob", "operator"),
        "viewer": Principal("vera", "viewer"),
    }


def open_incident(service, ctx, principals, fault="F04", severity="high", seed=4, investigate=True):
    """Ingest a faulty batch, monitor it, optionally investigate. Returns (incident, investigation)."""
    scenario = ctx.scenario(fault, severity, seed)
    if scenario.artifact_path is not None:  # F10: simulate a bad deploy of the serving artifact
        import shutil

        from aegis.schemas.deployment import DeploymentState

        target = service.settings.artifacts_dir / "serving" / scenario.artifact_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(scenario.artifact_path, target)
        service.set_deployment(
            principals["operator"],
            "adult",
            DeploymentState(
                dataset="adult",
                model_version=ctx.record.version,
                serving_artifact=f"serving/{target.name}",
            ),
            "test: bad deploy",
        )
    batch = service.register_batch(
        principals["operator"], "adult", "current", scenario.current, source="test"
    )
    out = service.run_monitoring(principals["operator"], "adult", batch.id)
    investigation = None
    if investigate and out["incident"] is not None:
        investigation = service.start_investigation(
            principals["operator"], out["incident"].id, use_llm=False, wait=True
        )
    return out["incident"], investigation
