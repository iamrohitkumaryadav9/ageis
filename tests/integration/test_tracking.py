"""MLflow tracking (`aegis.observability.tracking`) against real local MLflow backends.

No mocks of MLflow itself: runs are written to a SQLite store or to a local `mlflow server`
process and read back through MLflow's own client, which is what "tracking works" means.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from mlflow.tracking import MlflowClient

from aegis.config import Settings
from aegis.observability import tracking
from aegis.observability.tracking import log_run


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Settings pointing at a private SQLite store with a private artifact directory."""
    monkeypatch.chdir(tmp_path)  # default artifact root ./mlruns lands in the temp directory
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")
    uri = f"sqlite:///{tmp_path / 'nested' / 'mlflow.db'}"
    return Settings(mlflow_tracking_uri=uri, mlflow_enabled=True), MlflowClient(tracking_uri=uri)


@pytest.fixture
def warnings(monkeypatch):
    """Warnings emitted by the tracking module, independent of global logging configuration."""
    seen: list[str] = []
    monkeypatch.setattr(
        tracking.log, "warning", lambda message, *args, **kw: seen.append(message % args)
    )
    return seen


def one_run(client: MlflowClient, experiment: str):
    exp = client.get_experiment_by_name(experiment)
    assert exp is not None, f"experiment {experiment!r} was not created"
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    return runs[0]


def test_run_is_created_logged_and_terminated(store, tmp_path):
    settings, client = store
    artifact = tmp_path / "report.txt"
    artifact.write_text("confusion matrix goes here")
    run_id = log_run(
        settings,
        experiment="aegis-test",
        run_name="baseline",
        params={"model": "hgb", "max_depth": 6, "features": ["a", "b"]},
        metrics={"auc": 0.91, "test": {"f1": 0.7, "n": 100}},
        tags={"dataset": "adult"},
        artifacts=[artifact],
    )
    assert run_id
    run = client.get_run(run_id)  # visible through the configured backend
    assert run.info.status == "FINISHED" and run.info.end_time is not None
    assert run.info.run_name == "baseline"
    assert run.data.params == {"model": "hgb", "max_depth": "6", "features": "['a', 'b']"}
    assert run.data.metrics == {"auc": 0.91, "test.f1": 0.7, "test.n": 100.0}
    assert run.data.tags["dataset"] == "adult"
    assert [a.path for a in client.list_artifacts(run_id)] == ["report.txt"]
    local = client.download_artifacts(run_id, "report.txt", str(tmp_path / "dl"))
    assert Path(local).read_text() == "confusion matrix goes here"
    assert one_run(client, "aegis-test").info.run_id == run_id


def test_store_directory_is_created_from_the_configured_uri(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'deep' / 'er' / 'mlflow.db'}"
    assert not (tmp_path / "deep").exists()
    settings = Settings(mlflow_tracking_uri=uri, mlflow_enabled=True)
    assert log_run(settings, experiment="e", run_name="r", params={}, metrics={})
    assert (tmp_path / "deep" / "er" / "mlflow.db").exists()


def test_repeated_logging_creates_separate_runs_in_one_experiment(store):
    settings, client = store
    ids = [
        log_run(settings, experiment="repeat", run_name=f"r{i}", params={"i": i}, metrics={"m": i})
        for i in range(3)
    ]
    assert len(set(ids)) == 3 and all(ids)
    exp = client.get_experiment_by_name("repeat")
    runs = client.search_runs([exp.experiment_id])
    assert sorted(r.data.params["i"] for r in runs) == ["0", "1", "2"]
    assert {r.info.status for r in runs} == {"FINISHED"}
    assert len(client.search_experiments(filter_string="name = 'repeat'")) == 1


def test_sklearn_model_is_logged_and_loadable(store):
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    settings, client = store
    model = LogisticRegression().fit(np.array([[0.0], [1.0], [2.0], [3.0]]), [0, 0, 1, 1])
    run_id = log_run(
        settings, experiment="models", run_name="lr", params={}, metrics={}, sklearn_model=model
    )
    assert run_id and client.get_run(run_id).info.status == "FINISHED"
    logged = client.search_logged_models(
        experiment_ids=[client.get_experiment_by_name("models").experiment_id]
    )
    assert len(logged) == 1 and logged[0].source_run_id == run_id


def test_disabled_tracking_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'off.db'}"
    settings = Settings(mlflow_tracking_uri=uri, mlflow_enabled=False)
    assert (
        log_run(settings, experiment="e", run_name="r", params={"a": 1}, metrics={"m": 1}) is None
    )
    assert not (tmp_path / "off.db").exists()


def test_failure_inside_a_run_marks_it_failed_and_does_not_raise(store, tmp_path, warnings):
    settings, client = store
    result = log_run(
        settings,
        experiment="failing",
        run_name="broken",
        params={"p": 1},
        metrics={"m": 1.0},
        artifacts=[tmp_path / "does-not-exist.txt"],
    )
    assert result is None
    assert len(warnings) == 1 and "mlflow tracking failed" in warnings[0]
    assert "experiment=failing" in warnings[0] and "run=broken" in warnings[0]
    run = one_run(client, "failing")
    assert run.info.status == "FAILED" and run.info.end_time is not None  # closed, not RUNNING
    assert run.data.params == {"p": "1"}  # what was logged before the failure is kept
    # the failed run did not leave an active run behind that would swallow the next one
    assert log_run(settings, experiment="failing", run_name="next", params={}, metrics={})
    exp = client.get_experiment_by_name("failing")
    assert sorted(r.info.status for r in client.search_runs([exp.experiment_id])) == [
        "FAILED",
        "FINISHED",
    ]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_missing_server_degrades_quickly_to_none(monkeypatch, tmp_path, warnings):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")
    settings = Settings(mlflow_tracking_uri=f"http://127.0.0.1:{free_port()}", mlflow_enabled=True)
    started = time.perf_counter()
    assert log_run(settings, experiment="e", run_name="r", params={}, metrics={"m": 1}) is None
    assert time.perf_counter() - started < 30
    assert len(warnings) == 1 and "mlflow tracking failed" in warnings[0]


def test_retry_and_timeout_defaults_are_bounded(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", raising=False)
    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_TIMEOUT", raising=False)
    settings = Settings(mlflow_tracking_uri=f"sqlite:///{tmp_path / 'd.db'}", mlflow_enabled=True)
    log_run(settings, experiment="e", run_name="r", params={}, metrics={})
    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "1"
    assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "10"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"experiment": "", "run_name": "r", "params": {}, "metrics": {}},
        {"experiment": "   ", "run_name": "r", "params": {}, "metrics": {}},
        {"experiment": "e", "run_name": "", "params": {}, "metrics": {}},
        {"experiment": "e", "run_name": "r", "params": ["not", "a", "dict"], "metrics": {}},
        {"experiment": "e", "run_name": "r", "params": {}, "metrics": 0.5},
        {"experiment": None, "run_name": "r", "params": {}, "metrics": {}},
    ],
)
def test_caller_mistakes_raise_instead_of_being_swallowed(store, kwargs):
    settings, _ = store
    with pytest.raises(ValueError):
        log_run(settings, **kwargs)


def test_unloggable_values_are_handled_explicitly(store):
    settings, client = store
    run_id = log_run(
        settings,
        experiment="odd",
        run_name="values",
        params={"long": "x" * 5000, "none": None, 7: "int key"},
        metrics={
            "ok": 1.0,
            "nan": float("nan"),
            "inf": float("inf"),
            "flag": True,
            "text": "ignored",
            "nested": {"none": None, "v": 2},
        },
    )
    run = client.get_run(run_id)
    assert len(run.data.params["long"]) == 250 and run.data.params["none"] == "None"
    assert run.data.params["7"] == "int key"
    assert run.data.metrics == {"ok": 1.0, "nested.v": 2.0}  # booleans, text, None are not metrics
    assert run.data.tags["aegis.dropped_metrics"] == "inf,nan"  # dropped, and said so


def test_flatten_ignores_booleans_and_non_numbers():
    out: dict[str, float] = {}
    tracking._flatten("", {"a": 1, "b": {"c": 2.5, "d": True, "e": "x", "f": [1, 2]}}, out)
    assert out == {"a": 1.0, "b.c": 2.5}


def test_tracking_uri_comes_from_configuration(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    first = Settings(mlflow_tracking_uri=f"sqlite:///{tmp_path / 'one.db'}", mlflow_enabled=True)
    second = Settings(mlflow_tracking_uri=f"sqlite:///{tmp_path / 'two.db'}", mlflow_enabled=True)
    a = log_run(first, experiment="e", run_name="r", params={}, metrics={})
    b = log_run(second, experiment="e", run_name="r", params={}, metrics={})
    assert MlflowClient(first.mlflow_tracking_uri).get_run(a).info.run_id == a
    assert MlflowClient(second.mlflow_tracking_uri).get_run(b).info.run_id == b
    with pytest.raises(Exception, match=r"not found|RESOURCE_DOES_NOT_EXIST"):
        MlflowClient(first.mlflow_tracking_uri).get_run(b)
    monkeypatch.setenv("AEGIS_MLFLOW_TRACKING_URI", "sqlite:////somewhere/else.db")
    assert Settings().mlflow_tracking_uri == "sqlite:////somewhere/else.db"


@pytest.fixture
def mlflow_server(tmp_path):
    """A real `mlflow server` process on a free local port."""
    port = free_port()
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--backend-store-uri",
            f"sqlite:///{tmp_path / 'server.db'}",
            "--default-artifact-root",
            str(tmp_path / "artifacts"),
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 90
        while time.time() < deadline:
            if process.poll() is not None:
                output = (
                    process.stdout.read().decode(errors="replace")[-800:] if process.stdout else ""
                )
                pytest.fail(f"mlflow server exited early: {output}")
            try:
                if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        else:
            pytest.fail("mlflow server did not become healthy within 90 s")
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()


def test_runs_logged_over_http_are_visible_on_the_tracking_server(
    mlflow_server, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    settings = Settings(mlflow_tracking_uri=mlflow_server, mlflow_enabled=True)
    run_id = log_run(
        settings,
        experiment="served",
        run_name="over-http",
        params={"k": "v"},
        metrics={"auc": 0.9},
        tags={"source": "test"},
    )
    assert run_id
    # read back through the server's REST API, independently of the MLflow client
    response = httpx.get(
        f"{mlflow_server}/api/2.0/mlflow/runs/get", params={"run_id": run_id}, timeout=10
    )
    assert response.status_code == 200
    run = response.json()["run"]
    assert run["info"]["status"] == "FINISHED"
    assert {"key": "k", "value": "v"} in run["data"]["params"]
    assert any(m["key"] == "auc" and m["value"] == 0.9 for m in run["data"]["metrics"])
    found = httpx.post(
        f"{mlflow_server}/api/2.0/mlflow/experiments/search", json={"max_results": 10}, timeout=10
    ).json()
    assert "served" in [e["name"] for e in found["experiments"]]
