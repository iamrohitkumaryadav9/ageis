"""MLflow experiment tracking, isolated behind one function so that a missing or
unreachable tracking server degrades to a logged warning instead of a failed run."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

from aegis.config import Settings

log = logging.getLogger(__name__)

SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


def _flatten(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, inner in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), inner, out)
    elif isinstance(value, bool):
        return
    elif isinstance(value, int | float):
        out[prefix] = float(value)


def log_run(
    settings: Settings,
    *,
    experiment: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    tags: dict[str, str] | None = None,
    artifacts: list[Path] | None = None,
    sklearn_model: Any = None,
    registered_model_name: str | None = None,
) -> str | None:
    """Log one run. Returns the MLflow run id, or ``None`` if tracking is off/unavailable.

    Tracking is an observer: an unreachable server or a rejected value must not fail training or
    a benchmark, so backend failures are logged and ``None`` is returned. Caller mistakes are a
    different matter and raise ``ValueError`` before anything is sent: an empty experiment or
    run name, or parameters/metrics that are not mappings.

    A run that fails part-way is closed with status ``FAILED`` (MLflow's context manager does
    this), never left ``RUNNING``.
    """
    if not isinstance(experiment, str) or not experiment.strip():
        raise ValueError("experiment name must be a non-empty string")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run name must be a non-empty string")
    if not isinstance(params, dict) or not isinstance(metrics, dict):
        raise ValueError("params and metrics must be dictionaries")
    if not settings.mlflow_enabled:
        return None
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    # MLflow retries a dead HTTP server for minutes by default; an observer may not do that.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    try:
        import mlflow

        uri = settings.mlflow_tracking_uri
        if uri.startswith("sqlite:///"):
            Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        flat: dict[str, float] = {}
        _flatten("", metrics, flat)
        finite = {k: v for k, v in flat.items() if math.isfinite(v)}
        dropped = sorted(set(flat) - set(finite))
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({str(k)[:250]: str(v)[:250] for k, v in params.items()})
            mlflow.log_metrics(finite)
            mlflow.set_tags(
                {
                    **(tags or {}),
                    **({"aegis.dropped_metrics": ",".join(dropped)[:250]} if dropped else {}),
                }
            )
            for path in artifacts or []:
                mlflow.log_artifact(str(path))
            if sklearn_model is not None:
                # MLflow serialises with skops, which refuses types it cannot prove safe.
                # These two are produced by our own training code and were reviewed.
                mlflow.sklearn.log_model(
                    sklearn_model,
                    name="model",
                    registered_model_name=registered_model_name,
                    skops_trusted_types=SKOPS_TRUSTED_TYPES,
                )
            return run.info.run_id
    except Exception as exc:
        log.warning(
            "mlflow tracking failed (%s): %s; uri=%s experiment=%s run=%s",
            type(exc).__name__,
            str(exc)[:300],
            settings.mlflow_tracking_uri,
            experiment,
            run_name,
        )
        return None
