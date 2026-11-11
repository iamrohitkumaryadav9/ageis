"""MLflow experiment tracking, isolated behind one function so that a missing or
unreachable tracking server degrades to a logged warning instead of a failed run."""

from __future__ import annotations

import logging
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
    """Log one run. Returns the MLflow run id, or ``None`` if tracking is off/unavailable."""
    if not settings.mlflow_enabled:
        return None
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    try:
        import mlflow

        uri = settings.mlflow_tracking_uri
        if uri.startswith("sqlite:///"):
            Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        flat: dict[str, float] = {}
        _flatten("", metrics, flat)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            mlflow.log_metrics(flat)
            mlflow.set_tags(tags or {})
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
    except Exception as exc:  # tracking must never break training or the benchmark
        log.warning("mlflow tracking unavailable: %s", exc)
        return None
