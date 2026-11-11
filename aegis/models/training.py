"""Baseline model training with a documented selection protocol.

Protocol
--------
1. Candidates are compared with stratified 5-fold cross-validation on the
   ``train`` split only. Preprocessing lives inside the pipeline, so every fold
   fits its own imputer/scaler/encoder and no statistic leaks across folds.
2. The primary criterion is mean CV ROC-AUC (threshold-free, robust to the
   class imbalance of both datasets). A candidate must beat the simpler model
   by more than ``AUC_MARGIN`` to be preferred; otherwise the simpler, better
   calibrated logistic regression wins. Accuracy is reported but never used
   for selection: a majority-class predictor scores 76-88% on these datasets.
3. The winner is refit on all of ``train``; its decision threshold is chosen
   on ``validation`` to maximise F1.
4. ``test`` is evaluated exactly once, for the reported numbers.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aegis.ingestion.contracts import DataContract
from aegis.ingestion.datasets import TARGET, DatasetSpec
from aegis.ingestion.splits import frame_fingerprint
from aegis.models.metrics import best_f1_threshold, classification_metrics
from aegis.models.registry import ModelRecord, ModelRegistry

AUC_MARGIN = 0.005
CV_FOLDS = 5
SIMPLEST = "logistic_regression"


def candidate_estimators(seed: int) -> dict[str, tuple[Any, dict[str, Any]]]:
    return {
        "logistic_regression": (
            LogisticRegression(C=1.0, max_iter=1000),
            {"C": 1.0, "max_iter": 1000},
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=seed
            ),
            {"max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 31},
        ),
    }


def build_pipeline(spec: DatasetSpec, estimator: Any) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False
                ),
            ),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("num", numeric, spec.numeric_features),
            ("cat", categorical, spec.categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def predict_proba(model: Pipeline, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict_proba(frame[features])[:, 1]


def measure_latency(model: Pipeline, frame: pd.DataFrame, features: list[str]) -> dict[str, float]:
    """Batch throughput and single-row latency percentiles (wall clock, this machine)."""
    x = frame[features]
    start = time.perf_counter()
    model.predict_proba(x)
    batch_s = time.perf_counter() - start
    singles = []
    for i in range(min(200, len(x))):
        row = x.iloc[[i]]
        t0 = time.perf_counter()
        model.predict_proba(row)
        singles.append((time.perf_counter() - t0) * 1000)
    return {
        "batch_rows": float(len(x)),
        "batch_ms": batch_s * 1000,
        "batch_us_per_row": batch_s / len(x) * 1e6,
        "single_row_p50_ms": float(np.percentile(singles, 50)),
        "single_row_p95_ms": float(np.percentile(singles, 95)),
    }


@dataclass
class TrainingResult:
    selected: ModelRecord
    records: dict[str, ModelRecord]
    cv: dict[str, dict[str, float]]
    selection_reason: str


def _version(dataset: str, algorithm: str, params: dict, fingerprint: str, seed: int) -> str:
    digest = hashlib.sha256(
        json.dumps([algorithm, params, fingerprint, seed], sort_keys=True).encode()
    ).hexdigest()[:8]
    short = {"logistic_regression": "lr", "hist_gradient_boosting": "hgb"}.get(algorithm, algorithm)
    return f"{dataset}-{short}-{digest}"


def select(cv: dict[str, dict[str, float]]) -> tuple[str, str]:
    """Apply the documented selection rule to cross-validation summaries."""
    best = max(cv, key=lambda k: cv[k]["roc_auc_mean"])
    if best != SIMPLEST and SIMPLEST in cv:
        gain = cv[best]["roc_auc_mean"] - cv[SIMPLEST]["roc_auc_mean"]
        if gain <= AUC_MARGIN:
            return SIMPLEST, (
                f"{best} leads CV ROC-AUC by only {gain:.4f} (<= margin {AUC_MARGIN}); "
                f"preferring the simpler {SIMPLEST}"
            )
        return best, f"{best} beats {SIMPLEST} on CV ROC-AUC by {gain:.4f} (> margin {AUC_MARGIN})"
    return best, f"{best} has the highest CV ROC-AUC"


def train_baseline(
    spec: DatasetSpec,
    splits: dict[str, pd.DataFrame],
    contract: DataContract,
    registry: ModelRegistry,
    seed: int,
) -> TrainingResult:
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    fingerprint = frame_fingerprint(train)
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    cv: dict[str, dict[str, float]] = {}
    fitted: dict[str, tuple[Pipeline, dict, float]] = {}

    for name, (estimator, params) in candidate_estimators(seed).items():
        scores = cross_validate(
            build_pipeline(spec, estimator),
            train[spec.features],
            train[TARGET],
            cv=folds,
            scoring={"roc_auc": "roc_auc", "f1": "f1", "neg_brier": "neg_brier_score"},
        )
        cv[name] = {
            "roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "roc_auc_std": float(scores["test_roc_auc"].std()),
            "f1_mean": float(scores["test_f1"].mean()),
            "brier_mean": float(-scores["test_neg_brier"].mean()),
        }
        pipeline = build_pipeline(spec, estimator)
        start = time.perf_counter()
        pipeline.fit(train[spec.features], train[TARGET])
        fitted[name] = (pipeline, params, time.perf_counter() - start)

    chosen, reason = select(cv)
    records: dict[str, ModelRecord] = {}
    for name, (pipeline, params, fit_s) in fitted.items():
        threshold = best_f1_threshold(
            validation[TARGET].to_numpy(), predict_proba(pipeline, validation, spec.features)
        )
        metrics = {
            "cv": cv[name],
            "validation": classification_metrics(
                validation[TARGET].to_numpy(),
                predict_proba(pipeline, validation, spec.features),
                threshold,
            ),
            "test": classification_metrics(
                test[TARGET].to_numpy(), predict_proba(pipeline, test, spec.features), threshold
            ),
            "training_seconds": fit_s,
            "latency": measure_latency(pipeline, test, spec.features),
        }
        records[name] = registry.register(
            {
                "dataset": spec.name,
                "version": _version(spec.name, name, params, fingerprint, seed),
                "algorithm": name,
                "threshold": threshold,
                "features": spec.features,
                "contract_version": contract.version,
                "train_fingerprint": fingerprint,
                "seed": seed,
                "params": params,
                "metrics": metrics,
                "library_versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__},
                "notes": "selected baseline"
                if name == chosen
                else "baseline candidate (not selected)",
            },
            pipeline,
        )
    registry.set_baseline(spec.name, records[chosen].version)
    return TrainingResult(selected=records[chosen], records=records, cv=cv, selection_reason=reason)
