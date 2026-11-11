"""Ten deterministic fault injectors.

Each injector is a pure function of (clean batch, RNG, severity) and returns
the corrupted batch plus the ground-truth parameters. Severity scales how
*large* the fault is, so the benchmark covers faint and blatant versions of
each category.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from aegis.ingestion.datasets import TARGET, DatasetSpec
from aegis.schemas.diagnosis import RootCause

SEVERITIES = ("low", "medium", "high")


@dataclass
class Injection:
    frame: pd.DataFrame
    affected_features: list[str]
    params: dict[str, Any] = field(default_factory=dict)
    artifact_variant: str | None = None  # only F10: which wrong artifact to deploy


Injector = Callable[[pd.DataFrame, DatasetSpec, np.random.Generator, str], Injection]


def _level(severity: str, low: Any, medium: Any, high: Any) -> Any:
    return {"low": low, "medium": medium, "high": high}[severity]


def _pick(rng: np.random.Generator, options: list[str]) -> str:
    return str(options[int(rng.integers(len(options)))])


def numerical_mean_shift(df, spec, rng, severity) -> Injection:
    """F01: add a constant offset of k reference standard deviations to one numeric feature."""
    out = df.copy()
    feature = _pick(rng, spec.numeric_features)
    k = _level(severity, 0.35, 0.7, 1.5)
    direction = 1.0 if rng.random() < 0.5 else -1.0
    offset = direction * k * float(out[feature].std())
    out[feature] = np.round(out[feature] + offset)
    return Injection(out, [feature], {"offset": offset, "std_multiple": k * direction})


def categorical_frequency_shift(df, spec, rng, severity) -> Injection:
    """F02: overwrite a fraction of one categorical feature with a single (non-modal) category."""
    out = df.copy()
    feature = _pick(rng, spec.categorical_features)
    counts = out[feature].value_counts()
    target = str(counts.index[1] if len(counts) > 1 else counts.index[0])
    fraction = _level(severity, 0.15, 0.30, 0.60)
    mask = rng.random(len(out)) < fraction
    out.loc[mask, feature] = target
    return Injection(out, [feature], {"to_category": target, "fraction": fraction})


def missing_value_spike(df, spec, rng, severity) -> Injection:
    """F03: blank out a fraction of one or two features (completely at random)."""
    out = df.copy()
    n_features = _level(severity, 1, 1, 2)
    features = [str(f) for f in rng.choice(spec.features, size=n_features, replace=False)]
    fraction = _level(severity, 0.12, 0.30, 0.60)
    for feature in features:
        out.loc[rng.random(len(out)) < fraction, feature] = np.nan
    return Injection(out, features, {"fraction": fraction})


def feature_scaling_error(df, spec, rng, severity) -> Injection:
    """F04: multiply one numeric feature by a unit-conversion-like factor."""
    out = df.copy()
    feature = _pick(rng, spec.numeric_features)
    factor = float(
        _level(
            severity,
            _pick_num(rng, [2.2, 0.45]),
            _pick_num(rng, [12, 60, 0.1]),
            _pick_num(rng, [1000, 0.001, 100]),
        )
    )
    out[feature] = out[feature] * factor
    return Injection(out, [feature], {"factor": factor})


def _pick_num(rng: np.random.Generator, options: list[float]) -> float:
    return float(options[int(rng.integers(len(options)))])


def target_relationship_reversal(df, spec, rng, severity) -> Injection:
    """F05: invert the label for a fraction of rows; features are untouched (pure concept change)."""
    out = df.copy()
    fraction = _level(severity, 0.30, 0.55, 0.90)
    mask = rng.random(len(out)) < fraction
    out.loc[mask, TARGET] = 1 - out.loc[mask, TARGET]
    return Injection(out, [], {"flipped_fraction": fraction})


def duplicate_records(df, spec, rng, severity) -> Injection:
    """F06: overwrite a fraction of rows with exact copies of other rows (batch size unchanged)."""
    out = df.copy()
    fraction = _level(severity, 0.10, 0.25, 0.50)
    n_dup = int(len(out) * fraction)
    positions = rng.permutation(len(out))
    victims, sources = positions[:n_dup], rng.choice(positions[n_dup:], size=n_dup, replace=True)
    out.iloc[victims] = out.iloc[sources].to_numpy()
    return Injection(out.astype(df.dtypes.to_dict()), [], {"duplicated_fraction": fraction})


def feature_leakage(df, spec, rng, severity) -> Injection:
    """F07: re-order one numeric feature so that it encodes the label.

    The feature's marginal distribution is preserved exactly (the same values
    are permuted), so there is no covariate-drift signal to hide behind: the
    only trace is an implausibly strong feature/label relationship.
    """
    out = df.copy()
    feature = _pick(rng, spec.numeric_features)
    aligned_fraction = _level(severity, 0.80, 0.92, 1.0)
    aligned = rng.random(len(out)) < aligned_fraction
    idx = np.flatnonzero(aligned)
    values = np.sort(out[feature].to_numpy()[idx])
    key = out[TARGET].to_numpy()[idx] + rng.random(len(idx)) * 0.5  # label order, random ties
    reordered = np.empty_like(values)
    reordered[np.argsort(key, kind="stable")] = values
    column = out[feature].to_numpy().copy()
    column[idx] = reordered
    out[feature] = column
    return Injection(out, [feature], {"aligned_fraction": aligned_fraction})


def schema_change(df, spec, rng, severity) -> Injection:
    """F08: break the data contract - re-type, rename, or drop columns."""
    out = df.copy()
    if severity == "low":
        feature = _pick(rng, spec.numeric_features)
        out[feature] = (
            out[feature].map(lambda v: f"{v:g} units" if pd.notna(v) else v).astype(object)
        )
        return Injection(out, [feature], {"change": "retyped_numeric_to_text"})
    if severity == "medium":
        feature = _pick(rng, spec.features)
        new_name = "".join(p.capitalize() for p in feature.split("_")) + "V2"
        return Injection(
            out.rename(columns={feature: new_name}),
            [feature],
            {"change": "renamed", "new_name": new_name},
        )
    features = [str(f) for f in rng.choice(spec.features, size=2, replace=False)]
    return Injection(out.drop(columns=features), features, {"change": "dropped"})


def delayed_labels(df, spec, rng, severity) -> Injection:
    """F09: labels for the most recent rows have not arrived yet."""
    out = df.copy()
    out[TARGET] = out[TARGET].astype("float64")
    fraction = _level(severity, 0.50, 0.80, 1.0)
    n_missing = round(len(out) * fraction)
    if n_missing:
        out.iloc[len(out) - n_missing :, out.columns.get_loc(TARGET)] = np.nan
    return Injection(out, [], {"unlabelled_fraction": fraction})


def model_artifact_mismatch(df, spec, rng, severity) -> Injection:
    """F10: the serving artifact is not the registered one. Data is untouched."""
    variant = _level(
        severity, "other_registered_version", "unregistered_retrain", "corrupted_bytes"
    )
    return Injection(df.copy(), [], {"variant": variant}, artifact_variant=variant)


@dataclass(frozen=True)
class FaultType:
    fault_id: str
    name: str
    root_cause: RootCause
    injector: Injector
    expected_signal: str


FAULTS: dict[str, FaultType] = {
    f.fault_id: f
    for f in [
        FaultType(
            "F01",
            "Numerical mean shift",
            RootCause.NUMERICAL_SHIFT,
            numerical_mean_shift,
            "statistical distribution change",
        ),
        FaultType(
            "F02",
            "Categorical frequency shift",
            RootCause.CATEGORICAL_SHIFT,
            categorical_frequency_shift,
            "category proportion change",
        ),
        FaultType(
            "F03",
            "Missing-value spike",
            RootCause.MISSING_VALUES,
            missing_value_spike,
            "data quality degradation",
        ),
        FaultType(
            "F04",
            "Feature scaling error",
            RootCause.SCALING_ERROR,
            feature_scaling_error,
            "range/scale anomaly",
        ),
        FaultType(
            "F05",
            "Target relationship reversal",
            RootCause.CONCEPT_DRIFT,
            target_relationship_reversal,
            "model performance / concept change",
        ),
        FaultType(
            "F06",
            "Duplicate records",
            RootCause.DUPLICATE_RECORDS,
            duplicate_records,
            "duplicate-rate anomaly",
        ),
        FaultType(
            "F07",
            "Feature leakage",
            RootCause.FEATURE_LEAKAGE,
            feature_leakage,
            "suspicious feature/label relationship",
        ),
        FaultType(
            "F08", "Schema change", RootCause.SCHEMA_CHANGE, schema_change, "contract violation"
        ),
        FaultType(
            "F09",
            "Delayed labels",
            RootCause.LABEL_DELAY,
            delayed_labels,
            "label availability degradation",
        ),
        FaultType(
            "F10",
            "Model artifact mismatch",
            RootCause.ARTIFACT_MISMATCH,
            model_artifact_mismatch,
            "model version/hash inconsistency",
        ),
    ]
}
