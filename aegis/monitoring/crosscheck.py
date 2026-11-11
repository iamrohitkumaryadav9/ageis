"""Independent cross-check of AEGIS's drift statistics against Evidently.

AEGIS computes drift itself (it needs per-method verdicts, Benjamini-Hochberg
adjustment and effect-size gating). A second implementation of the same
statistics is the cheapest way to catch a mistake in the first, so this module
runs Evidently's per-column drift metrics on the same frames. It is used by
the test suite and by ``experiments/drift_crosscheck.py``; it is not on the
request path.
"""

from __future__ import annotations

import pandas as pd

from aegis.ingestion.datasets import DatasetSpec

NUMERIC_METHODS = ("ks", "wasserstein", "psi")
CATEGORICAL_METHODS = ("chisquare", "psi")


def evidently_drift(
    spec: DatasetSpec, reference: pd.DataFrame, current: pd.DataFrame
) -> dict[str, dict[str, float]]:
    """Return ``{feature: {method: value}}``. KS and chi-square values are p-values;
    Wasserstein is normalised by the reference standard deviation; PSI is the index."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.metrics import ValueDrift

    numeric = [
        f
        for f in spec.numeric_features
        if f in current.columns and pd.api.types.is_numeric_dtype(current[f])
    ]
    categorical = [f for f in spec.categorical_features if f in current.columns]
    definition = DataDefinition(numerical_columns=numeric, categorical_columns=categorical)
    metrics = [ValueDrift(column=f, method=m) for f in numeric for m in NUMERIC_METHODS]
    metrics += [ValueDrift(column=f, method=m) for f in categorical for m in CATEGORICAL_METHODS]
    columns = [*numeric, *categorical]
    # Compare like with like: AEGIS computes drift on non-null values, per feature.
    snapshot = Report(metrics).run(
        Dataset.from_pandas(current[columns].dropna(), data_definition=definition),
        Dataset.from_pandas(reference[columns].dropna(), data_definition=definition),
    )
    out: dict[str, dict[str, float]] = {}
    for item in snapshot.dict()["metrics"]:
        config = item["config"]
        out.setdefault(config["column"], {})[config["method"]] = float(item["value"])
    return out
