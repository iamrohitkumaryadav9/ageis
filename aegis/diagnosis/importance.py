"""Which features does the model actually rely on?

Three views, all on bounded samples so cost stays predictable on a laptop CPU:

* model-native importance is not used (HGB exposes none; LR coefficients are
  scale dependent),
* permutation importance: drop in ROC-AUC when one raw feature is shuffled,
* SHAP: mean |attribution| per raw feature, aggregated over one-hot columns.

SHAP is computed on at most ``max_rows`` rows (default 200). TreeExplainer on
the fitted booster is exact and takes well under a second at that size;
explaining every production row would cost minutes and add no diagnostic value.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from aegis.ingestion.datasets import TARGET


def _sample(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    return frame.iloc[np.random.default_rng(seed).choice(len(frame), size=max_rows, replace=False)]


def permutation_importances(
    model: Pipeline,
    frame: pd.DataFrame,
    features: list[str],
    *,
    max_rows: int = 1000,
    repeats: int = 3,
    seed: int = 0,
) -> dict:
    labelled = frame[frame[TARGET].notna()]
    sample = _sample(labelled, max_rows, seed)
    if len(sample) < 50 or sample[TARGET].nunique() < 2:
        return {"available": False, "reason": "not enough labelled rows of both classes"}
    start = time.perf_counter()
    result = permutation_importance(
        model,
        sample[features],
        sample[TARGET].astype(int),
        scoring="roc_auc",
        n_repeats=repeats,
        random_state=seed,
    )
    ranked = sorted(zip(features, result.importances_mean, strict=True), key=lambda kv: -kv[1])
    return {
        "available": True,
        "metric": "roc_auc_drop",
        "rows": len(sample),
        "repeats": repeats,
        "seconds": round(time.perf_counter() - start, 3),
        "importances": {name: round(float(value), 5) for name, value in ranked},
    }


def shap_importances(
    model: Pipeline, frame: pd.DataFrame, features: list[str], *, max_rows: int = 200, seed: int = 0
) -> dict:
    """Mean |SHAP| per raw feature on a bounded sample."""
    import shap

    sample = _sample(frame, max_rows, seed)
    preprocess, estimator = model.named_steps["preprocess"], model.named_steps["model"]
    start = time.perf_counter()
    transformed = preprocess.transform(sample[features])
    names = preprocess.get_feature_names_out()
    if hasattr(estimator, "coef_"):
        background = shap.maskers.Independent(transformed, max_samples=len(sample))
        explainer = shap.LinearExplainer(estimator, background)
        kind = "linear"
    else:
        explainer = shap.TreeExplainer(estimator)
        kind = "tree"
    values = np.asarray(explainer.shap_values(transformed))
    if values.ndim == 3:  # (rows, columns, classes) for some estimators
        values = values[..., -1]
    per_column = np.abs(values).mean(axis=0)
    totals: dict[str, float] = dict.fromkeys(features, 0.0)
    for column, value in zip(names, per_column, strict=True):
        raw = column.split("__", 1)[1]
        owner = next(
            (
                f
                for f in sorted(features, key=len, reverse=True)
                if raw == f or raw.startswith(f + "_")
            ),
            None,
        )
        if owner is not None:
            totals[owner] += float(value)
    scale = sum(totals.values()) or 1.0
    ranked = sorted(totals.items(), key=lambda kv: -kv[1])
    return {
        "available": True,
        "explainer": kind,
        "rows": len(sample),
        "seconds": round(time.perf_counter() - start, 3),
        "mean_abs_shap_share": {name: round(value / scale, 5) for name, value in ranked},
    }
