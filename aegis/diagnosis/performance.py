"""Performance diagnostics: version comparison and error slicing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aegis.ingestion.datasets import TARGET, DatasetSpec
from aegis.models.metrics import classification_metrics
from aegis.models.registry import ModelRegistry
from aegis.models.training import predict_proba

KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc", "brier")


def evaluate_version(
    registry: ModelRegistry, spec: DatasetSpec, version: str, frame: pd.DataFrame
) -> dict | None:
    """Metrics of a *registered, hash-verified* version on the labelled rows of ``frame``."""
    labelled = frame[frame[TARGET].notna()] if TARGET in frame.columns else frame.iloc[0:0]
    if len(labelled) < 50 or not set(spec.features) <= set(labelled.columns):
        return None
    record = registry.get(spec.name, version)
    model = registry.load(spec.name, version, verify=True)
    try:
        scores = predict_proba(model, labelled, spec.features)
    except Exception:
        return None
    metrics = classification_metrics(
        labelled[TARGET].to_numpy().astype(int), scores, record.threshold
    )
    return {k: metrics[k] for k in (*KEYS, "n")}


def compare_versions(
    registry: ModelRegistry,
    spec: DatasetSpec,
    version_a: str,
    version_b: str,
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> dict:
    out: dict = {"version_a": version_a, "version_b": version_b, "datasets": {}}
    for name, frame in (("reference", reference), ("current", current)):
        a = evaluate_version(registry, spec, version_a, frame)
        b = evaluate_version(registry, spec, version_b, frame)
        if a is None or b is None:
            out["datasets"][name] = {
                "available": False,
                "reason": "not enough labelled, contract-conforming rows",
            }
            continue
        out["datasets"][name] = {
            "available": True,
            "n": a["n"],
            "a": a,
            "b": b,
            "delta_b_minus_a": {
                k: (None if a[k] is None or b[k] is None else round(b[k] - a[k], 5)) for k in KEYS
            },
        }
    return out


def error_slices(
    spec: DatasetSpec,
    reference: pd.DataFrame,
    reference_errors: np.ndarray,
    current: pd.DataFrame,
    current_errors: np.ndarray,
    *,
    min_support: int = 50,
    top: int = 5,
) -> list[dict]:
    """Segments whose error rate rose the most. A uniform rise across segments points at
    a global cause (concept change, wrong model); a concentrated rise points at a slice."""
    rows = []
    for feature in spec.features:
        if feature not in current.columns:
            continue
        if feature in spec.numeric_features:
            if not pd.api.types.is_numeric_dtype(current[feature]):
                continue
            edges = np.unique(np.quantile(reference[feature].dropna(), [0.25, 0.5, 0.75]))
            ref_seg = np.searchsorted(edges, reference[feature].fillna(np.inf), side="right")
            cur_seg = np.searchsorted(edges, current[feature].fillna(np.inf), side="right")
            labels = {i: f"quartile-bin {i + 1}" for i in range(len(edges) + 1)}
        else:
            ref_seg = reference[feature].astype(str).to_numpy()
            cur_seg = current[feature].astype(str).to_numpy()
            labels = None
        for segment in pd.unique(cur_seg):
            cur_mask, ref_mask = cur_seg == segment, ref_seg == segment
            if cur_mask.sum() < min_support or ref_mask.sum() < min_support:
                continue
            rise = float(current_errors[cur_mask].mean() - reference_errors[ref_mask].mean())
            rows.append(
                {
                    "feature": feature,
                    "segment": labels[int(segment)] if labels else str(segment)[:40],
                    "support": int(cur_mask.sum()),
                    "reference_error_rate": round(float(reference_errors[ref_mask].mean()), 4),
                    "current_error_rate": round(float(current_errors[cur_mask].mean()), 4),
                    "rise": round(rise, 4),
                }
            )
    # ties go to the broader segment: it explains more of the errors
    return sorted(rows, key=lambda r: (-r["rise"], -r["support"]))[:top]
