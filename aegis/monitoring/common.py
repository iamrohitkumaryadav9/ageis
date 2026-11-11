"""Small shared statistical helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from aegis.schemas.findings import EpistemicLevel, Finding, Severity, finding_id


def make_finding(
    report_id: str,
    monitor: str,
    check: str,
    *,
    metric_name: str,
    triggered: bool,
    severity: Severity,
    level: EpistemicLevel,
    interpretation: str,
    feature: str | None = None,
    observed: Any = None,
    reference: Any = None,
    threshold: Any = None,
    evidence: dict[str, Any] | None = None,
) -> Finding:
    return Finding(
        finding_id=finding_id(report_id, monitor, check, feature),
        monitor=monitor,  # type: ignore[arg-type]
        check=check,
        metric_name=metric_name,
        feature=feature,
        observed_value=_scalar(observed),
        reference_value=_scalar(reference),
        threshold=_scalar(threshold),
        triggered=bool(triggered),
        severity=severity if triggered else Severity.INFO,
        level=level,
        interpretation=interpretation,
        evidence=_jsonable(evidence or {}),
    )


def _scalar(value: Any) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return None if not np.isfinite(value) else round(value, 6)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | np.ndarray):
        return [_jsonable(v) for v in value]
    return _scalar(value)


def psi(expected: np.ndarray, actual: np.ndarray, floor: float = 1e-4) -> float:
    """Population Stability Index between two proportion vectors over the same bins."""
    e = np.clip(np.asarray(expected, dtype=float), floor, None)
    a = np.clip(np.asarray(actual, dtype=float), floor, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """BH step-up adjusted p-values (controls the false discovery rate across features)."""
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return []
    order = np.argsort(p)
    scaled = p[order] * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out.tolist()


def two_proportion_z(x1: int, n1: int, x2: int, n2: int) -> float:
    """One-sided p-value for H1: p2 > p1 (pooled two-proportion z-test)."""
    from scipy.stats import norm

    if n1 == 0 or n2 == 0:
        return 1.0
    pooled = (x1 + x2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    return float(norm.sf((x2 / n2 - x1 / n1) / se))


def matched_reference_stat(
    reference: pd.DataFrame, n: int, stat, repeats: int = 5, seed: int = 0
) -> float:
    """Mean of ``stat`` over reference subsamples of size ``n`` (size-matched expectation)."""
    if n >= len(reference):
        return float(stat(reference))
    rng = np.random.default_rng(seed)
    values = [
        stat(reference.iloc[rng.choice(len(reference), size=n, replace=False)])
        for _ in range(repeats)
    ]
    return float(np.mean(values))
