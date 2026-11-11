"""Data-quality monitoring against the data contract.

Schema checks are observed facts. Rate checks (missingness, duplicates, range,
unexpected categories) compare an observed rate with the reference rate and
require both a minimum absolute excess and a binomial significance test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from aegis.ingestion.contracts import DataContract
from aegis.monitoring.common import make_finding, matched_reference_stat
from aegis.monitoring.config import MonitoringConfig
from aegis.schemas.findings import EpistemicLevel, Finding, Severity

FACT, ANOMALY = EpistemicLevel.OBSERVED_FACT, EpistemicLevel.STATISTICAL_ANOMALY


def check_schema(
    report_id: str, current: pd.DataFrame, contract: DataContract, allowed_extra: set[str]
) -> tuple[list[Finding], list[str]]:
    """Return schema findings and the list of features that are present and correctly typed."""
    findings: list[Finding] = []
    usable: list[str] = []
    expected = contract.by_name
    missing = [c for c in expected if c not in current.columns]
    extra = [c for c in current.columns if c not in expected and c not in allowed_extra]

    findings.append(
        make_finding(
            report_id,
            "quality",
            "schema_missing_columns",
            metric_name="missing_column_count",
            observed=len(missing),
            reference=0,
            threshold=0,
            triggered=bool(missing),
            severity=Severity.HIGH,
            level=FACT,
            interpretation=(
                f"{len(missing)} contracted column(s) absent: {missing}"
                if missing
                else "all contracted columns are present"
            ),
            evidence={"missing_columns": missing},
        )
    )
    findings.append(
        make_finding(
            report_id,
            "quality",
            "schema_unexpected_columns",
            metric_name="unexpected_column_count",
            observed=len(extra),
            reference=0,
            threshold=0,
            triggered=bool(extra),
            severity=Severity.LOW,
            level=FACT,
            interpretation=(
                f"{len(extra)} column(s) not in the contract: {[_safe(c) for c in extra]}"
                if extra
                else "no unexpected columns"
            ),
            evidence={"unexpected_columns": [_safe(c) for c in extra]},
        )
    )
    for name, column in expected.items():
        if name in missing:
            continue
        series = current[name]
        if column.kind == "numeric" and not pd.api.types.is_numeric_dtype(series):
            coerced = pd.to_numeric(series, errors="coerce")
            unparseable = float((coerced.isna() & series.notna()).mean())
            findings.append(
                make_finding(
                    report_id,
                    "quality",
                    "schema_type_mismatch",
                    feature=name,
                    metric_name="unparseable_rate",
                    observed=unparseable,
                    reference=0.0,
                    threshold=0.0,
                    triggered=True,
                    severity=Severity.HIGH,
                    level=FACT,
                    interpretation=(
                        f"'{name}' is contracted as numeric but arrived as {series.dtype}; "
                        f"{unparseable:.1%} of non-null values do not parse as numbers"
                    ),
                    evidence={"observed_dtype": str(series.dtype), "expected_kind": "numeric"},
                )
            )
        else:
            usable.append(name)
    return findings, usable


def _safe(text: object, limit: int = 40) -> str:
    """Render untrusted text for inclusion in findings: printable ASCII, bounded length."""
    cleaned = "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in str(text))
    return cleaned[:limit]


def _rate_finding(
    report_id, check, feature, observed, reference, n, cfg, *, delta, delta_high, what
) -> Finding:
    excess = observed - reference
    p_value = (
        binomtest(round(observed * n), n, min(max(reference, 1e-6), 1 - 1e-6), "greater").pvalue
        if n > 0
        else 1.0
    )
    triggered = excess >= delta and p_value < cfg.quality_alpha
    return make_finding(
        report_id,
        "quality",
        check,
        feature=feature,
        metric_name=check,
        observed=observed,
        reference=reference,
        threshold=reference + delta,
        triggered=triggered,
        severity=Severity.HIGH if excess >= delta_high else Severity.MEDIUM,
        level=ANOMALY,
        interpretation=(
            f"{what} is {observed:.1%} versus {reference:.1%} expected "
            f"(excess {excess:+.1%}, binomial p={p_value:.2g})"
        ),
        evidence={"excess": excess, "p_value": p_value, "n": n},
    )


def check_quality(
    report_id: str,
    current: pd.DataFrame,
    reference: pd.DataFrame,
    contract: DataContract,
    usable: list[str],
    cfg: MonitoringConfig,
) -> list[Finding]:
    findings: list[Finding] = []
    n = len(current)
    columns = contract.by_name

    # duplicates: expectation is size-matched because the chance of an exact repeat grows with n
    present = [c for c in columns if c in current.columns]
    dup_rate = float(current[present].astype(str).duplicated().mean()) if n else 0.0
    ref_cols = [c for c in present if c in reference.columns]
    expected_dup = matched_reference_stat(
        reference[ref_cols], n, lambda frame: float(frame.astype(str).duplicated().mean())
    )
    findings.append(
        _rate_finding(
            report_id,
            "duplicate_rate",
            None,
            dup_rate,
            expected_dup,
            n,
            cfg,
            delta=cfg.duplicate_rate_excess,
            delta_high=cfg.duplicate_rate_excess_high,
            what="exact-duplicate row rate",
        )
    )

    for name in usable:
        column, series = columns[name], current[name]
        findings.append(
            _rate_finding(
                report_id,
                "missing_rate",
                name,
                float(series.isna().mean()),
                column.nullable_rate,
                n,
                cfg,
                delta=cfg.missing_rate_delta,
                delta_high=cfg.missing_rate_delta_high,
                what=f"missing rate of '{name}'",
            )
        )
        values = series.dropna()
        if values.empty:
            continue
        if column.kind == "numeric":
            numeric = values.to_numpy(dtype=float)
            non_finite = float((~np.isfinite(numeric)).mean())
            span = (column.maximum - column.minimum) or 1.0
            low = column.minimum - cfg.range_tolerance * span
            high = column.maximum + cfg.range_tolerance * span
            finite = numeric[np.isfinite(numeric)]
            out_rate = float(((finite < low) | (finite > high)).mean()) if finite.size else 0.0
            invalid = out_rate + non_finite
            findings.append(
                make_finding(
                    report_id,
                    "quality",
                    "range_violation_rate",
                    feature=name,
                    metric_name="range_violation_rate",
                    observed=invalid,
                    reference=0.0,
                    threshold=cfg.range_violation_rate,
                    triggered=invalid >= cfg.range_violation_rate,
                    severity=Severity.HIGH if invalid >= 0.2 else Severity.MEDIUM,
                    level=FACT,
                    interpretation=(
                        f"{invalid:.1%} of '{name}' values are non-finite or outside the tolerated "
                        f"reference range [{low:.4g}, {high:.4g}] "
                        f"(observed min {finite.min():.4g}, max {finite.max():.4g})"
                        if finite.size
                        else f"'{name}' has no finite values"
                    ),
                    evidence={
                        "reference_min": column.minimum,
                        "reference_max": column.maximum,
                        "observed_min": float(finite.min()) if finite.size else None,
                        "observed_max": float(finite.max()) if finite.size else None,
                        "non_finite_rate": non_finite,
                    },
                )
            )
        else:
            known = set(column.categories or [])
            as_text = values.astype(str)
            unknown_mask = ~as_text.isin(known)
            unknown_rate = float(unknown_mask.mean())
            examples = [_safe(v) for v in as_text[unknown_mask].value_counts().index[:3]]
            findings.append(
                make_finding(
                    report_id,
                    "quality",
                    "unexpected_category_rate",
                    feature=name,
                    metric_name="unexpected_category_rate",
                    observed=unknown_rate,
                    reference=0.0,
                    threshold=cfg.unexpected_category_rate,
                    triggered=unknown_rate >= cfg.unexpected_category_rate,
                    severity=Severity.MEDIUM,
                    level=FACT,
                    interpretation=(
                        f"{unknown_rate:.1%} of '{name}' values are categories absent from the contract"
                    ),
                    evidence={"example_values": examples, "n_unknown": int(unknown_mask.sum())},
                )
            )
            cardinality = int(as_text.nunique())
            expected_card = matched_reference_stat(
                reference[[name]].dropna(),
                len(values),
                lambda frame: float(frame.iloc[:, 0].nunique()),
            )
            ratio = cardinality / expected_card if expected_card else float("inf")
            off = ratio >= cfg.cardinality_ratio or ratio <= 1 / cfg.cardinality_ratio
            findings.append(
                make_finding(
                    report_id,
                    "quality",
                    "cardinality_change",
                    feature=name,
                    metric_name="cardinality",
                    observed=cardinality,
                    reference=expected_card,
                    threshold=cfg.cardinality_ratio,
                    triggered=off and abs(cardinality - expected_card) >= 2,
                    severity=Severity.LOW,
                    level=FACT,
                    interpretation=(
                        f"'{name}' has {cardinality} distinct values versus ~{expected_card:.1f} "
                        "expected at this sample size"
                    ),
                    evidence={"ratio": ratio},
                )
            )
    return findings
