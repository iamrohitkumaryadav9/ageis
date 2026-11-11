"""Model performance monitoring with explicit label-availability handling.

Supervised metrics are computed only on rows whose label has arrived, and only
when there are enough of them; otherwise the check is *skipped with a reason*
rather than reported with a number that means nothing. The prediction
distribution is label-free and is always monitored.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

from aegis.models.metrics import classification_metrics
from aegis.monitoring.common import make_finding, two_proportion_z
from aegis.monitoring.config import MonitoringConfig
from aegis.monitoring.drift import numeric_psi
from aegis.schemas.findings import EpistemicLevel, Finding, Severity, SkippedCheck

SUPERVISED = ("accuracy", "precision", "recall", "f1", "roc_auc")


def check_label_availability(
    report_id: str, labelled: int, total: int, cfg: MonitoringConfig
) -> Finding:
    fraction = labelled / total if total else 0.0
    return make_finding(
        report_id,
        "performance",
        "label_availability",
        metric_name="labelled_fraction",
        observed=fraction,
        reference=1.0,
        threshold=cfg.min_label_fraction,
        triggered=fraction < cfg.min_label_fraction,
        severity=Severity.HIGH if fraction < cfg.low_label_fraction else Severity.MEDIUM,
        level=EpistemicLevel.OBSERVED_FACT,
        interpretation=f"{labelled} of {total} rows ({fraction:.1%}) have labels",
        evidence={"labelled": labelled, "total": total},
    )


def check_prediction_distribution(
    report_id: str,
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    threshold: float,
    cfg: MonitoringConfig,
) -> Finding:
    value = numeric_psi(reference_scores, current_scores, cfg.psi_bins)
    ks = ks_2samp(reference_scores, current_scores, method="asymp")
    ref_rate = float((reference_scores >= threshold).mean())
    cur_rate = float((current_scores >= threshold).mean())
    triggered = value >= cfg.prediction_psi and ks.pvalue < cfg.drift_alpha
    return make_finding(
        report_id,
        "performance",
        "prediction_distribution",
        metric_name="score_psi",
        observed=value,
        reference=0.0,
        threshold=cfg.prediction_psi,
        triggered=triggered,
        severity=Severity.MEDIUM,
        level=EpistemicLevel.STATISTICAL_ANOMALY,
        interpretation=(
            f"model score PSI {value:.3f} (KS p={ks.pvalue:.2g}); predicted-positive rate "
            f"{ref_rate:.3f} -> {cur_rate:.3f}"
        ),
        evidence={
            "ks_statistic": float(ks.statistic),
            "p_value": float(ks.pvalue),
            "reference_positive_rate": ref_rate,
            "current_positive_rate": cur_rate,
            "reference_mean_score": float(reference_scores.mean()),
            "current_mean_score": float(current_scores.mean()),
        },
    )


def check_performance(
    report_id: str,
    reference_y: np.ndarray,
    reference_scores: np.ndarray,
    current_y: np.ndarray,
    current_scores: np.ndarray,
    threshold: float,
    cfg: MonitoringConfig,
) -> tuple[list[Finding], list[SkippedCheck], dict]:
    """``current_y`` may contain NaN for rows whose label has not arrived."""
    findings: list[Finding] = []
    skipped: list[SkippedCheck] = []
    has_label = ~np.isnan(current_y.astype(float))
    n_labelled = int(has_label.sum())
    findings.append(check_label_availability(report_id, n_labelled, len(current_y), cfg))

    ref_metrics = classification_metrics(reference_y, reference_scores, threshold)
    summary = {"reference": ref_metrics, "current": None, "n_labelled": n_labelled}
    if n_labelled < cfg.min_labelled:
        skipped.append(
            SkippedCheck(
                monitor="performance",
                check="supervised_metrics",
                reason=f"only {n_labelled} labelled rows (< {cfg.min_labelled}); "
                "supervised metrics would be unreliable",
            )
        )
        return findings, skipped, summary

    y, scores = current_y[has_label].astype(int), current_scores[has_label]
    cur_metrics = classification_metrics(y, scores, threshold)
    summary["current"] = cur_metrics
    ref_errors = round((1 - ref_metrics["accuracy"]) * ref_metrics["n"])
    cur_errors = round((1 - cur_metrics["accuracy"]) * cur_metrics["n"])
    p_value = two_proportion_z(ref_errors, ref_metrics["n"], cur_errors, cur_metrics["n"])
    significant = p_value < cfg.performance_alpha
    partial = n_labelled < len(current_y)

    for metric in SUPERVISED:
        ref_value, cur_value = ref_metrics[metric], cur_metrics[metric]
        if ref_value is None or cur_value is None:
            skipped.append(
                SkippedCheck(
                    monitor="performance",
                    check=metric,
                    reason="undefined: only one class present among labelled rows",
                )
            )
            continue
        drop = ref_value - cur_value
        findings.append(
            make_finding(
                report_id,
                "performance",
                f"{metric}_drop",
                metric_name=metric,
                observed=cur_value,
                reference=ref_value,
                threshold=ref_value - cfg.metric_drop,
                triggered=significant and drop >= cfg.metric_drop,
                severity=Severity.CRITICAL if drop >= cfg.metric_drop_critical else Severity.HIGH,
                level=EpistemicLevel.STATISTICAL_ANOMALY,
                interpretation=(
                    f"{metric} {ref_value:.3f} -> {cur_value:.3f} (change {-drop:+.3f}) on "
                    f"{n_labelled} labelled rows; error-rate increase p={p_value:.2g}"
                    + (
                        "; labels are partial, so this subset may not be representative"
                        if partial
                        else ""
                    )
                ),
                evidence={
                    "drop": drop,
                    "error_rate_p_value": p_value,
                    "n_labelled": n_labelled,
                    "partial_labels": partial,
                },
            )
        )
    return findings, skipped, summary
