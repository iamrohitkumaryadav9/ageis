"""Concept drift on the model's error stream, using River.

Both detectors watch the 0/1 prediction-error sequence: the trailing reference
errors initialise them, then the current batch is replayed in arrival order.

* ADWIN keeps an adaptive window and signals when two sub-windows have means
  that differ by more than a Hoeffding-style bound (confidence ``delta``).
  Assumes bounded, independent observations; has formal false-positive
  guarantees; adapts to both gradual and abrupt change; needs labels.
* Page-Hinkley is a sequential CUSUM-type test for an upward shift of the mean
  beyond a tolerance ``delta``; alarms when the cumulative deviation exceeds
  ``threshold``. Cheap and good at abrupt change; its parameters are
  scale-dependent and it only looks for increases here.

Neither can say *why* errors rose, and neither works without labels: an error
stream cannot be built from unlabelled rows.
"""

from __future__ import annotations

import numpy as np
from river.drift import ADWIN, PageHinkley

from aegis.monitoring.common import make_finding, two_proportion_z
from aegis.monitoring.config import MonitoringConfig
from aegis.schemas.findings import EpistemicLevel, Finding, Severity


def first_detection(detector, warmup: np.ndarray, stream: np.ndarray) -> int | None:
    """Index in ``stream`` (0-based) of the first drift signal, or None."""
    for value in warmup:
        detector.update(float(value))
    for i, value in enumerate(stream):
        detector.update(float(value))
        if detector.drift_detected:
            return i
    return None


def check_concept(
    report_id: str,
    reference_errors: np.ndarray,
    current_errors: np.ndarray,
    cfg: MonitoringConfig,
) -> list[Finding]:
    warmup = reference_errors[-cfg.concept_warmup :]
    ref_rate, cur_rate = float(reference_errors.mean()), float(current_errors.mean())
    detectors = {
        "adwin": ADWIN(delta=cfg.adwin_delta),
        "page_hinkley": PageHinkley(
            min_instances=cfg.ph_min_instances,
            delta=cfg.ph_delta,
            threshold=cfg.ph_threshold,
            mode="up",
        ),
    }
    # Sequential detectors have no protection against the many looks they take at the
    # stream, so an alarm only counts if the overall error rate also rose significantly.
    p_value = two_proportion_z(
        int(reference_errors.sum()),
        len(reference_errors),
        int(current_errors.sum()),
        len(current_errors),
    )
    findings = []
    for name, detector in detectors.items():
        index = first_detection(detector, warmup, current_errors)
        # ADWIN is two-sided; an *improvement* in error rate is not an incident.
        worse = index is not None and cur_rate > ref_rate and p_value < cfg.concept_alpha
        findings.append(
            make_finding(
                report_id,
                "concept",
                name,
                metric_name="error_rate",
                observed=cur_rate,
                reference=ref_rate,
                threshold=cfg.adwin_delta if name == "adwin" else cfg.ph_threshold,
                triggered=worse,
                severity=Severity.MEDIUM,
                level=EpistemicLevel.STATISTICAL_ANOMALY,
                interpretation=(
                    f"{name} signalled a change in the error stream after {index + 1} labelled rows "
                    f"(error rate {ref_rate:.3f} -> {cur_rate:.3f})"
                    if worse
                    else f"{name} saw no upward change in the error stream "
                    f"(error rate {ref_rate:.3f} -> {cur_rate:.3f})"
                ),
                evidence={
                    "detection_index": index if worse else None,
                    "detector_alarm": index is not None,
                    "error_rate_p_value": p_value,
                    "n_labelled": len(current_errors),
                    "warmup": len(warmup),
                },
            )
        )
    return findings
