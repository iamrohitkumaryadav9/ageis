"""All monitoring thresholds in one place.

Two ideas recur. (1) A statistical test answers "is this difference real?";
an effect size answers "is it big enough to care?". A drift finding requires
both, because with thousands of rows almost any difference is significant.
(2) Rates that depend on sample size (duplicates, cardinality) are compared
with the reference *resampled to the same size*, not with a fixed constant.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class MonitoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = "mon-1"
    min_samples: int = 100  # below this, statistical checks are skipped rather than guessed

    # data quality
    missing_rate_delta: float = 0.05
    missing_rate_delta_high: float = 0.20
    duplicate_rate_excess: float = 0.03
    duplicate_rate_excess_high: float = 0.20
    range_violation_rate: float = 0.01
    range_tolerance: float = 0.10  # fraction of the reference range allowed beyond min/max
    unexpected_category_rate: float = 0.01
    cardinality_ratio: float = 1.5
    quality_alpha: float = 0.001

    # data drift
    drift_alpha: float = 0.01  # applied to Benjamini-Hochberg adjusted p-values
    psi_bins: int = 10
    psi_medium: float = 0.10
    psi_high: float = 0.25
    ks_effect: float = 0.10
    wasserstein_effect: float = 0.10  # in units of reference standard deviation
    cramers_v_effect: float = 0.10
    tvd_effect: float = 0.10  # total variation: share of probability mass that moved

    # concept drift (error stream)
    adwin_delta: float = 0.002
    ph_delta: float = 0.02
    ph_threshold: float = 30.0
    ph_min_instances: int = 30
    concept_warmup: int = 2000  # trailing reference errors used to initialise the detectors
    concept_alpha: float = 0.01  # a detector alarm must be backed by a significant error-rate rise

    # performance
    min_labelled: int = 200
    min_label_fraction: float = 0.70
    low_label_fraction: float = 0.30
    metric_drop: float = 0.05
    metric_drop_critical: float = 0.15
    performance_alpha: float = 0.001
    prediction_psi: float = 0.10

    # feature/label relationship
    leakage_strength_gain: float = 0.10
    leakage_z: float = 3.3  # gain must exceed this many standard errors (~p < 0.0005)
    leakage_strength_high: float = 0.80
    reversal_min_strength: float = 0.10
