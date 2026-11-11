---
id: RB-004
title: Feature scaling and unit errors
version: 1.0
applies_to: [scaling_error]
---
## Symptoms
A numeric feature's values are off by a constant factor: quantiles of the current data equal the reference quantiles multiplied by a constant such as 60, 100, 1000 or 0.001. Range violations appear when the factor is above one; when it is below one the values stay inside the contracted range and only drift tests fire.

## Triage
Identify the factor from the quantile ratio. Typical causes are a unit change upstream (hours to minutes, currency units to cents), a normalisation step applied twice or dropped, or a percentage sent as a fraction.

## Diagnosis
A multiplicative change leaves zeros at zero and preserves rank correlations with other features. An additive shift moves the zeros and is handled by runbook RB-002. Confirm with the feature owner which unit is now being sent.

## Remediation
Revert the preprocessing configuration by applying the inverse factor in ingestion, validated on a held-out part of the affected batch: after correction the feature must match the reference distribution and model performance must return to its reference level. The correction is conditional on the upstream unit staying as it is now, so record a rollback path and remove the correction when the source is fixed.

## Verification
Re-run monitoring on corrected data. The drift and range findings for the feature must clear and no new finding may appear.
