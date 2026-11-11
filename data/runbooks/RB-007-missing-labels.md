---
id: RB-007
title: Delayed or missing labels
version: 1.0
applies_to: [label_delay]
---
## Symptoms
The share of rows with a ground-truth label falls below the expected level. Supervised metrics are either unavailable or computed on a partial subset.

## Triage
Check the label pipeline before the model. Delayed labels are common when outcomes take time to materialise. A sudden drop to zero suggests a broken join or feed rather than natural delay.

## Diagnosis
Do not compute or report supervised metrics when too few labels are available: a number computed on a handful of rows is noise, and the earliest labels to arrive are often not representative. Rely on label-free signals meanwhile: input drift, data quality and the prediction distribution.

## Remediation
No model action is justified by missing labels alone. Defer, keep monitoring label-free signals, and re-run performance monitoring when labels arrive. Escalate to the label pipeline owner if availability does not recover within the expected delay.

## Verification
Label availability returns above the configured minimum and supervised metrics are recomputed on the full batch.
