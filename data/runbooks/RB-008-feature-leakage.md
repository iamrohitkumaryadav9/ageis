---
id: RB-008
title: Suspicious feature leakage
version: 1.0
applies_to: [feature_leakage]
---
## Symptoms
A single feature's association with the label rises far above its reference level, sometimes to near-perfect separation, while its own distribution may be unchanged. Offline metrics computed with the feature look too good.

## Triage
Features do not become more predictive by themselves. Suspect that the feature is now derived from, or joined after, the outcome: a status field updated post-decision, a backfilled value, or a join on a table that contains the label.

## Diagnosis
Check the feature's lineage and the time at which its value is known relative to prediction time. Inspect rank correlations with other features: tampering at row level disturbs them even when the marginal distribution is intact.

## Remediation
Quarantine the batch and block retraining on it. A model retrained on leaked data will look excellent offline and fail in production. Do not roll the model back: the deployed model is not at fault.

## Verification
After the upstream fix the feature's association with the label must return to its reference level on a fresh batch.
