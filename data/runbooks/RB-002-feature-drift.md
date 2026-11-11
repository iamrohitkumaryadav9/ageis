---
id: RB-002
title: Feature distribution drift (numerical and categorical)
version: 1.1
applies_to: [numerical_shift, categorical_shift]
---
## Symptoms
One or more features differ from the reference window by a material effect size (PSI at or above 0.1, Wasserstein distance at or above 0.1 standard deviations, or a large change in category proportions) with a significant adjusted p-value.

## Triage
Drift is a change in the inputs, not a verdict on the model. First establish whether performance on labelled rows moved. If labels are delayed, look at the prediction distribution instead and say explicitly that performance impact is unknown.

## Diagnosis
For a numeric feature, inspect the quantile relationship to the reference: a constant offset suggests a population or location shift, a constant ratio suggests a unit or scaling error and belongs to runbook RB-004. For a categorical feature, list the categories that gained and lost share and check for labels that are new to the contract. Weigh the feature's importance to the model: drift in a feature the model barely uses rarely matters.

## Remediation
If performance is unaffected, take no model action: record the drift, keep monitoring and consider refreshing the reference window after review. If performance degraded and labels are available, propose retraining on recent data, validate the candidate on held-out recent data and against the reference set, and require approval before promotion. Never promote a retrained model on drift evidence alone.

## Verification
A candidate model must beat the deployed model on held-out recent data by a meaningful margin and its regression on the reference set must be reported to the approver.
