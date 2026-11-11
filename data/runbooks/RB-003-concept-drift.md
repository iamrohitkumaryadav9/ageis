---
id: RB-003
title: Concept drift and unexplained performance degradation
version: 1.3
applies_to: [concept_drift]
---
## Symptoms
Error rate, F1 or ROC-AUC degrades and ADWIN or Page-Hinkley signal a change in the error stream, while feature distributions and data quality look like the reference. The relationship between features and the label changed rather than the features themselves.

## Triage
Rule out the cheaper explanations first: wrong model artifact in serving, a schema or scaling change, a labelling problem. Concept drift is the diagnosis that remains when the inputs are normal and the model is the registered one.

## Diagnosis
Check whether the error rise is uniform or concentrated in a segment; a concentrated rise suggests a policy or market change affecting that slice. Check whether the sign of feature-label associations flipped. Verify label quality for the period: mislabelled outcomes look exactly like concept drift to every detector.

## Remediation
Retraining on recent labelled data is the usual response, but only when a candidate actually improves on held-out recent data. If the candidate does not improve, the labels are probably noisy or the sample is too small; continue the investigation instead of promoting. Retraining and promotion are high-risk actions and always need human approval.

## Verification
Compare candidate and deployed model on a held-out slice of recent data that the candidate was not trained on. Report the expected regression on the old reference set so the approver can see the trade.
