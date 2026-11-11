---
id: RB-001
title: Data quality incidents (missing values, duplicates, invalid ranges)
version: 1.2
applies_to: [missing_values, duplicate_records]
---
## Symptoms
A missing-value rate, duplicate-row rate or out-of-range rate rises above its reference level. Non-null values of the affected feature usually keep their reference distribution, so drift tests stay quiet while quality checks fire.

## Triage
Confirm which features are affected and whether the rise is confined to one source, partition or time window. A missing-value spike in a single feature points at an upstream field that stopped being populated. Duplicates that are exact copies across all columns point at a replayed or double-delivered batch rather than repeat customers.

## Diagnosis
Compare the missing rate with the contract's nullable rate, not with zero: some features are legitimately sparse. For duplicates compare with the size-matched reference expectation, because the chance of an exact repeat grows with batch size. Check whether model performance on the labelled portion moved; imputation hides missing values from the model but shifts predictions toward the median profile.

## Remediation
For duplicates, enable deduplication in ingestion and re-run monitoring on the deduplicated batch. For a missing-value spike, quarantine the batch so it is not used for decisions or training, and ask the owning team to backfill the field. Do not retrain on a batch with a missing-value spike: the model would learn the outage.

## Verification
After the fix the quality rate must return to its reference level on a fresh batch and no new finding may appear. Keep the quarantine until a clean batch has been observed.
