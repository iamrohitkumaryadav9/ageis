---
id: RB-005
title: Schema changes and data contract violations
version: 1.1
applies_to: [schema_change]
---
## Symptoms
A contracted column is missing, renamed, or arrives with a different type, for example a numeric column delivered as text with unit suffixes. The model cannot score the batch, so performance metrics are unavailable rather than normal.

## Triage
A contract violation is an observed fact and needs no statistical judgement. Determine whether the producer shipped an intentional schema version or an accidental change. Look for an unexpected new column that appeared together with a missing one: that pattern is a rename.

## Diagnosis
List missing columns, unexpected columns and type mismatches. Do not attempt to coerce or guess a mapping automatically; a silent coercion turns a loud failure into a quiet wrong answer.

## Remediation
Reject the incompatible batch by quarantining it and keep serving from the last conforming data. Ask the producer to restore the contracted schema or to version the change so the contract and model can be updated deliberately.

## Verification
The next batch must pass contract validation with zero missing columns and zero type mismatches before the quarantine is lifted.
