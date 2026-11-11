---
id: RB-006
title: Model artifact and version mismatches
version: 1.0
applies_to: [artifact_mismatch]
---
## Symptoms
The hash of the artifact in serving differs from the hash recorded in the model registry for the declared version. The prediction distribution may shift without any change in the inputs.

## Triage
Treat this as an integrity incident. Establish whether the served bytes equal another registered version, which indicates a deployment of the wrong version, or match nothing in the registry, which indicates an unreviewed or corrupted file. Never load an artifact that matches no registry record.

## Diagnosis
If the served artifact is a known version, compare it with the declared version on reference and recent labelled data to quantify the impact. If it is unknown, impact cannot be measured safely; assume predictions made since deployment are untrusted.

## Remediation
Restore the registered artifact for the declared version and verify its hash. If the declared version itself is suspect, roll back to the previously validated version. Both are state-changing actions and need human approval.

## Verification
After restoration the artifact hash must equal the registry record and the prediction distribution on current data must match the reference.
