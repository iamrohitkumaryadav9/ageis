# Remediation lifecycle and its accounting

How a remediation moves from proposal to rollback, which records exist at each step, and how
every count in the benchmark report is derived. Code: `aegis/remediation/reconciliation.py`,
`aegis/remediation/lifecycle.py`, `aegis/evaluation/benchmark.py`.

## The lifecycle

```
PROPOSED
  -> POLICY_DECIDED (pre_validation)     DEFER | INVESTIGATE_MORE | REJECT | PROPOSE_ACTION
       -> VALIDATION                     passed | failed          (only after PROPOSE_ACTION)
            -> POLICY_DECIDED (post_validation)   REJECT | REQUEST_HUMAN_APPROVAL
                 -> APPROVAL_REQUESTED
                      -> APPROVAL_DECIDED          approved | rejected      (a human, via the API)
                           -> EXECUTED             succeeded | failed
                                -> VERIFIED        resolved | unresolved
                                     -> ROLLED_BACK succeeded | failed      (optional)
```

Every proposal ends in exactly one terminal status:

| Terminal status | Meaning |
|---|---|
| `deferred_no_action` | the proposal was `no_action`; policy `DEFER`. Nothing to validate |
| `investigate_more` | evidence requirements not met; policy `INVESTIGATE_MORE` |
| `rejected_by_policy_before_validation` | policy `REJECT` on its first pass (score not positive) |
| `rejected_validation_failed` | sandbox validation failed |
| `rejected_by_policy_after_validation` | validation passed, policy still rejected |
| `pending_approval` | waiting for a human. Open, not failed, in a live system |
| `rejected_by_approver` | a human said no |
| `execution_failed` | approved, execution did not succeed |
| `executed_resolved` / `executed_unresolved` | executed; the incident batch is / is not healthy |
| `rolled_back` / `rollback_failed` | executed, then rolled back |
| `cancelled` | cancelled before completion |

Two statuses are never legitimate and are reported as reconciliation failures:
`executed_unverified` (an execution nobody checked) and `unclassifiable`.

## Identifiers

Each stage has its own identifier because each is a different record. They are not merged into
one id.

| Id | Names | Benchmark form | Database |
|---|---|---|---|
| `scenario_id` | the benchmark scenario | `SCN-<hash>` | not applicable |
| `incident_id` | the incident | `INC-<scenario>-<config>` | `incidents.id` |
| `proposal_id` | one remediation proposal | `PRP-<scenario>-<config>` | `remediations.id` |
| `policy_decision_id` | one policy evaluation; up to two per proposal | `POL-...-pre`, `POL-...-post` | not a row; stored as JSON on the remediation |
| `validation_id` | one sandbox validation | `VAL-...` | derived `VAL-<remediation id>` |
| `approval_id` | one approval request and its decision | `APR-...` | derived `APR-<remediation id>`; `approvals` row is unique per remediation |
| `execution_id` | one execution | `EXE-...` | derived `EXE-<remediation id>`; the deployment revision is in the event detail |
| `rollback_id` | one rollback | `RBK-...` | derived `RBK-<remediation id>` |

Events are linked by `proposal_id`, and an event names the earlier record it depends on (an
execution carries its `approval_id`, a rollback its `execution_id`). A reference to a record that
does not exist in the same lifecycle is a `dangling_reference` failure.

Events carry two honesty flags. `simulated`: no human, API or database was involved (benchmark
approvals, executions and rollbacks). `derived`: the event was reconstructed from summary fields
of a run that predates event recording, not captured live.

## Reconciliation

`reconcile(events)` accounts for every record and returns a JSON-serialisable report.
`reconciled` is true only when there are no failures. Nothing is excluded from a count for being
inconsistent; it is counted and reported.

| Failure code | Raised when |
|---|---|
| `malformed_event` | a record does not fit the schema |
| `duplicate_event_id` | the same event id appears twice |
| `duplicate_stage` | a stage recorded more often than it can occur (two validations; two policy decisions in one phase) |
| `missing_link` | a stage without its prerequisite: validation without permission, approval request without a passed validation, execution without approval, rollback without execution, any event without a `PROPOSED` |
| `out_of_order` | a stage recorded before its prerequisite |
| `dangling_reference` | an event references an id not present in its lifecycle |
| `missing_stage_id`, `invalid_outcome` | a stage without its own id, or with an outcome it cannot have |
| `inconsistent_incident`, `inconsistent_scenario` | events of one proposal name different incidents or scenarios |
| `incomplete_lifecycle` | the lifecycle stops mid-way: never decided, validated but never decided, approved but never executed, executed but never verified; or still pending when the run requires closure |
| `proposal_count_mismatch` | fewer or more lifecycles than investigations (a crashed investigation shows up here) |
| `accounting_identity_violated` | one of the identities below does not hold |

Identities checked on every reconciliation:

1. every proposal has exactly one terminal status;
2. proposals = actionable + `no_action` + `investigate_further`;
3. validation attempts = passed + failed;
4. approval requests = approved + rejected + still pending;
5. execution attempts = approvals granted;
6. actionable proposals = rejected before validation + investigate-more + validation attempts.

## Metric definitions

Every rate is stored with its numerator, denominator and definition.

| Metric | Numerator | Denominator |
|---|---|---|
| `validation_success_rate` | validation attempts that passed | all validation attempts |
| `regression_failure_rate` | validation attempts with a failed regression check | all validation attempts |
| `supported_fault_remediation_success` | detected scenarios of a fault with an automatable fix where the expected action was proposed and its validation passed | all detected scenarios of such faults (F03, F04, F06, F07, F08, F10) |
| `execution_resolution_rate` | executed fixes after which the incident batch was healthy | executed fixes |
| `rollback_success_rate` | rollbacks that restored the previous state | rollback attempts |

`supported_fault_remediation_success` was previously called "supported-fault validation success".
The old name suggested its denominator was validation attempts. It is not: it counts scenarios,
and a scenario fails if the wrong action was proposed or nothing was validated. That is why its
denominator (72) differs from the validation attempts (85).

## Run `main`, reconciled

Run `main` predates event recording, so its 748 events are derived from the recorded summary
fields of each row. Source: `results/aegis-ir/main/lifecycle_reconciliation.json`.

| Stage | Count | Derivation |
|---|---|---|
| Faulty scenarios detected and investigated (configuration B) | 119 | 120 faulty, 1 undetected |
| Proposals | 119 | one per investigation; 0 crashed |
| `no_action`, policy `DEFER` | 32 | |
| Actionable proposals | **87** | 119 - 32 |
| Rejected by policy before validation | 2 | |
| Validation attempts | **85** | 87 - 2 |
| Validation failed | 17 | |
| Validation passed, approval requested | 68 | 85 - 17 |
| Executed (simulated approval) | 68 | |
| Incident resolved after execution | 67 | |
| Rollback attempts / succeeded | 68 / 68 | |

**Why 87 proposals and 85 validations.** Two `retrain_candidate` proposals never reached the
sandbox: an F04 and an F07 scenario, both misdiagnosed as concept drift with evidence fit 0.65.
The policy returned `REJECT` on its first pass, and the workflow only validates after
`PROPOSE_ACTION`. The reason given by the code for a first-pass `REJECT` is a non-positive score
(0.65 x 0.55 efficacy, minus risk, minus 0.20 cost). The historical rows did not store policy
reasons, so for `main` that reason is inferred from the code; runs made after this change record
the reasons in the event detail (see the benchmark document for the new run).

**The other numbers.** 68/85 is validation attempts that passed. 67/72 is the supported-fault
metric with its different denominator: of 72 detected supported-fault scenarios, five failed:
three were given `retrain_candidate` instead of the expected fix, one was given `no_action`, and
one correct scale fix failed validation. 67/68 is executions that resolved the incident. 68/68
is rollbacks over rollback attempts.

## What is simulated and what is not

In the benchmark, the path up to and including the approval *request* is the real workflow
(LangGraph nodes, policy engine, sandbox validation). The approval *decision*, the execution and
the rollback are simulated in-process: the benchmark builds the new deployment state, re-runs
monitoring to see whether the incident clears, and checks that restoring the base state
reproduces the original incident. No HTTP request, database row or human is involved.

The real path is exercised by the end-to-end tests (`tests/e2e`), which drive approval, execution
and rollback through the API and PostgreSQL and then call
`GET /remediation-lifecycle/reconciliation`, which builds lifecycle events from the append-only
audit trail and reconciles them with the same code. Denied attempts (an operator trying to
approve) are access-control records, not lifecycle stages; the endpoint reports how many it
excluded. The audit trail does not record policy decisions as separate rows, so that
reconciliation runs without the policy-event requirements.

## Tests

`tests/unit/test_lifecycle_reconciliation.py` (35 tests): duplicates, missing links, rejected and
deferred proposals, validation failures, approval rejection, execution and rollback failures,
incomplete lifecycles, crashed investigations, the audit-trail adapter, and a regression test
that pins the 87-versus-85 explanation to the historical data.
