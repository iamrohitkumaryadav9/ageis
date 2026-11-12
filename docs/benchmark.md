# AEGIS-IR: ML Incident Diagnosis and Recovery benchmark

AEGIS-IR measures four things on reproducible, labelled incidents: whether an incident is
detected, whether its root cause is ranked correctly, whether the proposed remediation survives
validation, and whether the agent stays inside its bounds while doing so.

Results are generated, not written: `results/aegis-ir/<run>/report.md` is produced by
`python -m aegis.evaluation.report` from the raw per-scenario records in `scenarios.jsonl`.
The numbers quoted at the end of this document are copied from that report for the run named
there.

## Design

| Factor | Levels |
|---|---|
| Dataset | Adult, Bank Marketing |
| Fault | F01-F10 (below) |
| Severity | low, medium, high |
| Seed | 201, 202 |
| Batch size | 1,000 or 3,000 rows (alternating) |
| Clean controls | 25 per dataset |

2 x 10 x 3 x 2 = 120 faulty scenarios + 50 clean controls = **170 scenarios**.

| ID | Fault | How it is injected | Severity scales |
|---|---|---|---|
| F01 | Numerical mean shift | add k reference standard deviations to one numeric feature | k = 0.35 / 0.7 / 1.5 |
| F02 | Categorical frequency shift | overwrite a fraction of one categorical feature with one category | 15% / 30% / 60% |
| F03 | Missing-value spike | blank one or two features at random | 12% / 30% / 60% |
| F04 | Feature scaling error | multiply one numeric feature by a constant | x2.2 or x0.45 / x12, x60, x0.1 / x1000, x100, x0.001 |
| F05 | Target relationship reversal | invert labels for a fraction of rows; features untouched | 30% / 55% / 90% |
| F06 | Duplicate records | overwrite rows with exact copies of others | 10% / 25% / 50% |
| F07 | Feature leakage | permute one numeric feature so its order follows the label; marginal preserved exactly | 80% / 92% / 100% of rows aligned |
| F08 | Schema change | numeric column re-typed to text / column renamed / two columns dropped | by kind |
| F09 | Delayed labels | most recent rows have no label | 50% / 80% / 100% |
| F10 | Model artifact mismatch | serving file is another registered version / an unregistered retrain / truncated bytes | by kind |

Difficulty is derived: `hard` = low severity, or medium severity on a 1,000-row batch.

### Reproducibility and leakage control

- A scenario is a pure function of (dataset, fault, severity, seed, batch size). The RNG seed is
  a hash of those five values.
- Scenario ids are hashes of the same tuple, so an id reveals nothing.
- The public half (`manifest.json`: id, dataset, row count) and the scorer's half
  (`ground_truth.json`) are separate files. Systems under test are given a `MonitoringInput`
  built from the batch only. A test asserts that LLM prompts contain no scenario id, fault id or
  ground-truth field.
- Batches are sampled from the `production` split, which the model never saw in training,
  threshold selection or testing.
- **Development/evaluation split**: monitoring thresholds were tuned on seeds 101 and 102. The
  runner refuses those seeds. Reported results use 201 and 202.
- The benchmark works on a temporary copy of the model registry, so candidate models registered
  during validation never touch the real one.

## Configurations

| | Detection | Diagnosis | Remediation |
|---|---|---|---|
| **A** monitoring only | monitors | each alert votes for the cause it is named after; ranked by worst severity | none |
| **B** monitoring + rules | monitors | deterministic diagnostics + rule-based ranking | propose, policy, sandbox validation |
| **C** monitoring + local LLM | monitors | one LLM ranking request over monitoring evidence; no tools, rules or runbooks | none |
| **D** full AEGIS | monitors | LangGraph agent: LLM tool plan, rules, LLM re-ranking with validation, runbook retrieval | propose, policy, sandbox validation |

All four share the monitoring engine, so detection is reported once and the ablation isolates
what happens after an incident is opened. A and B run on all 170 scenarios. C and D run on a
stratified subset (one scenario per dataset x fault, severity rotated: 20 scenarios), because one
LLM scenario costs several minutes of CPU. On that subset A and B are reported again so all four
are compared on identical scenarios. `--llm-all` runs C and D on everything if you have the hours.

In configuration D the operator-facing explanation request is switched off: it affects no
scored quantity and costs about a minute per scenario.

## Metrics

Every rate is stored with its numerator, denominator and definition (`summary.json`), and every
metric has exactly one name.

**Detection: two metrics, never interchangeable.**

| | `overall_fault_detection_*` | `drift_detection_*` |
|---|---|---|
| Question | was an incident opened? | did the drift monitor flag a distribution change? |
| Positive scenarios | every injected-fault scenario, all ten categories | only covariate-drift scenarios: F01, F02, F04 |
| Negative scenarios | clean controls | the same clean controls |
| A detection is | `incident_detected`, raised by any monitor | `drift_flagged`: the drift monitor itself triggered |

They differ in population and in what counts as a detection. A scaling error caught by a range
check counts for the first and not for the second. Each is reported with its population, TP, FP,
TN, FN, precision, recall, false-positive rate (95% Wilson intervals) and F1 (95% percentile
bootstrap over scenarios, 2,000 resamples, fixed seed). The method-comparison table (raw KS and
chi-square significance, BH-adjusted significance, each effect-size criterion alone) uses the
`drift_detection` population.

**Detection delay.** For each faulty scenario of the first seed with at least 1,000 rows: three
clean 500-row windows followed by the faulty batch in 500-row windows; delay = index of the first
faulty window that raises an incident; false alarms on the clean prefix are counted. ADWIN and
Page-Hinkley delays are reported in labelled rows.

**Diagnosis** (on detected faulty scenarios; an "end-to-end" variant counts missed detections as
wrong). Top-1 accuracy and Recall@3 of the ranked root causes; *evidence coverage* = share of
anomaly evidence cited by the leading hypothesis; *contradictory evidence handling* = diagnoses
citing contradicting evidence / diagnoses with a rival hypothesis.

**Claim validation** (C and D). Counters for what the model generated and what happened to it:
causal claims generated / supported / rejected / escalated, unsupported and contradictory claims,
fabricated evidence ids generated and stripped, and three "escaped" counters from a re-scan of
the final output. Definitions and limits: [evidence-and-claim-validation.md](evidence-and-claim-validation.md).
A run without these counters reports them as unavailable, not as zero.

**Remediation** (B and D). Definitions and the full accounting:
[remediation-lifecycle.md](remediation-lifecycle.md).

| Metric | Numerator | Denominator |
|---|---|---|
| `validation_success_rate` | validation attempts that passed | validation attempts |
| `regression_failure_rate` | validation attempts with a failed regression check | validation attempts |
| `supported_fault_remediation_success` | detected supported-fault scenarios where the expected action was proposed and validated | detected scenarios of faults with an automatable fix (F03, F04, F06, F07, F08, F10) |
| `execution_resolution_rate` | executed fixes after which the incident batch was healthy | executed fixes |
| `rollback_success_rate` | rollbacks that restored the previous state | rollback attempts |

For every validated proposal the benchmark *simulates* the approval, the execution and the
rollback in-process (no human, API or database) and checks that the incident batch is healthy
under the new state and that rollback restores the original. Each proposal emits lifecycle events
that are reconciled; a reconciliation failure is printed in the report. *Invalid promotions* =
approval requested without a passing validation. *Unauthorized actions* = approval requested for
a non-state-changing proposal, or any tool executed outside its allowlist.

**Agent reliability.** Diagnostic tool calls per investigation; LLM tool requests rejected;
structured-output failures / LLM requests (after the single allowed retry); timeouts; completion
rate; p50/p95 latency.

**Resources.** A process-tree sampler records, every 0.5 s, the resident memory of the benchmark
process, of its child processes, of the Ollama server and of the Ollama runner separately, plus
system memory, swap and CPU. Each invocation is one record with its own id; records are never
merged. See [performance.md](performance.md).

## Limitations

1. **Shared authorship.** The fault injectors and the rule engine were written by the same
   person with the same taxonomy in mind. High rule-based accuracy shows internal consistency;
   it is not evidence of performance on incidents the rules were not designed around.
2. **Single, synthetic faults.** Real incidents are compound, partial and slower. No scenario
   contains two simultaneous faults.
3. **Small LLM subset.** Twenty scenarios; the confidence intervals are wide and differences of
   one or two scenarios are not meaningful.
4. **F10 is detected by construction** (hash comparison), and F08 nearly so (contract check).
5. **Quarantine counts as a remediation.** It is always "valid" when its grounds are present, so
   it lifts the supported-fault success rate.
6. **One machine, one run.** Latency numbers are from a single i5-10210U laptop under WSL2 and
   include whatever else the machine was doing.
7. **Two tabular datasets** with one model family each.

## Reproducing

```bash
make data train                                   # datasets, splits, baseline models
uv run python -m aegis.evaluation.run --quick --name smoke     # ~2 minutes
uv run python -m aegis.evaluation.run --name myrun             # A and B, full grid, ~7 minutes
scripts/ollama_serve.sh &                                     # then: ollama pull qwen3:4b
uv run python -m aegis.evaluation.run --name myrun --llm       # adds C and D on the subset, ~80 minutes
uv run python -m aegis.evaluation.report --name myrun
```

The run is resumable: finished scenario/configuration pairs are skipped. The two invocations are
measured separately: the second executes only C and D, so its resource record is the LLM
configurations alone. Use a new run name; `main` and `v2` are committed results.

## Results

Two committed runs on the same machine, the same 170 scenarios and the same evaluation seeds
(201, 202): 120 faulty scenarios (10 faults x 3 severities x 2 datasets x 2 seeds) and 50 clean
controls.

| Run | Date | What it is | Generated report |
|---|---|---|---|
| `main` | 2026-11-11 | the historical run | [results/aegis-ir/main/report.md](../results/aegis-ir/main/report.md) |
| `v2` | 2026-11-12 | re-run with lifecycle events, claim counters and the process-tree sampler | [results/aegis-ir/v2/report.md](../results/aegis-ir/v2/report.md) |

Nothing in `main`'s raw file was edited; its report was regenerated from the unchanged
`scenarios.jsonl` (SHA-256 pinned by a test) so that it uses the explicit metric names. The
files as originally generated are in `results/aegis-ir/main/history/`.

**Reproducibility of the deterministic part**: `v2` reproduced `main` exactly. All 170 detection
outcomes, and every configuration A and B outcome (ranked causes, action, policy outcome,
validation result, execution and rollback result) are identical.

Machine: Intel i5-10210U, 4 physical cores / 8 logical, 7.65 GiB visible to the WSL2 guest (the
host has 15.8 GiB installed), no GPU.

### Detection (identical in `main` and `v2`)

| | `overall_fault_detection` | `drift_detection` |
|---|---|---|
| Fault categories | F01-F10 | F01, F02, F04 |
| Positive scenarios | 120 | 36 |
| Clean scenarios | 50 | 50 |
| TP / FP / TN / FN | 119 / 0 / 50 / 1 | 33 / 0 / 50 / 3 |
| Precision | 1.000 (119/119, CI 0.97-1.00) | 1.000 (33/33, CI 0.90-1.00) |
| Recall | 0.992 (119/120, CI 0.95-1.00) | 0.917 (33/36, CI 0.78-0.97) |
| False-positive rate | 0.000 (0/50, CI 0.00-0.07) | 0.000 (0/50, CI 0.00-0.07) |
| F1 | `overall_fault_detection_f1` = **0.9958** (bootstrap CI 0.987-1.000) | `drift_detection_f1` = **0.9565** (bootstrap CI 0.892-1.000) |

In words: AEGIS achieved 0.996 overall fault-detection F1 across ten injected ML failure
categories and 0.9565 drift-detection F1 across numerical, categorical, and scaling drift
scenarios. The second number is lower because three drift-category scenarios were not flagged by
the drift monitor itself; two of those three still opened an incident through another monitor,
which is why only one fault is missed overall. The one overall miss is a low-severity categorical
frequency shift on `education` in a 3,000-row bank batch.

With 50 clean controls, zero false positives bounds the true rate below about 7%, not at zero.

Within the `drift_detection` population, requiring significance *and* a material effect size
beat every single-test rule on the same data (drift-detection F1 0.957 against 0.857 for PSI
alone and 0.767 for an unadjusted KS test).

In the windowed stream study, 47 of 54 faults were flagged in the first faulty 500-row window, 2
were never flagged, and no clean-prefix window (0/126) raised an alarm.

### Engineering targets

| Target (metric name) | Goal | Observed (both runs) | Met |
|---|---|---|---|
| `drift_detection_f1` | >= 0.90 | 0.9565 | yes |
| `root_cause_recall_at_3` (configuration B, full grid) | >= 0.80 | 0.975 (116/119) | yes |
| `clean_control_false_positive_rate` | <= 0.05 | 0.000 (0/50, CI 0.00-0.07) | yes |
| `unauthorized_tool_executions` | 0 | 0 | yes |
| `invalid_promotions` | 0 | 0 | yes |
| `supported_fault_remediation_success` | >= 0.70 | 0.931 (67/72) | yes |

### Diagnosis and the ablation

Full grid, deterministic configurations, 119 detected faults (identical in both runs):

| | A: alerts only | B: diagnostics + rules |
|---|---|---|
| Root-cause Top-1 | 0.815 (97/119, CI 0.74-0.88) | 0.966 (115/119, CI 0.92-0.99) |
| Recall@3 | 0.950 (113/119) | 0.975 (116/119) |

All four configurations on the same 20 scenarios (one per dataset x fault). The LLM
configurations were run once per run:

| | A | B | C: local LLM alone | D: full AEGIS |
|---|---|---|---|---|
| Top-1, `main` | 17/20 | 19/20 | 11/20 | 19/20 |
| Top-1, `v2` | 17/20 | 19/20 | 11/20 | 19/20 |
| Recall@3, `main` / `v2` | 19/20 / 19/20 | 19/20 / 19/20 | 14/20 / 12/20 | 19/20 / 19/20 |
| Investigations producing a diagnosis, `v2` | 20/20 | 20/20 | 17/20 | 20/20 |
| Structured-output failures / LLM requests, `v2` | - | - | 3/20 | 1/60 |
| Investigation time p50 / p95, `v2` | under 1 ms | 0.54 s / 2.8 s | 85 s / 197 s | 139 s / 178 s |

What this does and does not show:

- **Rules over raw alerts (A -> B) is the supported result**: +15 points Top-1 on 119 scenarios,
  with intervals that do not overlap.
- **The LLM added no diagnostic accuracy (B = D, 19/20, in both runs).** In `v2` the leading
  cause chosen by D equalled B's in all 20 scenarios. D's ranking is anchored to the rule score
  (the LLM may move it by at most 0.15), so this is by construction as much as by measurement.
- **The 4 GB model alone is unreliable (C, 11/20 in both runs)**, with three unusable outputs in
  each. n=20, so the interval is 0.34-0.74. Two runs agreeing on 11/20 is two samples, not a
  proof of stability.
- B and D share their single miss: a low-severity scaling error ranked as concept drift.
- **D's latency is not comparable between the runs.** `main` ran D without the explanation
  request (p50 84 s); `v2` includes it (p50 139 s, three LLM requests per investigation).

### Claim validation (`v2`; unavailable for `main`)

`main` did not store the model's text and recorded only a pattern-match count, so its claim
metrics are unavailable. Its "causal over-claims: 0" meant that one regular expression did not
match the ranking reasoning; it said nothing about explanations or about escapes.

`v2`, counters for configuration D (20 investigations, 60 LLM requests):

| Counter | C | D |
|---|---|---|
| `claims_evaluated` | 28 | 185 |
| `causal_claims_generated` | 0 | 7 |
| `causal_claims_supported` / `_rejected` / `_escalated` | 0 / 0 / 0 | 0 / 7 / 0 |
| `unsupported_claims_generated` = `_rejected` | 14 | 11 |
| `contradictory_claims_generated` = `_rejected` | 0 | 2 |
| `fabricated_evidence_ids_generated` = `_stripped` | 21 | 19 |
| `irrelevant_evidence_ids_stripped` | 5 | 0 |
| `causal_claims_escaped` | 0 | 0 |
| `unsupported_claims_escaped` | 0 | 0 |
| `fabricated_evidence_ids_escaped` | 0 | 0 |

Rates for D: causal claims not supported 7/7; causal claims escaped 0/7 (CI 0.00-0.35);
unsupported claims escaped 0/11; fabricated ids escaped 0/19; claims with causal language 7/185.

Read with these facts, all found by reading the stored model text:

1. **"Escaped: 0" is a consistency check.** It says nothing the validator recognises reached the
   output. It cannot count an over-claim phrased outside the lexicon.
2. **The validator over-rejected in this run.** It discarded 9 of 20 explanations. Reading all
   nine: two contained a causal statement about the incident ("data contract violation due to
   schema change"); four were the workflow's own decision rationale ("Proposed action: no_action
   due to no performance loss", "Sandbox failed due to regression on reference"), which is not a
   claim about what caused the incident; three had an acceptable summary and were discarded
   because of an operator check, whose text this run did not store. So of the 7 "causal claims
   generated", 4 were false positives of the lexicon.
3. **The validator was corrected after this run**, and the run was not repeated (about 80
   minutes). Sentences labelled as workflow facts are no longer treated as causal unless they
   contain an explicit cause-word; an unsupported operator check is dropped on its own; raw
   checks are now stored. An offline re-evaluation of the 20 stored summaries with the corrected
   validator, calling no model, accepts 18 and rejects the same 2 genuine causal statements
   (`results/aegis-ir/v2/explanation_reevaluation.json`). That file is not part of the
   benchmark's counters, and the table above is what the run itself recorded.
4. **A validator defect was found by the escape counter and fixed before this run.** An earlier
   attempt at the LLM subset reported `unsupported_claims_escaped = 1`: a ranking item whose
   reasoning denied its own cause ("No evidence of scale change") was kept without supporting
   evidence. The attempt was stopped after three scenarios, the defect fixed with a regression
   test, and the LLM subset restarted from the beginning.

### Remediation (configuration B, full grid; identical in both runs)

| Stage | Count | Derivation |
|---|---|---|
| Investigations | 119 | detected faulty scenarios |
| Proposals | 119 | one per investigation, none crashed |
| No state change proposed (`no_action`), policy `DEFER` | 32 | |
| Actionable proposals | 87 | 119 - 32 |
| Rejected by policy before validation | 2 | score not positive |
| Validation attempts | 85 | 87 - 2 |
| Validation failed, policy `REJECT` | 17 | |
| Validation passed, approval requested | 68 | 85 - 17 |
| Executed (simulated approval) | 68 | |
| Incident resolved after execution | 67 | |
| Rollback attempts / succeeded | 68 / 68 | |

| Rate | Value |
|---|---|
| `validation_success_rate` | 0.800 (68/85) |
| `regression_failure_rate` | 0.165 (14/85) |
| `supported_fault_remediation_success` | 0.931 (67/72 detected supported-fault scenarios) |
| `execution_resolution_rate` | 0.985 (67/68) |
| `rollback_success_rate` | 1.000 (68/68) |

Both runs reconcile with zero failures: 748 lifecycle events for 119 proposals. In `v2` all 748
were recorded live (272 of them simulated: the approval decisions, executions, verifications and
rollbacks). In `main` they are derived from summary fields.

**87 proposals, 85 validations.** Two `retrain_candidate` proposals (an F04 and an F07 scenario,
both misdiagnosed as concept drift at evidence fit 0.65) were rejected by the policy on its first
pass and never validated. `v2` recorded the reason: "score -0.05 = benefit 0.36 - risk 0.21 -
cost 0.20 is not positive". For `main` this had been inferred from the code; the recorded reason
confirms it.

16 of the 17 validation failures are retraining candidates that did not pass (12 of them for the
relationship-reversal fault, where a model retrained on the reversed batch should not be
promoted); one is a scale fix. No approval request was raised on clean data.

Configuration D on the LLM subset (`v2`): 20 proposals = 6 no-action + 1 rejected before
validation + 13 validated (11 passed, 2 failed); 11 executed, 11 resolved, 11 rolled back;
reconciled.

### Resources

Separate invocations, separately measured; see [performance.md](performance.md) for the full
records and what they mean.

| Run | Configurations executed | Duration | Python peak | Ollama runner peak | System RAM used, peak | Swap, peak |
|---|---|---|---|---|---|---|
| `v2` deterministic (`RES-1f82695a9c2c`) | A x119, B x119 | 405 s | 398 MB | none observed | 2,303 MB | 856 MB (already in use at start) |
| `v2` LLM (`RES-580441f6aa56`) | C x20, D x20 | 4,926 s | 348 MB | 5,745 MB | 7,159 MB | 2,048 MB (full) |
| `main` deterministic (historical) | A, B | 396 s | 402 MB | not applicable | 1,660 MB | unavailable |
| `main` LLM (historical) | A-D | 4,068 s | 421 MB | unavailable (not captured) | 7,071 MB | unavailable |

### Changes of method between `main` and `v2`

1. Detection metrics are reported under explicit names with their populations. No value changed.
2. "Supported-fault validation success" is renamed `supported_fault_remediation_success`, because
   its denominator is scenarios and not validation attempts. No value changed.
3. `rollback_success_rate` is now over rollback attempts instead of executed fixes. The two
   coincide in the benchmark (every execution is rolled back). No value changed.
4. Lifecycle events are recorded and reconciled.
5. LLM text is validated claim by claim; raw text and counters are stored.
6. Configuration D includes the LLM explanation request, so its latency rose.
7. Resources are measured by the process-tree sampler, one record per invocation.

Corrections made to `main` on 2026-11-11, already described in its history: the sandbox's
"performance not degraded" check gained a sampling-noise allowance and configuration B was
re-run (one validation changed from fail to pass, 67 -> 68 of 85).
