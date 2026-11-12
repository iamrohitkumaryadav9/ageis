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

**Detection.** A scenario is "detected" if monitoring opens an incident (any triggered finding of
medium severity or above). Precision, recall, F1 over faulty versus clean; false-positive rate on
clean controls; recall by fault, severity, difficulty and dataset. *Drift F1* is computed on the
drift monitor alone: positives are covariate-drift faults (F01, F02, F04), negatives are clean
controls. A method-comparison table reports the same quantities for raw KS/chi-square
significance, BH-adjusted significance, and each effect-size criterion alone.

**Detection delay.** For each faulty scenario of the first seed with at least 1,000 rows: three
clean 500-row windows followed by the faulty batch in 500-row windows; delay = index of the first
faulty window that raises an incident; false alarms on the clean prefix are counted. ADWIN and
Page-Hinkley delays are reported in labelled rows.

**Diagnosis** (on detected faulty scenarios; an "end-to-end" variant counts missed detections as
wrong). Top-1 accuracy and Recall@3 of the ranked root causes; *evidence coverage* = share of
anomaly evidence cited by the leading hypothesis; *contradictory evidence handling* = share of
diagnoses with a rival hypothesis in which contradicting evidence is cited; *unsupported claim
rate* = hypotheses citing no existing anomaly evidence / hypotheses proposed (for C and D this is
counted before validation removes them); fabricated evidence ids; causal over-claims.

**Remediation** (B and D). Validation success rate over validations run; regression failure
rate; *supported-fault validation success* = detected scenarios of faults with an automatable
remediation (F03, F04, F06, F07, F08, F10) that end with the expected action validated. For every
validated proposal the scorer simulates the approval, executes, and checks that the incident
batch is healthy under the new state and that rollback restores the original state. *Invalid
promotions* = approval requested without a passing validation. *Unauthorized actions* = approval
requested for a non-state-changing proposal, or any tool executed outside its allowlist.

**Agent reliability.** Diagnostic tool calls per investigation; LLM tool requests rejected;
structured-output failure rate (failed LLM requests / requests, after the single allowed retry);
timeouts; completion rate; p50/p95 latency.

**Resources.** A sampler thread records, once a second, the RSS of the benchmark process and of
the Ollama processes, and system CPU. Peak and mean are reported per run.

Proportions carry 95% Wilson intervals.

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
uv run python -m aegis.evaluation.run --name main              # A and B, full grid, ~10 minutes
scripts/ollama_serve.sh &                                     # then: ollama pull qwen3:4b
uv run python -m aegis.evaluation.run --name main --llm        # adds C and D on the subset (resumable)
uv run python -m aegis.evaluation.report --name main
```

The run is resumable: finished scenario/configuration pairs are skipped.

## Results

Run `main`, 2026-11-11, on the build machine (Intel i5-10210U, 4 cores / 8 threads, 7.7 GB RAM,
no GPU, WSL2). 170 scenarios: 120 faulty (10 faults x 3 severities x 2 datasets x 2 evaluation
seeds) and 50 clean controls. The full generated report, with every table, is
[results/aegis-ir/main/report.md](../results/aegis-ir/main/report.md); the per-scenario rows are in
`scenarios.jsonl` next to it. The numbers below are copied from that report.

### Engineering targets

| Target | Goal | Observed | Met |
|---|---|---|---|
| Drift detection F1 (F01/F02/F04 vs clean controls) | >= 0.90 | 0.957 | yes |
| Root-cause Recall@3 (configuration B, full grid) | >= 0.80 | 0.975 | yes |
| Clean-control false-positive rate | <= 0.05 | 0.000 (0/50, 95% CI 0.00-0.07) | yes |
| Unauthorized tool executions | 0 | 0 | yes |
| Invalid promotions | 0 | 0 | yes |
| Validation success on supported faults | >= 0.70 | 0.931 (67/72) | yes |

With 50 clean controls, zero false positives bounds the true rate below about 7%, not at zero.

### Detection

Precision 1.000, recall 0.992 (119/120). The one miss is a low-severity categorical frequency shift
on `education` in a 3,000-row bank batch: significant on no test after Benjamini-Hochberg
adjustment. Requiring significance *and* a material effect size beat every single-test rule on
the same data (F1 0.957 against 0.857 for PSI alone and 0.767 for an unadjusted KS test).

In the windowed stream study, 47 of 54 faults were flagged in the first faulty 500-row window, 2
were never flagged, and no clean-prefix window (0/126) raised an alarm.

### Diagnosis and the ablation

Full grid, deterministic configurations (119 detected faults):

| | A: alerts only | B: diagnostics + rules |
|---|---|---|
| Root-cause Top-1 | 0.815 (97/119, CI 0.74-0.88) | 0.966 (115/119, CI 0.92-0.99) |
| Recall@3 | 0.950 | 0.975 |
| Contradicting evidence cited when a rival exists | 0/21 | 29/31 |
| Investigation time p50 / p95 | under 1 ms | 0.93 s / 4.2 s |

All four configurations on the same 20 scenarios (one per dataset x fault):

| | A | B | C: local LLM alone | D: full AEGIS |
|---|---|---|---|---|
| Root-cause Top-1 | 17/20 | 19/20 | 11/20 | 19/20 |
| Recall@3 | 19/20 | 19/20 | 14/20 | 19/20 |
| Investigations that produced a diagnosis | 20/20 | 20/20 | 17/20 | 20/20 |
| LLM claims dropped as unsupported | - | - | 6/31 | 13/30 |
| Fabricated evidence ids stripped | - | - | 9 | 17 |
| Investigation time p50 / p95 | under 1 ms | 0.87 s / 4.3 s | 79 s / 194 s | 84 s / 125 s |

What this does and does not show:

- **Rules over raw alerts (A -> B) is the supported result**: +15 points Top-1 on 119 scenarios,
  with intervals that do not overlap. The gain is where symptoms overlap - scaling against shift,
  and a performance drop that an input change explains.
- **The LLM added no diagnostic accuracy (B = D, 19/20).** D's ranking is anchored to the rule
  score (the LLM may move it by at most 0.15), so this is by construction as much as by
  measurement. What the LLM contributes is tool selection and the written explanation, at a cost
  of about 84 s per investigation on this CPU.
- **The 4 GB model alone is unreliable (C, 11/20)**: it named concept drift as the top cause in
  most of its misses with a stated confidence of 0.95, and returned unusable output three times.
  With n=20 the interval is 0.34-0.74; it is clearly worse than B, but the size of the gap is not
  pinned down.
- **Validation is doing real work**: in D, 13 of the model's 30 ranked claims were dropped for
  lacking anomaly evidence and 17 cited evidence ids did not exist. None reached the output. These
  rows count what the model *attempted*, not what the system reported.
- B's four misses: three low-severity scaling errors and one medium leakage case, each ranked
  below concept drift or a plain shift.

### Remediation (configuration B, full grid)

Of 119 investigations: 68 reached a human approval request, 19 were rejected by policy (17 after a
failed sandbox validation, 2 retraining proposals before validation), and 32 were deferred with no
action proposed. Sandbox validation passed for 68 of 85 proposals; 16 of the 17 failures are
retraining candidates that did not pass (12 of them for the relationship-reversal fault, where a
model retrained on the reversed batch should not be promoted), and one is a scale fix. An executed fix cleared the incident in 67 of 68 cases and rollback restored the
previous deployment state in 68 of 68. No approval request was raised on clean data.

### Resources

The LLM run took 68 minutes and peaked at 7.1 GB of system RAM in use on a 7.7 GB machine, with
swap in use: this machine is at the lower limit for `qwen3:4b`. The deterministic run took 6.6
minutes and about 0.4 GB. See [performance.md](performance.md).

### Two corrections made after the first run

Reported because they change what the numbers mean:

1. The sandbox's "performance not degraded" check compared AUC before and after a fix with no
   allowance for sampling noise, and rejected a correct scale fix over a 0.004 dip (found by the
   demo on a non-benchmark seed). It now allows two Hanley-McNeil standard errors. Configuration B
   was re-run in full with the corrected check; one validation changed from fail to pass (67 -> 68
   of 85). Configuration D's remediation rows predate the change; all of D's non-retraining
   validations had already passed and the allowance can only turn a fail into a pass, so its
   figures are unaffected.
2. The resource sampler matched process names and missed the `llama-server` child that holds the
   model, so the Ollama memory figure for the LLM run is withheld rather than reported wrong. The
   sampler now follows child processes; a separately measured figure is in `performance.md`.

A model warm-up step (loading the model on its own time budget before the first request) was also
added after the LLM run. It does not change any model output; D's latencies above were measured
with the model already loaded for all but the first scenario.
