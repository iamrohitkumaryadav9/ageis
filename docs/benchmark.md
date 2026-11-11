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

<!-- RESULTS -->
_Pending: filled from `results/aegis-ir/main/report.md` once the run completes._
