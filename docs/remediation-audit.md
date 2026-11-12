# Remediation audit: five defects

Audit of the repository at commit `a471962`, written before any implementation change. Each
finding names the code or data it was read from. "Historical" means benchmark run `main`
(2026-11-11); its raw files are not edited by this work.

## Summary

| # | Problem | Root cause | Risk | Fix |
|---|---|---|---|---|
| 1 | Two F1 scores presented without their populations | One target key, `drift_f1`, and an unnamed "F1" in the report; no test protects either | Medium: a reader can quote the wrong number | Explicit metric names, a population block per metric, tests |
| 2 | Remediation counts do not visibly reconcile | Rates use four different denominators and none is printed; no lifecycle events are recorded | Medium: numbers are right but unauditable | Lifecycle event schema, reconciliation, named denominators |
| 3 | `causal_overclaims = 0` is uninterpretable | The counter is a regex match on one short field; nothing measures what was generated or what escaped | High: implies a causal check that does not exist | Deterministic claim validator with separate generated / rejected / escaped counters |
| 4 | Memory figures inconsistent and incomplete | GB-versus-GiB mix-up; sampler matched process names and missed the runner | Medium | Process-tree sampler, environment record, separate runs |
| 5 | Release checks incomplete | Never done: browser check, frontend tests, Compose `llm` profile, fallback model, tracking tests | Medium | Do each for real; one release-verification script |

## Problem 1: benchmark metric ambiguity

Read: `aegis/evaluation/report.py` (`detection_metrics`, `evaluate_targets`, `write_markdown`),
`results/aegis-ir/main/summary.json`, `README.md`, `docs/benchmark.md`, `apps/dashboard/src`.

How each number is computed today:

- **0.996**: `detection["overall"]`. Positives are all 120 faulty scenarios (ten fault
  categories), negatives the 50 clean controls. A scenario is a hit when the monitoring run sets
  `incident_detected`, by any monitor. TP 119, FN 1, FP 0, TN 50.
- **0.9565**: `detection["drift"]["methods"]["monitor"]`. Positives are only the 36 scenarios of
  F01, F02 and F04; negatives the same 50 clean controls. A hit is `drift_flagged`: the *drift
  monitor specifically* triggered, which is a stricter condition than "an incident was opened".
  TP 33, FN 3, FP 0, TN 50.

So the two differ in both population (120 against 36 positives) and in the positive condition
(any monitor against the drift monitor). Neither fact is stated next to the numbers.

Defects:

1. The engineering-target key is `drift_f1` and the report prints "F1 0.996" with no name.
   `summary.json` has no field describing either population.
2. TN is never reported; the drift entry has no confidence intervals and no counts of positives.
3. `docs/benchmark.md` and `README.md` label the 0.957 row correctly but never set it against the
   overall figure.
4. There are **no tests** for anything in `aegis/evaluation` (coverage 0%). The instruction to
   "run the existing metric tests" cannot be followed; the historical numbers will be reproduced
   by recomputing them from the raw file instead.
5. The dashboard shows no benchmark metrics at all, so it has no wrong label; it also means
   "evaluation results render" cannot currently be verified (see Problem 5).

A fact about the historical file that must stay visible: configuration B in `main` was re-run
after the sandbox noise-allowance fix. The pre-fix raw file exists only in a scratch directory.
It will be archived in the repository as `results/aegis-ir/main/history/`.

Fix: rename to `overall_fault_detection_*` and `drift_detection_*`; emit a `population` block
(definition, included faults, positives, clean controls, TP/FP/TN/FN, intervals) for both; keep
reading the old key names so the historical summary still loads; add tests that build synthetic
records where the two populations give different answers.

Verification: recompute from the unmodified `scenarios.jsonl` and compare with 0.996 / 0.9565.

## Problem 2: remediation lifecycle accounting

Read: `aegis/evaluation/benchmark.py` (`_remediation_row`, `_agent_result`),
`report.py::remediation_metrics`, `aegis/remediation/policy.py`, `lifecycle.py`,
`aegis/agents/graph.py`, `aegis/storage/models.py`, the raw rows.

Traced from the raw rows (configuration B, full grid):

| Stage | Count | Source |
|---|---|---|
| Faulty scenarios detected and investigated | 119 | rows with a `B` entry |
| Proposal is `no_action`, policy `DEFER` | 32 | `action == no_action` |
| Proposal is a state-changing action | 87 | 119 - 32 |
| Policy `REJECT` before validation | 2 | `validated is None`, outcome `REJECT` |
| Sandbox validation attempted | 85 | `validated is not None` |
| Validation failed, policy `REJECT` | 17 | |
| Validation passed, approval requested | 68 | |
| Simulated execution resolved the incident | 67 of 68 | |
| Simulated rollback restored state | 68 of 68 | |

**87 - 85 = 2**: two `retrain_candidate` proposals (an F04 and an F07 scenario, both misdiagnosed
as concept drift with evidence fit 0.65) that the policy rejected on its first pass, so the graph
never sent them to validation. From `policy.py` the only pre-validation `REJECT` is a
non-positive score (0.65 x 0.55 efficacy, minus risk, minus 0.20 cost). The historical rows do
not store the policy reasons, so this cause is **inferred from the code, not read from the data**;
the new run records the reasons and will confirm or refute it.

Defects:

1. Four denominators are in use and none is printed: validation attempts (85), detected
   supported-fault *scenarios* (72), executed fixes (68), and all proposals (119, which includes
   the 32 `no_action`).
2. "Supported-fault validation success 67/72" is not a share of validation attempts. Its
   denominator is detected scenarios whose fault has an automatable fix; a scenario counts as a
   failure if the wrong action was proposed or nothing was validated. Five failed: three got
   `retrain_candidate`, one got `no_action`, one correct scale fix failed validation.
3. `rollback_success_rate` has denominator "executed fixes", not "rollback attempts"; they
   coincide only because the benchmark rolls back every execution.
4. In the benchmark, approval, execution and rollback are **simulated in-process**
   (`state_to_execute` plus a monitoring re-run), not driven through the API and database. The
   report does not say so.
5. No identifiers link proposal, validation, policy decision, approval, execution and rollback.
   In the database a remediation row carries its validation and decision as JSON columns; only
   the remediation and the approval have ids.
6. A crashed configuration (`{"error": ...}`) is dropped from `remediation_metrics` silently.
   None occurred in `main`, but nothing would report it.

Categories that exist in the code: deferred (`no_action`), rejected before validation (policy
score), rejected after failed validation, `INVESTIGATE_MORE` (possible, did not occur), crashed
(possible, did not occur). Duplicates and cancellations cannot occur in the benchmark path;
cancellation exists in the service (`cancel`) and is covered there.

Fix: a `LifecycleEvent` schema with linked ids; the benchmark emits events; a reconciler reports
duplicates, missing links, illegal orderings and incomplete lifecycles as failures; every rate
carries its numerator and denominator by name; an adapter builds the same events from the
database's audit trail.

## Problem 3: causal over-claim metric

Read: `aegis/agents/llm_tasks.py` (`OVERCLAIM`, `validate_ranking`, `validate_explanation`),
`graph.py`, `benchmark.py`, `tests/adversarial`.

What exists: one regular expression (`caused by|is the cause|the root cause is|proves|...`)
applied to (a) the `reasoning` string of each ranked cause, at most 15 words by prompt, and
(b) the explanation text. A match replaces the reasoning or discards the explanation.

What `causal_overclaims = 0` in the historical report means: the expression matched zero of the
ranking `reasoning` strings in configurations C and D. Nothing more.

- It does **not** show the model made no causal claims: phrasings outside the expression
  ("due to", "led to", "because of", "is responsible for") are not seen at all.
- It does **not** cover explanations: the benchmark ran D with `use_llm_explain=False`.
- The raw reasoning strings were not stored, so the historical run **cannot be re-audited**.
  Generated, rejected and escaped counts for `main` are unavailable and will be labelled so.
- Evidence checking today is existence (is the id real) and polarity (is one cited item an
  anomaly). There is no check of relevance, evidence type, or whether evidence is strong enough
  for the wording.

So the system validates evidence references; it does not validate causal claims. The fix is a
deterministic claim validator (`aegis/agents/claims.py`): sentence-level classification by
lexicon, an evidence-strength policy (observation < association < mechanism < intervention), and
the rule that causal wording needs intervention-grade evidence, which only a passed sandbox
validation provides. Raw model text is stored in benchmark rows from now on.

Limit that will remain and be documented: a lexicon cannot recognise every paraphrase. "Escaped"
can be measured only against (a) a labelled adversarial corpus and (b) a re-scan of final output.

## Problem 4: resource measurement

Read: `aegis/evaluation/run.py::ResourceSampler`, `resources.json`, `manifest.json`, `/proc`,
and the Windows host through `powershell.exe`.

- **7.7 against 8.2**: the same quantity in two units. `MemTotal` is 8,023,708 kB, which is
  7.65 GiB (what `free` shows) and 8.2 GB in decimal units (what `manifest.json` stores as
  `total / 1e9`).
- **Verified today**: the Windows host has 16,957,034,496 bytes installed (15.8 GiB). There is no
  `.wslconfig`, so WSL2 (version 3.0.5) uses its default of half the host memory, and 8 logical
  processors (4 physical cores with hyper-threading) with 2 GiB of swap. "4 cores" in earlier
  documents means physical cores.
- **Why the runner was missed**: the original sampler summed processes whose *name* contains
  "ollama". Ollama 0.40 runs the model in a child process named `llama-server`. The later fix
  follows children, but still classifies nothing: server and runner are one number, there is no
  swap figure, no sampling-error count, a fixed 1 s interval, PID identity is not checked, and
  the sampler has no tests.
- The historical LLM run's runner memory is **unrecoverable**. The 3.1 GB figure is from a
  different process (a demo run) and stays labelled as such.

Fix: `aegis/observability/resources.py` with an injectable process source, identity by
(pid, create time), separate server / runner / AEGIS-tree series, swap, error accounting, and an
environment record. New measurements go in new, separately identified runs.

## Problem 5: release verification

- **Browser**: no browser or Playwright in the WSL environment; Chromium's system libraries
  `libnss3` and `libasound2` are absent and there is no root. Options, in order: user-local
  libraries for a headless Chromium, or the Windows host's Chrome/Edge over the DevTools protocol.
- **Frontend tests**: none; `package.json` has no test runner.
- **Evaluation in the dashboard**: no page and no API endpoint exist for benchmark results.
- **Compose `llm` profile**: never started. Reading it, `ollama-pull` waits a fixed 3 s, the
  `ollama` service has no health check, and the API does not wait for either.
- **Fallback model**: never pulled; fallback selection is tested only against a fake server.
- **MLflow**: `observability/tracking.py` has 0% test coverage.
- **Release workflow**: none; verification was a sequence of manual commands.

## Dependencies between fixes

- 3 before the new LLM benchmark (new counters need raw text recorded).
- 4 before any new benchmark run (new runs must use the new sampler).
- 2 before the deterministic re-run (the run must emit lifecycle events).
- 1 can be verified immediately against historical data.
- 5E (release script) last, since it invokes everything else.

## Planned tests

| Area | File |
|---|---|
| Metric populations and labels | `tests/unit/test_evaluation_metrics.py` |
| Lifecycle reconciliation | `tests/unit/test_lifecycle_reconciliation.py`, addition to `tests/e2e` |
| Claim validation | `tests/unit/test_claims.py`, `tests/adversarial/test_claims_adversarial.py` |
| Resource sampler | `tests/unit/test_resources.py`, `tests/integration/test_resources_live.py` |
| MLflow tracking | `tests/integration/test_tracking.py` |
| Dashboard | `apps/dashboard/src/*.test.tsx`, browser script under `apps/dashboard/e2e` |

## New dependencies expected

Frontend development only: a test runner and DOM test utilities, and a browser driver for the
browser check. No new Python runtime dependency is expected.
