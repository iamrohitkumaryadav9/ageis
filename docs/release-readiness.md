# Release readiness

What was verified, how, and what was not. The gate-by-gate result of the last full run is
generated, not written here: [results/release/release-report.md](../results/release/release-report.md)
(machine-readable: `release-report.json`).

Vocabulary used throughout, and never interchangeably:

| Term | Means |
|---|---|
| implemented | the code exists |
| unit tested | exercised in isolation, collaborators faked |
| integration tested | exercised against a real PostgreSQL, a real MLflow store or a real process table |
| end-to-end tested | driven through the HTTP API from upload to rollback |
| browser verified | driven in a real browser against a running stack |
| hardware verified | run with a real local model on this machine |
| benchmark verified | measured on AEGIS-IR evaluation seeds |

## The release workflow

```bash
make test-db-up                                   # PostgreSQL for the tests
export AEGIS_API_TOKENS=...                       # or .env; approver, operator and viewer tokens
uv run python -m scripts.release_verify           # every gate
uv run python -m scripts.release_verify --skip-optional   # mandatory gates only
uv run python -m scripts.release_verify --list
```

Each gate ends as `passed`, `failed`, `blocked` (a prerequisite is missing) or `skipped`
(excluded on the command line). Only `passed` counts. The verdict is RELEASE READY when every
gate passed, CONDITIONALLY READY when every mandatory gate passed and an optional one did not,
NOT READY otherwise. A partial run says so in its verdict.

### Mandatory gates

| Gate | What it runs |
|---|---|
| `lint` | `ruff check`, `ruff format --check` |
| `backend_unit` | `tests/unit` |
| `postgres_integration` | API and pipeline tests on PostgreSQL 17 |
| `mlflow_tracking` | `tests/integration/test_tracking.py` |
| `resource_sampler` | sampler tests on fake process trees and on the real process table |
| `adversarial` | prompt injection, tool misuse, claim validation, approval bypass |
| `security_policy` | approval enforcement, tool allowlist, policy engine, lifecycle reconciliation |
| `e2e` | full lifecycle over HTTP on the deterministic path |
| `benchmark_integrity` | raw reference results match their pinned SHA-256; headline metrics recompute to the published values; committed summaries are not stale; lifecycle reconciles for every committed run; metric and lifecycle tests |
| `frontend_tests` | Vitest component and API-client tests |
| `frontend_build` | TypeScript check and production build |
| `compose_default` | `scripts.verify_compose`: default services plus tracing, from `docker compose up` to a deterministic investigation through the API |
| `browser` | `apps/dashboard/e2e/browser-check.mjs` against that stack |

### Optional gates (hardware-dependent)

These need a local LLM: several GB of downloads, about 3 GB of free RAM, and minutes of CPU time
per investigation. A machine without them can still pass every mandatory gate, because the
system is specified to work without an LLM.

| Gate | What it runs |
|---|---|
| `ollama_qwen3_4b` | `scripts.verify_llm --model qwen3:4b` |
| `ollama_llama3_2_3b` | `scripts.verify_llm --model llama3.2:3b` |
| `e2e_real_llm` | the end-to-end lifecycle and the sampler's runner attribution with the real model |
| `compose_llm` | `scripts.verify_compose --llm`: Ollama in a container, connectivity, fallback, recovery |

## Result of the last full run

2026-11-13, commit `9cfd559`: **RELEASE READY**. 13 of 13 mandatory gates and 4 of 4 optional
gates passed, in 2,056 s. Backend: 423 unit, 63 PostgreSQL integration, 18 MLflow, 38 sampler,
120 adversarial, 142 security and policy, 3 end-to-end tests. Frontend: 51 tests and the build.
Compose default: 9 of 9 checks. Browser: 24 of 24. Each model: 10 of 10. Real-model end-to-end:
2 tests. Compose `llm` profile: 17 of 17.

It took five full runs to get there, and the earlier ones are part of the evidence:

| Pass | Outcome | What it found |
|---|---|---|
| 1 | 13/13 mandatory, 3/4 optional | With the fallback model, an omitted rule hypothesis was ranked below the model's weaker guess. A real defect; fixed (section C). Report kept as `release-report.pass1-before-ordering-fix.json` |
| 2 | stopped | An existing adversarial test still asserted the old ordering. The test was updated to the stronger property (decision D24) |
| 3 | stopped | A unit test of the retriever was flaky: its stub embedder used Python's per-process salted `hash()`. Pre-existing, unrelated to this work, made deterministic and checked under ten hash seeds |
| 4 | 13/13 mandatory, 3/4 optional | The image rebuild could not reach the container registry, so the `llm` profile did not start. Environmental, but it exposed two defects in the verification scripts themselves: the Compose check crashed instead of reporting, and the release report showed the previous run's check counts next to the failed gate. Both fixed; report kept as `release-report.pass4-registry-fetch-failure.json` |
| 5 | 13/13 mandatory, 4/4 optional | nothing |

Two things follow. A verdict from this script is a statement about one run on one machine: LLM
output varies (in pass 5 the primary model returned an unusable ranking once and the
investigation correctly kept the deterministic one), and the Compose gates need the network.
And the optional gates earned their place: the two product defects found during verification
were both found by running a real model, not by the mandatory suite.

## A. Dashboard: browser verification

**Status: browser verified** in headless Chromium 131 at 1440x900 and 390x844, against the
Docker Compose stack (nginx, API, PostgreSQL, MLflow, Jaeger), with a real incident created
through the API. Evidence: `results/release/browser/browser-report.json` and the screenshots next
to it.

What the 24 checks do:

- load the page; reject an invalid token with a visible 401; sign in; overview tiles show live data;
- loading state under a delayed response; "API unavailable" when the request is aborted; a 500
  shown with its status and request id;
- incident list, status filter, the empty state, navigation to an incident;
- incident detail: findings, feature findings, performance, evidence records;
- investigation tab: ranked hypotheses worded as hypotheses, evidence expandable, tool calls;
- remediation tab: a pending proposal shown as "recommendation only" with validation and policy;
- a viewer clicks Approve: the server's 403 is displayed and nothing is shown as executed;
- an approver approves: executed banner, verification, the audit trail including the denied attempt;
- rollback is disabled until a reason is entered, then restores the previous state;
- a non-existent incident shows 404;
- the evaluation page shows `overall_fault_detection_f1` and `drift_detection_f1` under their own
  names, and the values and populations on the page equal what the API serves;
- no horizontal page overflow on desktop or at 390 px on four pages; tabs operable on the phone;
- no uncaught exception and no console error other than the 4xx/5xx responses the script provokes.

**Defects the browser run found, now fixed**: the tab bar overflowed the page at phone width;
tables in cards squeezed their columns into very tall rows on a phone; the evaluation page
defaulted to an arbitrary run and described a run with LLM configurations as having none; the
policy block printed its score line twice. None of these was visible to the type-check or the
build.

**Frontend tests**: 51 Vitest tests (API client, state and error handling, app shell and token
handling, incident detail, approval and rollback controls, evaluation page). Added development
dependencies, each for a stated reason: `vitest` (test runner that reuses the Vite config),
`jsdom` (DOM for component tests), `@testing-library/react`, `@testing-library/dom`,
`@testing-library/user-event` (render and interact as a user would), `playwright` (drives the
browser check). No runtime dependency was added.

**Limits**: on a phone, wide tables scroll sideways inside their card and long evidence
statements make tall rows: usable, not polished. Chromium only; Firefox and Safari were not exercised. Layout is checked for overflow
and visibility by the script, and the screenshots were reviewed once by eye; there is no visual
regression baseline. Accessibility was not audited beyond labels and roles the tests rely on.

**Running the browser check**: needs headless Chromium.
`cd apps/dashboard && npx playwright install --with-deps chromium-headless-shell` on a machine
with root. Without root (as on the build machine), install the browser with
`PLAYWRIGHT_BROWSERS_PATH=.local/ms-playwright npx playwright install chromium-headless-shell`,
then `apt-get download libnss3 libnspr4 libasound2t64`, unpack the three packages with
`dpkg-deb -x <deb> .local/browser-libs/root`, and the release script picks both locations up. If
Chromium cannot start, the gate reports `blocked` with this instruction; it does not pass.

## B. Docker Compose `llm` profile

**Status: integration tested on real containers.** Evidence:
`results/release/compose-llm.json` and, for the run that started from an empty model volume,
`results/release/compose-llm.first-run-clean-pull.json`.

Defects found by reading and then starting the profile, and fixed:

| Defect | Fix |
|---|---|
| `ollama-pull` slept 3 s and hoped the server was up | `ollama` has a health check (`ollama list`); the pull `depends_on: service_healthy` |
| image tag `latest` | pinned to `ollama/ollama:0.40.16`, overridable with `OLLAMA_IMAGE` |
| no way to pull the fallback model | `AEGIS_OLLAMA_PULL_EXTRA` |
| the pull exited silently | it lists the models at the end and fails the container on error |

Measured on the first run (empty model volume; other volumes already populated):

| Check | Result |
|---|---|
| Ollama container healthy | yes |
| In-container pull of `qwen3:4b` and `llama3.2:3b` | 888 s (network-bound, about 6 MB/s) |
| API reaches Ollama at `http://ollama:11434` over the Compose network | yes; `/ready` reports `qwen3:4b via Ollama 0.40.16` |
| Ollama published to the host | no (the check misread Docker's `:0` output on that run and is the one recorded failure; the check was corrected and the port was never published) |
| Investigation with the LLM through the API | completed in 267 s; plan 87 s, rank 97 s, explain 61 s, all three accepted |
| Ollama stopped | `/ready` stays `ready` and reports the LLM `unavailable` |
| Investigation requested with the LLM while Ollama is down | completed in 6.5 s on the deterministic path, `llm_status = unavailable`, ranked by rules, with the reason in `errors` |
| Ollama started again | LLM reported `ok` after 3 s |

Default services from empty volumes (first start downloads the datasets, trains both baselines,
migrates, bootstraps): 191 s to API healthy.

The API deliberately does not wait for the model pull. Until it finishes, `/ready` reports the
LLM as unavailable and investigations take the deterministic path.

**Limits**: one machine, one run from a clean model volume. During the in-container LLM
investigation the guest's 2 GiB of swap was almost full; on an 8 GiB allocation the `llm`
profile leaves no headroom for anything else.

## C. Fallback model `llama3.2:3b`

**Status: hardware verified.** Both models were pulled into the existing user-local Ollama
0.40.16 and run through `scripts.verify_llm`, which uses no fake server. Evidence:
`results/release/llm-llama3.2-3b.json`, `results/release/llm-qwen3-4b.json`.

Figures are from each model's first verification run on 2026-11-12; later runs overwrote the
JSON files and differ in detail, as noted below the table.

| | `qwen3:4b` (primary) | `llama3.2:3b` (fallback) |
|---|---|---|
| Parameters / quantisation / size on disk | 4.0B / Q4_K_M / 2.50 GB | 3.2B / Q4_K_M / 2.02 GB |
| Digest (first 12) | `359d7dd4bcda` | `a80c4f17acd5` |
| Cold load | 7.5 s | 11.7 s |
| Structured-output request | valid on the first attempt, 13.0 s | valid on the first attempt, 13.1 s |
| - answer to the toy classification | `scaling_error` (correct) | `concept_drift` (wrong; schema-valid) |
| Full investigation (plan, rank, explain) | 225 s; 3 of 3 requests accepted | 144 s; 3 of 3 requests schema-valid |
| - per request | 76 / 81 / 64 s | 41 / 63 / 39 s |
| - tools requested by the model | 4 | 1 |
| - what survived claim validation | ranking kept (`ranked_by = llm`), explanation kept | **ranking discarded**: one item unsupported, one contradictory; the rule engine's ranking was used. Explanation kept |
| Final top cause / policy outcome | scaling error / request approval | scaling error / request approval (from the rules) |
| Runner memory, peak | 3,318 MB | 2,677 MB |
| Process-tree total, peak | 3,704 MB | 3,024 MB |
| Chosen as fallback when the primary is not pulled | n/a | yes, and it answered |
| 1-second budget | `timeout` after 1.0 s | `timeout` after 1.0 s |
| Reply truncated to 8 tokens | `invalid_output`, retried once, then refused | same |
| Neither model pulled | reported unavailable, no exception | same |
| Dead endpoint, then the real one | unavailable, then works | same |
| Same investigation with the LLM unreachable | completes on rules in 2 to 3 s | same |

Small-model output varies between runs. Across the five passes the primary model's full
investigation took 225 to 359 s and once returned an unusable ranking (the deterministic ranking
was kept); the fallback's took 133 to 144 s. A second run of `llama3.2:3b` behaved differently
in a way that exposed a defect:

**Defect found with the fallback model, and fixed.** In the second run the model ranked a single
weak guess (concept drift, capped at 0.45 because the rule engine does not support it) and
omitted the cause the rule engine supported at 0.88 (scaling error). The merged ranking listed
the model's item first and the omitted rule hypothesis after it, so the displayed leading cause
was the model's. The policy then returned `INVESTIGATE_MORE` and nothing was proposed, so the
outcome was safe, but the diagnosis shown was not the best-supported one: a model could displace
deterministic evidence by not mentioning it. The merged ranking is now ordered by evidence fit.
Two regression tests reproduce the case (both fail without the fix). An offline replay of the
model output stored by benchmark run `v2` shows the fix changes none of the 20 configuration D
diagnoses. The report of the verification pass that caught it is kept as
`results/release/release-report.pass1-before-ordering-fix.json`.

Assessment. `llama3.2:3b` is **usable as a fallback on this machine in the sense that matters
for safety**: it loads, returns schema-valid JSON, is about a third faster and 0.6 GB lighter, and
every failure mode degrades to the deterministic path. It is **not a like-for-like substitute**:
across the runs its ranking was either discarded by validation or contributed only a weak guess,
and it answered the standalone classification wrongly. With the fallback active, expect the
LLM's contribution to shrink to tool planning and the explanation, with the diagnosis coming from
the rule engine. These are single scenarios; they are not a measurement of the model's accuracy,
and the fallback was not run on the benchmark.

## D. MLflow tracking

**Status: integration tested**, 18 tests, 100% line coverage of
`aegis/observability/tracking.py`. The tests use real MLflow backends and no MLflow mocks: a
SQLite store per test, and one test that starts a real `mlflow server` process and reads the run
back over its REST API.

Covered: run creation, parameter, metric, tag and artifact logging, run termination, a failure
inside a run (closed as `FAILED`, and the next run is unaffected), a missing server, caller
mistakes, unloggable values, repeated logging, tracking URI configuration, disabled tracking,
model logging.

Defects found while writing the tests, and fixed:

- An unreachable tracking server could stall training for minutes, because MLflow retries with
  backoff by default. Retries and timeout are now bounded (1 retry, 10 s) unless set explicitly.
- Non-finite metrics were passed straight through. They are now dropped, and the run is tagged
  with the names that were dropped.
- An empty experiment or run name was swallowed by the catch-all and returned `None`. Caller
  mistakes now raise `ValueError`; only backend failures degrade to a warning.

Also verified in the Compose stack: the MLflow server lists both training experiments.

## What was not verified

| Item | Impact | To close it |
|---|---|---|
| Browsers other than Chromium | a layout or script difference in Firefox or Safari would be unnoticed | add `firefox` and `webkit` to the browser script; needs their system libraries |
| Visual regression | a later CSS change could degrade layout without failing a check | store baseline screenshots and compare |
| The fallback model on the benchmark | its diagnostic quality is known from one scenario only | `AEGIS_OLLAMA_MODEL=llama3.2:3b python -m aegis.evaluation.run --name fallback --llm` (about 60 to 90 minutes) |
| The `llm` profile from a completely empty Docker state, repeatedly | startup timing is from one run | `python -m scripts.verify_compose --llm --clean` |
| LLM benchmark repeatability | configurations C and D were run once per benchmark run; small-model output varies | repeat `--llm` under new run names and compare |
| Authentication beyond static bearer tokens | adequate for a local tool, not for production | out of scope; see the threat model |
