# Implementation plan

Eight phases, each with an exit criterion that is verified by running something.

| # | Phase | Exit criterion | Verification |
|---|---|---|---|
| 1 | Foundation: repo, deps, datasets, splits, baseline model | baseline trains, tests pass | `pytest tests/unit`, `scripts.train_baseline` |
| 2 | Monitoring + fault injection | deterministic monitoring benchmark runs | small AEGIS-IR run, config A |
| 3 | Root-cause engine | diagnosis works with no LLM | config B Top-1 / Recall@3 |
| 4 | Agentic investigation | agent investigates a generated incident via Ollama | live run + fallback tests |
| 5 | RAG + remediation | remediation cannot bypass validation or approval | adversarial tests |
| 6 | Backend + persistence | full workflow reachable through the API on PostgreSQL | API integration tests |
| 7 | Dashboard + observability | incident lifecycle inspectable in the UI; traces exist | frontend build, span assertions |
| 8 | Benchmark + release | fresh setup reproduces the demo | Compose up, full benchmark, docs |

Order rationale: fault injection precedes the agent so there is ground truth to measure it
against; the rules engine precedes the LLM so there is a baseline the LLM must justify itself
against, and a fallback when it is unavailable.

Verification status is recorded in the table at the end of this document; a row is marked done
only after the command next to it was actually executed on the build machine.

## Verification status (Definition of Done)

Recorded on 2026-11-12 on the build machine (i5-10210U, 4 cores, 7.7 GB RAM, no GPU, WSL2). A row
is "verified" only if the command or check named next to it was executed and passed.

| # | Requirement | Status | How it was verified |
|---|---|---|---|
| 1 | Repository initialized and documented | verified | git history per phase; README and `docs/` |
| 2 | Dependencies resolve reproducibly | verified | `uv sync --frozen` locally and inside the API image build |
| 3 | Datasets download reproducibly | verified | `scripts.download_data` (SHA-256 checked) locally and on container first start |
| 4 | Baseline ML model trains | verified | `scripts.train_baseline` locally and in the container |
| 5 | Ten fault injectors work | verified | `tests/unit/test_faults.py`; benchmark grid uses all ten |
| 6 | Data quality monitoring | verified | unit tests; F03/F06/F08 recall 12/12 each |
| 7 | Statistical drift monitoring | verified | unit tests; drift F1 0.957 on evaluation seeds |
| 8 | Concept drift monitoring | verified | unit tests; F05 recall 12/12 |
| 9 | Performance monitoring | verified | unit tests; F09 (delayed labels) handled 12/12 |
| 10 | Incidents persist in PostgreSQL | verified | integration tests against PostgreSQL 17 |
| 11 | Evidence records are traceable | verified | tests assert hypothesis evidence ids resolve to findings |
| 12 | Root-cause ranking works without an LLM | verified | configuration B: Top-1 0.966, Recall@3 0.975 |
| 13 | Local Ollama integration | verified | `pytest -m ollama` (1 passed, 291 s); real-model demo |
| 14 | LangGraph investigation executes | verified | `tests/unit/test_graph.py`; live runs |
| 15 | Tool allowlist enforced | verified | adversarial tests; 0 unauthorized executions in the benchmark |
| 16 | Structured model outputs validated | verified | unit and adversarial tests; benchmark counters |
| 17 | RAG retrieves relevant runbooks | verified | 12-query evaluation: lexical hit@1 0.83, hit@3 1.0 |
| 18 | Prompt injection tests pass | verified | `tests/adversarial` (38 tests) |
| 19 | Remediation proposals generated | verified | unit tests; 87 proposals in the benchmark |
| 20 | Candidate validation | verified | unit tests; sandbox validation in the benchmark |
| 21 | Human approval enforced server-side | verified | adversarial API tests; demo shows HTTP 403 for a non-approver |
| 22 | Rollback | verified | tests; rollback restored state 68/68 in the benchmark |
| 23 | Complete audit trail | verified | integration tests, including denied attempts; append-only trigger test |
| 24 | FastAPI endpoints | verified | `tests/integration/test_api.py`; demo over HTTP |
| 25 | React dashboard consumes real backend data | **partly verified** | type-check, production build, and its proxied API calls return live data in dev and behind nginx in Compose. It was **not viewed in a browser**: layout and interaction are unchecked |
| 26 | MLflow tracking | verified | training logs runs; MLflow server in Compose lists both experiments. No automated test covers `observability/tracking.py` |
| 27 | OpenTelemetry tracing | verified | span assertions in tests; spans exported to Jaeger in Compose (service `aegis` listed) |
| 28 | Unit tests pass | verified | 272 passed |
| 29 | Integration tests pass | verified | 55 passed |
| 30 | Adversarial tests pass | verified | 38 passed |
| 31 | End-to-end tests pass | verified | 3 passed, plus 1 real-model test run separately |
| 32 | Benchmark runs reproducibly | **partly verified** | full run completed and is resumable with fixed seeds; deterministic configurations were re-run and matched except for the one documented fix. The LLM configurations were run once; small-model output is not guaranteed identical across runs |
| 33 | CPU/RAM measurements documented | verified with a gap | `docs/performance.md`; per-process Ollama memory was not captured during the benchmark's LLM run (measured separately) |
| 34 | Docker Compose starts the application | verified | built and started from empty volumes; API, dashboard, MLflow and the `tracing` profile checked; deterministic demo passed against it. The `llm` profile (Ollama in a container) was **not started** |
| 35 | README supports a fresh local installation | verified (LLM steps excepted) | fresh `git clone` into an empty directory, then the README quick start as written: `uv sync`, `make data`, `make train`, `make bootstrap` on an empty database, `.env` with generated tokens, `make api`, the demo with `--no-llm`, and `npm ci && npm run build`. The optional Ollama install steps were not repeated in the clone |

Whole suite: 368 passed, 1 deselected (the real-model test), 83% line coverage over `aegis` and
`apps` (the benchmark runner and report generator are exercised by running them, not by tests).

Known gaps, stated plainly:

- No browser-level check of the dashboard and no frontend tests.
- The `llm` Compose profile is untested; the LLM was exercised through a host-installed Ollama.
- The fallback model `llama3.2:3b` was never pulled; fallback selection is tested with a fake server only.
- The LLM ablation covers 20 scenarios, run once.
- Authentication is static bearer tokens with roles: adequate for a local tool, not for production.


## Addendum, 2026-11-13: five defects remediated

The gaps listed above were closed or measured in a second round of work; the audit that started
it is `docs/remediation-audit.md`. Status of each row above that changed:

| # | Requirement | Status now | How it was verified |
|---|---|---|---|
| 25 | React dashboard consumes real backend data | browser verified | 24 checks in headless Chromium against the Compose stack at desktop and phone widths; 51 component tests. Chromium only |
| 26 | MLflow tracking | integration tested | 18 tests on real MLflow backends, 100% of `observability/tracking.py` |
| 32 | Benchmark runs reproducibly | verified for the deterministic configurations | run `v2` reproduced run `main` exactly on all 170 detections and every A and B outcome. LLM configurations agree on Top-1 (11/20 and 19/20 in both runs) but are single runs each |
| 33 | CPU/RAM measurements documented | verified for new runs; historical gap stated | process-tree sampler; separate deterministic and LLM records in `v2`. The historical run's runner memory is unavailable and is labelled so |
| 34 | Docker Compose starts the application | verified including the `llm` profile | `scripts.verify_compose` and `--llm`: 9 and 17 checks |
| - | Fallback model `llama3.2:3b` | hardware verified | `scripts.verify_llm`: 10 checks against the real model |

Whole release workflow: `python -m scripts.release_verify`, last result RELEASE READY, 13 of 13
mandatory and 4 of 4 optional gates. See `docs/release-readiness.md` for what that does and does
not establish, and `results/release/release-report.md` for the generated table.
