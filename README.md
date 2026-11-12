# AEGIS — Autonomous ML Incident Commander

AEGIS watches a deployed machine-learning model, opens an incident when data or model behaviour
departs from a reference window, investigates it, ranks root-cause hypotheses from evidence,
proposes a remediation, validates that remediation in a sandbox, and executes it only after an
authenticated human approves. Everything runs on a laptop CPU with no paid API.

```
 batch ─▶ monitoring ─▶ incident ─▶ investigation ─▶ hypotheses ─▶ proposal ─▶ sandbox validation
          (deterministic)            (LangGraph +                               │
                                      local LLM, bounded)                       ▼
                    audit trail ◀── rollback ◀── execution ◀── human approval ◀─ policy engine
```

Two rules shape the whole system:

1. **Deterministic core.** Statistics, metrics, hashes, authorisation and state transitions are
   plain Python and SQL. The LLM never computes or decides any of them.
2. **Bounded agent.** A local 4B model chooses among five read-only diagnostic tools, re-ranks
   hypotheses and writes the summary. Its output is schema-validated and checked against the
   evidence store. If Ollama is down, the same workflow runs without it and says so.

> **Scope.** A portfolio and research system built on two public datasets. It is not evidence
> that any model is fit for real lending or employment decisions, and it has not run in
> production. See [Limitations](#limitations).

## What is in the box

| Area | Implementation |
|---|---|
| Datasets | UCI Adult and Bank Marketing; checksum-pinned download; four leakage-safe splits |
| Model | Logistic regression vs. HistGradientBoosting, selected by 5-fold CV ROC-AUC with a simplicity margin; MLflow-tracked; hash-verified registry |
| Fault injection | 10 deterministic fault categories x 3 severities, clean controls, ground truth kept apart |
| Monitoring | quality, drift (KS, chi-square, PSI, Wasserstein, BH-adjusted), concept drift (River ADWIN, Page-Hinkley), performance with delayed labels, feature/label relationship, artifact integrity |
| Diagnosis | evidence records, shift-vs-scale quantile fitting, permutation importance, bounded SHAP, version comparison, error slices, rule-based hypothesis ranking |
| Agent | one LangGraph workflow, 10 nodes, <= 5 diagnostic tool calls, <= 3 LLM requests, timeouts, cancellation |
| RAG | 8 versioned runbooks, lexical or hybrid retrieval, retrieval evaluation, injection flagging |
| Remediation | proposals, sandbox validation, risk-scored policy, state machine, approval, execution, rollback |
| Backend | FastAPI, PostgreSQL 17, Alembic, bearer-token roles, append-only audit table |
| Dashboard | React + TypeScript + Vite |
| Observability | OpenTelemetry traces, Prometheus metrics, structured logs, MLflow |
| Benchmark | AEGIS-IR: 170 scenarios, four ablation configurations |
| Tests | backend unit, integration (real PostgreSQL, real MLflow), adversarial and end-to-end tests; dashboard component tests; a real-browser check; one release-verification command |

## Requirements

- Linux or WSL2 (developed on Ubuntu under WSL2: Intel i5-10210U, 4 physical cores, 7.65 GiB visible to the guest)
- [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you)
- Docker with Compose (PostgreSQL; and the full stack if you want it)
- Optional: Node.js 20+ (dashboard development), Ollama (LLM path)

No GPU. No hosted inference API. About 3 GB of disk for the Python environment and models, plus
2.5 GB if you pull `qwen3:4b`. The LLM path needs about 3.7 GB of free RAM for one investigation
and more for long runs; see [docs/performance.md](docs/performance.md).

## Quick start (local development)

```bash
git clone <this repository> aegis && cd aegis

# 1. Python environment, exactly as locked
uv sync

# 2. Data and baseline models (about 3 minutes; downloads ~1.6 MB, verifies SHA-256)
make data
make train

# 3. PostgreSQL 17 on port 55432, migrations, reference windows and deployments
make test-db-up
make bootstrap

# 4. Configuration: create tokens
cp .env.example .env
python3 -c "import secrets; [print(secrets.token_urlsafe(24)) for _ in range(3)]"
#    put the three values into AEGIS_API_TOKENS in .env (approver, operator, viewer)

# 5. API on http://localhost:8000  (OpenAPI docs at /docs)
make api

# 6. Dashboard on http://localhost:5173  (second terminal)
cd apps/dashboard && npm install && npm run dev
```

Open the dashboard, paste a token (top right), and you are looking at an empty incident list.
Create an incident with the demo below.

### Optional: the local LLM

```bash
# Without root: a user-local, CPU-only Ollama (77 MB instead of the 1.5 GB full archive)
uv run python -m scripts.install_ollama_local
scripts/ollama_serve.sh &                         # or use a system-wide `ollama serve`
.local/ollama/bin/ollama pull qwen3:4b            # 2.5 GB
.local/ollama/bin/ollama pull llama3.2:3b         # optional fallback model
```

`GET /ready` shows `llm: ok` when the model is reachable. Nothing else changes: with the LLM
unavailable every investigation takes the deterministic path and records
`llm_status = unavailable`.

## Demonstration scenario

A numeric feature silently changes scale upstream. AEGIS notices, works out the factor, proposes
the inverse correction, proves on held-out rows that it works, and waits for a human.

```bash
uv run python -m scripts.demo --operator-token <operator token> --approver-token <approver token> --no-llm
```

Drop `--no-llm` to let the local model plan and rank (about 3-4 minutes on a 4-core laptop).
The script uploads a CSV batch, runs monitoring, investigates, prints hypotheses with their
evidence, shows the sandbox validation checks, demonstrates that an operator token cannot
approve, approves with the approver token, re-monitors the same batch under the new deployment,
prints the audit trail, and rolls back. A real transcript is in
[docs/demo-transcript.md](docs/demo-transcript.md).

## Docker Compose

```bash
cp .env.example .env            # set AEGIS_API_TOKENS
docker compose up -d --build
```

| Service | URL |
|---|---|
| Dashboard | http://localhost:8080 |
| API | http://localhost:8000 (`/docs`, `/ready`, `/metrics`) |
| MLflow | http://localhost:5000 |

On first start the API container downloads the datasets, trains the baselines, migrates the
database and registers the reference windows (about three minutes; watch `docker compose logs -f api`).

Optional profiles:

```bash
docker compose --profile llm up -d        # Ollama in a container; pulls qwen3:4b once it is healthy (2.5 GB)
docker compose --profile tracing up -d    # Jaeger at http://localhost:16686; set AEGIS_OTLP_ENDPOINT=http://jaeger:4318
```

Without the `llm` profile the stack is fully functional and `/ready` reports the LLM as
unavailable. To use an Ollama running on the host instead, start it with `OLLAMA_HOST=0.0.0.0`
and set `AEGIS_OLLAMA_BASE_URL=http://host.docker.internal:11434`.

## Tests and checks

```bash
make test-db-up                                  # PostgreSQL for integration, adversarial and e2e tests
uv run pytest tests/unit tests/integration tests/adversarial tests/e2e -m "not ollama"
make test-frontend                               # dashboard tests (Vitest)
uv run ruff check . && uv run ruff format --check .
```

One command runs every release gate and writes a machine-readable report:

```bash
export AEGIS_API_TOKENS=...                      # or .env
make release-verify                              # all gates; the optional ones need a local LLM
make release-verify-mandatory                    # mandatory gates only
```

It runs the backend suites, the frontend tests and build, starts the Docker Compose stack and
drives the dashboard in a real browser, checks benchmark integrity, and (optionally) verifies
both local models and the Compose `llm` profile. Gates end as passed, failed, blocked or skipped;
only passed counts. What each gate does and the last result:
[docs/release-readiness.md](docs/release-readiness.md),
[results/release/release-report.md](results/release/release-report.md).

Tests run on the real datasets, a real PostgreSQL 17 and real MLflow backends. The test double
for the LLM is a scripted Ollama HTTP endpoint, so the real client code is exercised; the real
models are exercised by `scripts.verify_llm` and the `-m ollama` tests.

## Benchmark: AEGIS-IR

```bash
uv run python -m aegis.evaluation.run --name myrun          # A and B on 170 scenarios, ~7 minutes
uv run python -m aegis.evaluation.run --name myrun --llm    # adds C and D on a 20-scenario subset, ~80 minutes
uv run python -m aegis.evaluation.report --name myrun       # summary.json, report.md, lifecycle reconciliation, charts
```

Methodology, limitations and results: [docs/benchmark.md](docs/benchmark.md).

AEGIS achieved 0.996 overall fault-detection F1 across ten injected ML failure categories and
0.9565 drift-detection F1 across numerical, categorical, and scaling drift scenarios.

Those are two different metrics and are never interchangeable:

| Metric | Positives | Clean controls | A detection is | TP / FP / TN / FN | F1 |
|---|---|---|---|---|---|
| `overall_fault_detection_f1` | 120 scenarios, all ten fault categories | 50 | an incident opened by any monitor | 119 / 0 / 50 / 1 | 0.9958 |
| `drift_detection_f1` | 36 scenarios: F01, F02, F04 only | 50 | the drift monitor itself triggering | 33 / 0 / 50 / 3 | 0.9565 |

Measured on an Intel i5-10210U (4 physical cores, 8 logical), 7.65 GiB of RAM visible to WSL2
(the host has 15.8 GiB), no GPU; 170 scenarios; evaluation seeds never used for tuning.

| | Result |
|---|---|
| Overall fault detection | recall 0.992 (119/120 faults); 0 false alarms on 50 clean batches (95% CI 0.00-0.07) |
| Root cause, rules (B) vs alerts only (A) | Top-1 0.966 vs 0.815 on 119 diagnosed scenarios; Recall@3 0.975 |
| Root cause, local LLM alone (C) | Top-1 11/20 - the reason the LLM is not the source of truth |
| Full agent (D) vs rules (B) | same accuracy (19/20), at tens of seconds per investigation instead of about one |
| Remediation lifecycle | 119 proposals: 32 no-action, 2 rejected before validation, 85 validated (68 passed), 68 approval requests; every record reconciles |
| Executed fixes (simulated approval) | incident cleared 67/68; rollback restored state 68/68 |
| Safety counters | 0 unauthorized tool executions, 0 invalid promotions |

The LLM did not improve diagnostic accuracy over the rule engine. Its measured contribution is
tool selection and the written explanation. What it claims is validated sentence by sentence
against the evidence; see [docs/evidence-and-claim-validation.md](docs/evidence-and-claim-validation.md).
Every count above is derived in [docs/benchmark.md](docs/benchmark.md) and
[docs/remediation-lifecycle.md](docs/remediation-lifecycle.md).

## Repository layout

```
aegis/
  ingestion/      datasets, splits, data contracts
  models/         training protocol, metrics, hash-verified registry
  faults/         fault injectors, scenario generation, ground truth
  monitoring/     quality, drift, concept, performance, relationship, integrity, engine
  diagnosis/      evidence, feature inspection, importance, rules, naive baseline
  agents/         Ollama client, tool allowlist, LLM tasks + validators, LangGraph workflow
  rag/            runbook retriever and retrieval evaluation
  remediation/    proposals, sandbox validation, policy, lifecycle, executor
  services/       the application service: every state change, in one transaction, audited
  storage/        SQLAlchemy models, Alembic migrations
  evaluation/     AEGIS-IR runner and report generator
  observability/  logging, tracing, metrics, MLflow tracking
apps/api/         FastAPI application
apps/dashboard/   React + TypeScript dashboard
data/runbooks/    versioned runbooks and labelled retrieval queries
experiments/      development sweeps and a live-LLM investigation script
infrastructure/   Dockerfiles, nginx
scripts/          data, training, bootstrap, demo, Ollama helpers
tests/            unit / integration / adversarial / e2e
docs/             architecture, decisions, threat model, API, benchmark, interview preparation
```

## Documentation

- [Architecture](docs/architecture.md)
- [Decision log](docs/decisions.md) — including version compatibility and every measured trade-off
- [Threat model](docs/threat-model.md)
- [API reference](docs/api.md)
- [Benchmark](docs/benchmark.md) and [performance report](docs/performance.md)
- [Remediation lifecycle and its accounting](docs/remediation-lifecycle.md)
- [Evidence and claim validation](docs/evidence-and-claim-validation.md)
- [Release readiness](docs/release-readiness.md) and the [audit that led to it](docs/remediation-audit.md)
- [Implementation plan and verification status](docs/implementation-plan.md)
- [Interview preparation](docs/interview-preparation.md)

## Dataset notes

Both datasets are from the UCI Machine Learning Repository (CC BY 4.0) and are downloaded by
script; raw archives are kept unmodified with a provenance record and pinned SHA-256.

- **Adult** (Becker & Kohavi, 1996): 48,842 census records from 1994. Contains sensitive
  attributes (age, race, sex, native country, marital status). `fnlwgt` and `education` are
  dropped.
- **Bank Marketing** (Moro, Rita & Cortez, 2014): 45,211 phone-campaign records from one
  Portuguese bank, 2008-2010. `duration` is dropped because it is only known after the outcome.

Both are old, come from one place and time, and encode historical bias. They are used here as
realistic tabular streams to break on purpose, nothing more.

## Limitations

- Batch monitoring, not streaming. Detection delay is measured in 500-row windows.
- The benchmark's faults are synthetic, single, and written by the same author as the rules that
  diagnose them. See the limitations section of the benchmark document before quoting a number.
- CPU inference is slow: about 3.5 minutes for a full LLM investigation on the development
  machine. The deterministic path takes about a second.
- Authentication is static bearer tokens. There is no SSO, token rotation or rate limiting; do
  not expose the API to a network you do not control.
- Retraining validation is deliberately conservative and will refuse a model fitted to a genuine
  permanent concept change; a human has to re-baseline the reference window.
- One API process, one investigation at a time.
