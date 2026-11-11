# AEGIS architecture

AEGIS detects ML incidents, investigates them, proposes a remediation, validates it in
isolation, and executes it only after an authenticated human approves.

## Guiding split: deterministic core, bounded agent

```
                 ┌────────────────────────── deterministic core ───────────────────────────┐
 reference ─┐    │                                                                         │
 current  ──┼──▶ │ monitoring ──▶ findings ──▶ diagnosis (rules) ──▶ hypotheses            │
 model    ──┘    │   quality / drift / concept / performance / integrity                   │
                 │                                   │                                     │
                 │                                   ▼                                     │
                 │   policy engine ◀── remediation proposal ──▶ sandbox validation         │
                 │        │                                                                │
                 │        ▼                                                                │
                 │   approval (authenticated human) ──▶ execution ──▶ rollback             │
                 └───────────────▲───────────────────────────────────────▲─────────────────┘
                                 │ allowlisted, schema-validated tools   │ never
                         ┌───────┴────────┐                              │
                         │ LangGraph agent│  interprets evidence, picks  │
                         │ + local Ollama │  tools, re-ranks, summarises ┘
                         └────────────────┘
```

Everything that must be *true* (statistics, metrics, hashes, authorisation, state transitions)
is computed by plain Python. The LLM only reads structured evidence and writes structured
suggestions; each suggestion is validated against a schema and against the evidence IDs that
actually exist. If Ollama is down, the same graph runs with deterministic fallbacks and records
`llm_status = unavailable`.

## Epistemic levels

Every record carries the level of claim it is allowed to make:

| Level | Produced by | Example |
|---|---|---|
| Observed fact | monitor, direct measurement | "column `age` is absent" |
| Statistical anomaly | monitor, test + effect size | "`hours_per_week` PSI 0.41, BH-adjusted p < 1e-6" |
| Diagnostic hypothesis | rules engine (optionally LLM re-ranked) | "consistent with a unit/scaling error" |
| Validated proposal | sandbox validation | "rescaling by 1/60 clears the anomaly on a held-out half" |
| Approved action | authenticated approver | approval row with identity + timestamp |
| Executed action | executor | new deployment revision |

A hypothesis is never promoted to a cause. Confidence scores are evidence-fit scores, not
probabilities of causation.

## Modules

| Package | Responsibility |
|---|---|
| `aegis.ingestion` | dataset download + checksums, leakage-safe splits, data contracts |
| `aegis.models` | training/selection protocol, metrics, hash-verified model registry |
| `aegis.faults` | ten deterministic fault injectors, scenario generation, ground truth kept apart |
| `aegis.monitoring` | quality, drift, concept drift, performance, integrity monitors -> `Finding`s |
| `aegis.diagnosis` | evidence records, feature analysis, importance/SHAP, rule-based hypothesis ranking |
| `aegis.agents` | Ollama client, tool allowlist + schemas, LangGraph investigation workflow |
| `aegis.rag` | versioned runbooks, hybrid lexical/embedding retrieval, retrieval evaluation |
| `aegis.remediation` | proposals, sandbox validation, risk policy, lifecycle state machine, executor |
| `aegis.storage` | SQLAlchemy models, repositories, Alembic migrations (PostgreSQL 17) |
| `aegis.evaluation` | AEGIS-IR benchmark, ablations, reports |
| `aegis.observability` | structured logs, OpenTelemetry spans, Prometheus metrics, MLflow tracking |
| `apps.api` | FastAPI application |
| `apps.dashboard` | React + TypeScript dashboard |

## Data layout and leakage control

One seed yields four disjoint stratified partitions: `train` (fit + CV selection),
`validation` (threshold selection, and the monitoring **reference** window), `test` (reported
once), `production` (simulated live traffic; every scenario is sampled from it). The model never
sees `production`, so clean scenarios are true out-of-sample controls.

## Deployment state

What remediation may change is one small, versioned object per dataset: the **deployment
state** (`model_version`, expected artifact hash, ingestion config such as per-feature unit
scale, deduplication, quarantine flags). Executing an approved action writes a new revision;
rollback re-activates the previous one. Nothing else is mutable by the remediation path.

## Runtime topology (Docker Compose)

`postgres` (state) · `api` (FastAPI + monitoring + agent, one process) · `dashboard` (static
build behind nginx) · `mlflow` (tracking UI) · `ollama` (optional profile). One API process is
deliberate: on a 4-core laptop the bottleneck is LLM inference, which is serialised anyway.
