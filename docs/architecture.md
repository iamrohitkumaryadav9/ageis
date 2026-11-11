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

## Investigation workflow

One LangGraph state machine, typed state, explicit transitions:

```
collect_incident -> run_diagnostics -> build_evidence -> rank_hypotheses -> validate_hypotheses
        ^                                                                          |
        +--------------- at most one loop, only while tool budget remains ---------+
                                                                                   v
retrieve_runbook -> plan_remediation -> validate_remediation -> policy_check -> request_approval -> END
```

| Node | Deterministic part | LLM part (optional) |
|---|---|---|
| collect_incident | findings -> evidence records; LLM health check | - |
| run_diagnostics | executes tools through the allowlisting executor; fallback plan | chooses which diagnostic tools to call |
| build_evidence | merges tool evidence by id | - |
| rank_hypotheses | rule engine scores every candidate cause | re-ranks, citing evidence ids |
| validate_hypotheses | drops unknown causes, fabricated ids, unsupported claims, causal wording; bounds confidence | - |
| retrieve_runbook | lexical (or hybrid) retrieval; injection flagging | - |
| plan_remediation | cause -> action mapping | - |
| validate_remediation | sandbox validation when policy allows it | - |
| policy_check | Score = benefit - risk - cost; outcome | - |
| request_approval | records the request; deterministic summary | writes the operator-facing explanation |

`validate_remediation` is an addition to the originally sketched node list: validation is a
distinct, slow, failure-prone step and deserves its own node, span and audit event.

Bounds: five diagnostic tool calls (one reserved for runbook retrieval), one diagnostic loop,
three LLM requests each with at most one retry, per-request and per-tool timeouts, a LangGraph
recursion limit, and cooperative cancellation checked at every node. A node that raises records
the error and the workflow continues with the partial state.

State that must outlive the process (incidents, evidence, traces, hypotheses, remediations,
approvals) is written to PostgreSQL by the service layer after the graph returns. The graph does
not use a LangGraph checkpointer: an investigation is a few minutes of work that can simply be
re-run, and human approval is a durable database state rather than a suspended graph.

## Request path for a decision

```
POST /remediation/{id}/approve          bearer token -> Principal(identity, role)
  -> role >= approver?                  else 403 + audit(denied)
  -> BEGIN; SELECT remediation FOR UPDATE
  -> no prior approval row              else replay (same idempotency key) or 409
  -> state == PENDING_APPROVAL          else 409
  -> expected_version == version        else 409 "stale decision"
  -> INSERT approval (UNIQUE remediation_id)
  -> SELECT active deployment FOR UPDATE; revision == validated revision   else 409 "stale"
  -> candidate state re-derived from proposal == validated candidate state else 409
  -> INSERT deployment revision, deactivate previous
  -> re-monitor the incident batch under the new state -> COMPLETED or EXECUTED(unverified)
  -> audit rows for approve / activate / execute / verify
  -> COMMIT   (any exception: ROLLBACK everything, including the approval)
```

## Observability

- **Traces** (OpenTelemetry): FastAPI requests, `service.*` operations, `investigation.run` with
  one child span per node, `tool.execute` per tool call, `llm.chat` per LLM request with model,
  outcome, attempts and token counts. Spans go to an in-process ring buffer (served at
  `/observability/traces` and aggregated at `/observability/latency`) and to OTLP when configured.
- **Metrics** (Prometheus, `/metrics`): monitoring, tool, LLM and investigation duration
  histograms; tool calls and LLM requests by outcome; incidents; remediation transitions;
  denied privileged actions; process RSS.
- **Logs**: structured JSON (structlog). Tokens and data rows are never logged.
- **Experiments** (MLflow): parameters, CV/validation/test metrics, artifacts and registered
  model versions for every trained baseline.

## Runtime topology (Docker Compose)

`postgres` (state) · `api` (FastAPI + monitoring + agent, one process) · `dashboard` (static
build behind nginx) · `mlflow` (tracking UI) · `ollama` (optional profile). One API process is
deliberate: on a 4-core laptop the bottleneck is LLM inference, which is serialised anyway.
