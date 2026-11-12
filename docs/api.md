# API reference

Base URL `http://localhost:8000` (or `/api` behind the dashboard's nginx). Interactive OpenAPI
documentation is served at `/docs`. All bodies are JSON unless noted.

## Authentication and roles

Every endpoint except `/health`, `/ready` and `/metrics` needs `Authorization: Bearer <token>`.
Tokens are configured on the server (`AEGIS_API_TOKENS=<token>:<identity>:<role>,...`).

| Role | May |
|---|---|
| `viewer` | read everything |
| `operator` | + upload data, run monitoring, investigate, create and validate remediations |
| `approver` | + approve, reject, roll back |

The reviewer's identity and role come from the token. No request body field can assert them.

## Errors

Every error has the same shape and an `x-request-id` response header:

```json
{"error": {"code": "conflict", "message": "stale decision: ...", "details": {"expected_version": 3, "current_version": 5}, "request_id": "9f2c..."}}
```

| HTTP | `code` | Meaning |
|---|---|---|
| 401 | `unauthenticated` | missing or invalid token |
| 403 | `forbidden` | role too low |
| 404 | `not_found` | unknown id or route |
| 409 | `conflict` | illegal state transition, duplicate or stale decision, stale deployment |
| 413 | `payload_too_large` | body over the upload limit |
| 422 | `invalid_request` / `invalid_input` | schema or semantic validation failed (unknown fields are rejected) |
| 503 | `database_unavailable` | the database is unreachable; nothing was changed |
| 500 | `internal_error` | unexpected; details are logged, not returned |

## Endpoints

### Health

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | liveness; always 200 while the process runs |
| GET | `/ready` | 200 `ready` / 503 `not_ready`. Components: `database` and `models` are required; `llm` and `runbooks` are optional and never make the service unready |
| GET | `/metrics` | Prometheus exposition |

### Data and monitoring (operator)

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/data/reference` | multipart: `dataset`, `file` (.csv) | 201 batch. Must contain every contracted feature and a fully labelled 0/1 `target` |
| POST | `/data/current` | multipart: `dataset`, `file` (.csv) | 201 batch. Any columns; `target` may be absent or partly empty |
| POST | `/monitor/run` | `{"dataset": "adult", "batch_id": "BAT-..."}` (`batch_id` optional: latest) | `{run, incident \| null, replayed}`. Re-running the same batch under the same deployment is a replay |

Limits: 20 MB and 200,000 rows per upload (configurable). Uploaded filenames are stored as text
and never used as paths.

### Incidents and investigations

| Method | Path | Role | Notes |
|---|---|---|---|
| GET | `/incidents?limit&offset&status&severity&dataset` | viewer | paginated `{items, total, limit, offset}`, newest first, `limit` <= 100 |
| GET | `/incidents/{id}` | viewer | findings, skipped checks, performance, evidence, investigations, remediations |
| POST | `/incidents/{id}/investigate` | operator | body `{"use_llm": true, "wait": false}`. 202 with a running investigation (poll it), or 200 when `wait` is true. One investigation runs per incident at a time |
| GET | `/investigations/{id}` | viewer | status, LLM status/model, hypotheses, tool traces, LLM requests, workflow events, runbook references, explanation, `trace_id` |
| POST | `/investigations/{id}/cancel` | operator | cooperative cancellation; partial results are kept |

### Remediation

| Method | Path | Role | Notes |
|---|---|---|---|
| POST | `/incidents/{id}/remediation` | operator | create from the latest investigation's recommendation. 201, or 200 if an open one exists. 409 if the recommendation changes no state |
| GET | `/remediation/{id}` | viewer | proposal, validation checks, policy decision, approval, verification, `allowed_events`, `executed` |
| POST | `/remediation/{id}/validate` | operator | sandbox validation of a `PROPOSED` remediation; moves to `PENDING_APPROVAL` when policy allows. Replays are no-ops |
| POST | `/remediation/{id}/approve` | **approver** | body `{"expected_version": n, "comment": "..."}`; optional `Idempotency-Key` header. Approves, executes and verifies in one transaction |
| POST | `/remediation/{id}/reject` | **approver** | same body |
| POST | `/remediation/{id}/rollback` | **approver** | body `{"reason": "..."}`; re-activates the pre-remediation deployment state as a new revision |

`expected_version` is the `version` the reviewer was looking at. If the remediation changed
since, the decision is refused as stale. A repeated request with the same `Idempotency-Key`,
decision and reviewer returns the recorded outcome without acting again; any other second
decision is a 409.

Lifecycle:

```
PROPOSED -> VALIDATING -> VALIDATED -> PENDING_APPROVAL -> APPROVED -> EXECUTED -> COMPLETED
                      \-> VALIDATION_FAILED          \-> REJECTED        \-> ROLLED_BACK <-/
```

### Observability (viewer)

| Method | Path | Notes |
|---|---|---|
| GET | `/overview` | incident counts, pending approvals, per-dataset deployment, last run, drift status and performance (`null` where not measurable) |
| GET | `/audit?incident_id&entity_id&limit&offset` | append-only audit events, oldest first |
| GET | `/models` | registered model versions with hashes and test metrics |
| GET | `/observability/latency` | p50/p95/max and failure rate per span name; process RSS and CPU |
| GET | `/observability/traces?trace_id&prefix&limit` | recent spans from the in-process buffer |

### Lifecycle reconciliation and benchmark results (viewer)

| Method | Path | Returns |
|---|---|---|
| GET | `/remediation-lifecycle/reconciliation?incident_id` | every remediation record in the audit trail accounted for: `reconciled`, stage and terminal-status counts, rates with numerator / denominator / definition, and `failures` (missing, duplicated or out-of-order records). Denied attempts are excluded and counted in `denied_attempts_excluded`. See `docs/remediation-lifecycle.md` |
| GET | `/evaluation/runs` | names of benchmark runs with a generated summary under `AEGIS_RESULTS_DIR` |
| GET | `/evaluation/runs/{name}` | one run: the two detection metrics, each under its own name (`overall_fault_detection_*`, `drift_detection_*`) with definition, population and confusion counts; engineering targets; diagnosis rates; remediation counts, rates and reconciliation status; claim-validation counters (or `available: false`). 404 for an unknown or malformed name; 409 if the summary was written by an older report version |

The API serves these numbers; it never computes or edits them. They are written by
`python -m aegis.evaluation.report`.

Investigation responses now also carry, inside `llm_ranking_stats` and `explanation`: the model's
raw text (`raw_items`, `claim_validation.raw_summary`), one record per validated claim, the claim
counters, and `explanation.output_audit` (a re-scan of the final output). See
`docs/evidence-and-claim-validation.md`.

## Example

```bash
T_OP=...; T_AP=...; API=http://localhost:8000
BATCH=$(curl -s -H "Authorization: Bearer $T_OP" -F dataset=adult -F file=@batch.csv $API/data/current | jq -r .id)
INC=$(curl -s -H "Authorization: Bearer $T_OP" -H 'content-type: application/json' \
      -d "{\"dataset\":\"adult\",\"batch_id\":\"$BATCH\"}" $API/monitor/run | jq -r .incident.id)
curl -s -H "Authorization: Bearer $T_OP" -H 'content-type: application/json' \
     -d '{"use_llm": false, "wait": true}' $API/incidents/$INC/investigate
REM=$(curl -s -H "Authorization: Bearer $T_OP" $API/incidents/$INC | jq -r '.remediations[0]')
curl -s -H "Authorization: Bearer $T_AP" -H 'content-type: application/json' \
     -d "{\"expected_version\": $(echo $REM | jq .version), \"comment\": \"ok\"}" \
     $API/remediation/$(echo $REM | jq -r .id)/approve
```

`uv run python -m scripts.demo` does the same thing with commentary.
