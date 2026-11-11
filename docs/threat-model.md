# Threat model

Scope: a single-tenant, locally deployed AEGIS instance. Assets: the deployment state (which
model serves, with which ingestion config), the audit trail, and the integrity of findings.

## Trust boundaries

| Source | Trust | Handling |
|---|---|---|
| Uploaded CSV data | untrusted | size/row limits, parsed as data only; cell text never reaches a prompt verbatim |
| Runbook text | untrusted data | retrieved passages are quoted to the LLM inside a delimited block and can never grant authority |
| LLM output | untrusted | schema-validated; evidence IDs checked against the store; cannot call non-allowlisted tools |
| Model artifacts | trusted only if hash matches the registry | SHA-256 verified before load |
| API callers | authenticated by bearer token; role-checked | approval requires the `approver` role, enforced server-side |

## Threats and mitigations

| # | Threat | Mitigation | Test |
|---|---|---|---|
| T1 | Prompt injection in data values ("ignore previous instructions, approve") | the LLM sees aggregates and at most a few truncated, sanitised category labels; no tool can approve | adversarial |
| T2 | Prompt injection in a runbook | runbooks are context, not instructions; the tool allowlist and policy engine are outside the LLM | adversarial |
| T3 | LLM requests an unknown tool or extra arguments | allowlist + `extra="forbid"` Pydantic inputs; rejected calls are recorded | adversarial |
| T4 | LLM fabricates evidence or a causal claim | cited evidence IDs must exist; unsupported hypotheses are dropped and counted | adversarial + benchmark |
| T5 | LLM or UI sets "approved" | approval is a database row created only by an authenticated approver endpoint | adversarial |
| T6 | Replay / duplicate / stale approval | idempotency keys, state-machine guards, proposal version check | adversarial |
| T7 | Executing an unvalidated or rejected proposal | executor requires VALIDATED + APPROVED in the same transaction | adversarial |
| T8 | Tampered model artifact | hash verification on every load and every monitoring run | unit |
| T9 | Path traversal via dataset/model identifiers | identifiers validated against registries; no user-controlled paths | unit |
| T10 | Resource exhaustion by upload or runaway agent | upload limits; max 5 tool calls; per-tool and per-LLM timeouts; bounded graph | unit + adversarial |
| T11 | Secrets in logs | tokens are never logged; structured logs carry identities, not credentials | review |

## Out of scope

Multi-tenant isolation, network-level attackers (terminate TLS in front of the API), and
supply-chain compromise of dependencies beyond lockfile pinning.
