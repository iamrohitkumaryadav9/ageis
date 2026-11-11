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

## How the controls are layered

No single check is trusted to hold alone.

1. **Capability**: the tool allowlist contains nothing that executes code, reads paths, approves
   or deploys. A model that is fully compromised by injected text still has only read-only
   diagnostics to call, at most five times.
2. **Validation**: tool arguments (`extra="forbid"`), LLM output (JSON schema at decode time,
   Pydantic after), cited evidence ids, causal wording and status claims are each checked;
   anything that fails is dropped and counted, and the deterministic result stands.
3. **Policy**: the decision engine reads evidence and rule-engine scores, not LLM prose or
   runbook text. An LLM-only hypothesis cannot reach the action threshold.
4. **State machine**: there is no edge from `PROPOSED` to `EXECUTED` that avoids `VALIDATED` and
   `APPROVED`; the test suite walks the table exhaustively.
5. **Authorisation**: approval is a database row written by an endpoint that checks the caller's
   role from a bearer token; the row is unique per remediation at the schema level.
6. **Transaction**: approve, execute and verify commit together or not at all, with row locks,
   a version check against stale reviews and a revision check against stale validations.
7. **Audit**: every transition, and every refused attempt, is an append-only row; a database
   trigger rejects UPDATE and DELETE on the audit table.

## Residual risks

- Bearer tokens are static and shared-secret; there is no rotation, expiry or SSO. Suitable for
  a local deployment, not for a network-exposed one.
- A database superuser can disable the audit trigger. The audit trail is tamper-evident against
  the application, not against the DBA.
- The injection detector for runbooks is a phrase list. It reduces exposure; the real control is
  that runbook text cannot reach tools, policy or approval.
- Registered models are still pickles. The hash check proves a file is the one that was
  registered, not that the registered file was benign.
- Rate limiting is not implemented; the single investigation worker bounds LLM load but not
  request volume.

## Out of scope

Multi-tenant isolation, network-level attackers (terminate TLS in front of the API), and
supply-chain compromise of dependencies beyond lockfile pinning.
