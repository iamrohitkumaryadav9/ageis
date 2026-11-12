# Demo transcript

Unedited output of `uv run python -m scripts.demo` against a locally running API
(PostgreSQL 17 in Docker, Ollama 0.40.16 with `qwen3:4b`, Intel i5-10210U, 4 physical cores, 7.65 GiB RAM visible to WSL2, no GPU),
recorded on 2026-11-12. The only post-processing is removal of the repeated
`... running (Ns)` progress markers. IDs and timings will differ on every run.

The model was unloaded before the run, so the 219 s investigation includes a 23 s cold model load.
Each step is an HTTP call to the API; the fault is injected into the uploaded CSV and the ground
truth printed on the last line is never sent to AEGIS.

To reproduce: `make test-db-up bootstrap`, start Ollama (`scripts/ollama_serve.sh`) and the API
(`make api`), then run the demo with an operator and an approver token (see the README).
Add `--no-llm` for the deterministic path, which completes in a few seconds.

```text

=== 0. readiness ===
  database  ok           required  
  models    ok           required  2 active deployment(s)
  runbooks  ok           optional  40 passages, corpus ed30f7acf68c
  llm       ok           optional  qwen3:4b via Ollama 0.40.16

=== 1. upload a production batch (2000 rows; fault injected: F04, hidden from AEGIS) ===
  batch BAT-2a40262acf94  rows=2000  sha256=f83434fe3aaf88a5...

=== 2. run monitoring ===
  run REP-a39ca6a91521: incident_detected=True severity=medium (1 findings, 699 ms)
  incident INC-196d44cf8466 opened: 4.4% of 'capital_loss' values are non-finite or outside the tolerated reference range [-368.3, 4051] (observed min 0, max 2.614e+05)

=== 3. investigate (local LLM; this takes minutes on a CPU) ===











































  ... completed (221s)
  status=completed llm_status=available model=qwen3:4b duration=219.2s
    llm plan     ok=True 61.8s error=None
    llm rank     ok=True 70.0s error=None
    llm explain  ok=True 63.1s error=None
    tool get_data_quality_report        ok       by llm      32 ms
    tool get_data_drift_report          ok       by llm      2 ms
    tool inspect_feature_distribution   ok       by rules    36 ms
    tool retrieve_runbook               ok       by workflow 7 ms
    tool create_remediation_proposal    ok       by workflow 0 ms
    tool run_candidate_validation       ok       by workflow 1141 ms
    tool request_human_approval         ok       by workflow 0 ms

=== 4. root-cause hypotheses (ranked by evidence fit; not established causes) ===
  #1 scaling_error      fit=0.88 by=llm support=2 contra=2
  conclusion: The evidence is most consistent with scaling error (evidence fit 0.88) affecting ['capital_loss']. This is a ranked hypothesis, not established causation.
  runbooks: ['RB-004#symptoms', 'RB-002#diagnosis', 'RB-003#symptoms']

=== 5. remediation proposal, sandbox validation, policy ===
  REM-429938736685: fix_ingestion_scale {'factor': 0.017344497607655503, 'feature': 'capital_loss', 'observed_factor': 57.6551724137931}  state=PENDING_APPROVAL
    [pass] precondition parameters_valid: feature='capital_loss', factor=0.017344497607655503
    [pass] effect       factor_reproducible_on_independent_half: half-sample estimate x57.82752217866721 (scale) vs proposed inverse 0.01734
    [pass] effect       targeted_findings_cleared: all targeted findings cleared
    [pass] regression   no_new_findings: no new findings
    [pass] contract     data_contract_valid: corrected data conforms to the contract
    [pass] regression   performance_not_degraded: ROC-AUC after 0.898 vs before 0.902, reference 0.933 (floor 0.874; sampling-noise allowance 0.028)
  policy: REQUEST_HUMAN_APPROVAL  ['score 0.72 = benefit 0.79 - risk 0.02 - cost 0.05', 'validated in sandbox; medium-risk state change requires human approval']

=== 6. approval is enforced by the backend ===
  operator tries to approve -> HTTP 403: the approver role is required to decide a remediation
  approver approves -> state=COMPLETED executed=True deployment revision #6
  verification: {'passed': True, 'summary': 'incident batch is healthy under the new deployment state', 'remaining_findings': []}

=== 7. the same batch under the new deployment ===
  incident_detected=False (deployment revision #6)

=== 8. audit trail ===
  04:08:11 bob     monitoring.run                     ok
  04:08:11 bob     incident.opened                    ok
  04:08:11 bob     investigation.started              ok
  04:11:51 bob     remediation.proposed               ok
  04:11:51 bob     remediation.start_validation       ok
  04:11:51 bob     remediation.validation_passed      ok
  04:11:51 bob     remediation.request_approval       ok
  04:11:51 bob     investigation.completed            ok
  04:11:52 bob     remediation.approve                denied
  04:11:52 alice   remediation.approve                ok
  04:11:52 alice   deployment.revision.activated      ok
  04:11:52 alice   remediation.execute                ok
  04:11:53 alice   remediation.verify                 ok

=== 9. rollback (restores the pre-remediation deployment state) ===
  state=ROLLED_BACK rollback revision #7

ground truth (never shown to AEGIS): F04 on ['capital_loss'] {'factor': 60.0}
```
