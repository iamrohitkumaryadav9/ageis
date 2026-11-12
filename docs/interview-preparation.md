# Interview preparation: explaining AEGIS

A walkthrough of the design, the reasoning behind each decision, and the questions a senior
engineer is likely to ask. Numbers quoted here are from `docs/benchmark.md`; re-run the
benchmark and check them before quoting.

## 1. Thirty-second version

AEGIS watches a deployed classifier. When incoming data or model behaviour departs from a
reference window it opens an incident, gathers evidence with statistical monitors, ranks
root-cause hypotheses, retrieves the matching runbook, proposes a remediation, tests that
remediation in a sandbox, and waits for an authenticated human to approve before changing
anything. A local 4B-parameter LLM plans diagnostics, re-ranks hypotheses and writes the
summary, but everything that has to be true is computed by ordinary code, and the system works
without the LLM.

## 2. Architecture walkthrough

Follow one batch:

1. **Ingestion** stores the CSV as Parquet with a checksum and provenance row.
2. **Monitoring** runs six monitors against the reference window and data contract: quality,
   drift, integrity, performance, concept drift, feature/label relationship. Output: typed
   `Finding`s, each labelled *observed fact* or *statistical anomaly*.
3. **Incident**: any triggered finding of medium severity or above opens a persistent incident
   and stores each finding as an evidence record with a stable id.
4. **Investigation** (LangGraph): choose diagnostics, run them through an allowlist, rank
   hypotheses, validate what the LLM said, retrieve runbooks, propose, sandbox-validate, apply
   policy, request approval.
5. **Remediation** is a state machine persisted in PostgreSQL. Approval is an endpoint that
   checks the caller's role, then approves, executes and verifies in one transaction.
6. **Audit**: every transition and every refused attempt is an append-only row.

The single mutable thing remediation can touch is a small versioned `DeploymentState` (model
version, ingestion config, quarantine flags). Executing writes a new revision; rollback
re-activates the old one. That narrowness is what makes "controlled" a property rather than an
aspiration.

## 3. Why each technology

| Choice | Reason | What I would say against it |
|---|---|---|
| scikit-learn, HistGradientBoosting | strong tabular baseline, trains in seconds on CPU, native NaN handling | not the point of the project; any calibrated classifier would do |
| SciPy for KS / chi-square / Wasserstein | reference implementations; I own the decision logic on top | - |
| River (ADWIN, Page-Hinkley) | maintained streaming detectors with the right update API | they need labels and say nothing about cause |
| Evidently | independent second implementation of the drift statistics, used as a cross-check | not on the request path, by choice |
| SHAP | bounded TreeExplainer attributions to say whether a drifted feature matters to the model | 200 rows only; global attribution, not causal |
| LangGraph | explicit typed state graph with conditional edges: the loop bound is visible in the graph | a plain function pipeline would also work; the graph earns its place through the conditional loop and per-node tracing |
| Ollama + qwen3:4b | local, no paid API, structured output via JSON-schema-constrained decoding | slow on CPU (about 3.5 minutes per investigation) |
| FastAPI + Pydantic v2 | request/response validation is the same mechanism as tool and LLM-output validation | - |
| PostgreSQL | row locks, unique constraints and triggers do real safety work here (single approval, append-only audit) | SQLite would not give `FOR UPDATE` semantics |
| MLflow | experiment and model-version tracking | serving integrity needs a content hash, so a small registry sits beside it |
| OpenTelemetry | one trace from HTTP request to tool call to LLM request | in-process buffer is a convenience, not a tracing backend |

**Evidently**: AEGIS's monitors are implemented directly on SciPy because the benchmark needs
per-method verdicts, BH adjustment and effect-size gating that I wanted to own and test. Evidently
is used where a second opinion is worth most: `aegis/monitoring/crosscheck.py` runs its per-column
drift metrics on the same frames, and `tests/unit/test_crosscheck.py` asserts that the two
implementations agree (KS p-values, Wasserstein distance in reference standard deviations,
categorical PSI, and the set of flagged features). I would rather say that than pretend a
dashboard library is the monitoring engine.

## 4. Statistics: methods and assumptions

- **Kolmogorov-Smirnov** (numeric): largest ECDF gap. Non-parametric, assumes i.i.d. continuous
  data; conservative with ties; most sensitive near the median, weak in the tails.
- **Chi-square homogeneity** (categorical): needs adequate expected counts, so rare levels are
  pooled. Sensitive to n, blind to magnitude.
- **PSI**: symmetrised KL over bins. An effect size with no sampling distribution; depends on
  binning; the 0.1 / 0.25 bands are convention, not theory.
- **Wasserstein-1**, in reference standard deviations: "how far mass moved". Catches scale
  changes that saturate KS and PSI; sensitive to outliers.
- **Multiple testing**: one test per feature, so p-values are Benjamini-Hochberg adjusted (false
  discovery rate), which is less conservative than Bonferroni and appropriate when several
  features may truly drift together.
- **Significance is not importance**: at n = 10,000 a 0.09-sd shift is significant. Every alarm
  therefore requires a significant test *and* a material effect. There is a test that feeds a
  significant-but-trivial shift and asserts it is not flagged.
- **Size-matched expectations**: duplicate rate and cardinality depend on batch size, so the
  reference value is computed on reference subsamples of the same size.
- **Proportions in the benchmark** are reported with Wilson 95% intervals, because the LLM
  configurations run on about twenty scenarios and a bare percentage would overstate certainty.

## 5. Data drift versus concept drift

- **Covariate (data) drift**: P(X) changes. Detectable without labels. Not automatically a
  problem: the model may be robust to it.
- **Concept drift**: P(Y | X) changes. Needs labels. The inputs look normal and the errors rise.
- **Label (prior) shift**: P(Y) changes; shows up in the prediction distribution.

AEGIS tells them apart by what is *absent*: the rule for concept drift is boosted when inputs
look like the reference and discounted when a drifted feature, a quality problem, a wrong
artifact or a leakage signal could explain the loss. That is why "normal" results are kept as
evidence. Limits to state plainly: with delayed labels concept drift is invisible until they
arrive; label noise is indistinguishable from concept drift to every detector here.

ADWIN keeps an adaptive window and compares sub-window means with a Hoeffding-style bound: it
has false-positive guarantees and handles gradual change. Page-Hinkley is a CUSUM-type test for
an upward mean shift: cheap, good at abrupt change, parameters are scale-dependent. Both take
many looks at the stream, so an alarm only counts if the overall error rate also rose
significantly.

## 6. Why the LLM is not the source of truth

Three reasons, in order of importance:

1. **It cannot be made reliable enough for decisions that change production.** A statistic is
   reproducible and testable; a 4B model's arithmetic is neither.
2. **Its input is partly attacker-controlled** (data values, runbook text). Anything it decides
   could be decided by whoever can write a category label.
3. **It is optional infrastructure.** If the diagnosis lived in the LLM, an Ollama outage would
   be an AEGIS outage.

So the LLM gets three bounded jobs (plan, rank, explain), each with a schema and a validator,
and each with a deterministic fallback. What it says is checked against the evidence store:
unknown causes, fabricated evidence ids, hypotheses with no anomaly evidence, causal wording
("caused by", "proves") and false status claims ("has been approved") are removed and counted.

Be honest about the corollary: on the benchmark the rule engine is already near its ceiling, so
the LLM mostly adds flexibility in tool choice and readable summaries, not accuracy. See the
ablation section.

## 7. Agent state management

- Typed `InvestigationState` (TypedDict); list fields that accumulate (events, errors, LLM
  calls) use reducer annotations; everything in the state is JSON-serialisable.
- Dataframes, the registry and the tool executor are *not* in the state: they travel in
  `config["configurable"]`, so the state can be persisted and inspected.
- One conditional edge creates the only loop, guarded by a loop counter and the tool budget.
- Each node is wrapped: cancellation check, span, timing, exception containment. A failing node
  appends to `errors` and the run continues.
- No checkpointer: the durable state is the database, and human approval is a row in it.

## 8. RAG design

Eight versioned Markdown runbooks, split at `##` headings into about forty passages with ids such
as `RB-004#remediation`. TF-IDF cosine retrieval by default; optional reciprocal-rank fusion with
local embeddings. Evaluated on 24 labelled queries (hit@1, hit@3, MRR) for lexical, dense and
hybrid; lexical already reaches hit@3 = 1.0, and hybrid costs a model swap on a small machine,
so lexical is the default (decision D10).

Runbooks are untrusted data. Passages with instruction-like phrasing are flagged, withheld from
LLM prompts and shown to humans with a warning. More important than the detector: retrieved text
has no path to the tool executor, the policy engine or approval. There is a test showing the
policy decision is byte-identical with a poisoned runbook and with no runbooks at all.

## 9. Tool security

- Ten tools, declared in one table; nothing runs code, shell or file reads.
- Pydantic inputs with `extra="forbid"`; then semantic checks (the feature must be in the
  contract, the version must be registered).
- The LLM may select only five read-only diagnostic tools; remediation tools are driven by the
  workflow. The caller-specific allowlist is enforced in the executor, not the prompt.
- Budget of five diagnostic calls per investigation, enforced by the executor; per-call timeout.
- Every attempt, including rejected ones, is traced with requested and validated arguments.

## 10. Human approval design

Walk through the decision path in `docs/architecture.md`. Points to make:

- The request body has no `approved` or `reviewer` field; unknown fields are rejected.
- One approval per remediation is a UNIQUE constraint, so two concurrent approvers cannot both
  win; there is a six-thread test.
- Two kinds of staleness are handled separately: the *review* is stale (`expected_version`) or
  the *validation* is stale (deployment revision moved).
- Approve, execute and verify share one transaction. A crash after the approval insert leaves
  no approval.
- Optional separation of duties: approver must differ from requester.
- Refused attempts are audited in their own transaction, because the main one rolls back.

## 11. Failure handling

| Failure | Behaviour |
|---|---|
| Ollama offline | health check fails fast; investigation runs the deterministic path; `llm_status = unavailable` is stored and shown |
| Ollama timeout | no retry (it would double the wait); remaining LLM steps are skipped for that investigation |
| Invalid LLM output | one retry with the validation error fed back, then the deterministic result |
| Primary model missing | configured local fallback model; never a cloud endpoint |
| Tool exception / timeout | traced as error/timeout; investigation continues |
| Node exception | recorded; later nodes work with partial state |
| Batch violates contract | scoring skipped with a reason; schema finding raised; no fabricated metrics |
| Too few labels | supervised metrics skipped, label-availability finding raised |
| Unverified artifact | never loaded |
| Database down | `/ready` 503, requests 503, no partial writes |
| Background investigation crash | investigation marked `failed`, incident returned to `open`, audited |

## 12. Benchmark methodology

AEGIS-IR: 2 datasets x 10 fault categories x 3 severities x 2 seeds = 120 faulty scenarios plus
50 clean controls. Scenario ids are hashes; ground truth lives in a separate file read only by
the scorer; prompts are tested not to contain it. Thresholds were tuned on seeds 101-102 and the
runner refuses those seeds; results use 201-202.

Four configurations on identical scenarios: A alerts only, B rules, C LLM alone, D full agent.
Detection is shared, so the ablation isolates diagnosis and remediation. A and B run on all 170
scenarios; C and D on a stratified subset of 20 because each LLM scenario costs minutes of CPU.

Weaknesses to volunteer before being asked:

- I wrote both the fault injectors and the rules. The rules are near ceiling by construction;
  the benchmark shows the system is internally consistent, not that it would diagnose incidents
  it was not designed around.
- Faults are single and synthetic. Real incidents are compound and messier.
- Twenty LLM scenarios give wide confidence intervals.
- F10 (artifact mismatch) is detected by a hash comparison: 100% recall is a tautology.
- "Supported-fault remediation success" counts quarantine as a remediation, which is a low bar.

## 13. Ablation findings

Measured on AEGIS-IR runs `main` and `v2` (see `docs/benchmark.md`). The deterministic
configurations reproduced exactly between the two runs.

- **Two detection numbers, two populations.** `overall_fault_detection_f1` = 0.9958: all ten
  fault categories, 120 faulty and 50 clean scenarios, a detection is any incident. And
  `drift_detection_f1` = 0.9565: only F01, F02 and F04 (36 scenarios) against the same 50 clean
  ones, a detection is the drift monitor itself triggering. Never quote one for the other.
- **A -> B (alerts -> diagnostics + rules): Top-1 0.815 -> 0.966 on 119 detected faults**,
  intervals 0.74-0.88 and 0.92-0.99. The one difference the benchmark has the power to support.
- **B -> D (add the LLM agent): 19/20 and 19/20, in both runs.** No accuracy gain, and partly by
  construction, because the LLM can move a rule score by at most 0.15. The cost is latency:
  p50 139 s with the explanation step against under 1 s.
- **C (the 4 GB model alone): 11/20 in both runs**, three unusable outputs each. n=20, interval
  0.34-0.74.
- **Claim validation in D (`v2`)**: 185 claims evaluated; 7 classified causal, all rejected, none
  supported; 11 unsupported and 2 contradictory claims dropped; 19 fabricated evidence ids
  stripped; nothing the validator recognises reached the output.
- **And the validator was too strict.** 9 of 20 explanations were discarded. Reading them: 2 had
  a real causal assertion, 4 were the system's own decision rationale ("no_action due to no
  performance loss"), 3 were discarded over an operator check. The validator was corrected and
  an offline re-check of the stored summaries accepts 18 of 20. I report the run as it was, and
  the correction separately.
- **Remediation accounting**: 119 proposals = 32 no-action + 2 rejected by policy before
  validation + 85 validated (68 passed, 17 failed). 68 approval requests, 68 simulated
  executions, 67 resolved, 68 rollbacks. Every record reconciles.

The useful sentences:

- "The LLM did not improve Top-1 over the rule engine on this benchmark. Its value is tool
  selection and explanation, and the validation counters show why it is not allowed to be the
  source of truth."
- "'Zero over-claims escaped' is a consistency check on my own validator, not a proof. It cannot
  see a paraphrase outside its lexicon, and I keep four such sentences in a test so the limit is
  on record."
- "My first benchmark report had a 'causal over-claims: 0' line that meant only that one regular
  expression did not match. I replaced it with counters whose numerator and denominator are
  stated, and marked the old run's figures as unavailable."

What I would not claim: that D is "as good as" B in general (n=20); that the false-positive rate
is zero (0/50 bounds it under about 7%); that the claim validator catches every over-claim; that
these synthetic faults represent production incidents.

## 14. Limitations and trade-offs

- Batch monitoring, not streaming; detection delay is measured in windows.
- Two public tabular datasets with sensitive attributes; nothing here supports real lending or
  employment decisions.
- Static bearer tokens; single tenant; no rate limiting.
- Retraining validation is conservative enough to reject a genuine permanent concept change.
- CPU inference makes the LLM path minutes long; prompt length is a first-class budget.
- The rule engine encodes ten causes; an eleventh needs a rule and a runbook.
- In-process span buffer and single API process: fine for a laptop, not a fleet.

## 15. Questions a senior engineer will ask

**"Why not let the LLM call the fix directly?"** Because the blast radius of a wrong answer
should not depend on the model's reliability or on who can write text into its context. The
tools it can reach are read-only by construction.

**"What stops a prompt injection in the CSV?"** Rows never enter a prompt. The LLM sees
aggregate statements; category examples are truncated to 40 printable characters and appear only
in stored findings. And even a fully hijacked model can only call read-only diagnostics and emit
a ranking that the validator bounds.

**"Your KS test fires on everything at scale. How do you handle that?"** Effect-size gate plus
BH adjustment; show the method-comparison table and the "significant but immaterial" test.

**"How do you know a fix worked?"** Three separate measurements: sandbox validation on a
held-out half of the batch before approval, post-execution verification on the incident batch,
and the next monitoring run. Rollback is one call if any of them was wrong.

**"What happens if two people approve at once?"** Row lock plus a UNIQUE constraint on the
approval row; one wins, the others get 409. Tested with six threads.

**"Drift was detected. Why didn't you retrain?"** Drift is a property of inputs; retraining is
justified by measured performance loss and a candidate that beats the deployed model on held-out
recent data without collapsing on the reference set. Otherwise the action is "monitor".

**"How would this scale to 500 models?"** Monitoring is cheap and stateless per batch
(hundreds of milliseconds): shard it. The registry and deployment state already key on dataset;
they would key on model id. The investigation worker becomes a queue. The LLM becomes a shared
service with a real batching runtime. The audit and approval design does not change.

**"What would you change with a GPU or a bigger model?"** Longer evidence listings, an
iterative tool loop instead of a single plan, and LLM-proposed hypotheses outside the fixed
taxonomy. The validation and policy layers would stay exactly as they are.

**"Where is the weakest part?"** The benchmark's independence: faults and rules share an
author. The next step would be incidents contributed by someone else, compound faults, and a
held-out fault category the rules have never seen, to measure the "insufficient evidence" path.

**"Why PostgreSQL JSONB columns instead of fully normalised tables?"** Facts that are queried
or constrained are columns (state, severity, reviewer, revision). Rich payloads that are written
once and read whole (reports, validation checks) are documents. Normalising a monitoring report
into forty tables would add joins and no integrity.

**"Is confidence a probability?"** No. It is an evidence-fit score used for ranking and
thresholds. It is not calibrated, and the system never says "caused by".
