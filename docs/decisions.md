# Decision log

Short ADR-style entries. Newest last.

### D01 - Environment has 7.65 GiB of RAM visible, not 16 GB
The WSL2 VM on the build machine exposes 7.65 GiB (8.22 GB in decimal units; the host has 15.8 GiB
and no `.wslconfig`, so WSL2 takes half) on an i5-10210U, 4 physical cores / 8 logical. All defaults are sized for
that: one LLM loaded at a time, bounded SHAP samples, a single API process.

### D02 - Pinned versions resolve without conflict
`scikit-learn 1.9.1`, `river 0.22.0`, `evidently 0.7.23`, `langgraph 1.2.20`, `mlflow 3.16.1`
resolve together on Python 3.12 under uv (`uv.lock` committed). No pin was changed. Unpinned
packages float to what uv resolved and are frozen by the lockfile.

### D03 - Reference window = validation split
Drift is measured against data the model was *not* fitted on, so reference score distributions
are not optimistically biased by training-set memorisation.

### D04 - Dropped columns
Adult: `fnlwgt` (sampling weight), `education` (duplicate of `education_num`).
Bank: `duration` - known only after the call outcome; using it is target leakage, as the dataset
authors note. Dropping it lowers headline F1 substantially and is the correct trade.

### D05 - Model selection by CV ROC-AUC with a simplicity margin
Accuracy is not used: majority-class accuracy is 76% (Adult) and 88% (Bank). A more complex
model must beat logistic regression by > 0.005 CV ROC-AUC. Threshold is tuned for F1 on
validation; test is touched once.

### D06 - Own file registry for integrity, MLflow for tracking
MLflow records experiments, metrics and model versions. The serving-integrity check (F10) needs
a content hash compared on every monitoring run, so a minimal registry stores
`record.json` + `model.joblib` and verifies SHA-256 on load. Two tools, two jobs.

### D07 - MLflow skops trusted types
MLflow 3.16 serialises scikit-learn models with skops and refuses unknown types. `numpy.dtype`
and HGB's `TreePredictor` are explicitly trusted for models produced by this repository's own
training code. Nothing downloaded is ever deserialised.

### D08 - Significance AND effect size for every statistical alarm
With 1,000-10,000 rows almost any difference is statistically significant. A drift finding
needs a Benjamini-Hochberg-adjusted p-value below alpha *and* a material effect (PSI,
Wasserstein in reference standard deviations, KS D, Cramer's V or total variation). The same
pattern gates the concept-drift detectors (sequential alarm AND a significant error-rate rise)
and the leakage check (gain in association AND several standard errors). The benchmark's
method-comparison table shows what each ingredient contributes on identical data.

### D09 - Thresholds were tuned on development seeds only
Monitoring thresholds were adjusted while looking at scenarios from seeds 101 and 102. The
benchmark refuses to run on those seeds; reported numbers use seeds 201 and 202. Three faint
misses seen on the development seeds were left as misses rather than tuned away.

### D10 - Lexical runbook retrieval by default; hybrid is optional
Measured on 24 labelled queries over 40 passages: lexical TF-IDF hit@1 0.83 / hit@3 1.00 /
MRR 0.91 at 2 ms per query; dense (nomic-embed-text) 0.75 / 0.92 / 0.85; hybrid (reciprocal rank
fusion) 0.92 / 1.00 / 0.95 at 56 ms. Three passages are retrieved, and both lexical and hybrid
put a relevant runbook in the top three every time. Hybrid's hit@1 gain costs loading a second
model, which on an 8 GB machine evicts the chat model (about 15 s to reload on each swap). So
`AEGIS_RAG_METHOD=lexical` is the default and `hybrid` is one setting away. The sample is small;
the decision should be revisited if the corpus grows. No vector database: 40 passages are one
matrix in memory.

### D11 - Prompt size is a latency budget on CPU
Measured on the i5-10210U with qwen3:4b (Q4, 4 threads): about 14 tokens/s prompt evaluation,
5.7 tokens/s generation, 14.5 s cold load, 3.2 GB resident. The first prompts (about 1,000
tokens in, 350 out) could not finish inside 120 s. Prompts were cut to one static system prefix
(rules, cause glossary, tool list - byte-identical across requests so the runtime can reuse its
evaluation) plus a short evidence listing; outputs are capped at 120/260/200 tokens for
plan/rank/explain. A full three-request investigation now takes 3.5 minutes. The default
request timeout stays at 120 s as specified; `.env.example` recommends 180 s for this CPU class
because the ranking request sits close to the limit.

### D12 - The LLM re-ranks, the rules bound
The agent's ranking is accepted only for hypotheses that cite existing anomaly evidence. Its
confidence may move at most 0.15 from the rule engine's score for the same cause, and a cause
the rules do not support is capped at 0.45, below both the "sufficient evidence" line and the
policy's action threshold. The policy engine additionally takes the minimum of the two
confidences. An LLM can therefore change the order of explanations and the wording, and can
argue for more investigation; it cannot manufacture grounds for action.

### D13 - Approval executes; there is no separate execute endpoint
`POST /remediation/{id}/approve` records the approval, executes, and verifies in one database
transaction with the remediation and deployment rows locked. Splitting approval from execution
would create a window in which an approved-but-stale remediation could be executed later. If
anything fails (stale deployment, verification crash) the whole transaction rolls back,
including the approval row.

### D14 - Retraining is validated against the reference set too
A candidate retrained on recent data must beat the deployed model on held-out recent data and
must not lose more than 0.10 ROC-AUC on the reference set. Under the benchmark's label-flip
concept fault the second condition fails, so AEGIS reports "retraining not validated" and asks
for investigation instead of promoting a model fitted to corrupted labels. This is deliberately
conservative: a genuine, permanent concept change would also fail it and needs a human to
re-baseline the reference window.

### D15 - Unverified artifacts are never deserialised
Model files are joblib pickles, and loading a pickle executes it. Monitoring hashes the serving
artifact first; if the bytes match no registry record the file is not opened and scoring is
skipped with a recorded reason. The cost is that prediction-level evidence is unavailable for
that incident; the benefit is that an integrity incident cannot become code execution.

### D16 - One API process, one investigation worker
Investigations run on a single background worker thread. The LLM is the bottleneck and
`OLLAMA_NUM_PARALLEL=1`; concurrent investigations would only queue inside Ollama while holding
dataframes in memory. Monitoring and API reads are unaffected.

### D17 - User-local tool installs
The build machine had no root access, no Node.js in WSL and no Ollama.
`scripts/install_ollama_local.py` streams the official archive and keeps only the CPU runtime
(77 MB instead of 1.5 GB); Node.js LTS was unpacked under `.local/`. Both are git-ignored.

### D18 - Evidently as a cross-check, not the engine
Drift statistics are implemented on SciPy (per-method verdicts, BH adjustment and effect-size
gating are needed by the benchmark and are easier to test when owned). Evidently 0.7.23 computes
the same statistics independently in `aegis/monitoring/crosscheck.py`; a test asserts agreement.
It is deliberately off the request path.


### D19 - Claim validation is lexical and deterministic, and says so
Causal wording in LLM text is recognised by fixed lexicons and compared with an evidence-strength
policy (observation < association < mechanism < intervention); only a passed sandbox validation
for the same cause supports a causal sentence. A model-based classifier was rejected: the thing
being checked is a model's output, and a second model would add an unverifiable judge. The cost
is recall on paraphrases, which is measured by a labelled corpus that includes sentences the
lexicon misses. First live use over-rejected (decision rationale read as causation); that was
corrected and both the run and the correction are reported.

### D20 - Lifecycle stages keep separate identifiers
Proposal, policy decision, validation, approval, execution and rollback each have their own id,
linked by the proposal id and by explicit references. One merged "remediation id" would have
hidden exactly the mismatch this was built to expose (87 proposals, 85 validations).

### D21 - Benchmark approval, execution and rollback stay simulated, and are labelled
Driving 119 scenarios through the API and database would measure the web stack, not the
remediation logic, and would slow the benchmark several-fold. The real path is covered by the
end-to-end tests, which reconcile the audit trail with the same code. Every simulated event
carries `simulated: true`.

### D22 - Hardware-dependent checks are optional release gates
The system is specified to work without an LLM, so a machine that cannot run one can still pass
every mandatory gate. A blocked or failed optional gate downgrades the verdict to "conditionally
ready" and is listed. Browser and Docker gates are mandatory: without them the dashboard and the
deployment are unverified.

### D23 - New development dependencies
Frontend only: `vitest`, `jsdom`, `@testing-library/react`, `@testing-library/dom`,
`@testing-library/user-event` (component tests on the existing Vite configuration) and
`playwright` (the browser check). No Python dependency and no runtime dependency was added;
`uv.lock` is unchanged.
