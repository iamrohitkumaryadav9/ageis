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

Status is tracked in the README "Definition of done" table; a box is ticked only after the
corresponding command has been executed.
