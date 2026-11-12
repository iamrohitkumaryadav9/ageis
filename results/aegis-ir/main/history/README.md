# History of run `main`

Raw and derived files of AEGIS-IR run `main` as they stood before later changes. Nothing in this
directory is regenerated.

| File | What it is |
|---|---|
| `scenarios.2026-11-11.before-noise-allowance.jsonl` | Raw per-scenario results of the first complete run, before configuration B was re-run with the sandbox noise-allowance fix. Detection, A, C and D records are identical to the current `../scenarios.jsonl`; only B differs (one validation: fail -> pass). |
| `summary.as-generated-2026-11-11.json`, `report.as-generated-2026-11-11.md`, `scenarios.as-generated-2026-11-11.csv` | The derived files exactly as generated on 2026-11-11 by the report code of that day (metric keys `drift_f1`, unnamed "F1", no populations, no lifecycle reconciliation). |

The current `../summary.json` and `../report.md` are regenerated from the unchanged
`../scenarios.jsonl` by the current report code. They restate the same raw results under explicit
metric names; no raw value was edited. SHA-256 of `../scenarios.jsonl`:
`910170c1f1119b6d119b7b361767fcf4b0c8d32d43d3ccde94767c9d715374f5` (pinned by a test).

Known defect in `../manifest.json`: its `llm_model` and `llm_subset` fields are empty, because the
manifest was rewritten when configuration B was re-run without the LLM flag. The LLM
configurations C and D of this run were executed with `qwen3:4b` on 20 scenarios; that is visible
in `../scenarios.jsonl` (records carrying `C` and `D`) and in the generated report, which counts
them from the raw records. The manifest is left as it is; the runner no longer overwrites these
fields on a resumed run.
