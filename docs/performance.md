# Performance and resource report

Everything here was measured on the build machine. Nothing is estimated or extrapolated.

| | |
|---|---|
| CPU | Intel Core i5-10210U, 4 cores / 8 threads, no GPU |
| RAM | 7.7 GB visible to WSL2, 2 GB swap |
| OS | Linux 6.18 (WSL2), Python 3.12 |
| LLM | `qwen3:4b` on Ollama 0.40.16 (llama.cpp runner, CPU only), context 4096, one request at a time |
| Database | PostgreSQL 17 in Docker |

## Deterministic path

Source: AEGIS-IR run `main`, 170 batches of 1,000-3,000 rows.

| Stage | p50 | p95 |
|---|---|---|
| Whole monitoring run per batch | 588 ms | 939 ms |
| - relationship monitor (per-feature AUC against a size-matched reference) | 274 ms | 467 ms |
| - data quality | 125 ms | 225 ms |
| - drift tests | 78 ms | 199 ms |
| - performance metrics | 64 ms | 134 ms |
| - concept drift (ADWIN, Page-Hinkley) | 20 ms | 39 ms |
| - artifact integrity (SHA-256) | 1.5 ms | 4 ms |
| Investigation without the LLM (diagnostics, rules, runbook retrieval, sandbox validation, policy) | 0.93 s | 4.2 s |

The deterministic benchmark run (170 scenarios, configurations A and B, plus the detection-delay
study) took 396 s, peaked at 402 MB resident in the Python process and 1.7 GB system-wide, at a
mean 35% of total CPU.

## LLM path

| Measurement | Value | Source |
|---|---|---|
| Model on disk | 2.6 GB | `~/.ollama/models` |
| Model in memory | 3.1 GB resident (Ollama server + runner) | demo run, sampler following child processes |
| API process | 256-340 MB resident | demo run |
| Cold model load | 23 s to more than 120 s | see below |
| Prompt evaluation | 11-14 tokens/s | Ollama server timings |
| Generation | 3.0-4.5 tokens/s | Ollama server timings |
| One LLM request, model loaded | 44 s mean (configuration D), 97 s mean (configuration C, longer output) | benchmark |
| Full investigation with LLM (plan, rank, explain) | 215-219 s | two demo runs |
| Investigation in the benchmark (plan and rank only) | p50 84 s, p95 125 s | benchmark, configuration D |

The benchmark's LLM run (40 + 20 model requests among 170 scenarios) took 68 minutes at a mean
64% of total CPU, and **system memory in use peaked at 7.1 GB of 7.7 GB**, with swap active.

### What this means in practice

- **8 GB is the floor, not a comfortable size.** With the model (3.1 GB), the API, PostgreSQL, an
  editor and a browser, this machine swaps. 16 GB is the realistic recommendation; on 8 GB, close
  other applications or use the smaller fallback model.
- **Cold starts are slow and variable.** Ollama unloads the model after two idle minutes. Reloading
  3 GB took 23 s when the file was still in the page cache and more than two minutes when it was
  not. Two demo runs failed their first LLM request for exactly this reason before model loading
  was given its own budget (`AEGIS_OLLAMA_LOAD_TIMEOUT_S`, default 300 s) separate from the
  per-request timeout. When the load or a request does time out, the investigation completes on
  the deterministic path and says so.
- **The default request timeout of 120 s is tight on this CPU.** The benchmark and demo used 180 s
  (`AEGIS_OLLAMA_TIMEOUT_S=180`), which Docker Compose also sets. At 120 s the ranking request,
  which generates the most tokens, occasionally times out.
- **Prompt caching matters.** The system prompt is static, so the runner reuses its evaluated
  prefix; consecutive requests evaluate about 380-430 new prompt tokens instead of 700+.
- The LLM adds roughly 80-220 s to an investigation that otherwise takes about one second, and did
  not improve diagnostic accuracy on the benchmark (see [benchmark.md](benchmark.md)). Running
  with `AEGIS_LLM_ENABLED=false` is a legitimate configuration.

## Known measurement gap

During the benchmark's LLM run the resource sampler matched processes by name and missed the
`llama-server` runner that holds the model, so that run has no valid per-process Ollama figure; the
generated report withholds it. The system-wide figure for that run is valid. The sampler has since
been fixed to follow child processes, and the 3.1 GB figure above comes from a demo run measured
with the fixed sampler. The benchmark was not re-run for this (68 minutes).

Not measured: dense-embedding retrieval latency under load, dashboard rendering performance, and
behaviour with more than one concurrent investigation (the service runs one at a time by design).
