# Performance and resource report

Everything here was measured on the build machine. Nothing is estimated or extrapolated, and
measurements from different runs are never combined into one figure.

## Environment (verified 2026-11-12)

| | Value | Source |
|---|---|---|
| CPU | Intel Core i5-10210U @ 1.60 GHz, 4 physical cores, 8 logical | `/proc/cpuinfo`, `psutil` |
| GPU | none | |
| Physical host RAM | 16,957,034,496 bytes = 15.79 GiB | Windows host, `Win32_ComputerSystem.TotalPhysicalMemory` |
| WSL2 configuration | no `.wslconfig`; WSL 3.0.5 defaults apply (half of host memory, all logical processors) | Windows host |
| Memory visible to the Linux guest | 8,023,708 kB = **7.65 GiB = 8.22 GB (decimal)** | `/proc/meminfo` |
| Swap | 2.0 GiB | `/proc/swaps` |
| OS | Ubuntu 26.04 under WSL2, Linux 6.18; Python 3.12 | |
| LLM runtime | Ollama 0.40.16 (llama.cpp runner), CPU only, context 4096, one request at a time | |
| Database | PostgreSQL 17 in Docker | |

**"7.7 GB" and "8.2 GB" were the same machine.** Earlier documents quoted the guest's memory as
7.7 (rounded GiB) in one place and 8.2 (decimal GB, in a benchmark manifest) in another. Both
described 8,023,708 kB. Memory is now recorded in bytes with both units. "4 cores" meant physical
cores; the guest sees 8 logical processors.

WSL2 limits on what these numbers mean: guest memory is the VM's allocation, not the host's;
Windows-side memory pressure is invisible from inside; the guest returns freed memory lazily;
"system used" excludes the page cache.

## How resources are measured

`aegis/observability/resources.py`, a sampler thread started by each benchmark invocation.

| Series | What it is |
|---|---|
| `python_rss` | the benchmark process |
| `aegis_tree_rss` | that process plus every process it spawned |
| `ollama_server_rss` | processes running `ollama serve` |
| `ollama_runner_rss` | the server's descendants: the runner that holds the model |
| `process_tree_total_rss` | the three groups together, each process counted once |
| `system_used`, `swap_used` | machine-wide |
| `cpu_percent` | machine-wide, all logical processors |

- Processes are identified by (pid, create time), so a reused PID is not mistaken for the old
  process, and a child older than its "parent" is not adopted.
- A process that vanishes mid-sample is skipped. Any other sampler failure is counted in
  `monitoring_errors` and reported with the run.
- Requested interval 0.5 s. One sample walks the whole process table, so under load the achieved
  interval is longer: 0.70 s in the deterministic run (581 samples in 405 s), 0.53 s in the LLM
  run (9,331 in 4,926 s). A peak shorter than that can be missed.
- RSS is summed per group, so shared library pages are counted once per process that maps them.
  The group totals are upper bounds; `system_used` is the cross-check.

**Why the runner was missed before.** The original sampler summed processes whose name contains
"ollama". Ollama 0.40 runs the model in a child process named `llama-server`. The runner is now
found through the process tree, whatever it is called (tested with fake trees and against a real
loaded model).

## New measurements: benchmark run `v2` (2026-11-12)

Two separate invocations of the same run name. The second executed only the LLM configurations.

| | Deterministic | LLM-enabled |
|---|---|---|
| Resource run id | `RES-1f82695a9c2c` | `RES-580441f6aa56` |
| Work executed by this invocation | A x119, B x119, detection-delay study | C x20, D x20 (60 + 20 LLM requests), detection-delay study |
| Model | none | `qwen3:4b`, digest `359d7dd4bcda`, 4.0B, Q4_K_M |
| Wall-clock duration | 404.7 s | 4,926.3 s |
| Sampling interval requested / samples | 0.5 s / 581 | 0.5 s / 9,331 |
| Memory available at start | 5,636 MB | 6,244 MB |
| Python process RSS, peak | 398.3 MB | 347.5 MB |
| AEGIS process tree RSS, peak | 398.3 MB | 347.5 MB |
| Ollama server RSS, peak | 35.4 MB (idle server, no model) | 26.3 MB |
| Ollama runner RSS, peak | 0 (no runner observed) | **5,744.5 MB** |
| Process-tree total, peak | 433.8 MB | 6,001.1 MB |
| System-wide memory used, peak | 2,303.1 MB | **7,159.4 MB** of 7,835 MB |
| Swap used, peak | 855.6 MB (already in use at start) | **2,048.0 MB (full)** |
| CPU mean / p95 | 35.4% / 59.8% | 65.9% / 73.4% |
| Sampler errors | 0 | 0 |

### The runner grew to 5.7 GB, and why

A single investigation holds the runner at about 3.3 GB (below). Over the 80-request benchmark
it reached 5.7 GB. The Ollama server log shows the cause: the llama.cpp runner keeps a prompt
cache of evaluated prefixes, and it grew to "22 prompts, 3197 MiB (limits: 8192 MiB)" by the end
of the run. Model (about 3.1 GB) plus prompt cache (about 3.2 GB) is the observed peak.

Consequences:

- The LLM benchmark exhausted swap on this machine. Its latency figures were measured under
  memory pressure and would be lower with more RAM.
- This also explains the historical run's 7,071 MB system-wide peak, though that run's runner
  was not measured and the explanation is an inference for it.
- The service is less exposed than the benchmark: with `OLLAMA_KEEP_ALIVE=2m` the runner exits
  between incidents and starts again empty. A burst of investigations inside the keep-alive
  window would accumulate cache the same way.
- Not done: bounding the runner's prompt cache. It is a runner setting, and it was not changed
  or tested here.

## New measurements: single investigations (2026-11-12)

From `scripts.verify_llm`, one process per model, each sampled by the same sampler. These are
separate measurements from the benchmark above.

| | `qwen3:4b` | `llama3.2:3b` |
|---|---|---|
| Cold load | 7.5 s | 11.7 s |
| Full investigation (plan, rank, explain) | 225 s | 144 s |
| Ollama runner RSS, peak | 3,317.6 MB | 2,677.2 MB |
| Process-tree total, peak | 3,704.1 MB | 3,024.0 MB |
| System-wide memory used, peak | 5,013.9 MB | 4,370.0 MB |
| CPU mean / p95 | 65.3% / 75.5% | 63.9% / 75.7% |

Through Docker Compose (`llm` profile, Ollama in a container, 2026-11-12): one investigation with
the LLM took 267 s (plan 87 s, rank 97 s, explain 61 s); the guest's swap was almost full during
it. Per-process memory inside the container was not sampled.

Cold load time depends on the page cache: 7.5 to 23 s when the model file is cached, more than
two minutes when it is not (observed on 2026-11-11). Model loading therefore has its own budget
(`AEGIS_OLLAMA_LOAD_TIMEOUT_S`, default 300 s), separate from the per-request timeout.

## Deterministic path timings (`v2`)

| Stage | p50 | p95 |
|---|---|---|
| Monitoring run per batch (1,000 to 3,000 rows) | 704 ms | 1,186 ms |
| Investigation without the LLM (diagnostics, rules, runbook retrieval, sandbox validation, policy) | 0.78 s | 3.2 s |

## LLM path timings (`v2`)

| | Value |
|---|---|
| One LLM request, mean, configuration D (three requests per investigation) | 47.0 s |
| One LLM request, mean, configuration C (one longer request) | 98.9 s |
| Investigation, configuration D | p50 139 s, p95 178 s |
| Prompt / completion tokens per investigation, configuration D | 2,054 / 365 |

## Historical measurements (2026-11-11, original sampler)

Kept as recorded. Nothing below was re-measured, and no later figure is substituted into it.

| Run | Duration | Python peak RSS | Ollama | System-wide peak | Swap | CPU mean / p95 |
|---|---|---|---|---|---|---|
| `main`, LLM-enabled | 4,067.8 s | 421.0 MB | 25.2 MB recorded, which is the server only. **Runner memory: unavailable** (the sampler could not see it) | 7,071.1 MB | unavailable | 63.5% / 71.6% |
| `main`, deterministic | 395.6 s | 402.4 MB | 25.2 MB (idle server, no model loaded) | 1,660.2 MB | unavailable | 35.2% / 56.8% |

A demonstration run on 2026-11-11, measured separately with an interim sampler, recorded
3,126 MB for Ollama (server and runner as one figure) and 4,973 MB system-wide over 203 s. That
is a single investigation, consistent with the 3.3 GB measured above; it is not a measurement of
the historical benchmark and is not presented as one.

## What this means in practice

- **An 8 GiB allocation is the floor.** One investigation needs about 3.7 GB for AEGIS and the
  model together. A long run of LLM requests grows the runner toward 6 GB and fills swap.
  16 GiB for the guest is the realistic recommendation; on this host that means a `.wslconfig`
  raising the WSL2 memory limit.
- The LLM adds roughly 100 to 280 s to an investigation that otherwise takes about one second,
  and did not improve diagnostic accuracy on the benchmark. `AEGIS_LLM_ENABLED=false` is a
  legitimate configuration.
- The default request timeout of 120 s is tight on this CPU; the benchmark, the Compose file and
  `.env.example` use 180 s.

Not measured: dense-embedding retrieval under load, dashboard rendering performance, more than
one concurrent investigation (the service runs one at a time by design), per-process memory
inside containers.
