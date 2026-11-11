"""Run AEGIS-IR.

    uv run python -m aegis.evaluation.run                     # A and B on the full grid (minutes)
    uv run python -m aegis.evaluation.run --llm               # also C and D on a stratified subset (CPU: ~1-2 h)
    uv run python -m aegis.evaluation.run --quick             # small smoke run

Outputs under ``results/aegis-ir/<name>/``: ``scenarios.jsonl`` (one raw record per scenario),
``manifest.json`` (public design), ``ground_truth.json`` (scorer-only), ``delay.json``,
``resources.json``. The run is resumable: finished scenario/config pairs are skipped.
Build the report with ``python -m aegis.evaluation.report``.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import psutil

from aegis.agents.ollama import OllamaClient
from aegis.config import get_settings
from aegis.evaluation.benchmark import BenchmarkEnv, run_scenario
from aegis.faults.injectors import FAULTS
from aegis.faults.scenarios import CLEAN, scenario_grid, scenario_id
from aegis.ingestion.datasets import DATASETS
from aegis.monitoring.engine import MonitoringInput, run_monitoring

DEV_SEEDS = (101, 102)  # used while tuning thresholds; never used for reported results
EVAL_SEEDS = (201, 202)
WINDOW = 500


class ResourceSampler(threading.Thread):
    """Samples RSS of this process and of the Ollama runner, and system CPU, once a second."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._me = psutil.Process()

    def run(self) -> None:
        psutil.cpu_percent(interval=None)
        while not self._stop.wait(1.0):
            ollama = 0
            for proc in psutil.process_iter(["name", "memory_info"]):
                try:
                    if "ollama" in (proc.info["name"] or "").lower() and proc.info["memory_info"]:
                        ollama += proc.info["memory_info"].rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            memory = psutil.virtual_memory()
            self.samples.append(
                {
                    "python_rss": self._me.memory_info().rss,
                    "ollama_rss": ollama,
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "system_used": memory.used,
                }
            )

    def stop(self) -> dict:
        self._stop.set()
        if not self.samples:
            return {}
        mb = 1024 * 1024
        col = lambda key: np.array([s[key] for s in self.samples])  # noqa: E731
        return {
            "samples": len(self.samples),
            "python_peak_rss_mb": round(float(col("python_rss").max()) / mb, 1),
            "ollama_peak_rss_mb": round(float(col("ollama_rss").max()) / mb, 1),
            "combined_peak_rss_mb": round(
                float((col("python_rss") + col("ollama_rss")).max()) / mb, 1
            ),
            "system_used_peak_mb": round(float(col("system_used").max()) / mb, 1),
            "cpu_percent_mean": round(float(col("cpu_percent").mean()), 1),
            "cpu_percent_p95": round(float(np.percentile(col("cpu_percent"), 95)), 1),
        }


def llm_subset(grid: list[dict]) -> list[dict]:
    """One scenario per (dataset, fault), rotating severity: a stratified, deterministic subset."""
    severities = ("low", "medium", "high")
    chosen = []
    for dataset in sorted({g["dataset"] for g in grid}):
        for index, fault in enumerate(sorted(FAULTS)):
            wanted = severities[(index + (0 if dataset == "adult" else 1)) % 3]
            match = next(
                g
                for g in grid
                if g["dataset"] == dataset and g["fault_id"] == fault and g["severity"] == wanted
            )
            chosen.append(match)
    return chosen


def detection_delay(env: BenchmarkEnv, cells: list[dict]) -> list[dict]:
    """Stream clean windows followed by faulty windows; count windows until the first alarm."""
    rows = []
    for cell in cells:
        if cell["fault_id"] in (CLEAN, "F10") or cell["n_rows"] < 2 * WINDOW:
            continue
        scenario = env.scenario(cell)
        ctx = env.datasets[cell["dataset"]]
        clean = env.scenario(
            {**cell, "fault_id": CLEAN, "severity": "none", "seed": cell["seed"] + 5000}
        ).current
        prefix = [clean.iloc[i : i + WINDOW] for i in range(0, min(len(clean), 3 * WINDOW), WINDOW)]
        faulty = [
            scenario.current.iloc[i : i + WINDOW]
            for i in range(0, len(scenario.current) - WINDOW + 1, WINDOW)
        ]

        def alarm(frame, ctx=ctx, scenario=scenario):
            inp = MonitoringInput(
                ctx.spec,
                ctx.contract,
                ctx.reference,
                frame.reset_index(drop=True),
                env.registry,
                ctx.version,
                scenario.artifact_path,
            )
            return run_monitoring(inp).report.incident_detected

        false_alarms = sum(alarm(w) for w in prefix)
        delay = next((i for i, w in enumerate(faulty) if alarm(w)), None)
        rows.append(
            {
                "scenario_id": scenario.spec.scenario_id,
                "fault_id": cell["fault_id"],
                "severity": cell["severity"],
                "dataset": cell["dataset"],
                "window_rows": WINDOW,
                "clean_windows": len(prefix),
                "false_alarms_before_onset": int(false_alarms),
                "faulty_windows": len(faulty),
                "delay_windows": delay,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", default="main")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(EVAL_SEEDS))
    parser.add_argument("--clean", type=int, default=25, help="clean controls per dataset")
    parser.add_argument("--llm", action="store_true", help="also run configurations C and D")
    parser.add_argument(
        "--llm-all", action="store_true", help="run C/D on every scenario (very slow on CPU)"
    )
    parser.add_argument(
        "--quick", action="store_true", help="1 seed, 5 clean controls, no delay study"
    )
    parser.add_argument("--out", type=Path, default=Path("results/aegis-ir"))
    args = parser.parse_args()
    if set(args.seeds) & set(DEV_SEEDS):
        parser.error(
            f"seeds {DEV_SEEDS} were used for threshold tuning and may not be used for evaluation"
        )
    if args.quick:
        args.seeds, args.clean = args.seeds[:1], 5

    settings = get_settings()
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="aegis-ir-"))
    # The benchmark never touches the real registry: retraining candidates are registered in a copy.
    shutil.copytree(settings.model_dir, workdir / "models")
    llm = OllamaClient(settings) if args.llm else None
    if llm is not None and not llm.health().available:
        parser.error(
            f"--llm requested but the local model is not available: {llm.health().detail or 'unreachable'}"
        )
    env = BenchmarkEnv.create(settings, workdir / "models", workdir / "scenarios", llm)

    grid = scenario_grid(sorted(DATASETS), args.seeds, args.clean)
    for cell in grid:
        cell["scenario_id"] = scenario_id(
            cell["dataset"], cell["fault_id"], cell["severity"], cell["seed"], cell["n_rows"]
        )
    subset_ids = (
        {c["scenario_id"] for c in (grid if args.llm_all else llm_subset(grid))}
        if args.llm
        else set()
    )

    results_path = out / "scenarios.jsonl"
    done: dict[str, dict] = {}
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            record = json.loads(line)
            done[record["scenario_id"]] = record

    (out / "manifest.json").write_text(
        json.dumps(
            {
                "benchmark": "AEGIS-IR",
                "name": args.name,
                "seeds": args.seeds,
                "dev_seeds_excluded": list(DEV_SEEDS),
                "clean_controls_per_dataset": args.clean,
                "datasets": sorted(DATASETS),
                "faults": sorted(FAULTS),
                "n_scenarios": len(grid),
                "llm_subset": sorted(subset_ids),
                "scenarios": [
                    {
                        "scenario_id": c["scenario_id"],
                        "dataset": c["dataset"],
                        "n_rows": c["n_rows"],
                    }
                    for c in grid
                ],
                "model_versions": {name: ctx.version for name, ctx in env.datasets.items()},
                "llm_model": settings.ollama_model if args.llm else None,
                "llm_options": {
                    "num_ctx": settings.ollama_num_ctx,
                    "temperature": settings.ollama_temperature,
                    "timeout_s": settings.ollama_timeout_s,
                    "max_retries": settings.ollama_max_retries,
                },
                "machine": {
                    "cpu": platform.processor() or platform.machine(),
                    "logical_cpus": psutil.cpu_count(),
                    "physical_cpus": psutil.cpu_count(logical=False),
                    "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
                    "platform": platform.platform(),
                },
            },
            indent=2,
        )
        + "\n"
    )

    sampler = ResourceSampler()
    sampler.start()
    started = time.perf_counter()
    truths = []
    for index, cell in enumerate(grid, start=1):
        wanted = ("A", "B", "C", "D") if cell["scenario_id"] in subset_ids else ("A", "B")
        previous = done.get(cell["scenario_id"])
        missing = tuple(
            c
            for c in wanted
            if previous is None
            or (previous["detection"]["incident_detected"] and c not in previous["configs"])
        )
        if previous is not None and not missing:
            truths.append(previous["truth"])
            continue
        tick = time.perf_counter()
        record = run_scenario(env, cell, missing if previous is not None else wanted)
        if previous is not None:
            previous["configs"].update(record["configs"])
            record = previous
        done[cell["scenario_id"]] = record
        truths.append(record["truth"])
        with results_path.open("w") as handle:  # rewrite atomically enough for a single writer
            for item in done.values():
                handle.write(json.dumps(item, default=str) + "\n")
        print(
            f"[{index:>3}/{len(grid)}] {cell['scenario_id']} {cell['dataset']:<5} configs={''.join(wanted)} "
            f"detected={record['detection']['incident_detected']} {time.perf_counter() - tick:6.1f}s",
            flush=True,
        )

    (out / "ground_truth.json").write_text(json.dumps(truths, indent=2) + "\n")
    if not args.quick:
        delay_cells = [c for c in grid if c["seed"] == args.seeds[0]]
        (out / "delay.json").write_text(
            json.dumps(detection_delay(env, delay_cells), indent=2) + "\n"
        )
    resources = sampler.stop()
    resources["duration_s"] = round(time.perf_counter() - started, 1)
    resources["llm"] = bool(args.llm)
    previous_resources = {}
    if (out / "resources.json").exists():
        previous_resources = json.loads((out / "resources.json").read_text())
    key = "with_llm" if args.llm else "deterministic"
    previous_resources[key] = resources
    (out / "resources.json").write_text(json.dumps(previous_resources, indent=2) + "\n")
    shutil.rmtree(workdir, ignore_errors=True)
    print(f"done in {resources['duration_s']}s -> {out}")


if __name__ == "__main__":
    main()
