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
import shutil
import tempfile
import time
from pathlib import Path

from aegis.agents.ollama import OllamaClient
from aegis.config import get_settings
from aegis.evaluation.benchmark import BenchmarkEnv, run_scenario
from aegis.faults.injectors import FAULTS
from aegis.faults.scenarios import CLEAN, scenario_grid, scenario_id
from aegis.ingestion.datasets import DATASETS
from aegis.monitoring.engine import MonitoringInput, run_monitoring
from aegis.observability.resources import DEFAULT_INTERVAL_S, ResourceSampler, environment_info

DEV_SEEDS = (101, 102)  # used while tuning thresholds; never used for reported results
EVAL_SEEDS = (201, 202)
WINDOW = 500


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
    parser.add_argument(
        "--no-explain",
        action="store_true",
        help="configuration D skips the LLM explanation (as historical run 'main' did)",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=DEFAULT_INTERVAL_S,
        help="resource sampling, seconds",
    )
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
    env.llm_explain = not args.no_explain
    environment = environment_info()
    llm_identity = llm.model_info(settings.ollama_model) if llm is not None else None

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

    # A later invocation without --llm (for example a resumed deterministic run) must not erase
    # the record that LLM configurations were run earlier under this name.
    earlier: dict = {}
    if (out / "manifest.json").exists():
        try:
            earlier = json.loads((out / "manifest.json").read_text())
        except ValueError:
            earlier = {}
    keep_llm = not args.llm and bool(earlier.get("llm_model"))
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
                "llm_subset": earlier.get("llm_subset", []) if keep_llm else sorted(subset_ids),
                "scenarios": [
                    {
                        "scenario_id": c["scenario_id"],
                        "dataset": c["dataset"],
                        "n_rows": c["n_rows"],
                    }
                    for c in grid
                ],
                "model_versions": {name: ctx.version for name, ctx in env.datasets.items()},
                "llm_model": earlier.get("llm_model")
                if keep_llm
                else (settings.ollama_model if args.llm else None),
                "llm_options": {
                    "num_ctx": settings.ollama_num_ctx,
                    "temperature": settings.ollama_temperature,
                    "timeout_s": settings.ollama_timeout_s,
                    "max_retries": settings.ollama_max_retries,
                },
                "llm_model_identity": earlier.get("llm_model_identity")
                if keep_llm
                else llm_identity,
                "environment": environment,
                "methodology": [
                    "Detection metrics are reported under explicit names with their populations "
                    "(overall_fault_detection_*, drift_detection_*).",
                    "Remediation lifecycle events are recorded per proposal and reconciled.",
                    "LLM text is validated claim by claim; the raw text and the counters are stored.",
                    "Configuration D "
                    + (
                        "skips the LLM explanation step."
                        if args.no_explain
                        else "includes the LLM explanation step (run 'main' did not), so its "
                        "latency is not comparable with 'main'."
                    ),
                    "Resources are sampled by the process-tree sampler "
                    f"every {args.sample_interval} s; each invocation is a separate record.",
                ],
            },
            indent=2,
        )
        + "\n"
    )

    label = "llm" if args.llm else "deterministic"
    sampler = ResourceSampler(label, interval_s=args.sample_interval).start()
    started_configs: dict[str, int] = {}
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
        ran = missing if previous is not None else wanted
        record = run_scenario(env, cell, ran)
        for config in record["configs"]:
            if config in ran:
                started_configs[config] = started_configs.get(config, 0) + 1
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
    measured = sampler.stop().to_dict()
    measured["work"] = {
        "configs_executed": dict(sorted(started_configs.items())),
        "llm_enabled": bool(args.llm),
        "llm_explain": bool(args.llm and not args.no_explain),
        "delay_study": not args.quick,
        "note": "counts are scenario x configuration pairs executed by THIS invocation; "
        "pairs finished by an earlier invocation were skipped and are not in this measurement",
    }
    measured["llm"] = llm_identity
    measured["configuration"] = {
        "name": args.name,
        "seeds": args.seeds,
        "ollama_num_ctx": settings.ollama_num_ctx,
        "ollama_timeout_s": settings.ollama_timeout_s,
        "ollama_keep_alive": settings.ollama_keep_alive,
    }
    resources_path = out / "resources.json"
    document: dict = {"schema_version": "resources-2", "environment": environment, "runs": []}
    if resources_path.exists():
        existing = json.loads(resources_path.read_text())
        if "runs" in existing:
            document = existing
            document["environment"] = environment
        else:  # a file in the pre-process-tree format: keep it, never overwrite it
            resources_path.rename(out / "resources.legacy.json")
    document["runs"].append(measured)
    resources_path.write_text(json.dumps(document, indent=2) + "\n")
    shutil.rmtree(workdir, ignore_errors=True)
    print(f"done in {measured['duration_s']}s -> {out} (resource run {measured['run_id']})")


if __name__ == "__main__":
    main()
