"""Compare naive (A) and rule-based (B) diagnosis on dev scenarios.

Usage: uv run python -m experiments.diagnosis_sweep [--seeds 101 102]
"""

from __future__ import annotations

import argparse
import collections
import tempfile
from pathlib import Path

from aegis.config import get_settings
from aegis.diagnosis.naive import naive_diagnosis
from aegis.diagnosis.pipeline import deterministic_diagnosis
from aegis.faults.scenarios import CLEAN, build_scenario, scenario_grid
from aegis.ingestion.contracts import derive_contract
from aegis.ingestion.datasets import DATASETS
from aegis.ingestion.splits import read_split
from aegis.models.registry import ModelRegistry
from aegis.monitoring.engine import MonitoringInput, run_monitoring


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    registry = ModelRegistry(settings.model_dir)
    workdir = Path(tempfile.mkdtemp(prefix="aegis-sweep-"))
    ctx = {}
    for name, spec in DATASETS.items():
        ref = read_split(name, "validation", settings.processed_dir)
        prod = read_split(name, "production", settings.processed_dir)
        ctx[name] = (spec, ref, prod, derive_contract(spec, ref), registry.baseline(name).version)

    stats = {k: collections.defaultdict(lambda: [0, 0, 0]) for k in ("naive", "rules")}
    for g in scenario_grid(list(DATASETS), args.seeds, 0):
        if g["fault_id"] == CLEAN:
            continue
        spec, ref, prod, contract, version = ctx[g["dataset"]]
        scenario = build_scenario(
            spec,
            prod,
            registry,
            version,
            workdir=workdir,
            fault_id=g["fault_id"],
            severity=g["severity"],
            seed=g["seed"],
            n_rows=g["n_rows"],
        )
        inp = MonitoringInput(
            spec, contract, ref, scenario.current, registry, version, scenario.artifact_path
        )
        result = run_monitoring(inp)
        if not result.report.incident_detected:
            continue
        truth = scenario.truth.root_cause
        for name, diagnosis in (
            ("naive", naive_diagnosis(result.report)),
            ("rules", deterministic_diagnosis(inp, result)[1]),
        ):
            ranked = [h.root_cause.value for h in diagnosis.hypotheses]
            cell = stats[name][g["fault_id"]]
            cell[0] += 1
            cell[1] += ranked[:1] == [truth]
            cell[2] += truth in ranked[:3]
            if args.verbose and name == "rules" and ranked[:1] != [truth]:
                print(
                    "WRONG",
                    g,
                    scenario.truth.affected_features,
                    scenario.truth.params,
                    [(h.root_cause.value, h.confidence) for h in diagnosis.hypotheses[:3]],
                )
    for name, table in stats.items():
        n = sum(c[0] for c in table.values())
        print(
            f"\n{name}: top1={sum(c[1] for c in table.values()) / n:.3f} "
            f"recall@3={sum(c[2] for c in table.values()) / n:.3f} (n={n})"
        )
        for fault in sorted(table):
            c = table[fault]
            print(f"  {fault}: top1 {c[1]}/{c[0]}  r@3 {c[2]}/{c[0]}")


if __name__ == "__main__":
    main()
