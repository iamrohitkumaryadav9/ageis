"""Quick detection sweep used while developing monitors (dev seeds only).

Usage: uv run python -m experiments.detection_sweep [--seeds 101 102] [--clean 25]
"""

from __future__ import annotations

import argparse
import collections
import tempfile
import time
from pathlib import Path

from aegis.config import get_settings
from aegis.faults.scenarios import CLEAN, build_scenario, scenario_grid
from aegis.ingestion.contracts import derive_contract
from aegis.ingestion.datasets import DATASETS
from aegis.ingestion.splits import read_split
from aegis.models.registry import ModelRegistry
from aegis.monitoring.engine import MonitoringInput, run_monitoring


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102])
    parser.add_argument("--clean", type=int, default=25)
    args = parser.parse_args()
    settings = get_settings()
    registry = ModelRegistry(settings.model_dir)
    workdir = Path(tempfile.mkdtemp(prefix="aegis-sweep-"))
    ctx = {}
    for name, spec in DATASETS.items():
        ref = read_split(name, "validation", settings.processed_dir)
        prod = read_split(name, "production", settings.processed_dir)
        ctx[name] = (spec, ref, prod, derive_contract(spec, ref), registry.baseline(name).version)

    grid = scenario_grid(list(DATASETS), args.seeds, args.clean)
    hits = collections.defaultdict(list)
    clean_triggers: collections.Counter = collections.Counter()
    started = time.perf_counter()
    for g in grid:
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
        report = run_monitoring(
            MonitoringInput(
                spec, contract, ref, scenario.current, registry, version, scenario.artifact_path
            )
        ).report
        hits[(g["fault_id"], g["severity"])].append(report.incident_detected)
        if g["fault_id"] == CLEAN:
            for f in report.triggered:
                clean_triggers[(g["dataset"], f.check, f.feature, f.severity.value)] += 1
        elif not report.incident_detected:
            print("MISS", g, scenario.truth.affected_features, scenario.truth.params)
    print(f"{len(grid)} scenarios in {time.perf_counter() - started:.1f}s")
    for key in sorted(hits):
        print(f"  {key[0]:<6}{key[1]:<7} {sum(hits[key])}/{len(hits[key])}")
    print("clean triggers:", dict(clean_triggers))


if __name__ == "__main__":
    main()
