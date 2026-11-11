"""Scenario generation for experiments and the AEGIS-IR benchmark.

A scenario is fully determined by (dataset, fault, severity, seed, n_rows).
Its public half (``ScenarioSpec`` + the materialised batch) is what the system
under test receives. Its private half (``GroundTruth``) is written to a
separate file and is only ever read by the scorer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression

from aegis.faults.injectors import FAULTS, SEVERITIES
from aegis.ingestion.datasets import TARGET, DatasetSpec
from aegis.models.registry import ModelRegistry
from aegis.models.training import build_pipeline

CLEAN = "CLEAN"


class ScenarioSpec(BaseModel):
    """Public description. Deliberately contains nothing about the fault."""

    scenario_id: str
    dataset: str
    n_rows: int


class GroundTruth(BaseModel):
    scenario_id: str
    fault_id: str  # F01..F10 or CLEAN
    root_cause: str | None
    severity: Literal["none", "low", "medium", "high"]
    difficulty: Literal["control", "easy", "medium", "hard"]
    seed: int
    affected_features: list[str]
    params: dict


@dataclass
class Scenario:
    spec: ScenarioSpec
    truth: GroundTruth
    current: pd.DataFrame
    artifact_path: Path | None  # None => serve the registry artifact


def scenario_id(dataset: str, fault_id: str, severity: str, seed: int, n_rows: int) -> str:
    """Opaque id: a hash, so the id itself cannot leak the fault type."""
    raw = f"{dataset}|{fault_id}|{severity}|{seed}|{n_rows}"
    return "SCN-" + hashlib.sha256(raw.encode()).hexdigest()[:10]


def difficulty_of(severity: str, n_rows: int) -> str:
    if severity == "none":
        return "control"
    score = {"low": 2, "medium": 1, "high": 0}[severity] + (1 if n_rows < 2000 else 0)
    return ("easy", "medium", "hard", "hard")[score]


def _seed_int(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def _wrong_artifact(
    variant: str,
    spec: DatasetSpec,
    registry: ModelRegistry,
    declared: str,
    production: pd.DataFrame,
    seed: int,
    workdir: Path,
) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    if variant == "other_registered_version":
        others = [r for r in registry.list(spec.name) if r.version != declared]
        if not others:
            raise RuntimeError("F10 'other_registered_version' needs a second registered model")
        return registry.artifact_path(spec.name, others[0].version)
    path = workdir / f"serving-{seed}.joblib"
    if variant == "corrupted_bytes":
        original = registry.artifact_path(spec.name, declared).read_bytes()
        path.write_bytes(original[: len(original) // 2])
        return path
    # unregistered_retrain: a real, loadable model that nobody registered or validated
    rng = np.random.default_rng(seed)
    sample = production.iloc[
        rng.choice(len(production), size=min(1500, len(production)), replace=False)
    ]
    model = build_pipeline(spec, LogisticRegression(max_iter=300))
    model.fit(sample[spec.features], sample[TARGET])
    joblib.dump(model, path)
    return path


def build_scenario(
    spec: DatasetSpec,
    production: pd.DataFrame,
    registry: ModelRegistry,
    declared_version: str,
    *,
    fault_id: str,
    severity: str,
    seed: int,
    n_rows: int,
    workdir: Path,
) -> Scenario:
    if fault_id != CLEAN and (fault_id not in FAULTS or severity not in SEVERITIES):
        raise ValueError(f"unknown fault/severity: {fault_id}/{severity}")
    sid = scenario_id(spec.name, fault_id, severity, seed, n_rows)
    rng = np.random.default_rng(_seed_int(spec.name, fault_id, severity, seed, n_rows))
    rows = rng.choice(len(production), size=min(n_rows, len(production)), replace=False)
    batch = production.iloc[rows].reset_index(drop=True)
    batch[TARGET] = batch[TARGET].astype("float64")
    public = ScenarioSpec(scenario_id=sid, dataset=spec.name, n_rows=len(batch))

    if fault_id == CLEAN:
        truth = GroundTruth(
            scenario_id=sid,
            fault_id=CLEAN,
            root_cause=None,
            severity="none",
            difficulty="control",
            seed=seed,
            affected_features=[],
            params={},
        )
        return Scenario(public, truth, batch, None)

    fault = FAULTS[fault_id]
    injection = fault.injector(batch, spec, rng, severity)
    artifact = None
    if injection.artifact_variant:
        artifact = _wrong_artifact(
            injection.artifact_variant, spec, registry, declared_version, production, seed, workdir
        )
    truth = GroundTruth(
        scenario_id=sid,
        fault_id=fault_id,
        root_cause=fault.root_cause.value,
        severity=severity,
        difficulty=difficulty_of(severity, n_rows),
        seed=seed,
        affected_features=injection.affected_features,
        params=injection.params,
    )
    return Scenario(public, truth, injection.frame, artifact)


def scenario_grid(
    datasets: list[str],
    seeds: list[int],
    clean_per_dataset: int,
    sizes: tuple[int, ...] = (1000, 3000),
) -> list[dict]:
    """The benchmark's design matrix: every fault x severity x seed, plus clean controls."""
    grid: list[dict] = []
    for dataset in datasets:
        for fault_id in FAULTS:
            for s_idx, severity in enumerate(SEVERITIES):
                for k, seed in enumerate(seeds):
                    grid.append(
                        {
                            "dataset": dataset,
                            "fault_id": fault_id,
                            "severity": severity,
                            "seed": seed,
                            "n_rows": sizes[(k + s_idx) % len(sizes)],
                        }
                    )
        for k in range(clean_per_dataset):
            grid.append(
                {
                    "dataset": dataset,
                    "fault_id": CLEAN,
                    "severity": "none",
                    "seed": 10_000 + seeds[0] * 100 + k,
                    "n_rows": sizes[k % len(sizes)],
                }
            )
    return grid
