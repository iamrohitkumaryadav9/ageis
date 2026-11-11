"""Leakage-safe, reproducible data partitioning.

Four disjoint, stratified partitions are produced from one seed:

* ``train``       - model fitting and cross-validated model selection only.
* ``validation``  - threshold selection, and the monitoring *reference* window.
* ``test``        - touched once, for the final unbiased baseline report.
* ``production``  - simulated live traffic; scenarios are sampled from it.

The model never sees ``production`` rows, so a clean scenario is a genuine
out-of-sample control rather than data the model has memorised.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

from aegis.ingestion.datasets import TARGET

FRACTIONS = {"train": 0.50, "validation": 0.15, "test": 0.15, "production": 0.20}


class SplitManifest(BaseModel):
    dataset: str
    seed: int
    fractions: dict[str, float]
    sizes: dict[str, int]
    positive_rate: dict[str, float]
    index_sha256: dict[str, str]


def frame_fingerprint(df: pd.DataFrame) -> str:
    """Order-sensitive content hash of a frame (stable across runs and machines)."""
    hashed = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def make_splits(df: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    remaining = df
    out: dict[str, pd.DataFrame] = {}
    left = 1.0
    names = list(FRACTIONS)
    for i, name in enumerate(names[:-1]):
        take = FRACTIONS[name] / left
        part, remaining = train_test_split(
            remaining,
            train_size=take,
            stratify=remaining[TARGET],
            random_state=seed + i,
            shuffle=True,
        )
        out[name] = part
        left -= FRACTIONS[name]
    out[names[-1]] = remaining
    return out


def build_manifest(dataset: str, seed: int, splits: dict[str, pd.DataFrame]) -> SplitManifest:
    return SplitManifest(
        dataset=dataset,
        seed=seed,
        fractions=FRACTIONS,
        sizes={k: len(v) for k, v in splits.items()},
        positive_rate={k: round(float(v[TARGET].mean()), 6) for k, v in splits.items()},
        index_sha256={
            k: hashlib.sha256(np.sort(v.index.to_numpy()).astype("int64").tobytes()).hexdigest()
            for k, v in splits.items()
        },
    )


def write_splits(dataset: str, seed: int, splits: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    target = out_dir / dataset
    target.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(target / f"{name}.parquet", index=True)
    manifest = build_manifest(dataset, seed, splits)
    (target / "split_manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    return target


def read_split(dataset: str, name: str, processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / dataset / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m scripts.prepare_data` first")
    return pd.read_parquet(path)


def read_manifest(dataset: str, processed_dir: Path) -> SplitManifest:
    return SplitManifest.model_validate(
        json.loads((processed_dir / dataset / "split_manifest.json").read_text())
    )
