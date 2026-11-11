"""Data contracts: the schema a model was trained against, made explicit.

A contract is derived once from the reference window and versioned with the
model. Monitoring compares every incoming batch against it; a violation is an
observed fact (the column *is* missing), not a statistical judgement.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from aegis.ingestion.datasets import DatasetSpec


class ColumnContract(BaseModel):
    name: str
    kind: Literal["numeric", "categorical"]
    nullable_rate: float  # missing fraction observed in the reference window
    minimum: float | None = None
    maximum: float | None = None
    categories: list[str] | None = None


class DataContract(BaseModel):
    dataset: str
    columns: list[ColumnContract]
    reference_rows: int
    reference_duplicate_rate: float

    @property
    def by_name(self) -> dict[str, ColumnContract]:
        return {c.name: c for c in self.columns}

    @property
    def version(self) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


def derive_contract(spec: DatasetSpec, reference: pd.DataFrame) -> DataContract:
    columns: list[ColumnContract] = []
    for name in spec.numeric_features:
        series = reference[name]
        columns.append(
            ColumnContract(
                name=name,
                kind="numeric",
                nullable_rate=float(series.isna().mean()),
                minimum=float(series.min()),
                maximum=float(series.max()),
            )
        )
    for name in spec.categorical_features:
        series = reference[name]
        columns.append(
            ColumnContract(
                name=name,
                kind="categorical",
                nullable_rate=float(series.isna().mean()),
                categories=sorted(str(v) for v in series.dropna().unique()),
            )
        )
    return DataContract(
        dataset=spec.name,
        columns=columns,
        reference_rows=len(reference),
        reference_duplicate_rate=float(reference[spec.features].duplicated().mean()),
    )
