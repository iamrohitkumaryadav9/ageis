"""File-backed model registry with content-addressed integrity checks.

The registry is the source of truth for *which bytes* a model version is. A
serving artifact whose SHA-256 differs from its registry record is an
integrity incident (fault F10), detected by comparing hashes, never by asking
a language model.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

from aegis.ingestion.datasets import sha256_file


class ArtifactIntegrityError(RuntimeError):
    """A model artifact is missing, unreadable, or does not match its registered hash."""


class ModelRecord(BaseModel):
    dataset: str
    version: str
    algorithm: str
    artifact_sha256: str
    threshold: float
    features: list[str]
    contract_version: str
    train_fingerprint: str
    seed: int
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    library_versions: dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    parent_version: str | None = None
    notes: str = ""


class ModelRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _dir(self, dataset: str, version: str) -> Path:
        if not version.replace("-", "").replace("_", "").replace(".", "").isalnum():
            raise ValueError(f"illegal model version {version!r}")
        return self.root / dataset / version

    def artifact_path(self, dataset: str, version: str) -> Path:
        return self._dir(dataset, version) / "model.joblib"

    def register(self, record_fields: dict[str, Any], model: Pipeline) -> ModelRecord:
        dataset, version = record_fields["dataset"], record_fields["version"]
        target = self._dir(dataset, version)
        target.mkdir(parents=True, exist_ok=True)
        artifact = target / "model.joblib"
        joblib.dump(model, artifact)
        record = ModelRecord(**record_fields, artifact_sha256=sha256_file(artifact))
        (target / "record.json").write_text(record.model_dump_json(indent=2) + "\n")
        return record

    def get(self, dataset: str, version: str) -> ModelRecord:
        path = self._dir(dataset, version) / "record.json"
        if not path.exists():
            raise KeyError(f"model {dataset}/{version} is not registered")
        return ModelRecord.model_validate(json.loads(path.read_text()))

    def list(self, dataset: str) -> list[ModelRecord]:
        base = self.root / dataset
        if not base.exists():
            return []
        records = [
            ModelRecord.model_validate(json.loads(p.read_text()))
            for p in sorted(base.glob("*/record.json"))
        ]
        return sorted(records, key=lambda r: (r.created_at, r.version))

    def verify(self, dataset: str, version: str, artifact: Path | None = None) -> tuple[bool, str]:
        """Compare an artifact's hash with the registry. Returns (ok, actual_sha256)."""
        record = self.get(dataset, version)
        path = artifact or self.artifact_path(dataset, version)
        if not path.exists():
            return False, ""
        actual = sha256_file(path)
        return actual == record.artifact_sha256, actual

    def load(self, dataset: str, version: str, *, verify: bool = True) -> Pipeline:
        path = self.artifact_path(dataset, version)
        if verify:
            ok, actual = self.verify(dataset, version)
            if not ok:
                raise ArtifactIntegrityError(
                    f"{dataset}/{version}: artifact sha256 {actual or '<missing>'} does not match "
                    "the registry record"
                )
        return joblib.load(path)

    # -- baseline pointer: which version `train` selected as the initial production model
    def set_baseline(self, dataset: str, version: str) -> None:
        self.get(dataset, version)
        (self.root / dataset / "baseline.json").write_text(json.dumps({"version": version}) + "\n")

    def baseline(self, dataset: str) -> ModelRecord:
        path = self.root / dataset / "baseline.json"
        if not path.exists():
            raise KeyError(
                f"no baseline model for {dataset}; run `python -m scripts.train_baseline`"
            )
        return self.get(dataset, json.loads(path.read_text())["version"])
