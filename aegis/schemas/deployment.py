"""The small set of things remediation is allowed to change."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class IngestionConfig(BaseModel):
    """Declarative pre-model transforms applied to every incoming batch."""

    model_config = ConfigDict(extra="forbid")

    unit_scale: dict[str, float] = Field(default_factory=dict)  # feature -> multiplier
    deduplicate: bool = False


class DeploymentState(BaseModel):
    """What is live for one dataset/model. Each executed action writes a new revision."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    model_version: str
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    #: Artifact actually being served, relative to the artifacts directory. ``None`` means the
    #: registry copy of ``model_version``. Only deployment tooling sets this; no API accepts a path.
    serving_artifact: str | None = None
    quarantined: bool = False  # incoming batches are held and not used for decisions/training
    retraining_blocked: bool = False


def apply_ingestion(frame: pd.DataFrame, config: IngestionConfig) -> pd.DataFrame:
    if not config.unit_scale and not config.deduplicate:
        return frame
    out = frame.copy()
    for feature, factor in config.unit_scale.items():
        if feature in out.columns and pd.api.types.is_numeric_dtype(out[feature]):
            out[feature] = out[feature] * factor
    if config.deduplicate:
        out = out.drop_duplicates(keep="first")
    return out
