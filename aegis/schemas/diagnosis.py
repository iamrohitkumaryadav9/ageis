"""Diagnosis outputs: evidence records and ranked hypotheses."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from aegis.schemas.findings import EpistemicLevel, Severity


class RootCause(StrEnum):
    NUMERICAL_SHIFT = "numerical_shift"
    CATEGORICAL_SHIFT = "categorical_shift"
    MISSING_VALUES = "missing_values"
    SCALING_ERROR = "scaling_error"
    CONCEPT_DRIFT = "concept_drift"
    DUPLICATE_RECORDS = "duplicate_records"
    FEATURE_LEAKAGE = "feature_leakage"
    SCHEMA_CHANGE = "schema_change"
    LABEL_DELAY = "label_delay"
    ARTIFACT_MISMATCH = "artifact_mismatch"


INSUFFICIENT_EVIDENCE = "Insufficient evidence to determine the root cause."


class Evidence(BaseModel):
    """One citable unit of evidence. Hypotheses reference these by id and nothing else."""

    evidence_id: str
    source: str  # "monitoring:<monitor>" or "tool:<tool_name>"
    level: EpistemicLevel
    statement: str
    feature: str | None = None
    severity: Severity = Severity.INFO
    data: dict[str, Any] = Field(default_factory=dict)
    finding_id: str | None = None


class Hypothesis(BaseModel):
    hypothesis_id: str
    root_cause: RootCause
    affected_features: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    # Evidence-fit score in [0, 1]. It ranks explanations; it is NOT P(cause | data).
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    missing_evidence: list[str] = Field(default_factory=list)
    next_diagnostic_action: str = ""
    ranked_by: str = "rules"  # "rules" | "llm" | "monitoring-only"


class Diagnosis(BaseModel):
    hypotheses: list[Hypothesis]
    conclusion: str
    sufficient_evidence: bool
    engine: str = "rules"

    @property
    def top(self) -> Hypothesis | None:
        return self.hypotheses[0] if self.hypotheses else None
