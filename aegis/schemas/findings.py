"""Monitoring outputs.

A ``Finding`` is one evaluated check. ``level`` states what kind of claim it
is: an *observed fact* (a column is absent, a hash differs) or a *statistical
anomaly* (a test plus an effect size crossed a threshold). Neither is a cause.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return list(Severity).index(self)


class EpistemicLevel(StrEnum):
    OBSERVED_FACT = "observed_fact"
    STATISTICAL_ANOMALY = "statistical_anomaly"


Monitor = Literal["quality", "drift", "concept", "performance", "relationship", "integrity"]
Scalar = float | int | str | bool | None


def utcnow() -> datetime:
    return datetime.now(UTC)


class Finding(BaseModel):
    finding_id: str
    monitor: Monitor
    check: str
    metric_name: str
    feature: str | None = None
    observed_value: Scalar = None
    reference_value: Scalar = None
    threshold: Scalar = None
    triggered: bool
    severity: Severity
    level: EpistemicLevel
    interpretation: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utcnow)


class SkippedCheck(BaseModel):
    monitor: Monitor
    check: str
    feature: str | None = None
    reason: str


class MonitoringReport(BaseModel):
    report_id: str
    dataset: str
    model_version: str
    reference_version: str
    n_reference: int
    n_current: int
    findings: list[Finding]
    skipped: list[SkippedCheck] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    duration_ms: float = 0.0
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def triggered(self) -> list[Finding]:
        return [f for f in self.findings if f.triggered]

    @property
    def max_severity(self) -> Severity:
        return max(
            (f.severity for f in self.triggered), key=lambda s: s.rank, default=Severity.INFO
        )

    @property
    def incident_detected(self) -> bool:
        """An incident is opened for any triggered finding of at least MEDIUM severity."""
        return self.max_severity.rank >= Severity.MEDIUM.rank

    def by_id(self) -> dict[str, Finding]:
        return {f.finding_id: f for f in self.findings}


def finding_id(report_id: str, monitor: str, check: str, feature: str | None) -> str:
    """Deterministic id, so a re-run of the same report yields the same references."""
    digest = hashlib.sha256(f"{report_id}|{monitor}|{check}|{feature}".encode()).hexdigest()[:10]
    return f"FND-{digest}"
