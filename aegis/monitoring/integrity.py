"""Serving-artifact integrity: is the deployed file the registered model?"""

from __future__ import annotations

from pathlib import Path

from aegis.ingestion.datasets import sha256_file
from aegis.models.registry import ModelRegistry
from aegis.monitoring.common import make_finding
from aegis.schemas.findings import EpistemicLevel, Finding, Severity


def check_integrity(
    report_id: str, registry: ModelRegistry, dataset: str, declared_version: str, artifact: Path
) -> tuple[Finding, str | None]:
    """Return the finding and the *verified* version that the artifact actually is.

    ``None`` means the bytes match no registered version; such a file is never
    deserialised (joblib/pickle would execute whatever it contains).
    """
    record = registry.get(dataset, declared_version)
    actual = sha256_file(artifact) if artifact.exists() else ""
    if actual == record.artifact_sha256:
        return (
            make_finding(
                report_id,
                "integrity",
                "artifact_hash",
                metric_name="artifact_sha256_match",
                observed=True,
                reference=True,
                threshold=True,
                triggered=False,
                severity=Severity.CRITICAL,
                level=EpistemicLevel.OBSERVED_FACT,
                interpretation=f"serving artifact matches registry record for {declared_version}",
                evidence={"declared_version": declared_version},
            ),
            declared_version,
        )
    matches = [r.version for r in registry.list(dataset) if actual and r.artifact_sha256 == actual]
    serving = matches[0] if matches else None
    detail = (
        f"it is byte-identical to registered version {serving}"
        if serving
        else "it matches no registered version"
        if actual
        else "the artifact file is missing"
    )
    return (
        make_finding(
            report_id,
            "integrity",
            "artifact_hash",
            metric_name="artifact_sha256_match",
            observed=False,
            reference=True,
            threshold=True,
            triggered=True,
            severity=Severity.CRITICAL,
            level=EpistemicLevel.OBSERVED_FACT,
            interpretation=(
                f"serving artifact hash differs from the registry record for {declared_version}; {detail}"
            ),
            evidence={
                "declared_version": declared_version,
                "expected_sha256": record.artifact_sha256,
                "actual_sha256": actual or None,
                "matches_registered_version": serving,
            },
        ),
        serving,
    )
