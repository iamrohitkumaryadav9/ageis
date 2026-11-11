"""Turn monitoring findings into citable evidence records.

Evidence is the only vocabulary hypotheses may use. Triggered findings become
"anomaly" evidence; a handful of deliberately chosen *normal* results (no
drift, labels present, hash matches, ...) are kept too, because what did not
happen is what lets one explanation be ruled out in favour of another.
"""

from __future__ import annotations

from aegis.schemas.diagnosis import Evidence
from aegis.schemas.findings import EpistemicLevel, Finding, MonitoringReport, Severity


def evidence_id_for(finding_id: str) -> str:
    return "EV-" + finding_id.removeprefix("FND-")


def from_finding(finding: Finding) -> Evidence:
    return Evidence(
        evidence_id=evidence_id_for(finding.finding_id),
        source=f"monitoring:{finding.monitor}",
        level=finding.level,
        statement=finding.interpretation,
        feature=finding.feature,
        severity=finding.severity,
        finding_id=finding.finding_id,
        data={
            "check": finding.check,
            "polarity": "anomaly" if finding.triggered else "normal",
            "observed": finding.observed_value,
            "reference": finding.reference_value,
            **{k: v for k, v in finding.evidence.items() if not isinstance(v, dict | list)},
        },
    )


def _normal(report: MonitoringReport, name: str, statement: str, checks: list[Finding]) -> Evidence:
    return Evidence(
        evidence_id=f"EV-ok-{name}-{report.report_id.removeprefix('REP-')[:6]}",
        source="monitoring:summary",
        level=EpistemicLevel.OBSERVED_FACT,
        statement=statement,
        severity=Severity.INFO,
        data={"check": f"no_{name}", "polarity": "normal", "checks_evaluated": len(checks)},
    )


def build_evidence(report: MonitoringReport) -> list[Evidence]:
    """Anomaly evidence for every triggered finding plus summarised normal evidence."""
    evidence = [from_finding(f) for f in report.triggered]

    def group(*checks: str) -> list[Finding]:
        return [f for f in report.findings if f.check in checks]

    summaries = [
        (
            "covariate_drift",
            group("numeric_drift", "categorical_drift"),
            "no feature shows material distribution drift ({n} features tested)",
        ),
        (
            "quality_violation",
            group(
                "missing_rate", "duplicate_rate", "range_violation_rate", "unexpected_category_rate"
            ),
            "no data-quality rate exceeds its threshold ({n} checks)",
        ),
        (
            "schema_violation",
            group("schema_missing_columns", "schema_type_mismatch"),
            "the batch conforms to the data contract",
        ),
        (
            "performance_drop",
            group("accuracy_drop", "precision_drop", "recall_drop", "f1_drop", "roc_auc_drop"),
            "supervised performance is within tolerance of the reference ({n} metrics)",
        ),
        (
            "concept_alarm",
            group("adwin", "page_hinkley"),
            "neither ADWIN nor Page-Hinkley signalled a change in the error stream",
        ),
        (
            "association_gain",
            group("label_association_gain"),
            "no feature's association with the label increased ({n} features)",
        ),
        ("label_delay", group("label_availability"), "labels are available for the batch"),
        ("artifact_mismatch", group("artifact_hash"), "the serving artifact matches the registry"),
        (
            "prediction_shift",
            group("prediction_distribution"),
            "the model's score distribution matches the reference",
        ),
    ]
    for name, checks, text in summaries:
        if checks and not any(f.triggered for f in checks):
            evidence.append(_normal(report, name, text.format(n=len(checks)), checks))
    return evidence
