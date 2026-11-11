"""Ablation A baseline: read the root cause straight off the alert list.

No diagnostics, no reasoning about what is absent: each triggered check votes
for the cause it is named after, and causes are ranked by the worst severity
seen. This is what a monitoring dashboard alone gives an on-call engineer.
"""

from __future__ import annotations

from aegis.diagnosis.evidence import evidence_id_for
from aegis.diagnosis.rules import conclude
from aegis.schemas.diagnosis import Diagnosis, Hypothesis, RootCause
from aegis.schemas.findings import MonitoringReport, Severity

CHECK_TO_CAUSE: dict[str, RootCause] = {
    "numeric_drift": RootCause.NUMERICAL_SHIFT,
    "categorical_drift": RootCause.CATEGORICAL_SHIFT,
    "unexpected_category_rate": RootCause.CATEGORICAL_SHIFT,
    "missing_rate": RootCause.MISSING_VALUES,
    "range_violation_rate": RootCause.SCALING_ERROR,
    "duplicate_rate": RootCause.DUPLICATE_RECORDS,
    "schema_missing_columns": RootCause.SCHEMA_CHANGE,
    "schema_type_mismatch": RootCause.SCHEMA_CHANGE,
    "label_availability": RootCause.LABEL_DELAY,
    "artifact_hash": RootCause.ARTIFACT_MISMATCH,
    "label_association_gain": RootCause.FEATURE_LEAKAGE,
    "label_association_reversal": RootCause.CONCEPT_DRIFT,
    "adwin": RootCause.CONCEPT_DRIFT,
    "page_hinkley": RootCause.CONCEPT_DRIFT,
    "accuracy_drop": RootCause.CONCEPT_DRIFT,
    "precision_drop": RootCause.CONCEPT_DRIFT,
    "recall_drop": RootCause.CONCEPT_DRIFT,
    "f1_drop": RootCause.CONCEPT_DRIFT,
    "roc_auc_drop": RootCause.CONCEPT_DRIFT,
}


def naive_diagnosis(report: MonitoringReport) -> Diagnosis:
    votes: dict[RootCause, list] = {}
    for finding in report.triggered:
        cause = CHECK_TO_CAUSE.get(finding.check)
        if cause is not None:
            votes.setdefault(cause, []).append(finding)
    hypotheses = []
    for cause, findings in votes.items():
        worst = max(f.severity.rank for f in findings)
        confidence = (
            0.35 + 0.15 * worst / Severity.CRITICAL.rank * 4 + min(len(findings), 5) * 0.001
        )
        hypotheses.append(
            Hypothesis(
                hypothesis_id=f"HYP-naive-{cause.value}",
                root_cause=cause,
                affected_features=sorted({f.feature for f in findings if f.feature}),
                supporting_evidence_ids=[evidence_id_for(f.finding_id) for f in findings],
                confidence=round(min(confidence, 0.99), 3),
                reasoning=f"{len(findings)} alert(s) of type {sorted({f.check for f in findings})}.",
                ranked_by="monitoring-only",
            )
        )
    hypotheses.sort(key=lambda h: (-h.confidence, h.root_cause.value))
    return Diagnosis(hypotheses=hypotheses, engine="monitoring-only", **conclude(hypotheses))
