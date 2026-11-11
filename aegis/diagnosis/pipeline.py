"""The deterministic diagnosis path (ablation B, and the fallback when no LLM is available)."""

from __future__ import annotations

import hashlib

from aegis.diagnosis.evidence import build_evidence
from aegis.diagnosis.features import FeatureInspection, inspect_feature
from aegis.diagnosis.rules import rank_hypotheses
from aegis.monitoring.engine import MonitoringInput, MonitoringResult
from aegis.schemas.diagnosis import Diagnosis, Evidence
from aegis.schemas.findings import EpistemicLevel, Severity

MAX_INSPECTIONS = 3


def inspection_evidence(inspection: FeatureInspection) -> Evidence:
    digest = hashlib.sha256(inspection.feature.encode()).hexdigest()[:6]
    return Evidence(
        evidence_id=f"EV-insp-{digest}",
        source="tool:inspect_feature_distribution",
        level=EpistemicLevel.OBSERVED_FACT,
        statement=f"'{inspection.feature}': {inspection.summary}",
        feature=inspection.feature,
        severity=Severity.INFO,
        data={
            "check": "feature_pattern",
            "polarity": "normal" if inspection.pattern == "no_change" else "anomaly",
            "pattern": inspection.pattern,
            "scale_factor": inspection.scale_factor,
            "shift_offset": inspection.shift_offset,
            "scale_unexplained": inspection.scale_unexplained,
            "shift_unexplained": inspection.shift_unexplained,
            "max_correlation_change": inspection.max_correlation_change,
        },
    )


def features_worth_inspecting(evidence: list[Evidence], limit: int = MAX_INSPECTIONS) -> list[str]:
    """Numeric features whose change needs characterising (shift or scale?), strongest first.

    Drifted features come first, by PSI. Features with out-of-range values but no drift finding
    follow: a rescaled sparse feature (mostly zeros) can leave the bulk of its distribution
    untouched, so drift tests stay quiet while a few values land far outside the contract.
    """

    def anomalies(check: str) -> list[Evidence]:
        return [
            e
            for e in evidence
            if e.data.get("check") == check and e.data.get("polarity") == "anomaly"
        ]

    drifted = sorted(anomalies("numeric_drift"), key=lambda e: -(e.data.get("observed") or 0.0))
    ranged = sorted(
        anomalies("range_violation_rate"), key=lambda e: -(e.data.get("observed") or 0.0)
    )
    ordered = list(dict.fromkeys(e.feature for e in [*drifted, *ranged] if e.feature))
    return ordered[:limit]


def deterministic_diagnosis(
    inp: MonitoringInput, result: MonitoringResult
) -> tuple[list[Evidence], Diagnosis]:
    evidence = build_evidence(result.report)
    kinds = {c.name: c.kind for c in inp.contract.columns}
    for feature in features_worth_inspecting(evidence):
        inspection = inspect_feature(
            feature, kinds[feature], inp.reference, result.current, inp.spec.numeric_features
        )
        evidence.append(inspection_evidence(inspection))
    return evidence, rank_hypotheses(evidence)
