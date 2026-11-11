"""Monitoring orchestration: one batch in, one typed report out."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from aegis.ingestion.contracts import DataContract
from aegis.ingestion.datasets import TARGET, DatasetSpec
from aegis.models.registry import ModelRegistry
from aegis.models.training import predict_proba
from aegis.monitoring.concept import check_concept
from aegis.monitoring.config import MonitoringConfig
from aegis.monitoring.drift import check_drift
from aegis.monitoring.integrity import check_integrity
from aegis.monitoring.performance import check_performance, check_prediction_distribution
from aegis.monitoring.quality import check_quality, check_schema
from aegis.monitoring.relationship import check_relationships
from aegis.schemas.deployment import IngestionConfig, apply_ingestion
from aegis.schemas.findings import Finding, MonitoringReport, SkippedCheck


@dataclass
class MonitoringInput:
    spec: DatasetSpec
    contract: DataContract
    reference: pd.DataFrame  # features + target, fully labelled
    current: pd.DataFrame  # features (+ target; NaN where the label has not arrived)
    registry: ModelRegistry
    model_version: str  # the version the deployment *claims* to serve
    artifact_path: Path | None = None  # the file actually being served; default: registry copy
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)


@dataclass
class MonitoringResult:
    report: MonitoringReport
    performance: dict  # reference/current metric dictionaries (current is None without labels)
    serving_version: str | None  # verified identity of the serving artifact
    current: pd.DataFrame  # the batch after ingestion transforms
    current_scores: np.ndarray | None


_model_cache: dict[tuple[str, str, str], object] = {}
_score_cache: dict[tuple[str, str, int, int], np.ndarray] = {}


def _load(registry: ModelRegistry, dataset: str, version: str):
    key = (str(registry.root), dataset, version)
    if key not in _model_cache:
        _model_cache[key] = registry.load(dataset, version, verify=True)
    return _model_cache[key]


def _reference_scores(inp: MonitoringInput) -> np.ndarray:
    key = (str(inp.registry.root), inp.model_version, id(inp.reference), len(inp.reference))
    if key not in _score_cache:
        model = _load(inp.registry, inp.spec.name, inp.model_version)
        _score_cache[key] = predict_proba(model, inp.reference, inp.spec.features)
    return _score_cache[key]


def run_monitoring(
    inp: MonitoringInput, cfg: MonitoringConfig | None = None, report_id: str | None = None
) -> MonitoringResult:
    cfg = cfg or MonitoringConfig()
    report_id = report_id or f"REP-{uuid.uuid4().hex[:10]}"
    started = time.perf_counter()
    timings: dict[str, float] = {}
    findings: list[Finding] = []
    skipped: list[SkippedCheck] = []
    spec, contract = inp.spec, inp.contract

    def timed(name: str, start: float) -> None:
        timings[name] = round((time.perf_counter() - start) * 1000, 2)

    current = apply_ingestion(inp.current, inp.ingestion)
    record = inp.registry.get(spec.name, inp.model_version)

    t = time.perf_counter()
    schema_findings, usable = check_schema(report_id, current, contract, allowed_extra={TARGET})
    findings += schema_findings
    findings += check_quality(report_id, current, inp.reference, contract, usable, cfg)
    timed("quality", t)

    t = time.perf_counter()
    if len(current) < cfg.min_samples:
        skipped.append(
            SkippedCheck(
                monitor="drift",
                check="distribution",
                reason=f"batch has {len(current)} rows (< {cfg.min_samples})",
            )
        )
    else:
        drift_findings, drift_skipped = check_drift(
            report_id, current, inp.reference, contract, usable, cfg
        )
        findings += drift_findings
        skipped += drift_skipped
    timed("drift", t)

    t = time.perf_counter()
    artifact = inp.artifact_path or inp.registry.artifact_path(spec.name, inp.model_version)
    integrity_finding, serving_version = check_integrity(
        report_id, inp.registry, spec.name, inp.model_version, artifact
    )
    findings.append(integrity_finding)
    timed("integrity", t)

    t = time.perf_counter()
    performance: dict = {"reference": None, "current": None, "n_labelled": 0}
    current_scores: np.ndarray | None = None
    schema_ok = set(usable) == set(spec.features)
    if serving_version is None:
        skipped.append(
            SkippedCheck(
                monitor="performance",
                check="scoring",
                reason="serving artifact failed integrity verification; not loaded",
            )
        )
    elif not schema_ok:
        broken = sorted(set(spec.features) - set(usable))
        skipped.append(
            SkippedCheck(
                monitor="performance",
                check="scoring",
                reason=f"batch violates the data contract for {broken}; cannot be scored",
            )
        )
    elif len(current) == 0:
        skipped.append(SkippedCheck(monitor="performance", check="scoring", reason="empty batch"))
    else:
        reference_scores = _reference_scores(inp)
        reference_y = inp.reference[TARGET].to_numpy()
        try:
            current_scores = predict_proba(
                _load(inp.registry, spec.name, serving_version), current, spec.features
            )
        except Exception as exc:  # a model that cannot score its input is itself a finding
            skipped.append(
                SkippedCheck(
                    monitor="performance",
                    check="scoring",
                    reason=f"scoring failed: {type(exc).__name__}",
                )
            )
        if current_scores is not None:
            if len(current) >= cfg.min_samples:
                findings.append(
                    check_prediction_distribution(
                        report_id, reference_scores, current_scores, record.threshold, cfg
                    )
                )
            if TARGET in current.columns:
                current_y = current[TARGET].to_numpy(dtype=float)
            else:
                current_y = np.full(len(current), np.nan)
            perf_findings, perf_skipped, performance = check_performance(
                report_id,
                reference_y,
                reference_scores,
                current_y,
                current_scores,
                record.threshold,
                cfg,
            )
            findings += perf_findings
            skipped += perf_skipped
            timed("performance", t)

            labelled = ~np.isnan(current_y)
            if labelled.sum() >= cfg.min_labelled:
                t = time.perf_counter()
                y = current_y[labelled].astype(int)
                ref_err = (
                    (reference_scores >= record.threshold).astype(int) != reference_y
                ).astype(int)
                cur_err = ((current_scores[labelled] >= record.threshold).astype(int) != y).astype(
                    int
                )
                findings += check_concept(report_id, ref_err, cur_err, cfg)
                timed("concept", t)
                t = time.perf_counter()
                findings += check_relationships(
                    report_id,
                    current[labelled],
                    y,
                    inp.reference,
                    reference_y,
                    contract,
                    usable,
                    cfg,
                )
                timed("relationship", t)
            else:
                reason = f"only {int(labelled.sum())} labelled rows (< {cfg.min_labelled})"
                skipped.append(SkippedCheck(monitor="concept", check="error_stream", reason=reason))
                skipped.append(
                    SkippedCheck(monitor="relationship", check="label_association", reason=reason)
                )
    timings.setdefault("performance", round((time.perf_counter() - t) * 1000, 2))

    report = MonitoringReport(
        report_id=report_id,
        dataset=spec.name,
        model_version=inp.model_version,
        reference_version=contract.version,
        n_reference=len(inp.reference),
        n_current=len(current),
        findings=findings,
        skipped=skipped,
        timings_ms=timings,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return MonitoringResult(report, performance, serving_version, current, current_scores)
