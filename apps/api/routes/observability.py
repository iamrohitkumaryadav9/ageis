"""Read-only operational views: overview, audit trail, models, traces, latency, metrics."""

from __future__ import annotations

import numpy as np
import psutil
from fastapi import APIRouter, Depends, Query, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import func, select

from aegis.ingestion.datasets import DATASETS
from aegis.observability.tracing import recent_spans
from aegis.remediation.reconciliation import events_from_audit, reconcile
from aegis.services.core import AegisService
from aegis.services.errors import Principal
from aegis.storage import models as m
from apps.api.deps import get_service, require
from apps.api.schemas import AuditEventOut, Page

router = APIRouter(tags=["observability"])
ACTIVE = ("open", "investigating", "investigated", "awaiting_approval", "executed_unverified")


@router.get("/overview")
def overview(
    service: AegisService = Depends(get_service), _: Principal = Depends(require("viewer"))
) -> dict:
    with service.db.session() as session:
        total = session.scalar(select(func.count()).select_from(m.Incident)) or 0
        by_status = dict(
            session.execute(
                select(m.Incident.status, func.count()).group_by(m.Incident.status)
            ).all()
        )
        by_severity = dict(
            session.execute(
                select(m.Incident.severity, func.count()).group_by(m.Incident.severity)
            ).all()
        )
        active = sum(v for k, v in by_status.items() if k in ACTIVE)
        critical = (
            session.scalar(
                select(func.count())
                .select_from(m.Incident)
                .where(m.Incident.severity == "critical", m.Incident.status.in_(ACTIVE))
            )
            or 0
        )
        pending = (
            session.scalar(
                select(func.count())
                .select_from(m.Remediation)
                .where(m.Remediation.state == "PENDING_APPROVAL")
            )
            or 0
        )
        datasets = []
        for name in DATASETS:
            run = session.scalars(
                select(m.MonitoringRun)
                .where(m.MonitoringRun.dataset == name)
                .order_by(m.MonitoringRun.created_at.desc())
            ).first()
            deployment = session.scalars(
                select(m.DeploymentRevision).where(
                    m.DeploymentRevision.dataset == name, m.DeploymentRevision.active
                )
            ).first()
            entry: dict = {"dataset": name, "deployment": None, "last_run": None}
            if deployment is not None:
                entry["deployment"] = {
                    "revision": deployment.revision,
                    **deployment.state,
                    "activated_at": deployment.created_at,
                    "reason": deployment.reason,
                }
            if run is not None:
                drift = [f for f in run.report["findings"] if f["monitor"] == "drift"]
                current = run.performance.get("current")
                reference = run.performance.get("reference")
                keys = ("accuracy", "precision", "recall", "f1", "roc_auc")
                entry["last_run"] = {
                    "id": run.id,
                    "at": run.created_at,
                    "incident_detected": run.incident_detected,
                    "max_severity": run.max_severity,
                    "rows": run.report["n_current"],
                    "duration_ms": run.duration_ms,
                    "drift": {
                        "features_tested": len(drift),
                        "features_drifted": sorted(f["feature"] for f in drift if f["triggered"]),
                    },
                    # ``None`` means "not measurable" (e.g. labels have not arrived), never zero.
                    "performance": {
                        "current": {k: current[k] for k in keys} if current else None,
                        "reference": {k: reference[k] for k in keys} if reference else None,
                        "labelled_rows": run.performance.get("n_labelled", 0),
                    },
                }
            datasets.append(entry)
        return {
            "incidents": {
                "total": total,
                "active": active,
                "critical_active": critical,
                "by_status": by_status,
                "by_severity": by_severity,
            },
            "pending_approvals": pending,
            "datasets": datasets,
            "monitoring_runs": session.scalar(select(func.count()).select_from(m.MonitoringRun))
            or 0,
        }


@router.get("/audit", response_model=Page[AuditEventOut])
def audit(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    incident_id: str | None = Query(None, max_length=40),
    entity_id: str | None = Query(None, max_length=60),
    service: AegisService = Depends(get_service),
    _: Principal = Depends(require("viewer")),
) -> Page[AuditEventOut]:
    filters = []
    if incident_id:
        filters.append(m.AuditEvent.incident_id == incident_id)
    if entity_id:
        filters.append(m.AuditEvent.entity_id == entity_id)
    with service.db.session() as session:
        total = session.scalar(select(func.count()).select_from(m.AuditEvent).where(*filters)) or 0
        rows = session.scalars(
            select(m.AuditEvent)
            .where(*filters)
            .order_by(m.AuditEvent.id)
            .limit(limit)
            .offset(offset)
        ).all()
        return Page[AuditEventOut](
            items=[AuditEventOut.model_validate(r) for r in rows],
            total=total,
            limit=limit,
            offset=offset,
        )


@router.get("/remediation-lifecycle/reconciliation")
def lifecycle_reconciliation(
    incident_id: str | None = Query(None, max_length=40),
    service: AegisService = Depends(get_service),
    _: Principal = Depends(require("viewer")),
) -> dict:
    """Account for every remediation record in the audit trail (optionally for one incident).

    Missing, duplicated or out-of-order records are returned as ``failures``; ``reconciled`` is
    true only when there are none. Proposals still waiting for a human are open, not failed.
    """
    filters = [m.AuditEvent.entity_type == "remediation"]
    if incident_id:
        filters.append(m.AuditEvent.incident_id == incident_id)
    with service.db.session() as session:
        rows = session.scalars(select(m.AuditEvent).where(*filters).order_by(m.AuditEvent.id)).all()
        audit_rows = [
            {
                "id": r.id,
                "action": r.action,
                "entity_id": r.entity_id,
                "incident_id": r.incident_id,
                "outcome": r.outcome,
                "actor": r.actor,
                "detail": r.detail,
            }
            for r in rows
        ]
        remediations = session.scalar(
            select(func.count())
            .select_from(m.Remediation)
            .where(*([m.Remediation.incident_id == incident_id] if incident_id else []))
        )
    report = reconcile(
        events_from_audit(audit_rows),
        expected_proposals=int(remediations or 0),
        require_policy_events=False,
    )
    report["source"] = "audit_events"
    report["scope"] = {"incident_id": incident_id}
    report["denied_attempts_excluded"] = sum(r["outcome"] == "denied" for r in audit_rows)
    return report


@router.get("/models")
def models(
    service: AegisService = Depends(get_service), _: Principal = Depends(require("viewer"))
) -> dict:
    out = {}
    for name in DATASETS:
        records = service.registry.list(name)
        out[name] = [
            {
                "version": r.version,
                "algorithm": r.algorithm,
                "created_at": r.created_at,
                "artifact_sha256": r.artifact_sha256,
                "threshold": r.threshold,
                "notes": r.notes,
                "parent_version": r.parent_version,
                "test_metrics": {
                    k: (r.metrics.get("test") or {}).get(k)
                    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "brier")
                },
            }
            for r in records
        ]
    return out


@router.get("/observability/latency")
def latency(_: Principal = Depends(require("viewer"))) -> dict:
    """P50/P95 duration and failure rate per span name, from the in-process span buffer."""
    groups: dict[str, list[dict]] = {}
    for item in recent_spans.snapshot():
        key = item["name"]
        if key == "tool.execute":
            key = f"tool:{item['attributes'].get('tool', '?')}"
        elif key == "llm.chat":
            key = f"llm:{item['attributes'].get('purpose', '?')}"
        groups.setdefault(key, []).append(item)
    rows = []
    for name, items in sorted(groups.items()):
        durations = np.array([s["duration_ms"] for s in items])
        failed = sum(
            s["status"] == "ERROR" or any(e.startswith(("tool_", "llm_")) for e in s["events"])
            for s in items
        )
        rows.append(
            {
                "span": name,
                "count": len(items),
                "p50_ms": round(float(np.percentile(durations, 50)), 2),
                "p95_ms": round(float(np.percentile(durations, 95)), 2),
                "max_ms": round(float(durations.max()), 2),
                "failure_rate": round(failed / len(items), 4),
            }
        )
    process = psutil.Process()
    return {
        "spans": rows,
        "process": {
            "rss_mb": round(process.memory_info().rss / 1e6, 1),
            "cpu_percent": process.cpu_percent(interval=None),
            "threads": process.num_threads(),
        },
    }


@router.get("/observability/traces")
def traces(
    trace_id: str | None = Query(None, max_length=40),
    prefix: str = Query("", max_length=60),
    limit: int = Query(100, ge=1, le=500),
    _: Principal = Depends(require("viewer")),
) -> dict:
    spans = recent_spans.snapshot(prefix, trace_id)[-limit:]
    return {"spans": spans, "count": len(spans)}


@router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
