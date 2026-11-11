"""The tool allowlist.

Every capability the investigation can use is declared here with a Pydantic
input schema (``extra="forbid"``) and output schema. ``ToolExecutor`` is the
only way to run one: it rejects unknown tools and unknown or invalid arguments,
enforces a timeout and the per-investigation call budget, and records a trace
for every attempt including the rejected ones.

There is no tool that runs code, shell commands, or reads arbitrary paths, and
no tool that approves or executes a remediation.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aegis.diagnosis.features import inspect_feature
from aegis.diagnosis.importance import shap_importances
from aegis.diagnosis.performance import compare_versions, error_slices
from aegis.diagnosis.pipeline import inspection_evidence
from aegis.ingestion.datasets import TARGET
from aegis.monitoring.engine import MonitoringInput, MonitoringResult
from aegis.observability.metrics import TOOL_CALLS, TOOL_SECONDS
from aegis.observability.tracing import span
from aegis.rag.retriever import RunbookRetriever
from aegis.remediation.validation import validate_proposal
from aegis.schemas.deployment import DeploymentState
from aegis.schemas.diagnosis import Evidence
from aegis.schemas.findings import EpistemicLevel, Severity, utcnow
from aegis.schemas.remediation import RemediationProposal

MAX_TOOL_CALLS = 5


class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ schemas
class QualityReportIn(_In):
    pass


class DriftReportIn(_In):
    top_k: int = Field(default=5, ge=1, le=20)


class ModelMetricsIn(_In):
    include_slices: bool = True


class InspectFeatureIn(_In):
    feature: str = Field(min_length=1, max_length=64)


class CompareVersionsIn(_In):
    other_version: str | None = Field(default=None, max_length=80)


class RetrieveRunbookIn(_In):
    query: str = Field(min_length=3, max_length=300)
    top_k: int = Field(default=3, ge=1, le=5)


class ProposalIn(_In):
    proposal: RemediationProposal


class ApprovalRequestIn(_In):
    summary: str = Field(max_length=600)


class RollbackCandidateIn(_In):
    reason: str = Field(max_length=300)


class ToolOutput(BaseModel):
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)


class ToolTrace(BaseModel):
    call_id: str
    tool: str
    requested_by: Literal["llm", "rules", "workflow"]
    requested_args: dict[str, Any]
    validated_args: dict[str, Any] | None = None
    status: Literal["ok", "rejected", "error", "timeout"]
    output: dict[str, Any] | None = None
    error: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0
    started_at: datetime = Field(default_factory=utcnow)


@dataclass
class ToolContext:
    """Everything tools may touch. Built by the service layer, never by the LLM."""

    inp: MonitoringInput
    result: MonitoringResult
    retriever: RunbookRetriever
    state: DeploymentState
    validation_seed: int = 0
    sandbox: dict[str, Any] = field(
        default_factory=dict
    )  # proposal/validation of this investigation


@dataclass(frozen=True)
class ToolSpec:
    name: str
    kind: Literal["diagnostic", "remediation"]
    description: str
    input_model: type[BaseModel]
    fn: Callable[[Any, ToolContext], tuple[ToolOutput, list[Evidence]]]
    counts_against_budget: bool = True


def _tool_evidence(
    tool: str,
    key: str,
    statement: str,
    data: dict,
    feature: str | None = None,
    level=EpistemicLevel.OBSERVED_FACT,
) -> Evidence:
    digest = hashlib.sha256(f"{tool}|{key}".encode()).hexdigest()[:6]
    return Evidence(
        evidence_id=f"EV-{tool.split('_')[0][:4]}-{digest}",
        source=f"tool:{tool}",
        level=level,
        statement=statement,
        feature=feature,
        severity=Severity.INFO,
        data=data,
    )


# ------------------------------------------------------------- diagnostics
def _quality(args: QualityReportIn, ctx: ToolContext):
    report = ctx.result.report
    quality = [f for f in report.findings if f.monitor == "quality"]
    active = [f for f in quality if f.triggered]
    data = {
        "n_rows": report.n_current,
        "checks_evaluated": len(quality),
        "active": [
            {
                "check": f.check,
                "feature": f.feature,
                "observed": f.observed_value,
                "reference": f.reference_value,
                "severity": f.severity.value,
                "detail": f.interpretation,
            }
            for f in active
        ],
        "skipped": [s.model_dump() for s in report.skipped if s.monitor == "quality"],
    }
    return ToolOutput(
        summary=f"{len(active)} of {len(quality)} data-quality checks are active", data=data
    ), []


def _drift(args: DriftReportIn, ctx: ToolContext):
    drift = sorted(
        (f for f in ctx.result.report.findings if f.monitor == "drift"),
        key=lambda f: -(f.observed_value or 0.0),
    )
    rows = [
        {
            "feature": f.feature,
            "kind": f.check.split("_")[0],
            "psi": f.observed_value,
            "p_adjusted": f.evidence.get("p_adjusted"),
            "flagged": f.triggered,
            "ks": f.evidence.get("ks_statistic"),
            "wasserstein_sd": f.evidence.get("wasserstein_std"),
            "cramers_v": f.evidence.get("cramers_v"),
        }
        for f in drift[: args.top_k]
    ]
    evidence: list[Evidence] = []
    data: dict[str, Any] = {"features_tested": len(drift), "top": rows, "importance": None}
    flagged = [f.feature for f in drift if f.triggered]
    if (
        flagged
        and ctx.result.serving_version
        and set(ctx.inp.spec.features) <= set(ctx.result.current.columns)
    ):
        from aegis.monitoring.engine import _load

        model = _load(ctx.inp.registry, ctx.inp.spec.name, ctx.result.serving_version)
        shap_out = shap_importances(model, ctx.inp.reference, ctx.inp.spec.features, max_rows=200)
        shares = shap_out["mean_abs_shap_share"]
        data["importance"] = {
            "method": "mean |SHAP| share, 200 reference rows",
            "seconds": shap_out["seconds"],
            "drifted": {f: shares.get(f) for f in flagged},
        }
        for feature in flagged[:3]:
            rank = list(shares).index(feature) + 1
            evidence.append(
                _tool_evidence(
                    "get_data_drift_report",
                    f"imp|{feature}",
                    f"drifted feature '{feature}' carries {shares[feature]:.1%} of the model's attribution "
                    f"(rank {rank} of {len(shares)})",
                    {
                        "check": "feature_importance",
                        "polarity": "normal",
                        "share": shares[feature],
                        "rank": rank,
                    },
                    feature,
                )
            )
    return ToolOutput(
        summary=f"{len(flagged)} of {len(drift)} features flagged for drift: {flagged}", data=data
    ), evidence


def _metrics(args: ModelMetricsIn, ctx: ToolContext):
    result, record = ctx.result, ctx.inp.registry.get(ctx.inp.spec.name, ctx.inp.model_version)
    perf = result.performance
    keys = ("n", "accuracy", "precision", "recall", "f1", "roc_auc", "brier")
    data: dict[str, Any] = {
        "declared_version": ctx.inp.model_version,
        "serving_version": result.serving_version,
        "labelled_rows": perf.get("n_labelled", 0),
        "rows": result.report.n_current,
        "reference": {k: perf["reference"][k] for k in keys} if perf.get("reference") else None,
        "current": {k: perf["current"][k] for k in keys} if perf.get("current") else None,
        "confusion_matrix": perf["current"]["confusion_matrix"] if perf.get("current") else None,
        "unavailable_reasons": [
            s.reason for s in result.report.skipped if s.monitor in ("performance", "concept")
        ],
        "concept_detectors": [
            {
                "detector": f.check,
                "alarm": f.triggered,
                "detection_index": f.evidence.get("detection_index"),
            }
            for f in result.report.findings
            if f.monitor == "concept"
        ],
        "slices": None,
    }
    evidence: list[Evidence] = []
    if perf.get("current") is None:
        return ToolOutput(
            summary="supervised metrics unavailable: "
            + "; ".join(data["unavailable_reasons"])[:200],
            data=data,
        ), evidence
    if args.include_slices and result.current_scores is not None:
        from aegis.monitoring.engine import _reference_scores

        current = result.current
        labelled = current[TARGET].notna().to_numpy()
        cur_err = (
            (result.current_scores[labelled] >= record.threshold).astype(int)
            != current[TARGET].to_numpy()[labelled].astype(int)
        ).astype(float)
        ref_scores = _reference_scores(ctx.inp)
        ref_err = (
            (ref_scores >= record.threshold).astype(int) != ctx.inp.reference[TARGET].to_numpy()
        ).astype(float)
        slices = error_slices(ctx.inp.spec, ctx.inp.reference, ref_err, current[labelled], cur_err)
        data["slices"] = slices
        overall = float(cur_err.mean() - ref_err.mean())
        if slices and overall > 0.02:
            top = slices[0]
            concentrated = top["rise"] > 2.5 * overall
            evidence.append(
                _tool_evidence(
                    "get_model_metrics",
                    "slices",
                    f"overall error rate rose by {overall:+.3f}; the largest segment rise is {top['rise']:+.3f} "
                    f"({top['feature']}={top['segment']}), i.e. the rise is "
                    + (
                        "concentrated in a segment"
                        if concentrated
                        else "broadly uniform across segments"
                    ),
                    {
                        "check": "error_slices",
                        "polarity": "normal",
                        "overall_rise": overall,
                        "concentrated": concentrated,
                        "top_segment": f"{top['feature']}={top['segment']}",
                    },
                )
            )
    cur, ref = data["current"], data["reference"]
    return ToolOutput(
        summary=f"F1 {ref['f1']:.3f}->{cur['f1']:.3f}, ROC-AUC {ref['roc_auc']:.3f}->"
        + (f"{cur['roc_auc']:.3f}" if cur["roc_auc"] is not None else "n/a")
        + f" on {cur['n']} labelled rows",
        data=data,
    ), evidence


def _inspect(args: InspectFeatureIn, ctx: ToolContext):
    kinds = {c.name: c.kind for c in ctx.inp.contract.columns}
    if args.feature not in kinds:
        raise ToolArgumentError(
            f"unknown feature {args.feature!r}; contracted features: {sorted(kinds)}"
        )
    current = ctx.result.current
    if args.feature not in current.columns:
        raise ToolArgumentError(f"feature {args.feature!r} is absent from the current batch")
    inspection = inspect_feature(
        args.feature, kinds[args.feature], ctx.inp.reference, current, ctx.inp.spec.numeric_features
    )
    return (
        ToolOutput(
            summary=f"{args.feature}: {inspection.pattern} - {inspection.summary}",
            data=inspection.model_dump(),
        ),
        [inspection_evidence(inspection)],
    )


def _compare(args: CompareVersionsIn, ctx: ToolContext):
    registry, spec = ctx.inp.registry, ctx.inp.spec
    declared = ctx.inp.model_version
    other = args.other_version
    if other is None:
        serving = ctx.result.serving_version
        if serving and serving != declared:
            other = serving
        else:
            others = [r.version for r in registry.list(spec.name) if r.version != declared]
            other = others[0] if others else None
    if other is None:
        return ToolOutput(
            summary="no other registered version to compare with", data={"available": False}
        ), []
    try:
        registry.get(spec.name, other)
    except (KeyError, ValueError):
        known = [r.version for r in registry.list(spec.name)]
        raise ToolArgumentError(f"unknown model version {other!r}; registered: {known}") from None
    out = compare_versions(registry, spec, declared, other, ctx.inp.reference, ctx.result.current)
    evidence = []
    cur = out["datasets"]["current"]
    if cur.get("available"):
        delta = cur["delta_b_minus_a"]["roc_auc"]
        evidence.append(
            _tool_evidence(
                "compare_model_versions",
                other,
                f"on current labelled data, version {other} has ROC-AUC {cur['b']['roc_auc']:.3f} versus "
                f"{cur['a']['roc_auc']:.3f} for declared version {declared} (difference {delta:+.3f})",
                {
                    "check": "version_comparison",
                    "polarity": "normal",
                    "other_version": other,
                    "auc_delta": delta,
                },
            )
        )
    return ToolOutput(summary=f"compared declared {declared} with {other}", data=out), evidence


def _runbook(args: RetrieveRunbookIn, ctx: ToolContext):
    hits = ctx.retriever.search(args.query, top_k=args.top_k)
    return ToolOutput(
        summary=f"{len(hits)} runbook passage(s): {[h.passage_id for h in hits]}",
        data={
            "corpus_version": ctx.retriever.corpus_version,
            "passages": [h.model_dump() for h in hits],
        },
    ), []


# -------------------------------------------------------------- remediation
def _create_proposal(args: ProposalIn, ctx: ToolContext):
    ctx.sandbox["proposal"] = args.proposal
    ctx.sandbox.pop("validation", None)
    return ToolOutput(
        summary=f"proposal recorded: {args.proposal.action.value}",
        data={"proposal": args.proposal.model_dump(mode="json")},
    ), []


def _run_validation(args: _In, ctx: ToolContext):
    proposal = ctx.sandbox.get("proposal")
    if proposal is None:
        raise ToolArgumentError("no proposal has been created in this investigation")
    validation = validate_proposal(
        proposal, ctx.inp, ctx.result, ctx.state, seed=ctx.validation_seed
    )
    ctx.sandbox["validation"] = validation
    return ToolOutput(
        summary=f"validation {'passed' if validation.passed else 'FAILED'}: {validation.summary}",
        data={"validation": validation.model_dump(mode="json")},
    ), []


def _request_approval(args: ApprovalRequestIn, ctx: ToolContext):
    validation = ctx.sandbox.get("validation")
    if ctx.sandbox.get("proposal") is None or validation is None or not validation.passed:
        raise ToolArgumentError(
            "approval can only be requested for a proposal with a passing validation"
        )
    ctx.sandbox["approval_requested"] = True
    # This only *requests*. The approval itself is a database row written by an authenticated human.
    return ToolOutput(
        summary="human approval requested",
        data={"status": "PENDING_APPROVAL", "note": args.summary},
    ), []


def _rollback_candidate(args: RollbackCandidateIn, ctx: ToolContext):
    discarded = ctx.sandbox.pop("validation", None) is not None
    ctx.sandbox.pop("approval_requested", None)
    ctx.sandbox["candidate_discarded"] = args.reason
    return ToolOutput(
        summary="sandbox candidate discarded; live deployment state untouched",
        data={"had_validation": discarded, "reason": args.reason},
    ), []


class ToolArgumentError(ValueError):
    """Arguments are well-formed but refer to something that does not exist."""


TOOLS: dict[str, ToolSpec] = {
    t.name: t
    for t in [
        ToolSpec(
            "get_data_quality_report",
            "diagnostic",
            "Active data-quality checks: missing values, duplicates, ranges, schema.",
            QualityReportIn,
            _quality,
        ),
        ToolSpec(
            "get_data_drift_report",
            "diagnostic",
            "Per-feature drift statistics (PSI, KS, Wasserstein, chi-square) and importance of drifted features.",
            DriftReportIn,
            _drift,
        ),
        ToolSpec(
            "get_model_metrics",
            "diagnostic",
            "Model performance on labelled rows vs reference, concept-drift detectors, error slices.",
            ModelMetricsIn,
            _metrics,
        ),
        ToolSpec(
            "inspect_feature_distribution",
            "diagnostic",
            "How one feature changed: additive shift, multiplicative scale, category movement.",
            InspectFeatureIn,
            _inspect,
        ),
        ToolSpec(
            "compare_model_versions",
            "diagnostic",
            "Compare the declared model version with another registered version on the same data.",
            CompareVersionsIn,
            _compare,
        ),
        ToolSpec(
            "retrieve_runbook",
            "diagnostic",
            "Retrieve relevant runbook passages.",
            RetrieveRunbookIn,
            _runbook,
        ),
        ToolSpec(
            "create_remediation_proposal",
            "remediation",
            "Record a remediation proposal.",
            ProposalIn,
            _create_proposal,
            counts_against_budget=False,
        ),
        ToolSpec(
            "run_candidate_validation",
            "remediation",
            "Validate the proposal in the sandbox.",
            _In,
            _run_validation,
            counts_against_budget=False,
        ),
        ToolSpec(
            "request_human_approval",
            "remediation",
            "Ask a human to approve a validated proposal.",
            ApprovalRequestIn,
            _request_approval,
            counts_against_budget=False,
        ),
        ToolSpec(
            "rollback_candidate",
            "remediation",
            "Discard the sandbox candidate.",
            RollbackCandidateIn,
            _rollback_candidate,
            counts_against_budget=False,
        ),
    ]
}
DIAGNOSTIC_TOOLS = [name for name, t in TOOLS.items() if t.kind == "diagnostic"]
#: The only tools a language model may request. Remediation tools are driven by the workflow.
LLM_SELECTABLE_TOOLS = [n for n in DIAGNOSTIC_TOOLS if n != "retrieve_runbook"]


class ToolExecutor:
    def __init__(
        self, ctx: ToolContext, *, max_calls: int = MAX_TOOL_CALLS, timeout_s: float = 30.0
    ) -> None:
        self.ctx = ctx
        self.max_calls = min(max_calls, MAX_TOOL_CALLS)
        self.timeout_s = timeout_s
        self.diagnostic_calls = 0
        self.traces: list[ToolTrace] = []
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="aegis-tool"
        )

    @property
    def budget_left(self) -> int:
        return self.max_calls - self.diagnostic_calls

    def reject(
        self, name: str, args: dict[str, Any] | None, requested_by: str, reason: str
    ) -> ToolTrace:
        """Record a call that the workflow refused before execution (e.g. reserved budget)."""
        trace = ToolTrace(
            call_id=f"CALL-{uuid.uuid4().hex[:10]}",
            tool=str(name)[:80],
            requested_by=requested_by,
            requested_args=_bounded(args if isinstance(args, dict) else {}),
            status="rejected",
            error=reason,
        )
        TOOL_CALLS.labels(trace.tool if name in TOOLS else "unknown", "rejected").inc()
        self.traces.append(trace)
        return trace

    def execute(
        self,
        name: str,
        args: dict[str, Any] | None,
        requested_by: str = "workflow",
        allowed: list[str] | None = None,
    ) -> tuple[ToolTrace, list[Evidence]]:
        raw_args = args if isinstance(args, dict) else {}
        trace = ToolTrace(
            call_id=f"CALL-{uuid.uuid4().hex[:10]}",
            tool=str(name)[:80],
            requested_by=requested_by,
            requested_args=_bounded(raw_args),
            status="rejected",
        )
        evidence: list[Evidence] = []
        started = time.perf_counter()
        with span("tool.execute", tool=trace.tool, requested_by=requested_by) as current:
            spec = TOOLS.get(name) if isinstance(name, str) else None
            if spec is None or (allowed is not None and name not in allowed):
                trace.error = f"tool {str(name)[:60]!r} is not on the allowlist for this caller"
            elif not isinstance(args, dict | type(None)):
                trace.error = "arguments must be a JSON object"
            elif spec.counts_against_budget and self.budget_left <= 0:
                trace.error = f"diagnostic tool budget of {self.max_calls} calls is exhausted"
            else:
                try:
                    validated = spec.input_model.model_validate(raw_args)
                except ValidationError as exc:
                    trace.error = "invalid arguments: " + "; ".join(
                        f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}"
                        for e in exc.errors()[:4]
                    )
                else:
                    trace.validated_args = validated.model_dump(mode="json")
                    if spec.counts_against_budget:
                        self.diagnostic_calls += 1
                    future = self._pool.submit(spec.fn, validated, self.ctx)
                    try:
                        output, evidence = future.result(timeout=self.timeout_s)
                        trace.status, trace.output = "ok", _jsonable(output.model_dump())
                        trace.evidence_ids = [e.evidence_id for e in evidence]
                    except concurrent.futures.TimeoutError:
                        future.cancel()
                        trace.status, trace.error = (
                            "timeout",
                            f"no result within {self.timeout_s:.0f}s",
                        )
                    except ToolArgumentError as exc:
                        trace.status, trace.error = "rejected", str(exc)[:300]
                    except Exception as exc:
                        trace.status, trace.error = (
                            "error",
                            f"{type(exc).__name__}: {str(exc)[:200]}",
                        )
            trace.duration_ms = round((time.perf_counter() - started) * 1000, 2)
            current.set_attribute("tool.status", trace.status)
            if trace.status != "ok":
                current.add_event(f"tool_{trace.status}")
        TOOL_CALLS.labels(trace.tool if spec else "unknown", trace.status).inc()
        if spec:
            TOOL_SECONDS.labels(trace.tool).observe(trace.duration_ms / 1000)
        self.traces.append(trace)
        return trace, evidence if trace.status == "ok" else []


def _bounded(value: Any, depth: int = 0) -> Any:
    """Store untrusted requested arguments safely: bounded size, JSON types only."""
    if depth > 3:
        return "<truncated>"
    if isinstance(value, dict):
        return {str(k)[:60]: _bounded(v, depth + 1) for k, v in list(value.items())[:12]}
    if isinstance(value, list):
        return [_bounded(v, depth + 1) for v in value[:12]]
    if isinstance(value, str):
        return value[:300]
    if isinstance(value, bool | int | float) or value is None:
        return value
    return str(value)[:100]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return value
