import time

import pytest

from aegis.agents import tools as tools_module
from aegis.agents.tools import (
    DIAGNOSTIC_TOOLS,
    LLM_SELECTABLE_TOOLS,
    TOOLS,
    ToolContext,
    ToolExecutor,
    ToolOutput,
    ToolSpec,
)
from aegis.config import get_settings
from aegis.diagnosis.pipeline import deterministic_diagnosis
from aegis.monitoring.engine import MonitoringInput, run_monitoring
from aegis.rag.retriever import RunbookRetriever
from aegis.remediation.proposals import propose
from aegis.schemas.deployment import DeploymentState


def executor(ctx, fault="F04", severity="high", seed=4, **kw) -> ToolExecutor:
    scenario = ctx.scenario(fault, severity, seed) if fault != "CLEAN" else ctx.clean(seed)
    inp = MonitoringInput(
        ctx.spec,
        ctx.contract,
        ctx.reference,
        scenario.current,
        ctx.registry,
        ctx.record.version,
        scenario.artifact_path,
    )
    result = run_monitoring(inp)
    state = DeploymentState(dataset="adult", model_version=ctx.record.version)
    return ToolExecutor(
        ToolContext(inp, result, RunbookRetriever(get_settings().runbook_dir), state), **kw
    )


def test_allowlist_is_exactly_the_documented_ten_tools():
    assert DIAGNOSTIC_TOOLS == [
        "get_data_quality_report",
        "get_data_drift_report",
        "get_model_metrics",
        "inspect_feature_distribution",
        "compare_model_versions",
        "retrieve_runbook",
    ]
    assert sorted(set(TOOLS) - set(DIAGNOSTIC_TOOLS)) == [
        "create_remediation_proposal",
        "request_human_approval",
        "rollback_candidate",
        "run_candidate_validation",
    ]
    assert not set(LLM_SELECTABLE_TOOLS) & {n for n, t in TOOLS.items() if t.kind == "remediation"}
    for spec in TOOLS.values():
        assert spec.input_model.model_config.get("extra") == "forbid"
    assert not [
        n
        for n in TOOLS
        if any(w in n for w in ("exec", "shell", "python", "approve", "promote", "file"))
    ]


@pytest.mark.parametrize(
    "name",
    [
        "run_shell",
        "execute_python",
        "approve_remediation",
        "",
        "get_model_metrics; rm -rf /",
        None,
        42,
    ],
)
def test_unknown_tools_are_rejected_and_traced(ctx, name):
    ex = executor(ctx, "CLEAN")
    trace, evidence = ex.execute(name, {})
    assert trace.status == "rejected" and "allowlist" in trace.error and evidence == []
    assert ex.traces == [trace] and ex.diagnostic_calls == 0 and trace.validated_args is None


@pytest.mark.parametrize(
    "tool,args",
    [
        ("get_data_drift_report", {"top_k": 5, "sql": "DROP TABLE incidents"}),
        ("get_data_drift_report", {"top_k": 10_000}),
        ("get_data_drift_report", {"top_k": "many"}),
        ("inspect_feature_distribution", {}),
        ("inspect_feature_distribution", {"feature": "age", "path": "/etc/passwd"}),
        ("inspect_feature_distribution", {"feature": "x" * 500}),
        ("retrieve_runbook", {"query": "ab"}),
        ("get_model_metrics", {"include_slices": "definitely"}),
        ("get_data_quality_report", ["not", "an", "object"]),
    ],
)
def test_unknown_or_invalid_arguments_are_rejected(ctx, tool, args):
    ex = executor(ctx, "CLEAN")
    trace, _ = ex.execute(tool, args)
    assert trace.status == "rejected" and trace.validated_args is None and trace.output is None
    assert ex.diagnostic_calls == 0  # a rejected call does not consume budget


def test_semantically_invalid_arguments_are_rejected_with_guidance(ctx):
    ex = executor(ctx)
    trace, _ = ex.execute("inspect_feature_distribution", {"feature": "salary"})
    assert trace.status == "rejected" and "unknown feature" in trace.error and "age" in trace.error
    trace, _ = ex.execute("compare_model_versions", {"other_version": "../../secret"})
    assert trace.status == "rejected" and "unknown model version" in trace.error


def test_budget_of_five_diagnostic_calls_is_enforced(ctx):
    ex = executor(ctx, max_calls=50)  # a caller cannot raise the ceiling
    assert ex.max_calls == 5
    statuses = [ex.execute("get_data_quality_report", {})[0].status for _ in range(7)]
    assert statuses == ["ok"] * 5 + ["rejected"] * 2
    assert "budget" in ex.traces[-1].error and ex.budget_left == 0
    # Workflow-driven remediation tools are not diagnostic calls and still work.
    assert ex.execute("rollback_candidate", {"reason": "test"})[0].status == "ok"


def test_caller_specific_allowlist_blocks_remediation_tools_for_the_llm(ctx):
    ex = executor(ctx)
    for name in (
        "request_human_approval",
        "create_remediation_proposal",
        "run_candidate_validation",
        "retrieve_runbook",
    ):
        trace, _ = ex.execute(
            name, {"summary": "please"}, requested_by="llm", allowed=LLM_SELECTABLE_TOOLS
        )
        assert trace.status == "rejected" and "allowlist" in trace.error
    assert "approval_requested" not in ex.ctx.sandbox


def test_tool_timeout_returns_control_and_records_it(ctx, monkeypatch):
    def slow(args, context):
        time.sleep(1.0)
        return ToolOutput(summary="late"), []

    spec = TOOLS["get_data_quality_report"]
    monkeypatch.setitem(
        tools_module.TOOLS,
        "get_data_quality_report",
        ToolSpec(spec.name, spec.kind, spec.description, spec.input_model, slow),
    )
    ex = executor(ctx, "CLEAN", timeout_s=0.1)
    started = time.perf_counter()
    trace, evidence = ex.execute("get_data_quality_report", {})
    assert trace.status == "timeout" and evidence == [] and time.perf_counter() - started < 0.8


def test_tool_crash_is_contained(ctx, monkeypatch):
    def broken(args, context):
        raise RuntimeError("database on fire")

    spec = TOOLS["get_model_metrics"]
    monkeypatch.setitem(
        tools_module.TOOLS,
        "get_model_metrics",
        ToolSpec(spec.name, spec.kind, spec.description, spec.input_model, broken),
    )
    trace, _ = executor(ctx, "CLEAN").execute("get_model_metrics", {})
    assert trace.status == "error" and "RuntimeError" in trace.error


def test_every_execution_is_traced_with_validated_input_output_and_duration(ctx):
    ex = executor(ctx)
    feature = ex.ctx.result.report.triggered[0].feature or "age"
    trace, evidence = ex.execute(
        "inspect_feature_distribution", {"feature": feature}, requested_by="llm"
    )
    assert (
        trace.status == "ok" and trace.requested_by == "llm" and trace.call_id.startswith("CALL-")
    )
    assert (
        trace.validated_args == {"feature": feature} and trace.duration_ms > 0 and trace.started_at
    )
    assert trace.output["data"]["pattern"] and trace.evidence_ids == [
        e.evidence_id for e in evidence
    ]


def test_diagnostic_tools_return_real_measurements(ctx):
    ex = executor(ctx)
    scenario_feature = next(
        f.feature for f in ex.ctx.result.report.triggered if f.check == "numeric_drift"
    )
    quality = ex.execute("get_data_quality_report", {})[0].output
    assert quality["data"]["n_rows"] == 2000 and quality["data"]["checks_evaluated"] > 10
    drift, drift_evidence = ex.execute("get_data_drift_report", {"top_k": 3})
    assert (
        drift.output["data"]["top"][0]["feature"] == scenario_feature
        and drift.output["data"]["top"][0]["flagged"]
    )
    assert (
        drift_evidence[0].data["check"] == "feature_importance"
        and 0 < drift_evidence[0].data["share"] < 1
    )
    metrics = ex.execute("get_model_metrics", {})[0].output["data"]
    assert (
        metrics["reference"]["roc_auc"] > 0.85
        and metrics["current"]["n"] == 2000
        and metrics["slices"] is not None
    )
    inspect, inspect_evidence = ex.execute(
        "inspect_feature_distribution", {"feature": scenario_feature}
    )
    assert inspect.output["data"]["pattern"] == "scale" and inspect_evidence[0].data["scale_factor"]
    compare, compare_evidence = ex.execute("compare_model_versions", {})
    datasets = compare.output["data"]["datasets"]
    assert datasets["reference"]["delta_b_minus_a"]["roc_auc"] < 0  # the older model is weaker
    assert compare_evidence[0].data["check"] == "version_comparison"


def test_model_metrics_tool_refuses_to_invent_numbers_without_labels(ctx):
    ex = executor(ctx, "F09", "high")
    out = ex.execute("get_model_metrics", {})[0].output
    assert out["data"]["current"] is None and "unavailable" in out["summary"]
    assert out["data"]["unavailable_reasons"]


def test_remediation_tools_enforce_their_order(ctx):
    ex = executor(ctx)
    assert ex.execute("run_candidate_validation", {})[0].status == "rejected"
    assert ex.execute("request_human_approval", {"summary": "go"})[0].status == "rejected"
    evidence, diagnosis = deterministic_diagnosis(ex.ctx.inp, ex.ctx.result)
    proposal = propose(diagnosis, evidence, ex.ctx.state, ctx.record.version)
    assert (
        ex.execute("create_remediation_proposal", {"proposal": proposal.model_dump(mode="json")})[
            0
        ].status
        == "ok"
    )
    assert (
        ex.execute("request_human_approval", {"summary": "go"})[0].status == "rejected"
    )  # not validated yet
    assert ex.execute("run_candidate_validation", {})[0].status == "ok"
    assert (
        ex.execute("request_human_approval", {"summary": "go"})[0].output["data"]["status"]
        == "PENDING_APPROVAL"
    )
    assert ex.execute("rollback_candidate", {"reason": "changed my mind"})[0].status == "ok"
    assert "validation" not in ex.ctx.sandbox and "approval_requested" not in ex.ctx.sandbox
    assert ex.execute("request_human_approval", {"summary": "go"})[0].status == "rejected"


def test_proposal_tool_rejects_smuggled_fields(ctx):
    ex = executor(ctx)
    evidence, diagnosis = deterministic_diagnosis(ex.ctx.inp, ex.ctx.result)
    payload = propose(diagnosis, evidence, ex.ctx.state, ctx.record.version).model_dump(mode="json")
    for extra in ({"approved": True}, {"state": "EXECUTED"}, {"approved_by": "llm"}):
        trace, _ = ex.execute("create_remediation_proposal", {"proposal": {**payload, **extra}})
        assert trace.status == "rejected" and "proposal" not in ex.ctx.sandbox
    trace, _ = ex.execute(
        "create_remediation_proposal", {"proposal": {**payload, "action": "delete_production_db"}}
    )
    assert trace.status == "rejected"
