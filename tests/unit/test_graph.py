import httpx
import pytest

from aegis.agents import graph as graph_module
from aegis.agents.graph import MAX_DIAGNOSTIC_LOOPS, build_graph
from aegis.agents.tools import MAX_TOOL_CALLS
from aegis.observability.tracing import recent_spans
from aegis.schemas.diagnosis import INSUFFICIENT_EVIDENCE, Diagnosis
from tests.helpers import FakeOllama, good_plan, investigate, ranking_from_prompt

NODES = [
    "collect_incident",
    "run_diagnostics",
    "build_evidence",
    "rank_hypotheses",
    "validate_hypotheses",
    "retrieve_runbook",
    "plan_remediation",
    "validate_remediation",
    "policy_check",
    "request_approval",
]


def diagnostic_calls(tools):
    return [
        t
        for t in tools.traces
        if t.tool in [*graph_module.LLM_SELECTABLE_TOOLS, "retrieve_runbook"]
    ]


def test_graph_has_the_documented_nodes_and_one_bounded_loop():
    compiled = build_graph().get_graph()
    assert set(NODES) <= set(compiled.nodes)
    edges = {(e.source, e.target) for e in compiled.edges}
    assert ("validate_hypotheses", "run_diagnostics") in edges and (
        "validate_hypotheses",
        "retrieve_runbook",
    ) in edges
    assert ("request_approval", "__end__") in edges
    backward = [
        (s, t) for s, t in edges if s in NODES and t in NODES and NODES.index(t) < NODES.index(s)
    ]
    assert backward == [("validate_hypotheses", "run_diagnostics")]


@pytest.mark.parametrize(
    "fault,cause,action,outcome,status",
    [
        (
            "F04",
            "scaling_error",
            "fix_ingestion_scale",
            "REQUEST_HUMAN_APPROVAL",
            "awaiting_approval",
        ),
        (
            "F06",
            "duplicate_records",
            "enable_deduplication",
            "REQUEST_HUMAN_APPROVAL",
            "awaiting_approval",
        ),
        ("F08", "schema_change", "quarantine_batch", "REQUEST_HUMAN_APPROVAL", "awaiting_approval"),
        (
            "F10",
            "artifact_mismatch",
            "restore_model_version",
            "REQUEST_HUMAN_APPROVAL",
            "awaiting_approval",
        ),
        ("F05", "concept_drift", "retrain_candidate", "REJECT", "completed"),
        ("F09", "label_delay", "no_action", "DEFER", "completed"),
    ],
)
def test_investigation_without_llm_is_complete_and_deterministic(
    ctx, fault, cause, action, outcome, status
):
    final, tools, _, _ = investigate(ctx, ctx.scenario(fault, "high", seed=4))
    assert (
        final["llm_status"] == "disabled"
        and final["llm_model"] is None
        and not final.get("llm_calls")
    )
    assert [e["node"] for e in final["events"] if e["status"] == "ok"][:5] == NODES[:5]
    assert [e["node"] for e in final["events"]][-5:] == NODES[5:]
    assert final["diagnosis"]["hypotheses"][0]["root_cause"] == cause
    assert final["proposal"]["action"] == action and final["policy_decision"]["outcome"] == outcome
    assert final["status"] == status and final["approval_requested"] == (
        status == "awaiting_approval"
    )
    assert (
        all(t.status == "ok" for t in tools.traces)
        and len(diagnostic_calls(tools)) <= MAX_TOOL_CALLS
    )
    assert final["explanation"]["source"] == "deterministic"
    assert "nothing has been executed" in final["explanation"]["summary"]
    if status == "awaiting_approval":
        assert final["validation"]["passed"] and final["runbook_passages"]
        assert final["proposal"]["runbook_refs"]


def test_failed_validation_discards_the_candidate_and_never_requests_approval(ctx):
    final, tools, _, _ = investigate(ctx, ctx.scenario("F05", "high", seed=4))
    names = [t.tool for t in tools.traces]
    assert "rollback_candidate" in names and "request_human_approval" not in names
    assert final["validation"]["passed"] is False and not final["approval_requested"]
    assert "validation" not in tools.ctx.sandbox


def test_clean_batch_yields_insufficient_evidence_and_no_action(ctx):
    final, tools, _, _ = investigate(ctx, ctx.clean(seed=3))
    diagnosis = Diagnosis.model_validate(final["diagnosis"])
    assert diagnosis.conclusion == INSUFFICIENT_EVIDENCE and not diagnosis.hypotheses
    assert final["proposal"]["action"] == "investigate_further"
    assert (
        final["policy_decision"]["outcome"] == "INVESTIGATE_MORE" and final["status"] == "completed"
    )
    assert [t.tool for t in tools.traces] == ["create_remediation_proposal"]


def test_llm_path_plans_ranks_and_explains_with_three_bounded_requests(ctx):
    scenario = ctx.scenario("F04", "high", seed=4)
    feature = scenario.truth.affected_features[0]
    fake = FakeOllama(
        {
            "plan": good_plan(feature),
            "rank": ranking_from_prompt("scaling_error"),
            "explain": {
                "summary": "Monitoring found a constant-factor change in one feature, which is consistent "
                "with a unit error. A correction is proposed and is waiting for a human approver.",
                "operator_checks": ["Confirm the upstream unit with the feature owner."],
            },
        }
    )
    final, tools, _, _ = investigate(ctx, scenario, llm=fake.client())
    assert final["llm_status"] == "available" and final["llm_model"] == "qwen3:4b"
    assert [c["purpose"] for c in final["llm_calls"]] == ["plan", "rank", "explain"]
    assert all(
        c["ok"] and c["model"] == "qwen3:4b" and c["duration_ms"] >= 0 for c in final["llm_calls"]
    )
    by_llm = [t for t in tools.traces if t.requested_by == "llm"]
    assert [t.tool for t in by_llm] == ["inspect_feature_distribution", "get_model_metrics"]
    diagnosis = Diagnosis.model_validate(final["diagnosis"])
    assert diagnosis.engine == "llm+rules" and diagnosis.top.ranked_by == "llm"
    assert diagnosis.top.root_cause.value == "scaling_error"
    assert (
        final["llm_ranking_stats"]["accepted"] == 1
        and final["llm_ranking_stats"]["fabricated_evidence_ids"] == 0
    )
    assert final["explanation"]["source"] == "llm:qwen3:4b"
    assert final["explanation"][
        "deterministic_facts"
    ]  # the facts the text was written from are kept
    assert final["status"] == "awaiting_approval"


def test_prompts_contain_evidence_but_no_raw_rows_or_ground_truth(ctx):
    scenario = ctx.scenario("F04", "high", seed=4)
    fake = FakeOllama({"plan": good_plan(), "rank": ranking_from_prompt("scaling_error")})
    investigate(ctx, scenario, llm=fake.client(), use_llm_explain=False)
    prompts = "\n".join(m["content"] for r in fake.requests for m in r["messages"])
    assert "EV-" in prompts and "ANOMALY" in prompts
    for secret in (
        scenario.spec.scenario_id,
        "F04",
        "fault_id",
        "ground_truth",
        "GroundTruth",
    ):
        assert secret not in prompts
    assert (
        len(max((r["messages"][-1]["content"] for r in fake.requests), key=len)) < 9000
    )  # fits num_ctx 4096


def test_diagnostic_loop_fetches_missing_evidence_once_when_llm_plan_omits_it(ctx):
    """The LLM asks only for metrics; the rule engine still needs the shift-vs-scale pattern."""
    fake = FakeOllama({"plan": good_plan(), "rank": ranking_from_prompt("scaling_error")})
    final, tools, _, _ = investigate(
        ctx, ctx.scenario("F04", "high", seed=4), llm=fake.client(), use_llm_explain=False
    )
    runs = [e for e in final["events"] if e["node"] == "run_diagnostics"]
    assert len(runs) == 1 + MAX_DIAGNOSTIC_LOOPS and final["diagnostic_loops"] == 2
    assert [(t.tool, t.requested_by) for t in tools.traces[:2]] == [
        ("get_model_metrics", "llm"),
        ("inspect_feature_distribution", "rules"),
    ]
    assert [c["purpose"] for c in final["llm_calls"]].count(
        "plan"
    ) == 1  # the loop does not re-ask the LLM
    assert final["diagnosis"]["hypotheses"][0]["root_cause"] == "scaling_error"


def test_tool_budget_holds_even_when_the_llm_asks_for_more(ctx):
    greedy = {
        "calls": [
            {
                "tool": "get_data_quality_report",
                "feature": None,
                "other_version": None,
                "reason": "x",
            }
        ]
        * 6
    }
    fake = FakeOllama({"plan": greedy, "rank": ranking_from_prompt("scaling_error")})
    final, tools, _, _ = investigate(
        ctx, ctx.scenario("F04", "high", seed=4), llm=fake.client(), use_llm_explain=False
    )
    ok_diagnostics = [t for t in diagnostic_calls(tools) if t.status == "ok"]
    assert len(ok_diagnostics) == MAX_TOOL_CALLS
    assert ok_diagnostics[-1].tool == "retrieve_runbook"  # the reserved call was still available
    assert [t.error for t in tools.traces if t.status == "rejected"] and final["status"] in (
        "awaiting_approval",
        "completed",
    )


def test_cancellation_stops_work_and_preserves_partial_results(ctx):
    calls = {"n": 0}

    def cancelled():
        calls["n"] += 1
        return calls["n"] > 3  # cancel during the diagnostics phase

    final, tools, _, _ = investigate(
        ctx, ctx.scenario("F04", "high", seed=4), is_cancelled=cancelled
    )
    assert final["status"] == "cancelled"
    assert final["evidence"]  # what was collected before cancellation is kept
    assert final.get("proposal") is None and not final.get("approval_requested")
    assert "request_human_approval" not in [t.tool for t in tools.traces]
    assert [e["status"] for e in final["events"]].count("skipped_cancelled") >= 5


def test_node_failure_is_contained_and_reported(ctx, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("retriever index corrupted")

    scenario = ctx.scenario("F04", "high", seed=4)
    from aegis.rag.retriever import RunbookRetriever

    monkeypatch.setattr(RunbookRetriever, "search", boom)
    final, tools, _, _ = investigate(ctx, scenario)
    assert final["status"] == "awaiting_approval"  # the investigation still completes
    assert final["runbook_passages"] == [] and final["proposal"]["runbook_refs"] == []
    assert any(t.tool == "retrieve_runbook" and t.status == "error" for t in tools.traces)


def test_investigation_emits_nested_trace_spans(ctx):
    recent_spans.clear()
    fake = FakeOllama({"plan": good_plan(), "rank": ranking_from_prompt("scaling_error")})
    final, tools, _, _ = investigate(
        ctx, ctx.scenario("F04", "high", seed=4), llm=fake.client(), use_llm_explain=False
    )
    root = recent_spans.snapshot("investigation.run")[-1]
    in_trace = recent_spans.snapshot(trace_id=root["trace_id"])
    names = [s["name"] for s in in_trace]
    assert {f"investigation.{n}" for n in NODES} <= set(names)
    assert names.count("tool.execute") == len(tools.traces) and names.count("llm.chat") == 2
    tool_span = next(s for s in in_trace if s["name"] == "tool.execute")
    assert tool_span["attributes"]["tool.status"] == "ok" and tool_span["parent_span_id"]
    assert root["attributes"]["investigation.status"] == final["status"] and root["duration_ms"] > 0


def test_model_is_loaded_once_before_the_first_llm_request(ctx):
    scenario = ctx.scenario("F04", "high", seed=4)
    fake = FakeOllama(
        {
            "plan": good_plan(scenario.truth.affected_features[0]),
            "rank": ranking_from_prompt("scaling_error"),
            "explain": {
                "summary": "Monitoring found a constant-factor change in one feature, which is "
                "consistent with a unit error. A correction is proposed and awaits an approver.",
                "operator_checks": ["Confirm the upstream unit with the feature owner."],
            },
        }
    )
    final, _, _, _ = investigate(ctx, scenario, llm=fake.client())
    assert final["llm_status"] == "available"
    assert [load["model"] for load in fake.loads] == ["qwen3:4b"]
    assert "prompt" not in fake.loads[0]  # a pure load: nothing is generated
    assert fake.loads[0]["options"]["num_ctx"] == fake.client().settings.ollama_num_ctx


def test_model_that_cannot_be_loaded_degrades_to_the_deterministic_path(ctx):
    scenario = ctx.scenario("F04", "high", seed=4)
    fake = FakeOllama(load_error=httpx.ReadTimeout("model load timed out"))
    final, _, _, _ = investigate(ctx, scenario, llm=fake.client())
    assert final["status"] in {"completed", "awaiting_approval"}
    assert final["llm_status"] == "unavailable"
    assert fake.requests == []  # no chat request is attempted against a model that did not load
    assert any("did not load" in e for e in final["errors"])
    assert final["diagnosis"]["hypotheses"][0]["root_cause"] == "scaling_error"
    assert final["diagnosis"]["hypotheses"][0]["ranked_by"] == "rules"
