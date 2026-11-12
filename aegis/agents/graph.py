"""The bounded investigation workflow (one LangGraph state machine).

    collect_incident -> run_diagnostics -> build_evidence -> rank_hypotheses -> validate_hypotheses
          ^                                                                           |
          |                 (at most one loop, only while tool budget remains)        |
          +---------------------------------------------------------------------------+
                                                                                      v
    retrieve_runbook -> plan_remediation -> validate_remediation -> policy_check -> request_approval -> END

Bounds: <= 5 diagnostic tool calls (enforced by ToolExecutor), <= 1 diagnostic
loop, <= 3 LLM requests (plan, rank, explain; each with at most one retry), a LangGraph recursion limit, and per-call timeouts.
Every node records an event; a node that raises records the error and the
workflow continues with whatever partial results exist. Without a usable LLM
every node takes its deterministic branch.
"""

from __future__ import annotations

import operator
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from aegis.agents import claims as claim_rules
from aegis.agents import llm_tasks
from aegis.agents.ollama import OllamaClient
from aegis.agents.tools import LLM_SELECTABLE_TOOLS, ToolExecutor
from aegis.diagnosis.evidence import build_evidence
from aegis.diagnosis.pipeline import features_worth_inspecting
from aegis.diagnosis.rules import rank_hypotheses
from aegis.observability.logging import get_logger
from aegis.observability.metrics import INVESTIGATION_SECONDS
from aegis.observability.tracing import span
from aegis.remediation.policy import PolicyEngine
from aegis.remediation.proposals import propose
from aegis.schemas.diagnosis import Diagnosis, Evidence
from aegis.schemas.remediation import PolicyOutcome, RemediationProposal

log = get_logger(__name__)
RUNBOOK_RESERVE = 1  # one of the five tool calls is kept for runbook retrieval
MAX_DIAGNOSTIC_LOOPS = 1
RECURSION_LIMIT = 30


class InvestigationState(TypedDict, total=False):
    investigation_id: str
    incident_id: str
    status: str  # running | awaiting_approval | completed | cancelled
    llm_status: str  # available | unavailable | disabled | degraded
    llm_model: str | None
    llm_calls: Annotated[list[dict], operator.add]
    evidence: list[dict]
    pending_evidence: list[dict]
    diagnostic_loops: int
    rule_diagnosis: dict | None
    llm_ranking_stats: dict | None
    diagnosis: dict | None
    needs_more_evidence: bool
    runbook_passages: list[dict]
    proposal: dict | None
    validation: dict | None
    policy_pre_decision: dict | None  # the decision taken before sandbox validation
    policy_decision: dict | None
    approval_requested: bool
    explanation: dict | None
    events: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]


@dataclass
class AgentRuntime:
    """Non-serialisable collaborators. Passed through ``config['configurable']``."""

    tools: ToolExecutor
    policy: PolicyEngine
    llm: OllamaClient | None = None
    use_llm_plan: bool = True
    use_llm_rank: bool = True
    use_llm_explain: bool = True
    use_rules: bool = True  # False only in the "LLM diagnosis without rules" ablation
    use_runbooks: bool = True
    is_cancelled: Callable[[], bool] = lambda: False
    declared_version: str = ""
    scratch: dict[str, Any] = field(default_factory=dict)


def _rt(config: RunnableConfig) -> AgentRuntime:
    return config["configurable"]["runtime"]


def _evidence(state: InvestigationState) -> list[Evidence]:
    return [Evidence.model_validate(e) for e in state.get("evidence", [])]


def node(name: str):
    """Wrap a node: cancellation check, span, timing, and error containment."""

    def decorate(fn):
        def wrapped(state: InvestigationState, config: RunnableConfig) -> dict:
            runtime = _rt(config)
            if state.get("status") == "cancelled" or runtime.is_cancelled():
                return {
                    "status": "cancelled",
                    "events": [{"node": name, "status": "skipped_cancelled", "duration_ms": 0.0}],
                }
            started = time.perf_counter()
            with span(f"investigation.{name}", investigation_id=state.get("investigation_id", "")):
                try:
                    update = fn(state, runtime) or {}
                    status = "ok"
                except Exception as exc:  # keep partial results; never lose the investigation
                    log.error("investigation_node_failed", node=name, error=repr(exc)[:300])
                    update, status = (
                        {"errors": [f"{name}: {type(exc).__name__}: {str(exc)[:200]}"]},
                        "error",
                    )
            event = {
                "node": name,
                "status": status,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            }
            return {**update, "events": [*update.get("events", []), event]}

        wrapped.__name__ = name
        return wrapped

    return decorate


def _llm_usable(state: InvestigationState, runtime: AgentRuntime) -> bool:
    return runtime.llm is not None and state.get("llm_status") in ("available", "degraded")


def _degrade(result) -> str:
    return "unavailable" if result.error in ("unavailable", "timeout") else "degraded"


# --------------------------------------------------------------------- nodes
@node("collect_incident")
def collect_incident(state, runtime):
    evidence = build_evidence(runtime.tools.ctx.result.report)
    update: dict = {
        "status": "running",
        "evidence": [e.model_dump(mode="json") for e in evidence],
        "diagnostic_loops": 0,
        "pending_evidence": [],
    }
    if runtime.llm is None:
        return {**update, "llm_status": "disabled", "llm_model": None}
    health = runtime.llm.health()
    update["llm_status"] = (
        "available" if health.available else ("disabled" if not health.enabled else "unavailable")
    )
    update["llm_model"] = health.active_model
    if not health.available and health.enabled:
        update["errors"] = [
            f"LLM investigation unavailable: {health.detail or 'Ollama unreachable'}; "
            "continuing with deterministic diagnosis"
        ]
    elif health.available:
        # Load the model now, on its own time budget, so a cold start is not mistaken for a
        # slow inference call by the per-request timeout.
        loaded, seconds, detail = runtime.llm.warm(health.active_model)
        if not loaded:
            update["llm_status"] = "unavailable"
            update["errors"] = [
                f"LLM investigation unavailable: {detail} (after {seconds:.0f}s); "
                "continuing with deterministic diagnosis"
            ]
    return update


def _fallback_plan(
    evidence: list[Evidence], already: set[tuple], budget: int
) -> list[tuple[str, dict]]:
    """Deterministic tool plan: the calls that most sharply separate candidate causes."""

    def has(*checks):
        return any(
            e.data.get("check") in checks and e.data.get("polarity") == "anomaly" for e in evidence
        )

    plan: list[tuple[str, dict]] = [
        ("inspect_feature_distribution", {"feature": f})
        for f in features_worth_inspecting(evidence, 2)
    ]
    if has("artifact_hash"):
        plan.append(("compare_model_versions", {}))
    if has(
        "f1_drop", "roc_auc_drop", "accuracy_drop", "adwin", "page_hinkley", "label_availability"
    ):
        plan.append(("get_model_metrics", {"include_slices": True}))
    if has("numeric_drift", "categorical_drift"):
        plan.append(("get_data_drift_report", {"top_k": 5}))
    if has("missing_rate", "duplicate_rate", "schema_missing_columns", "schema_type_mismatch"):
        plan.append(("get_data_quality_report", {}))
    fresh = [(t, a) for t, a in plan if (t, tuple(sorted(a.items()))) not in already]
    return fresh[:budget]


@node("run_diagnostics")
def run_diagnostics(state, runtime):
    evidence = _evidence(state)
    tools = runtime.tools
    budget = max(tools.budget_left - RUNBOOK_RESERVE, 0)
    already = {
        (t.tool, tuple(sorted((t.validated_args or {}).items())))
        for t in tools.traces
        if t.status == "ok"
    }
    update: dict = {"diagnostic_loops": state.get("diagnostic_loops", 0)}
    calls: list[tuple[str, dict]] = []
    requested_by = "rules"
    first_pass = state.get("diagnostic_loops", 0) == 0
    if first_pass and runtime.use_llm_plan and _llm_usable(state, runtime) and budget > 0:
        result = llm_tasks.request_plan(
            runtime.llm, evidence, runtime.tools.ctx.inp.spec.features, budget
        )
        update["llm_calls"] = [result.audit()]
        if result.ok:
            calls, requested_by = llm_tasks.plan_to_calls(result.parsed), "llm"
        else:
            update["llm_status"] = _degrade(result)
            update["errors"] = [
                f"LLM tool planning failed ({result.error}); using the deterministic plan"
            ]
    if not calls:
        calls, requested_by = _fallback_plan(evidence, already, budget), "rules"

    new_evidence: list[dict] = []
    for name, args in calls:
        if runtime.is_cancelled():
            break
        if tools.budget_left <= RUNBOOK_RESERVE:
            tools.reject(
                name,
                args,
                requested_by,
                f"diagnostic budget exhausted ({RUNBOOK_RESERVE} call reserved for runbook retrieval)",
            )
            continue
        _, produced = tools.execute(
            name,
            args,
            requested_by=requested_by,
            allowed=LLM_SELECTABLE_TOOLS if requested_by == "llm" else None,
        )
        new_evidence += [e.model_dump(mode="json") for e in produced]
    update["pending_evidence"] = new_evidence
    update["diagnostic_loops"] = state.get("diagnostic_loops", 0) + 1
    return update


@node("build_evidence")
def build_evidence_node(state, runtime):
    merged = {e["evidence_id"]: e for e in state.get("evidence", [])}
    for item in state.get("pending_evidence", []):
        merged[item["evidence_id"]] = item
    return {"evidence": list(merged.values()), "pending_evidence": []}


@node("rank_hypotheses")
def rank_hypotheses_node(state, runtime):
    evidence = _evidence(state)
    rule_diagnosis = rank_hypotheses(evidence) if runtime.use_rules else None
    update: dict = {
        "rule_diagnosis": rule_diagnosis.model_dump(mode="json") if rule_diagnosis else None,
        "diagnosis": rule_diagnosis.model_dump(mode="json") if rule_diagnosis else None,
    }
    runtime.scratch.pop("llm_ranking", None)
    # If a cheap, well-defined piece of evidence is still missing, fetch it first (one bounded
    # loop) and let the LLM rank once, on the complete evidence, instead of twice.
    inspected = {e.feature for e in evidence if e.data.get("check") == "feature_pattern"}
    wanted = [f for f in features_worth_inspecting(evidence, 2) if f not in inspected]
    update["needs_more_evidence"] = bool(
        wanted
        and state.get("diagnostic_loops", 0) <= MAX_DIAGNOSTIC_LOOPS
        and runtime.tools.budget_left > RUNBOOK_RESERVE
    )
    if update["needs_more_evidence"]:
        return update
    if runtime.use_llm_rank and _llm_usable(state, runtime):
        result = llm_tasks.request_ranking(runtime.llm, evidence, rule_diagnosis)
        update["llm_calls"] = [result.audit()]
        if result.ok:
            runtime.scratch["llm_ranking"] = result.parsed
        else:
            update["llm_status"] = _degrade(result)
            update["errors"] = [
                f"LLM ranking failed ({result.error}); keeping the deterministic ranking"
            ]
    return update


@node("validate_hypotheses")
def validate_hypotheses(state, runtime):
    evidence = _evidence(state)
    rule_diagnosis = (
        Diagnosis.model_validate(state["rule_diagnosis"]) if state.get("rule_diagnosis") else None
    )
    update: dict = {}
    final = rule_diagnosis
    ranking = runtime.scratch.pop("llm_ranking", None)
    if ranking is not None:
        validated, stats = llm_tasks.validate_ranking(ranking, evidence, rule_diagnosis)
        update["llm_ranking_stats"] = stats
        if validated is not None:
            final = validated
        else:
            update["errors"] = [
                "LLM ranking contained no hypothesis backed by evidence; "
                "keeping the deterministic ranking"
            ]
    if final is None:
        final = rank_hypotheses([])  # "insufficient evidence": no rules and no valid LLM ranking
    update["diagnosis"] = final.model_dump(mode="json")
    return update


def _after_validation(state: InvestigationState) -> str:
    if state.get("status") == "cancelled":
        return "retrieve_runbook"  # falls straight through the remaining (skipping) nodes
    return "run_diagnostics" if state.get("needs_more_evidence") else "retrieve_runbook"


@node("retrieve_runbook")
def retrieve_runbook(state, runtime):
    diagnosis = Diagnosis.model_validate(state["diagnosis"])
    if not runtime.use_runbooks or diagnosis.top is None:
        return {"runbook_passages": []}
    top = diagnosis.top
    query = f"{top.root_cause.value.replace('_', ' ')} {llm_tasks.CAUSE_GLOSSARY[top.root_cause]} remediation"
    trace, _ = runtime.tools.execute("retrieve_runbook", {"query": query[:300], "top_k": 3})
    passages = (
        (trace.output or {}).get("data", {}).get("passages", []) if trace.status == "ok" else []
    )
    return {"runbook_passages": passages}


@node("plan_remediation")
def plan_remediation(state, runtime):
    diagnosis = Diagnosis.model_validate(state["diagnosis"])
    proposal = propose(
        diagnosis, _evidence(state), runtime.tools.ctx.state, runtime.declared_version
    )
    proposal.runbook_refs = [
        p["passage_id"] for p in state.get("runbook_passages", []) if not p.get("suspicious")
    ]
    trace, _ = runtime.tools.execute(
        "create_remediation_proposal", {"proposal": proposal.model_dump(mode="json")}
    )
    if trace.status != "ok":
        raise RuntimeError(f"proposal was rejected by its own schema: {trace.error}")
    return {"proposal": proposal.model_dump(mode="json")}


def _policy(state, runtime, validation=None):
    diagnosis = Diagnosis.model_validate(state["diagnosis"])
    rules = (
        Diagnosis.model_validate(state["rule_diagnosis"]) if state.get("rule_diagnosis") else None
    )
    proposal = RemediationProposal.model_validate(state["proposal"])
    return runtime.policy.evaluate(
        proposal, diagnosis, _evidence(state), rule_diagnosis=rules, validation=validation
    )


@node("validate_remediation")
def validate_remediation(state, runtime):
    if not state.get("proposal"):
        return {"validation": None}
    pre = _policy(state, runtime)
    recorded = {"policy_pre_decision": pre.model_dump(mode="json")}
    if pre.outcome is not PolicyOutcome.PROPOSE_ACTION:
        # policy does not allow this action to reach the sandbox
        return {"validation": None, **recorded}
    trace, _ = runtime.tools.execute("run_candidate_validation", {})
    validation = runtime.tools.ctx.sandbox.get("validation")
    if trace.status != "ok" or validation is None:
        raise RuntimeError(f"sandbox validation did not run: {trace.error}")
    if not validation.passed:
        runtime.tools.execute(
            "rollback_candidate", {"reason": f"validation failed: {validation.summary}"[:300]}
        )
    return {"validation": validation.model_dump(mode="json"), **recorded}


@node("policy_check")
def policy_check(state, runtime):
    if not state.get("proposal"):
        return {"policy_decision": None}
    from aegis.schemas.remediation import ValidationResult

    validation = (
        ValidationResult.model_validate(state["validation"]) if state.get("validation") else None
    )
    return {"policy_decision": _policy(state, runtime, validation).model_dump(mode="json")}


@node("request_approval")
def request_approval(state, runtime):
    decision = state.get("policy_decision") or {}
    diagnosis = Diagnosis.model_validate(state["diagnosis"])
    proposal = state.get("proposal") or {}
    requested = False
    if decision.get("outcome") == PolicyOutcome.REQUEST_HUMAN_APPROVAL.value:
        trace, _ = runtime.tools.execute(
            "request_human_approval",
            {"summary": f"{proposal.get('action')}: {proposal.get('rationale', '')}"[:600]},
        )
        requested = trace.status == "ok"
    facts = "\n".join(
        [
            f"Conclusion: {diagnosis.conclusion}",
            *(
                f"Hypothesis {i + 1}: {h.root_cause.value} (evidence fit {h.confidence:.2f})"
                for i, h in enumerate(diagnosis.hypotheses[:3])
            ),
            f"Proposed action: {proposal.get('action', 'none')} - {proposal.get('rationale', '')}",
            f"Sandbox validation: {(state.get('validation') or {}).get('summary', 'not run')}",
            f"Policy outcome: {decision.get('outcome', 'none')} ({'; '.join(decision.get('reasons', []))[:300]})",
            "Status: "
            + (
                "WAITING for a human approver; nothing has been executed"
                if requested
                else "no action is pending; nothing has been executed"
            ),
        ]
    )
    evidence = _evidence(state)
    rule_diagnosis = (
        Diagnosis.model_validate(state["rule_diagnosis"]) if state.get("rule_diagnosis") else None
    )
    intervention = claim_rules.Intervention.from_validation(
        state.get("proposal"), state.get("validation")
    )
    update: dict = {
        "approval_requested": requested,
        "status": "awaiting_approval" if requested else "completed",
        "explanation": {"source": "deterministic", "summary": facts, "operator_checks": []},
    }
    if runtime.use_llm_explain and _llm_usable(state, runtime):
        safe_passages = [p for p in state.get("runbook_passages", []) if not p.get("suspicious")]
        result = llm_tasks.request_explanation(runtime.llm, facts[:900], safe_passages)
        update["llm_calls"] = [result.audit()]
        if result.ok:
            review = llm_tasks.review_explanation(
                result.parsed, evidence, rule_diagnosis, intervention
            )
            claim_validation = {
                "counters": review.counters,
                "claims": [c.model_dump(mode="json") for c in review.claims],
                "escalated_claim_ids": review.escalated_claim_ids,
                "raw_summary": review.raw_summary,
                "accepted": review.ok,
                "reason": review.reason,
            }
            if review.ok:
                update["explanation"] = {
                    "source": f"llm:{result.model}",
                    "summary": review.summary,
                    "operator_checks": review.operator_checks,
                    "deterministic_facts": facts,
                }
                if review.escalated_claim_ids:
                    update["explanation"]["review_flags"] = [
                        "Causal wording in the model's summary was replaced: the evidence shows "
                        "how the data changed but no intervention confirms the cause."
                    ]
            else:
                update["errors"] = [f"LLM explanation discarded: {review.reason}"]
            update["explanation"]["claim_validation"] = claim_validation
        else:
            update["llm_status"] = _degrade(result)
    # Re-scan what actually leaves the investigation. Non-zero here means the pipeline let
    # something through that its own validator recognises: a defect, reported not hidden.
    update["explanation"]["output_audit"] = claim_rules.audit_final_output(
        diagnosis=diagnosis,
        explanation=update["explanation"],
        evidence=evidence,
        rule_diagnosis=rule_diagnosis,
        intervention=intervention,
    )
    return update


def build_graph():
    graph = StateGraph(InvestigationState)
    for name, fn in [
        ("collect_incident", collect_incident),
        ("run_diagnostics", run_diagnostics),
        ("build_evidence", build_evidence_node),
        ("rank_hypotheses", rank_hypotheses_node),
        ("validate_hypotheses", validate_hypotheses),
        ("retrieve_runbook", retrieve_runbook),
        ("plan_remediation", plan_remediation),
        ("validate_remediation", validate_remediation),
        ("policy_check", policy_check),
        ("request_approval", request_approval),
    ]:
        graph.add_node(name, fn)
    graph.add_edge(START, "collect_incident")
    graph.add_edge("collect_incident", "run_diagnostics")
    graph.add_edge("run_diagnostics", "build_evidence")
    graph.add_edge("build_evidence", "rank_hypotheses")
    graph.add_edge("rank_hypotheses", "validate_hypotheses")
    graph.add_conditional_edges(
        "validate_hypotheses",
        _after_validation,
        {"run_diagnostics": "run_diagnostics", "retrieve_runbook": "retrieve_runbook"},
    )
    graph.add_edge("retrieve_runbook", "plan_remediation")
    graph.add_edge("plan_remediation", "validate_remediation")
    graph.add_edge("validate_remediation", "policy_check")
    graph.add_edge("policy_check", "request_approval")
    graph.add_edge("request_approval", END)
    return graph.compile()


_GRAPH = None


def run_investigation(
    runtime: AgentRuntime, investigation_id: str, incident_id: str
) -> InvestigationState:
    """Run one investigation to completion and return the final state (always; never raises
    for LLM, tool or node failures)."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    started = time.perf_counter()
    with span(
        "investigation.run", investigation_id=investigation_id, incident_id=incident_id
    ) as current:
        final = _GRAPH.invoke(
            {"investigation_id": investigation_id, "incident_id": incident_id, "status": "running"},
            config={"configurable": {"runtime": runtime}, "recursion_limit": RECURSION_LIMIT},
        )
        if runtime.is_cancelled() and final.get("status") != "cancelled":
            final["status"] = "cancelled"
        current.set_attribute("investigation.status", final.get("status", ""))
        current.set_attribute("investigation.llm_status", final.get("llm_status", ""))
        current.set_attribute("investigation.tool_calls", len(runtime.tools.traces))
    final["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)  # type: ignore[typeddict-unknown-key]
    INVESTIGATION_SECONDS.labels(final.get("llm_status", "disabled")).observe(
        final["duration_ms"] / 1000
    )  # type: ignore[typeddict-item]
    return final
