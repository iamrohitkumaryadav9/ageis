"""AEGIS-IR scenario execution.

Four configurations are run on *identical* scenarios:

  A  monitoring only            alerts -> cause read straight off the alert list
  B  monitoring + rules         deterministic diagnostics + rule-based ranking + policy + validation
  C  monitoring + local LLM     the LLM ranks causes from monitoring evidence alone (no tools, rules, RAG)
  D  full AEGIS                 LangGraph agent: LLM tool planning + rules + LLM re-ranking + RAG + policy + validation

Detection is the same monitoring engine in all four (the ablation varies what
happens *after* detection), so detection metrics are reported once.

Ground truth is read by the scorer only. Systems under test receive a
``MonitoringInput`` built from the public half of each scenario.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from aegis.agents import claims as claim_rules
from aegis.agents import llm_tasks
from aegis.agents.graph import AgentRuntime, run_investigation
from aegis.agents.ollama import OllamaClient
from aegis.agents.tools import LLM_SELECTABLE_TOOLS, TOOLS, ToolContext, ToolExecutor
from aegis.config import Settings
from aegis.diagnosis.evidence import build_evidence
from aegis.diagnosis.naive import naive_diagnosis
from aegis.faults.scenarios import Scenario, build_scenario
from aegis.ingestion.contracts import DataContract, derive_contract
from aegis.ingestion.datasets import DATASETS, DatasetSpec
from aegis.ingestion.splits import read_split
from aegis.models.registry import ModelRegistry
from aegis.monitoring.engine import MonitoringInput, MonitoringResult, run_monitoring
from aegis.rag.retriever import RunbookRetriever
from aegis.remediation.executor import state_to_execute
from aegis.remediation.policy import PolicyEngine
from aegis.remediation.proposals import propose
from aegis.remediation.reconciliation import LifecycleEvent, Stage
from aegis.schemas.deployment import DeploymentState
from aegis.schemas.diagnosis import Diagnosis, Evidence
from aegis.schemas.findings import Severity
from aegis.schemas.remediation import (
    STATE_CHANGING,
    ActionType,
    RemediationProposal,
    ValidationResult,
)

#: Faults for which AEGIS defines an automatable, validatable remediation.
SUPPORTED_FAULTS = {
    "F03": ActionType.QUARANTINE_BATCH,
    "F04": ActionType.FIX_INGESTION_SCALE,
    "F06": ActionType.ENABLE_DEDUPLICATION,
    "F07": ActionType.QUARANTINE_BATCH,
    "F08": ActionType.QUARANTINE_BATCH,
    "F10": ActionType.RESTORE_MODEL_VERSION,
}
CONFIGS = ("A", "B", "C", "D")


@dataclass
class DatasetContext:
    spec: DatasetSpec
    reference: pd.DataFrame
    production: pd.DataFrame
    contract: DataContract
    version: str


@dataclass
class BenchmarkEnv:
    settings: Settings
    registry: ModelRegistry
    retriever: RunbookRetriever
    workdir: Path
    datasets: dict[str, DatasetContext] = field(default_factory=dict)
    llm: OllamaClient | None = None
    policy: PolicyEngine = field(default_factory=PolicyEngine)
    #: Whether configuration D also asks the LLM for the operator explanation. Historical run
    #: ``main`` had this off, which is why it has no claim metrics for explanations.
    llm_explain: bool = True

    @classmethod
    def create(
        cls, settings: Settings, registry_root: Path, workdir: Path, llm: OllamaClient | None
    ) -> BenchmarkEnv:
        registry = ModelRegistry(registry_root)
        env = cls(settings, registry, RunbookRetriever(settings.runbook_dir), workdir, llm=llm)
        for name, spec in DATASETS.items():
            reference = read_split(name, "validation", settings.processed_dir)
            env.datasets[name] = DatasetContext(
                spec,
                reference,
                read_split(name, "production", settings.processed_dir),
                derive_contract(spec, reference),
                registry.baseline(name).version,
            )
        return env

    def scenario(self, cell: dict) -> Scenario:
        ctx = self.datasets[cell["dataset"]]
        return build_scenario(
            ctx.spec,
            ctx.production,
            self.registry,
            ctx.version,
            fault_id=cell["fault_id"],
            severity=cell["severity"],
            seed=cell["seed"],
            n_rows=cell["n_rows"],
            workdir=self.workdir,
        )

    def monitoring_input(self, scenario: Scenario) -> MonitoringInput:
        ctx = self.datasets[scenario.spec.dataset]
        return MonitoringInput(
            ctx.spec,
            ctx.contract,
            ctx.reference,
            scenario.current,
            self.registry,
            ctx.version,
            scenario.artifact_path,
        )


def _diagnosis_row(
    diagnosis: Diagnosis | None, truth_cause: str | None, evidence: list[Evidence]
) -> dict[str, Any]:
    """Score one diagnosis against the ground truth (scorer side)."""
    hypotheses = diagnosis.hypotheses if diagnosis else []
    ranked = [h.root_cause.value for h in hypotheses]
    known = {e.evidence_id: e for e in evidence}
    cited = [i for h in hypotheses[:3] for i in h.supporting_evidence_ids]
    anomalies = {i for i, e in known.items() if e.data.get("polarity") == "anomaly"}
    top = hypotheses[0] if hypotheses else None
    top_support = set(top.supporting_evidence_ids) if top else set()
    return {
        "ranked": ranked[:3],
        "top1": bool(truth_cause) and ranked[:1] == [truth_cause],
        "recall_at_3": bool(truth_cause) and truth_cause in ranked[:3],
        "sufficient_evidence": bool(diagnosis and diagnosis.sufficient_evidence),
        "top_confidence": top.confidence if top else None,
        # share of anomaly evidence the leading hypothesis accounts for
        "evidence_coverage": len(top_support & anomalies) / len(anomalies)
        if anomalies and top
        else 0.0,
        # hypotheses (top 3) citing no existing anomaly evidence
        "unsupported_claims": sum(
            not any(i in anomalies for i in h.supporting_evidence_ids) for h in hypotheses[:3]
        ),
        "claims": len(hypotheses[:3]),
        "cited_unknown_evidence": sum(i not in known for i in cited),
        "contradictions_cited": sum(len(h.contradicting_evidence_ids) for h in hypotheses[:3]),
    }


def _lifecycle_events(
    *,
    scenario_id: str,
    config: str,
    proposal: RemediationProposal,
    pre_decision: dict | None,
    validation: ValidationResult | None,
    post_decision: dict | None,
    approval_requested: bool,
    execution: dict | None,
) -> list[dict[str, Any]]:
    """The canonical lifecycle of one benchmark proposal (see ``aegis.remediation.reconciliation``).

    Approval, execution and rollback are *simulated* in the benchmark (no human, API or
    database); those events say so. Ids are derived from the scenario and configuration, so a
    re-run produces the same ids.
    """
    key = f"{scenario_id.removeprefix('SCN-')}-{config}"
    common = {
        "proposal_id": f"PRP-{key}",
        "incident_id": f"INC-{key}",
        "scenario_id": scenario_id,
        "investigation_id": f"INV-{key}",
        "action": proposal.action.value,
    }
    events: list[dict[str, Any]] = []

    def emit(stage: Stage, outcome: str, **fields: Any) -> None:
        events.append(
            LifecycleEvent(
                event_id=f"EVT-{key}-{len(events)}",
                stage=stage,
                sequence=len(events),
                outcome=outcome,
                **common,
                **fields,
            ).model_dump(mode="json")
        )

    kind = (
        "actionable"
        if proposal.action in STATE_CHANGING
        else ("no_action" if proposal.action is ActionType.NO_ACTION else "investigate_further")
    )
    emit(Stage.PROPOSED, kind, detail={"risk_level": proposal.risk_level.value})
    validated = validation is not None
    first = pre_decision if pre_decision is not None else (None if validated else post_decision)
    if first is not None:
        emit(
            Stage.POLICY_DECIDED,
            first["outcome"],
            policy_decision_id=f"POL-{key}-pre",
            phase="pre_validation",
            detail={
                "score": first.get("score"),
                "reasons": first.get("reasons", [])[:3],
                "unmet_requirements": first.get("unmet_requirements", [])[:3],
            },
        )
    if validation is not None:
        emit(
            Stage.VALIDATION,
            "passed" if validation.passed else "failed",
            validation_id=f"VAL-{key}",
            detail={
                "failed_checks": [c.name for c in validation.checks if not c.passed],
                "regression_failed": validation.regression_failed,
            },
        )
        if post_decision is not None:
            emit(
                Stage.POLICY_DECIDED,
                post_decision["outcome"],
                policy_decision_id=f"POL-{key}-post",
                validation_id=f"VAL-{key}",
                phase="post_validation",
                detail={
                    "score": post_decision.get("score"),
                    "reasons": post_decision.get("reasons", [])[:3],
                },
            )
    if approval_requested:
        emit(
            Stage.APPROVAL_REQUESTED,
            "requested",
            approval_id=f"APR-{key}",
            validation_id=f"VAL-{key}" if validated else None,
        )
    if execution is not None:
        emit(Stage.APPROVAL_DECIDED, "approved", approval_id=f"APR-{key}", simulated=True)
        emit(
            Stage.EXECUTED,
            "succeeded" if execution["executed"] else "failed",
            execution_id=f"EXE-{key}",
            approval_id=f"APR-{key}",
            simulated=True,
            detail={"error": execution.get("error")} if execution.get("error") else {},
        )
        if execution["executed"]:
            emit(
                Stage.VERIFIED,
                "resolved" if execution["resolved"] else "unresolved",
                execution_id=f"EXE-{key}",
                simulated=True,
            )
            emit(
                Stage.ROLLED_BACK,
                "succeeded" if execution["rolled_back"] else "failed",
                rollback_id=f"RBK-{key}",
                execution_id=f"EXE-{key}",
                simulated=True,
            )
    return events


def _remediation_row(
    env: BenchmarkEnv,
    inp: MonitoringInput,
    result: MonitoringResult,
    diagnosis: Diagnosis,
    rule_diagnosis: Diagnosis | None,
    evidence: list[Evidence],
    state: DeploymentState,
    precomputed: dict | None = None,
    scenario_id: str = "SCN-unknown",
    config: str = "?",
) -> dict[str, Any]:
    """Propose -> policy -> sandbox validation -> (simulated approval) execute -> verify -> rollback."""
    from aegis.remediation.validation import validate_proposal

    if precomputed is not None:
        proposal = RemediationProposal.model_validate(precomputed["proposal"])
        validation = (
            ValidationResult.model_validate(precomputed["validation"])
            if precomputed.get("validation")
            else None
        )
        pre_decision = precomputed.get("policy_pre_decision")
        post_decision = precomputed.get("policy_decision")
        outcome = (post_decision or {}).get("outcome")
    else:
        proposal = propose(diagnosis, evidence, state, inp.model_version)
        decision = env.policy.evaluate(proposal, diagnosis, evidence, rule_diagnosis=rule_diagnosis)
        pre_decision = decision.model_dump(mode="json")
        validation = None
        if decision.outcome.value == "PROPOSE_ACTION":
            validation = validate_proposal(proposal, inp, result, state)
            decision = env.policy.evaluate(
                proposal, diagnosis, evidence, rule_diagnosis=rule_diagnosis, validation=validation
            )
        post_decision = decision.model_dump(mode="json")
        outcome = decision.outcome.value
    row: dict[str, Any] = {
        "action": proposal.action.value,
        "policy_outcome": outcome,
        "policy_reasons": (post_decision or {}).get("reasons", [])[:3],
        "validated": validation.passed if validation else None,
        "failed_checks": [c.name for c in validation.checks if not c.passed]
        if validation
        else None,
        "regression_failed": validation.regression_failed if validation else None,
        "validation_ms": validation.duration_ms if validation else None,
        "approval_requested": outcome == "REQUEST_HUMAN_APPROVAL",
        "executed_resolves_incident": None,
        "rollback_restores_state": None,
        "invalid_promotion": False,
        "unauthorized_action": False,
    }
    # Safety invariants, checked on every scenario:
    #  - an approval request without a passing validation would be an invalid promotion path
    #  - a state-changing action reaching "request approval" for a non-state-changing proposal is a bug
    if row["approval_requested"] and not (validation and validation.passed):
        row["invalid_promotion"] = True
    if row["approval_requested"] and proposal.action not in STATE_CHANGING:
        row["unauthorized_action"] = True
    execution: dict[str, Any] | None = None
    if row["approval_requested"] and validation and validation.passed:
        # Simulate the human approving, then check execution and rollback really work.
        execution = {"executed": False, "resolved": False, "rolled_back": False, "error": None}
        try:
            new_state = state_to_execute(state, proposal, validation)
        except Exception as exc:  # an execution that cannot be built is a failed execution
            execution["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            new_state = None
        if new_state is not None:
            execution["executed"] = True
            if proposal.action is ActionType.QUARANTINE_BATCH:
                execution["resolved"] = bool(new_state.quarantined)
            else:
                after = run_monitoring(
                    MonitoringInput(
                        inp.spec,
                        inp.contract,
                        inp.reference,
                        inp.current,
                        inp.registry,
                        new_state.model_version,
                        None,
                        new_state.ingestion,
                    )
                )
                execution["resolved"] = not after.report.incident_detected
            restored = DeploymentState.model_validate(
                state.model_dump()
            )  # rollback = re-activate the base revision
            before = run_monitoring(
                MonitoringInput(
                    inp.spec,
                    inp.contract,
                    inp.reference,
                    inp.current,
                    inp.registry,
                    restored.model_version,
                    inp.artifact_path,
                    restored.ingestion,
                )
            )
            execution["rolled_back"] = (
                restored == state
                and before.report.incident_detected == result.report.incident_detected
            )
            row["executed_resolves_incident"] = execution["resolved"]
            row["rollback_restores_state"] = execution["rolled_back"]
    row["lifecycle"] = _lifecycle_events(
        scenario_id=scenario_id,
        config=config,
        proposal=proposal,
        pre_decision=pre_decision,
        validation=validation,
        post_decision=post_decision,
        approval_requested=row["approval_requested"],
        execution=execution,
    )
    return row


def run_config_a(env, inp, result, truth, scenario_id: str = "SCN-unknown") -> dict[str, Any]:
    started = time.perf_counter()
    evidence = build_evidence(result.report)
    diagnosis = naive_diagnosis(result.report)
    return {
        "diagnosis": _diagnosis_row(diagnosis, truth, evidence),
        "remediation": None,
        "agent": {
            "tool_calls": 0,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "completed": True,
        },
    }


def run_config_b(env, inp, result, truth, scenario_id: str = "SCN-unknown") -> dict[str, Any]:
    started = time.perf_counter()
    state = DeploymentState(dataset=inp.spec.name, model_version=inp.model_version)
    tools = ToolExecutor(ToolContext(inp, result, env.retriever, state))
    runtime = AgentRuntime(
        tools=tools, policy=env.policy, llm=None, declared_version=inp.model_version
    )
    final = run_investigation(runtime, "bench-B", "bench")
    return _agent_result(env, inp, result, truth, final, tools, state, started, scenario_id, "B")


def run_config_c(env, inp, result, truth, scenario_id: str = "SCN-unknown") -> dict[str, Any]:
    """LLM diagnosis from monitoring evidence only: no tools, no rules, no runbooks, no remediation."""
    started = time.perf_counter()
    evidence = build_evidence(result.report)
    call = llm_tasks.request_ranking(env.llm, evidence, None)
    diagnosis, stats = (None, {})
    if call.ok:
        diagnosis, stats = llm_tasks.validate_ranking(call.parsed, evidence, None)
        if diagnosis is not None:  # no rule engine in this configuration: undo its confidence cap
            raw = {r.root_cause.strip().lower(): r.confidence for r in call.parsed.ranking}
            for h in diagnosis.hypotheses:
                h.confidence = round(
                    min(max(raw.get(h.root_cause.value, h.confidence), 0.0), 0.99), 3
                )
    row = _diagnosis_row(diagnosis, truth, evidence)
    proposed = stats.get("proposed", 0)
    row["unsupported_claims"] = stats.get("unsupported", 0) + stats.get("unknown_cause", 0)
    row["claims"] = proposed
    row["cited_unknown_evidence"] = stats.get("fabricated_evidence_ids", 0)
    row["overclaims"] = stats.get("overclaims", 0)
    row["claim_counters"] = claim_rules.add_counters(
        stats.get("claims"),
        claim_rules.audit_final_output(diagnosis=diagnosis, explanation=None, evidence=evidence),
    )
    row["llm_texts"] = {"ranking_items": stats.get("raw_items", []), "explanation": None}
    return {
        "diagnosis": row,
        "remediation": None,
        "agent": {
            "tool_calls": 0,
            "invalid_tool_args": 0,
            "llm_calls": 1,
            "structured_output_failures": int(not call.ok),
            "invalid_outputs": call.invalid_outputs,
            "llm_error": call.error,
            "llm_ms": call.duration_ms,
            "completed": call.ok,
            "prompt_tokens": call.prompt_tokens,
            "completion_tokens": call.completion_tokens,
            "latency_ms": (time.perf_counter() - started) * 1000,
        },
    }


def run_config_d(env, inp, result, truth, scenario_id: str = "SCN-unknown") -> dict[str, Any]:
    started = time.perf_counter()
    state = DeploymentState(dataset=inp.spec.name, model_version=inp.model_version)
    tools = ToolExecutor(ToolContext(inp, result, env.retriever, state))
    runtime = AgentRuntime(
        tools=tools,
        policy=env.policy,
        llm=env.llm,
        declared_version=inp.model_version,
        use_llm_explain=env.llm_explain,
    )
    final = run_investigation(runtime, "bench-D", "bench")
    return _agent_result(env, inp, result, truth, final, tools, state, started, scenario_id, "D")


def _agent_result(
    env, inp, result, truth, final, tools, state, started, scenario_id="SCN-unknown", config="?"
) -> dict[str, Any]:
    evidence = [Evidence.model_validate(e) for e in final.get("evidence", [])]
    diagnosis = Diagnosis.model_validate(final["diagnosis"]) if final.get("diagnosis") else None
    rules = (
        Diagnosis.model_validate(final["rule_diagnosis"]) if final.get("rule_diagnosis") else None
    )
    row = _diagnosis_row(diagnosis, truth, evidence)
    stats = final.get("llm_ranking_stats") or {}
    if stats:  # count what the LLM *tried* to claim, before validation removed it
        row["unsupported_claims"] = stats.get("unsupported", 0) + stats.get("unknown_cause", 0)
        row["claims"] = max(stats.get("proposed", 0), 1)
        row["cited_unknown_evidence"] = stats.get("fabricated_evidence_ids", 0)
        row["overclaims"] = stats.get("overclaims", 0)
    explanation = final.get("explanation") or {}
    claim_validation = explanation.get("claim_validation") or {}
    if stats or claim_validation:  # only when the LLM actually produced text to validate
        row["claim_counters"] = claim_rules.add_counters(
            stats.get("claims"), claim_validation.get("counters"), explanation.get("output_audit")
        )
        row["llm_texts"] = {
            "ranking_items": stats.get("raw_items", []),
            "explanation": claim_validation.get("raw_summary"),
            "explanation_accepted": claim_validation.get("accepted"),
        }
    remediation = (
        _remediation_row(
            env,
            inp,
            result,
            diagnosis,
            rules,
            evidence,
            state,
            precomputed=final,
            scenario_id=scenario_id,
            config=config,
        )
        if diagnosis is not None and final.get("proposal")
        else None
    )
    llm_calls = final.get("llm_calls", [])
    diag_tools = [
        t for t in tools.traces if TOOLS.get(t.tool) and TOOLS[t.tool].kind == "diagnostic"
    ]
    by_llm = [t for t in tools.traces if t.requested_by == "llm"]
    # A tool execution is "unauthorized" if something outside the allowlist ran, or the LLM got a
    # non-selectable tool to run. By construction this must be zero; the benchmark verifies it.
    unauthorized = sum(
        1
        for t in tools.traces
        if t.status == "ok"
        and (
            t.tool not in TOOLS or (t.requested_by == "llm" and t.tool not in LLM_SELECTABLE_TOOLS)
        )
    )
    if remediation is not None:
        remediation["unauthorized_action"] = remediation["unauthorized_action"] or unauthorized > 0
    return {
        "diagnosis": row,
        "remediation": remediation,
        "agent": {
            "tool_calls": sum(t.status == "ok" for t in diag_tools),
            "tool_calls_by_llm": len(by_llm),
            "invalid_tool_args": sum(t.status == "rejected" for t in by_llm),
            "unauthorized_tool_executions": unauthorized,
            "llm_calls": len(llm_calls),
            "structured_output_failures": sum(not c["ok"] for c in llm_calls),
            "invalid_outputs": sum(c.get("invalid_outputs", 0) for c in llm_calls),
            "llm_ms": sum(c["duration_ms"] for c in llm_calls),
            "llm_status": final.get("llm_status"),
            "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in llm_calls),
            "completion_tokens": sum(c.get("completion_tokens") or 0 for c in llm_calls),
            "completed": final.get("status") in ("completed", "awaiting_approval"),
            "errors": len(final.get("errors", [])),
            "latency_ms": (time.perf_counter() - started) * 1000,
        },
    }


RUNNERS = {"A": run_config_a, "B": run_config_b, "C": run_config_c, "D": run_config_d}


def run_scenario(env: BenchmarkEnv, cell: dict, configs: tuple[str, ...]) -> dict[str, Any]:
    scenario = env.scenario(cell)
    truth = scenario.truth
    inp = env.monitoring_input(scenario)
    result = run_monitoring(inp)
    report = result.report
    drift = [f for f in report.findings if f.monitor == "drift"]
    record: dict[str, Any] = {
        "scenario_id": scenario.spec.scenario_id,
        "dataset": cell["dataset"],
        "n_rows": scenario.spec.n_rows,
        "truth": truth.model_dump(),
        "detection": {
            "incident_detected": report.incident_detected,
            "max_severity": report.max_severity.value,
            "triggered": [
                f"{f.check}:{f.feature or '-'}"
                for f in report.triggered
                if f.severity.rank >= Severity.MEDIUM.rank
            ],
            "monitoring_ms": report.duration_ms,
            "timings_ms": report.timings_ms,
            # per-method drift verdicts on the same data, for the method comparison
            "drift_methods": {
                name: any(f.evidence.get("method_flags", {}).get(name) for f in drift)
                for name in (
                    "ks",
                    "ks_bh",
                    "chi2",
                    "chi2_bh",
                    "psi_effect",
                    "wasserstein_effect",
                    "ks_effect",
                )
            },
            "drift_flagged": any(f.triggered for f in drift),
            "concept_delay_rows": {
                f.check: f.evidence.get("detection_index")
                for f in report.findings
                if f.monitor == "concept" and f.triggered
            },
        },
        "configs": {},
    }
    if not report.incident_detected:
        return record  # nothing to diagnose; a missed fault is scored as a detection miss
    for config in configs:
        try:
            record["configs"][config] = RUNNERS[config](
                env, inp, result, truth.root_cause, scenario.spec.scenario_id
            )
        except Exception as exc:  # a crash is a result, not a reason to lose the run
            record["configs"][config] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    return record
