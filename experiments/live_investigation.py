"""Run one real investigation against the local Ollama model and print what happened.

Usage: uv run python -m experiments.live_investigation [--fault F04] [--severity high] [--dataset adult]
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from aegis.agents.graph import AgentRuntime, run_investigation
from aegis.agents.ollama import OllamaClient
from aegis.agents.tools import ToolContext, ToolExecutor
from aegis.config import get_settings
from aegis.faults.scenarios import build_scenario
from aegis.ingestion.contracts import derive_contract
from aegis.ingestion.datasets import DATASETS
from aegis.ingestion.splits import read_split
from aegis.models.registry import ModelRegistry
from aegis.monitoring.engine import MonitoringInput, run_monitoring
from aegis.rag.retriever import RunbookRetriever
from aegis.remediation.policy import PolicyEngine
from aegis.schemas.deployment import DeploymentState


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fault", default="F04")
    parser.add_argument("--severity", default="high")
    parser.add_argument("--dataset", default="adult", choices=sorted(DATASETS))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    spec = DATASETS[args.dataset]
    registry = ModelRegistry(settings.model_dir)
    reference = read_split(spec.name, "validation", settings.processed_dir)
    production = read_split(spec.name, "production", settings.processed_dir)
    contract = derive_contract(spec, reference)
    version = registry.baseline(spec.name).version
    scenario = build_scenario(
        spec,
        production,
        registry,
        version,
        fault_id=args.fault,
        severity=args.severity,
        seed=args.seed,
        n_rows=2000,
        workdir=Path(tempfile.mkdtemp(prefix="aegis-live-")),
    )
    inp = MonitoringInput(
        spec, contract, reference, scenario.current, registry, version, scenario.artifact_path
    )
    result = run_monitoring(inp)
    print(
        f"monitoring: incident={result.report.incident_detected} severity={result.report.max_severity.value} "
        f"({result.report.duration_ms:.0f} ms, {len(result.report.triggered)} findings triggered)"
    )

    llm = None if args.no_llm else OllamaClient(settings)
    if llm:
        print("llm health:", llm.health())
    retriever = RunbookRetriever(
        settings.runbook_dir,
        embedder=llm,
        cache_dir=settings.artifacts_dir / "rag",
        default_method=settings.rag_method,
    )
    tools = ToolExecutor(
        ToolContext(
            inp, result, retriever, DeploymentState(dataset=spec.name, model_version=version)
        )
    )
    final = run_investigation(
        AgentRuntime(tools=tools, policy=PolicyEngine(), llm=llm, declared_version=version),
        "INV-live",
        "INC-live",
    )

    print(
        f"\nstatus={final['status']} llm_status={final['llm_status']} model={final.get('llm_model')} "
        f"total={final['duration_ms'] / 1000:.1f}s"
    )
    for call in final.get("llm_calls", []):
        print(
            f"  llm {call['purpose']:<8} ok={call['ok']} {call['duration_ms'] / 1000:6.1f}s "
            f"tokens in/out={call['prompt_tokens']}/{call['completion_tokens']} attempts={call['attempts']} "
            f"error={call['error']}"
        )
    for t in tools.traces:
        print(
            f"  tool {t.tool:<30} {t.status:<8} by={t.requested_by:<8} {t.duration_ms:7.1f} ms "
            f"args={json.dumps(t.validated_args or t.requested_args)[:70]} {t.error or ''}"
        )
    print("ranking stats:", final.get("llm_ranking_stats"))
    for h in final["diagnosis"]["hypotheses"][:3]:
        print(
            f"  hypothesis {h['root_cause']:<18} fit={h['confidence']:.2f} by={h['ranked_by']} "
            f"support={len(h['supporting_evidence_ids'])} contra={len(h['contradicting_evidence_ids'])}\n"
            f"     {h['reasoning'][:200]}"
        )
    print(
        "rule top:",
        final["rule_diagnosis"]["hypotheses"][0]["root_cause"]
        if final["rule_diagnosis"]["hypotheses"]
        else None,
    )
    print("proposal:", final["proposal"]["action"], final["proposal"]["parameters"])
    print("validation:", (final.get("validation") or {}).get("summary"))
    print("policy:", final["policy_decision"]["outcome"], final["policy_decision"]["reasons"])
    print(
        "runbooks:",
        [p["passage_id"] for p in final["runbook_passages"]],
        "method:",
        {p["method"] for p in final["runbook_passages"]},
    )
    print("explanation source:", final["explanation"]["source"])
    print(final["explanation"]["summary"])
    print("checks:", final["explanation"].get("operator_checks"))
    print("errors:", final.get("errors"))
    print(
        "ground truth (scorer only):",
        scenario.truth.fault_id,
        scenario.truth.affected_features,
        scenario.truth.params,
    )


if __name__ == "__main__":
    main()
