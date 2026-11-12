"""Verify one local Ollama model against AEGIS, for real: no fake server.

    uv run python -m scripts.verify_llm --model qwen3:4b
    uv run python -m scripts.verify_llm --model llama3.2:3b

Checks, each recorded with what actually happened:

  metadata             the model is pulled; digest, size, quantisation
  load                 cold load on its own time budget
  structured_output    one schema-constrained request parses and validates
  investigation        a complete AEGIS investigation with this model (plan, rank, explain)
  fallback_selection   with a primary model that is not pulled, this model is chosen and used
  timeout              a 1-second budget produces `timeout`, not a hang or an exception
  malformed_output     a reply truncated to 8 tokens is rejected as `invalid_output`
  unavailable_model    neither configured model pulled -> reported unavailable, nothing raised
  unreachable_server   a dead endpoint -> reported unavailable; the real endpoint works after it
  deterministic_path   the same investigation completes with the LLM unreachable

Writes ``results/release/llm-<model>.json``. Exit code 0 only if every check passed.
Hardware-dependent: on a small CPU the investigation takes minutes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aegis.agents import claims as claim_rules
from aegis.agents.graph import AgentRuntime, run_investigation
from aegis.agents.ollama import OllamaClient
from aegis.agents.tools import ToolContext, ToolExecutor
from aegis.config import Settings, get_settings
from aegis.faults.scenarios import build_scenario
from aegis.ingestion.contracts import derive_contract
from aegis.ingestion.datasets import DATASETS
from aegis.ingestion.splits import read_split
from aegis.models.registry import ModelRegistry
from aegis.monitoring.engine import MonitoringInput, run_monitoring
from aegis.observability.resources import ResourceSampler, environment_info
from aegis.rag.retriever import RunbookRetriever
from aegis.remediation.policy import PolicyEngine
from aegis.schemas.deployment import DeploymentState


class Verdict(BaseModel):
    """Small schema for the structured-output check."""

    root_cause: str = Field(max_length=40)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(max_length=4)


def _settings(base: Settings, **overrides: Any) -> Settings:
    return base.model_copy(update=overrides)


def _investigate(settings: Settings, llm: OllamaClient | None) -> tuple[dict, list]:
    spec = DATASETS["adult"]
    registry = ModelRegistry(settings.model_dir)
    reference = read_split(spec.name, "validation", settings.processed_dir)
    version = registry.baseline(spec.name).version
    scenario = build_scenario(
        spec,
        read_split(spec.name, "production", settings.processed_dir),
        registry,
        version,
        fault_id="F04",
        severity="high",
        seed=4,
        n_rows=2000,
        workdir=Path(tempfile.mkdtemp(prefix="aegis-verify-")),
    )
    inp = MonitoringInput(
        spec, derive_contract(spec, reference), reference, scenario.current, registry, version, None
    )
    result = run_monitoring(inp)
    tools = ToolExecutor(
        ToolContext(
            inp,
            result,
            RunbookRetriever(settings.runbook_dir),
            DeploymentState(dataset=spec.name, model_version=version),
        )
    )
    final = run_investigation(
        AgentRuntime(tools=tools, policy=PolicyEngine(), llm=llm, declared_version=version),
        "INV-verify",
        "INC-verify",
    )
    return final, tools.traces


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, default=300.0, help="per-request budget, seconds")
    parser.add_argument("--out", type=Path, default=Path("results/release"))
    args = parser.parse_args()

    base = get_settings()
    model = args.model
    primary = _settings(
        base,
        ollama_model=model,
        ollama_fallback_model="aegis-no-such-model:0b",
        ollama_timeout_s=args.timeout,
        llm_enabled=True,
    )
    client = OllamaClient(primary)
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, passed: bool, /, **detail: Any) -> None:
        checks[name] = {"passed": bool(passed), **detail}
        print(f"  {'ok  ' if passed else 'FAIL'}  {name}  {json.dumps(detail, default=str)[:230]}")

    print(f"verifying {model} at {primary.ollama_base_url}")
    health = client.health()
    info = client.model_info(model)
    record(
        "metadata", info.get("available", False) and health.reachable, **info, server=health.version
    )
    if not checks["metadata"]["passed"]:
        _write(args.out, model, checks, None, None)
        sys.exit(
            f"model {model} is not available on {primary.ollama_base_url}; nothing else was run"
        )

    # Unload first so the load is measured cold.
    try:
        client._client.post("/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
    except Exception as exc:
        print(f"  note: could not unload before measuring ({type(exc).__name__})")
    time.sleep(2)
    sampler = ResourceSampler(f"verify-{model}", interval_s=0.5).start()
    loaded, seconds, why = client.warm(model)
    record("load", loaded, cold_load_s=round(seconds, 1), detail=why)

    started = time.perf_counter()
    answer = client.chat_json(
        system="You classify ML incidents. Reply with one JSON object only.",
        user=(
            "EVIDENCE\n[EV-1] ANOMALY: hours_per_week values are about 60x the reference "
            "(constant factor).\n[EV-2] normal: no other feature drifted.\n"
            "Choose root_cause from: scaling_error, concept_drift, missing_values. "
            'JSON: {"root_cause": str, "confidence": number, "evidence_ids": [str]}'
        ),
        schema=Verdict,
        purpose="verify",
        model=model,
        max_tokens=80,
    )
    record(
        "structured_output",
        answer.ok and answer.parsed is not None,
        parsed=answer.parsed.model_dump() if answer.parsed else None,
        error=answer.error,
        attempts=answer.attempts,
        invalid_outputs=answer.invalid_outputs,
        seconds=round(time.perf_counter() - started, 1),
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
        cites_only_given_ids=bool(answer.parsed)
        and set(answer.parsed.evidence_ids) <= {"EV-1", "EV-2"},
    )

    started = time.perf_counter()
    final, traces = _investigate(primary, client)
    calls = final.get("llm_calls", [])
    stats = final.get("llm_ranking_stats") or {}
    explanation = final.get("explanation") or {}
    top = (final["diagnosis"]["hypotheses"] or [{}])[0]
    counters = claim_rules.add_counters(
        stats.get("claims"),
        (explanation.get("claim_validation") or {}).get("counters"),
        explanation.get("output_audit"),
    )
    record(
        "investigation",
        final.get("status") in ("completed", "awaiting_approval")
        and final.get("llm_model") == model
        and any(c["ok"] for c in calls),
        status=final.get("status"),
        llm_status=final.get("llm_status"),
        llm_model=final.get("llm_model"),
        seconds=round(time.perf_counter() - started, 1),
        llm_calls=[
            {
                "purpose": c["purpose"],
                "ok": c["ok"],
                "error": c["error"],
                "seconds": round(c["duration_ms"] / 1000, 1),
                "prompt_tokens": c["prompt_tokens"],
                "completion_tokens": c["completion_tokens"],
                "invalid_outputs": c["invalid_outputs"],
            }
            for c in calls
        ],
        llm_calls_ok=sum(c["ok"] for c in calls),
        llm_calls_total=len(calls),
        top_cause=top.get("root_cause"),
        top_ranked_by=top.get("ranked_by"),
        rule_top_cause=(final["rule_diagnosis"]["hypotheses"] or [{}])[0].get("root_cause"),
        policy_outcome=(final.get("policy_decision") or {}).get("outcome"),
        explanation_source=explanation.get("source"),
        tools_requested_by_llm=sum(t.requested_by == "llm" for t in traces),
        tools_rejected=sum(t.status == "rejected" for t in traces),
        claim_counters=counters,
        errors=final.get("errors", []),
    )
    resources = sampler.stop().to_dict()

    # Fallback: a primary that is not pulled, this model as the fallback.
    fb = OllamaClient(
        _settings(
            base,
            ollama_model="aegis-no-such-model:0b",
            ollama_fallback_model=model,
            ollama_timeout_s=args.timeout,
            llm_enabled=True,
        )
    )
    fb_health = fb.health()
    fb_answer = fb.chat_json(
        system="Reply with one JSON object only.",
        user='Return {"root_cause": "scaling_error", "confidence": 0.5, "evidence_ids": []}',
        schema=Verdict,
        purpose="verify-fallback",
        max_tokens=60,
    )
    record(
        "fallback_selection",
        fb_health.available
        and fb_health.active_model == model
        and fb_answer.ok
        and fb_answer.model == model,
        active_model=fb_health.active_model,
        health_detail=fb_health.detail,
        answered_by=fb_answer.model,
        ok=fb_answer.ok,
        error=fb_answer.error,
    )

    quick = OllamaClient(_settings(primary, ollama_timeout_s=1.0, ollama_max_retries=0))
    started = time.perf_counter()
    slow = quick.chat_json(
        system="Reply with JSON.",
        user="Write a 300-word essay as JSON "
        + '{"root_cause": str, "confidence": number, "evidence_ids": [str]}',
        schema=Verdict,
        purpose="verify-timeout",
        model=model,
        max_tokens=400,
    )
    record(
        "timeout",
        (not slow.ok) and slow.error == "timeout",
        error=slow.error,
        detail=slow.detail,
        returned_after_s=round(time.perf_counter() - started, 1),
    )
    time.sleep(3)  # let the abandoned request finish server-side

    truncated = client.chat_json(
        system="Reply with one JSON object only.",
        user='JSON: {"root_cause": str, "confidence": number, "evidence_ids": [str]} for a scaling error.',
        schema=Verdict,
        purpose="verify-malformed",
        model=model,
        max_tokens=8,
    )
    record(
        "malformed_output",
        (not truncated.ok)
        and truncated.error == "invalid_output"
        and truncated.invalid_outputs >= 1,
        error=truncated.error,
        attempts=truncated.attempts,
        invalid_outputs=truncated.invalid_outputs,
        raw=truncated.raw[:80],
    )

    missing = OllamaClient(
        _settings(
            base, ollama_model="aegis-no-such-model:0b", ollama_fallback_model="aegis-none:0b"
        )
    )
    m_health = missing.health()
    m_answer = missing.chat_json(system="x", user="y", schema=Verdict, purpose="verify-missing")
    record(
        "unavailable_model",
        m_health.reachable
        and not m_health.available
        and not m_answer.ok
        and m_answer.error == "unavailable",
        health_detail=m_health.detail,
        error=m_answer.error,
    )

    dead = OllamaClient(_settings(primary, ollama_base_url="http://127.0.0.1:9"))
    d_health = dead.health()
    d_answer = dead.chat_json(
        system="x", user="y", schema=Verdict, purpose="verify-dead", model=model
    )
    after = client.chat_json(
        system="Reply with one JSON object only.",
        user='Return {"root_cause": "scaling_error", "confidence": 0.5, "evidence_ids": []}',
        schema=Verdict,
        purpose="verify-recovery",
        model=model,
        max_tokens=60,
    )
    record(
        "unreachable_server",
        (not d_health.reachable)
        and (not d_answer.ok)
        and d_answer.error == "unavailable"
        and after.ok,
        dead_health=d_health.detail,
        dead_error=d_answer.error,
        real_endpoint_ok_afterwards=after.ok,
    )

    started = time.perf_counter()
    offline, _ = _investigate(primary, dead)
    off_top = (offline["diagnosis"]["hypotheses"] or [{}])[0]
    record(
        "deterministic_path",
        offline.get("status") in ("completed", "awaiting_approval")
        and offline.get("llm_status") == "unavailable"
        and off_top.get("ranked_by") == "rules"
        and (offline.get("policy_decision") or {}).get("outcome")
        == (final.get("policy_decision") or {}).get("outcome"),
        status=offline.get("status"),
        llm_status=offline.get("llm_status"),
        top_cause=off_top.get("root_cause"),
        ranked_by=off_top.get("ranked_by"),
        policy_outcome=(offline.get("policy_decision") or {}).get("outcome"),
        seconds=round(time.perf_counter() - started, 1),
    )

    path = _write(args.out, model, checks, resources, environment_info())
    failed = [name for name, c in checks.items() if not c["passed"]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed -> {path}")
    sys.exit(1 if failed else 0)


def _write(out: Path, model: str, checks: dict, resources: dict | None, env: dict | None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"llm-{model.replace(':', '-').replace('/', '-')}.json"
    path.write_text(
        json.dumps(
            {
                "model": model,
                "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "passed": all(c["passed"] for c in checks.values()),
                "checks": checks,
                "resources": resources,
                "environment": env,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    return path


if __name__ == "__main__":
    main()
