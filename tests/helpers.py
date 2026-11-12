"""Test helpers: a scripted Ollama HTTP endpoint and investigation wiring.

The fake speaks Ollama's wire protocol through ``httpx.MockTransport``, so the
real ``OllamaClient`` (parsing, validation, retries, fallback) is what runs.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

from aegis.agents.graph import AgentRuntime, run_investigation
from aegis.agents.ollama import OllamaClient
from aegis.agents.tools import ToolContext, ToolExecutor
from aegis.config import Settings
from aegis.monitoring.engine import MonitoringInput
from aegis.rag.retriever import RunbookRetriever
from aegis.remediation.policy import PolicyEngine
from aegis.schemas.deployment import DeploymentState

Reply = str | dict | Callable[[dict], Any] | Exception


def purpose_of(body: dict) -> str:
    props = (body.get("format") or {}).get("properties", {})
    return (
        "plan"
        if "calls" in props
        else "rank"
        if "ranking" in props
        else "explain"
        if "summary" in props
        else "other"
    )


class FakeOllama:
    """Scripted replies per purpose. A reply may be a dict (sent as JSON), a raw string,
    a callable(request_body) -> reply, or an exception to raise (e.g. a timeout)."""

    def __init__(
        self,
        replies: dict[str, Reply | list[Reply]] | None = None,
        *,
        models: tuple[str, ...] = ("qwen3:4b",),
        reachable: bool = True,
        load_error: Exception | None = None,
    ):
        self.replies = replies or {}
        self.models = models
        self.reachable = reachable
        self.requests: list[dict] = []
        self.load_error = load_error
        self.loads: list[dict] = []  # model warm-up calls, kept apart from chat requests
        self.hosts: set[str] = set()

    def _next(self, purpose: str, body: dict) -> Any:
        reply = self.replies.get(purpose, self.replies.get("*"))
        if isinstance(reply, list):
            reply = reply.pop(0) if len(reply) > 1 else reply[0]
        if callable(reply) and not isinstance(reply, Exception):
            reply = reply(body)
        return reply

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.hosts.add(request.url.host)
        if not self.reachable:
            raise httpx.ConnectError("connection refused")
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "test"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in self.models]})
        body = json.loads(request.content)
        if request.url.path == "/api/generate":
            self.loads.append(body)
            if self.load_error is not None:
                raise self.load_error
            return httpx.Response(200, json={"model": body["model"], "done": True})
        self.requests.append(body)
        if body["model"] not in self.models:
            return httpx.Response(404, json={"error": f"model '{body['model']}' not found"})
        reply = self._next(purpose_of(body), body)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, httpx.Response):
            return reply
        if reply is None:
            reply = "I am not sure what you want."
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "message": {"role": "assistant", "content": content},
                "prompt_eval_count": 100,
                "eval_count": 50,
            },
        )

    def client(self, **overrides) -> OllamaClient:
        settings = Settings(ollama_base_url="http://ollama.test:11434", **overrides)
        return OllamaClient(settings, transport=httpx.MockTransport(self.handler))


def investigate(
    ctx,
    scenario,
    *,
    llm: OllamaClient | None = None,
    retriever: RunbookRetriever | None = None,
    state: DeploymentState | None = None,
    **runtime_flags,
):
    """Monitor a scenario and run the investigation graph. Returns (final_state, tools, inp, result)."""
    from aegis.config import get_settings

    inp = MonitoringInput(
        ctx.spec,
        ctx.contract,
        ctx.reference,
        scenario.current,
        ctx.registry,
        ctx.record.version,
        scenario.artifact_path,
    )
    from aegis.monitoring.engine import run_monitoring

    result = run_monitoring(inp)
    retriever = retriever or RunbookRetriever(get_settings().runbook_dir)
    state = state or DeploymentState(dataset=ctx.spec.name, model_version=ctx.record.version)
    tools = ToolExecutor(ToolContext(inp, result, retriever, state))
    runtime = AgentRuntime(
        tools=tools,
        policy=PolicyEngine(),
        llm=llm,
        declared_version=ctx.record.version,
        **runtime_flags,
    )
    final = run_investigation(runtime, "INV-test", "INC-test")
    return final, tools, inp, result


def good_plan(feature: str | None = None) -> dict:
    calls = [
        {
            "tool": "get_model_metrics",
            "feature": None,
            "other_version": None,
            "reason": "check performance",
        }
    ]
    if feature:
        calls.insert(
            0,
            {
                "tool": "inspect_feature_distribution",
                "feature": feature,
                "other_version": None,
                "reason": "characterise the change",
            },
        )
    return {"calls": calls}


def ranking_from_prompt(cause: str, confidence: float = 0.9) -> Callable[[dict], dict]:
    """A well-behaved ranking reply that cites the first anomaly evidence id found in the prompt."""
    import re

    def reply(body: dict) -> dict:
        prompt = body["messages"][-1]["content"]
        ids = re.findall(r"\[(EV-[\w-]+)\] ANOMALY", prompt)
        return {
            "ranking": [
                {
                    "root_cause": cause,
                    "confidence": confidence,
                    "supporting_evidence_ids": ids[:3],
                    "contradicting_evidence_ids": [],
                    "reasoning": "The cited findings are consistent with it.",
                }
            ],
            "insufficient_evidence": False,
        }

    return reply
