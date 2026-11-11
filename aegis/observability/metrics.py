"""Prometheus metrics. Buckets are chosen for the latencies this system really has:
milliseconds for monitoring and tools, tens of seconds for CPU LLM inference."""

from __future__ import annotations

import psutil
from prometheus_client import Counter, Gauge, Histogram

_FAST = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
_SLOW = (0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 240, 480)

MONITORING_SECONDS = Histogram(
    "aegis_monitoring_seconds", "Monitoring run duration", ["dataset"], buckets=_FAST
)
TOOL_SECONDS = Histogram("aegis_tool_seconds", "Tool execution duration", ["tool"], buckets=_FAST)
TOOL_CALLS = Counter("aegis_tool_calls_total", "Tool calls by outcome", ["tool", "status"])
LLM_SECONDS = Histogram(
    "aegis_llm_seconds", "LLM request duration", ["model", "purpose"], buckets=_SLOW
)
LLM_REQUESTS = Counter("aegis_llm_requests_total", "LLM requests by outcome", ["model", "outcome"])
INVESTIGATION_SECONDS = Histogram(
    "aegis_investigation_seconds", "Investigation duration", ["llm"], buckets=_SLOW
)
INCIDENTS = Counter("aegis_incidents_total", "Incidents opened", ["dataset", "severity"])
REMEDIATION_TRANSITIONS = Counter(
    "aegis_remediation_transitions_total", "Remediation state transitions", ["to_state"]
)
DENIED_ACTIONS = Counter("aegis_denied_actions_total", "Rejected privileged actions", ["reason"])
PROCESS_RSS = Gauge("aegis_process_rss_bytes", "Resident set size of the API process")
PROCESS_RSS.set_function(lambda: psutil.Process().memory_info().rss)
