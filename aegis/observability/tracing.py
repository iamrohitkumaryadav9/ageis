"""OpenTelemetry tracing.

Spans always go to a small in-process ring buffer (so the API can expose recent
traces and tests can assert on them without a collector) and additionally to an
OTLP/HTTP endpoint when ``AEGIS_OTLP_ENDPOINT`` is set.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanProcessor
from opentelemetry.trace import Status, StatusCode


class RecentSpans(SpanProcessor):
    """Keeps the last ``capacity`` finished spans in memory."""

    def __init__(self, capacity: int = 2000) -> None:
        self._spans: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def on_end(self, span: ReadableSpan) -> None:
        context = span.get_span_context()
        record = {
            "name": span.name,
            "trace_id": format(context.trace_id, "032x"),
            "span_id": format(context.span_id, "016x"),
            "parent_span_id": format(span.parent.span_id, "016x") if span.parent else None,
            "start": span.start_time,
            "duration_ms": round((span.end_time - span.start_time) / 1e6, 3),
            "status": span.status.status_code.name,
            "attributes": dict(span.attributes or {}),
            "events": [e.name for e in span.events],
        }
        with self._lock:
            self._spans.append(record)

    def snapshot(self, name_prefix: str = "", trace_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            spans = list(self._spans)
        return [
            s
            for s in spans
            if s["name"].startswith(name_prefix) and (trace_id is None or s["trace_id"] == trace_id)
        ]

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()


recent_spans = RecentSpans()
_provider: TracerProvider | None = None


def setup_tracing(otlp_endpoint: str = "", service_name: str = "aegis") -> TracerProvider:
    global _provider
    if _provider is not None:
        return _provider
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(recent_spans)
    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint.rstrip("/") + "/v1/traces"))
        )
    trace.set_tracer_provider(provider)
    _provider = provider
    return provider


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[trace.Span]:
    """Start a span; exceptions are recorded on it and re-raised."""
    if _provider is None:
        setup_tracing()
    tracer = trace.get_tracer("aegis")
    clean = {k: v for k, v in attributes.items() if isinstance(v, str | int | float | bool)}
    with tracer.start_as_current_span(name, attributes=clean) as current:
        try:
            yield current
        except Exception as exc:
            current.record_exception(exc)
            current.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise


def current_trace_id() -> str | None:
    context = trace.get_current_span().get_span_context()
    return format(context.trace_id, "032x") if context.is_valid else None
