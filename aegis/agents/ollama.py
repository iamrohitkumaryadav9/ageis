"""Local Ollama client with validated structured output.

Design rules
------------
* Only ever talks to the configured Ollama base URL. There is no cloud fallback.
* A response counts only if it parses as JSON *and* validates against the
  caller's Pydantic schema. Asking for JSON is not the same as getting it.
* Failures (offline, timeout, invalid output) are returned as data, never
  raised, so callers can continue with their deterministic result.
* Retries are bounded by ``ollama_max_retries`` and only happen for invalid
  output or a missing primary model (-> fallback model), not for timeouts:
  retrying a 120 s CPU timeout doubles the wait for the same outcome.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from aegis.config import Settings
from aegis.observability.logging import get_logger
from aegis.observability.metrics import LLM_REQUESTS, LLM_SECONDS
from aegis.observability.tracing import span

log = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResult[T]:
    ok: bool
    parsed: T | None = None
    model: str | None = None
    purpose: str = ""
    error: str | None = None  # unavailable | timeout | invalid_output | http_error | disabled
    detail: str = ""
    attempts: int = 0
    invalid_outputs: int = 0
    duration_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    prompt_eval_ms: float | None = None  # time spent reading the prompt (None if fully cached)
    generation_ms: float | None = None
    load_ms: float | None = None  # model load time, when the model was not already in memory
    raw: str = ""

    def audit(self) -> dict[str, Any]:
        return {
            k: getattr(self, k)
            for k in (
                "ok",
                "model",
                "purpose",
                "error",
                "detail",
                "attempts",
                "invalid_outputs",
                "duration_ms",
                "prompt_tokens",
                "completion_tokens",
                "prompt_eval_ms",
                "generation_ms",
                "load_ms",
            )
        }


@dataclass
class LLMHealth:
    enabled: bool
    reachable: bool
    version: str | None = None
    models: list[str] = field(default_factory=list)
    active_model: str | None = None  # the model that will actually be used, if any
    detail: str = ""

    @property
    def available(self) -> bool:
        return self.enabled and self.reachable and self.active_model is not None


def _ms(nanoseconds: int | None) -> float | None:
    return None if nanoseconds is None else round(nanoseconds / 1e6, 1)


def _has(models: list[str], name: str) -> bool:
    return (
        any(m == name or m.startswith(name + ":") or m.split(":")[0] == name for m in models)
        if ":" not in name
        else name in models
    )


class OllamaClient:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self._client = httpx.Client(
            base_url=settings.ollama_base_url,
            transport=transport,
            timeout=httpx.Timeout(settings.ollama_timeout_s, connect=3.0),
        )

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------ health
    def health(self) -> LLMHealth:
        s = self.settings
        if not s.llm_enabled:
            return LLMHealth(enabled=False, reachable=False, detail="LLM disabled by configuration")
        try:
            version = self._client.get("/api/version", timeout=3.0).json().get("version")
            tags = self._client.get("/api/tags", timeout=3.0).json()
            models = [m["name"] for m in tags.get("models", [])]
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            return LLMHealth(enabled=True, reachable=False, detail=f"{type(exc).__name__}")
        active = next(
            (m for m in (s.ollama_model, s.ollama_fallback_model) if _has(models, m)), None
        )
        detail = (
            ""
            if active == s.ollama_model
            else (
                f"primary model {s.ollama_model} not pulled; using fallback {active}"
                if active
                else f"neither {s.ollama_model} nor {s.ollama_fallback_model} is pulled"
            )
        )
        return LLMHealth(True, True, version, models, active, detail)

    # ------------------------------------------------------------ structured
    def chat_json(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        purpose: str,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult[T]:
        s = self.settings
        if not s.llm_enabled:
            return LLMResult(ok=False, error="disabled", purpose=purpose)
        models = [model] if model else [s.ollama_model, s.ollama_fallback_model]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        result: LLMResult[T] = LLMResult(ok=False, purpose=purpose)
        started = time.perf_counter()
        model_index = 0
        with span("llm.chat", purpose=purpose) as current:
            while result.attempts <= s.ollama_max_retries and model_index < len(models):
                name = models[model_index]
                result.attempts += 1
                result.model = name
                outcome = self._once(name, messages, schema, max_tokens or s.ollama_max_tokens)
                result.raw = outcome.get("raw", "")
                result.prompt_tokens = outcome.get("prompt_tokens", result.prompt_tokens)
                result.completion_tokens = outcome.get(
                    "completion_tokens", result.completion_tokens
                )
                for timing in ("prompt_eval_ms", "generation_ms", "load_ms"):
                    setattr(result, timing, outcome.get(timing, getattr(result, timing)))
                if outcome["status"] == "ok":
                    result.ok, result.parsed, result.error, result.detail = (
                        True,
                        outcome["parsed"],
                        None,
                        "",
                    )
                    break
                result.error, result.detail = outcome["status"], outcome.get("detail", "")
                if outcome["status"] == "model_missing":
                    model_index += 1  # try the fallback model; does not consume the retry budget
                    result.attempts -= 1
                    result.error = "unavailable"
                    continue
                if outcome["status"] == "invalid_output":
                    result.invalid_outputs += 1
                    messages = [
                        *messages[:2],
                        {"role": "assistant", "content": result.raw[:1500]},
                        {
                            "role": "user",
                            "content": "That reply was rejected: "
                            + result.detail[:400]
                            + " Reply again with ONLY a JSON object that satisfies the schema.",
                        },
                    ]
                    continue
                break  # unavailable / timeout / http_error: do not retry
            result.duration_ms = round((time.perf_counter() - started) * 1000, 1)
            current.set_attribute("llm.model", result.model or "")
            current.set_attribute("llm.outcome", "ok" if result.ok else result.error or "error")
            current.set_attribute("llm.attempts", result.attempts)
            current.set_attribute("llm.completion_tokens", result.completion_tokens or 0)
            if not result.ok:
                current.add_event(f"llm_{result.error}")
        outcome_label = "ok" if result.ok else (result.error or "error")
        LLM_REQUESTS.labels(result.model or "none", outcome_label).inc()
        LLM_SECONDS.labels(result.model or "none", purpose).observe(result.duration_ms / 1000)
        if not result.ok:
            log.warning(
                "llm_request_failed",
                purpose=purpose,
                model=result.model,
                error=result.error,
                detail=result.detail[:200],
                attempts=result.attempts,
            )
        return result

    def _once(
        self, model: str, messages: list[dict], schema: type[BaseModel], max_tokens: int
    ) -> dict:
        s = self.settings
        options: dict[str, Any] = {
            "temperature": s.ollama_temperature,
            "num_ctx": s.ollama_num_ctx,
            "num_predict": max_tokens,
        }
        if s.ollama_num_thread:
            options["num_thread"] = s.ollama_num_thread
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": schema.model_json_schema(),
            "options": options,
            "keep_alive": s.ollama_keep_alive,
            "think": False,
        }
        try:
            response = self._client.post("/api/chat", json=body)
            if response.status_code == 400 and "think" in response.text.lower():
                body.pop("think")  # model without a thinking mode
                response = self._client.post("/api/chat", json=body)
        except httpx.TimeoutException:
            return {"status": "timeout", "detail": f"no reply within {s.ollama_timeout_s:.0f}s"}
        except httpx.HTTPError as exc:
            return {"status": "unavailable", "detail": type(exc).__name__}
        if response.status_code == 404:
            return {"status": "model_missing", "detail": f"model {model} is not pulled"}
        if response.status_code >= 400:
            return {
                "status": "http_error",
                "detail": f"HTTP {response.status_code}: {response.text[:200]}",
            }
        try:
            payload = response.json()
            raw = payload["message"]["content"]
        except (ValueError, KeyError, TypeError):
            return {"status": "invalid_output", "detail": "response was not an Ollama chat payload"}
        tokens = {
            "raw": raw,
            "prompt_tokens": payload.get("prompt_eval_count"),
            "completion_tokens": payload.get("eval_count"),
            "prompt_eval_ms": _ms(payload.get("prompt_eval_duration")),
            "generation_ms": _ms(payload.get("eval_duration")),
            "load_ms": _ms(payload.get("load_duration")),
        }
        try:
            parsed = schema.model_validate(json.loads(raw))
        except json.JSONDecodeError as exc:
            return {"status": "invalid_output", "detail": f"not valid JSON ({exc.msg})", **tokens}
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:4]
            )
            return {"status": "invalid_output", "detail": f"schema violation: {problems}", **tokens}
        return {"status": "ok", "parsed": parsed, **tokens}

    # -------------------------------------------------------------- embeddings
    def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Embeddings from the local embedding model, or None if unavailable."""
        if not self.settings.llm_enabled:
            return None
        try:
            with span("llm.embed", count=len(texts)):
                response = self._client.post(
                    "/api/embed",
                    json={"model": self.settings.ollama_embed_model, "input": texts},
                    timeout=60.0,
                )
            if response.status_code >= 400:
                return None
            vectors = response.json().get("embeddings")
            return vectors if vectors and len(vectors) == len(texts) else None
        except (httpx.HTTPError, ValueError):
            return None
