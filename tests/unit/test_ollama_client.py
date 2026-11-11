import httpx
import pytest
from pydantic import BaseModel, Field

from aegis.agents.llm_tasks import LLMRanking
from tests.helpers import FakeOllama


class Answer(BaseModel):
    value: int = Field(ge=0, le=10)


def ask(client, **kw):
    return client.chat_json(system="s", user="u", schema=Answer, purpose="other", **kw)


def test_health_distinguishes_offline_missing_model_and_ready():
    assert FakeOllama().client().health().available
    offline = FakeOllama(reachable=False).client().health()
    assert offline.enabled and not offline.reachable and not offline.available
    fallback = FakeOllama(models=("llama3.2:3b",)).client().health()
    assert (
        fallback.available
        and fallback.active_model == "llama3.2:3b"
        and "fallback" in fallback.detail
    )
    none = FakeOllama(models=("mistral:7b",)).client().health()
    assert none.reachable and not none.available and none.active_model is None
    disabled = FakeOllama().client(llm_enabled=False).health()
    assert not disabled.enabled and not disabled.available


def test_valid_structured_output_is_parsed_and_measured():
    fake = FakeOllama({"*": {"value": 7}})
    result = ask(fake.client())
    assert result.ok and result.parsed == Answer(value=7) and result.model == "qwen3:4b"
    assert result.attempts == 1 and result.completion_tokens == 50 and result.duration_ms >= 0
    body = fake.requests[0]
    assert (
        body["format"]["properties"]["value"]["maximum"] == 10
    )  # the schema is sent, not just described
    assert body["options"] == {"temperature": 0.1, "num_ctx": 4096, "num_predict": 700}
    assert body["stream"] is False and body["think"] is False and body["keep_alive"] == "2m"


@pytest.mark.parametrize("bad", ["not json at all", '{"value": 99}', '{"other": 1}', "[1, 2]", ""])
def test_invalid_output_is_never_accepted(bad):
    fake = FakeOllama({"*": bad})
    result = ask(fake.client())
    assert not result.ok and result.parsed is None and result.error == "invalid_output"
    assert result.attempts == 2 and result.invalid_outputs == 2  # one retry, then stop
    assert len(fake.requests) == 2


def test_retry_feeds_back_the_validation_error_and_can_recover():
    fake = FakeOllama({"*": ['{"value": 99}', {"value": 3}]})
    result = ask(fake.client())
    assert (
        result.ok
        and result.parsed.value == 3
        and result.attempts == 2
        and result.invalid_outputs == 1
    )
    retry_messages = fake.requests[1]["messages"]
    assert len(retry_messages) == 4 and "rejected" in retry_messages[-1]["content"]
    assert "value" in retry_messages[-1]["content"]


def test_retry_limit_is_strict():
    fake = FakeOllama({"*": "garbage"})
    assert ask(fake.client(ollama_max_retries=0)).attempts == 1
    assert len(fake.requests) == 1


def test_timeout_is_reported_and_not_retried():
    fake = FakeOllama({"*": httpx.ReadTimeout("slow")})
    result = ask(fake.client())
    assert (
        not result.ok
        and result.error == "timeout"
        and result.attempts == 1
        and len(fake.requests) == 1
    )


def test_offline_is_reported_as_unavailable_without_raising():
    result = ask(FakeOllama(reachable=False).client())
    assert not result.ok and result.error == "unavailable"


def test_missing_primary_model_falls_back_to_the_configured_local_model():
    fake = FakeOllama({"*": {"value": 1}}, models=("llama3.2:3b",))
    result = ask(fake.client())
    assert result.ok and result.model == "llama3.2:3b"
    # primary is tried first (404), then the configured local fallback
    assert [r["model"] for r in fake.requests] == ["qwen3:4b", "llama3.2:3b"]
    nothing = ask(FakeOllama(models=()).client())
    assert not nothing.ok and nothing.error == "unavailable"


def test_disabled_llm_makes_no_request():
    fake = FakeOllama({"*": {"value": 1}})
    result = ask(fake.client(llm_enabled=False))
    assert result.error == "disabled" and fake.requests == [] and fake.hosts == set()


def test_only_the_configured_local_host_is_ever_contacted():
    fake = FakeOllama({"*": "garbage"}, models=())
    client = fake.client()
    client.health()
    ask(client)
    client.embed(["x"])
    assert fake.hosts == {"ollama.test"}


def test_http_error_is_surfaced():
    fake = FakeOllama({"*": httpx.Response(500, text="boom")})
    result = ask(fake.client())
    assert result.error == "http_error" and "500" in result.detail and result.attempts == 1


def test_model_without_thinking_mode_is_retried_without_the_flag():
    seen = []

    def reply(body):
        seen.append("think" in body)
        if "think" in body:
            return httpx.Response(400, json={"error": "model does not support thinking (think)"})
        return {"value": 2}

    assert ask(FakeOllama({"*": reply}).client()).ok and seen == [True, False]


def test_oversized_lists_violate_the_ranking_schema():
    item = {"root_cause": "x", "confidence": 0.5, "reasoning": "r"}
    fake = FakeOllama({"rank": {"ranking": [item] * 9}})
    result = fake.client().chat_json(system="s", user="u", schema=LLMRanking, purpose="rank")
    assert not result.ok and "ranking" in result.detail


def test_embed_returns_none_when_unavailable():
    assert FakeOllama(reachable=False).client().embed(["a"]) is None
