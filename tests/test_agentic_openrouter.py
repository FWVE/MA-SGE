"""OpenRouter routing, output and replay boundaries; no external calls."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI

from masge.agentic.batch import BatchGuard
from masge.agentic.models import Action, ExecutionLimits
from masge.agentic.openrouter import (
    ENDPOINT,
    MODEL,
    parse_structured_output,
    routing,
    structured_request,
)
from masge.agentic.provider import (
    ModelOutputError,
    NativeProvider,
    ProviderConfigurationError,
    validate_model_effort,
    write_json,
)


def decision() -> Action:
    return Action(action="view", plan=None, unit_ids=[], offset=0, lookup_text=None,
                  reason="Read current frontier", query_evidence=None, view_kind="frontier",
                  view_unit_id=None, memory_updates=[])


def reply() -> dict[str, Any]:
    return {"id": "gen-fixture", "object": "chat.completion", "created": 0, "model": MODEL,
            "provider": "Google AI Studio", "service_tier": "flex",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": decision().model_dump_json(), "reasoning": "Review state"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30,
                      "completion_tokens_details": {"reasoning_tokens": 4}, "cost": 0.00002}}


def test_openrouter_client_uses_only_designated_credential_and_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-key")
    monkeypatch.setattr("openai.AsyncOpenAI", lambda **kwargs: captured.update(kwargs))
    NativeProvider(MODEL, ExecutionLimits(), tmp_path, reasoning_effort="low")
    assert captured == {"api_key": "test-openrouter-key", "base_url": ENDPOINT,
                        "max_retries": 0, "timeout": 180}
    for effort in ("none", "high"):
        with pytest.raises(ValueError):
            validate_model_effort(MODEL, effort)
    with pytest.raises(ValueError):
        validate_model_effort("gpt-5.6-luna", "low")
    monkeypatch.delenv("OPENROUTER_API_KEY")
    captured.clear()
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        NativeProvider(MODEL, ExecutionLimits(), tmp_path, reasoning_effort="low")
    assert not captured


@pytest.mark.asyncio
async def test_openrouter_wire_replay_and_route_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from masge.agentic.replay import ReplayProvider

    expected = structured_request(MODEL, "low", [{"role": "system", "content": "fixture"}], Action, 10_000)

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == ENDPOINT + "/chat/completions"
        assert json.loads(request.content) == expected
        assert expected["provider"] == {"order": ["google-ai-studio/flex"], "only": ["google-ai-studio/flex"],
                                        "allow_fallbacks": False, "require_parameters": True}
        assert expected["reasoning"] == {"effort": "low"} and expected["service_tier"] == "flex"
        assert expected["response_format"]["json_schema"]["strict"] is True
        assert expected["response_format"]["json_schema"]["schema"]["$defs"]["Formula"]
        assert expected["response_format"]["json_schema"]["schema"]["$defs"]["Formula"]["type"] == "string"
        assert "children" in Action.model_json_schema()["$defs"]["Formula"]["required"]
        return httpx.Response(200, json=reply())

    provider = make_provider(tmp_path, respond)
    try:
        assert await provider.request("coordinator", "fixture", [], Action) == decision()
    finally:
        await provider.close()
    assert provider.calls == 1 and provider.tokens == 30
    assert provider.records[0]["usage"]["cost"] == 0.00002
    assert await ReplayProvider(tmp_path, MODEL, "low").request("coordinator", "fixture", [], Action) == decision()
    path = tmp_path / "calls/0001_coordinator/request.json"
    altered = json.loads(path.read_text(encoding="utf-8"))
    altered["provider_request"]["provider"]["allow_fallbacks"] = True
    write_json(path, altered)
    with pytest.raises(ValueError, match="pinned OpenRouter request"):
        await ReplayProvider(tmp_path, MODEL, "low").request("coordinator", "fixture", [], Action)
    assert routing()["allow_fallbacks"] is False


def make_provider(path: Path, respond: Any) -> NativeProvider:
    provider = NativeProvider.__new__(NativeProvider)
    provider.client = AsyncOpenAI(api_key="test-only", base_url=ENDPOINT, max_retries=0,
                                 http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    provider.model, provider.limits, provider.output_dir = MODEL, ExecutionLimits(), path
    provider.calls, provider.tokens, provider.records = 0, 0, []
    provider.reasoning_effort, provider.guard = "low", BatchGuard()
    return provider


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["model", "provider", "service_tier", "reasoning", "content"])
async def test_openrouter_route_fault_stops_batch_but_bad_content_is_query_local(tmp_path: Path, fault: str) -> None:
    raw = reply()
    if fault == "content":
        raw["choices"][0]["message"]["content"] = '{"action":"unsupported"}'
    else:
        raw[fault] = {"effort": "high"} if fault == "reasoning" else "different"
    provider = make_provider(tmp_path, lambda _: httpx.Response(200, json=raw))
    try:
        with pytest.raises(ModelOutputError if fault == "content" else ProviderConfigurationError):
            await provider.request("coordinator", "fixture", [], Action)
    finally:
        await provider.close()
    assert provider.calls == 1 and provider.records[0]["retry_count"] == 0
    assert provider.guard is not None
    assert (provider.guard.failure is None) == (fault == "content")
    assert (tmp_path / "calls/0001_coordinator/response.json").is_file()


def test_formula_serialization_preserves_depth_and_does_not_repair_missing_fields() -> None:
    from masge.agentic.models import Plan

    formula: dict[str, Any] = {"op": "condition", "condition_id": "c1", "children": [], "k": None}
    for _ in range(16):
        formula = {"op": "all", "condition_id": None, "children": [formula], "k": None}
    plan = Plan(operation="pattern", roles=[], relations=[], anchor_role="u", fixed_bindings=[],
                symmetric_role_groups=[], path_target_role=None, hop_count=None, direction="directed",
                core_mode=None, core_k=None, conditions=[], formula=formula, output_all_assignments=False)
    expected = decision().model_copy(update={"action": "plan", "plan": plan})
    wire = expected.model_dump(mode="json")
    wire["plan"]["formula"] = json.dumps(formula)
    raw = reply()
    raw["choices"][0]["message"]["content"] = json.dumps(wire)
    assert parse_structured_output(raw, Action) == expected
    wire["plan"]["formula"] = json.dumps({"op": "condition", "condition_id": "c1", "k": None})
    raw["choices"][0]["message"]["content"] = json.dumps(wire)
    with pytest.raises(ValueError, match="children"):
        parse_structured_output(raw, Action)
