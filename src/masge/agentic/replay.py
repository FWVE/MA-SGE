"""Replay recorded native provider calls without network access."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from masge.agentic.deepseek_strict import ENDPOINT, parse_tool_output, strict_request
from masge.agentic.models import ExecutionAction, ExecutionLimits
from masge.agentic.openrouter import ENDPOINT as OPENROUTER_ENDPOINT
from masge.agentic.openrouter import MODEL as OPENROUTER_MODEL
from masge.agentic.openrouter import parse_structured_output, structured_request, validate_response
from masge.agentic.provider import (
    DEEPSEEK_MODELS,
    parse_output,
    request_input,
    validate_model_effort,
)

T = TypeVar("T", bound=BaseModel)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class ReplayProvider:
    def __init__(self, path: Path, model: str, reasoning_effort: str = "high",
                 limits: ExecutionLimits | None = None) -> None:
        validate_model_effort(model, reasoning_effort)
        self.directories = sorted((path / "calls").iterdir())
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.calls = self.tokens = 0
        self.limits = limits or ExecutionLimits()

    async def request(self, role: str, instructions: str, messages: list[dict[str, Any]],
                      output_type: type[T]) -> T:
        require(self.calls < len(self.directories), "replay requested an unrecorded provider call")
        directory = self.directories[self.calls]
        request, response, accounting = (read(directory / name) for name in
                                         ("request.json", "response.json", "accounting.json"))
        require(directory.name.endswith("_" + role), "provider role/order mismatch")
        require(request["model"] == response["model"] == self.model, "actual native model mismatch")
        require(request["reasoning_effort"] == self.reasoning_effort and request["store"] is False, "request settings mismatch")
        deepseek = self.model in DEEPSEEK_MODELS
        openrouter = self.model == OPENROUTER_MODEL
        if openrouter:
            validate_response(response)
        if not deepseek and not openrouter:
            require(response["reasoning"]["effort"] == self.reasoning_effort, "response reasoning effort mismatch")
        require(accounting["model"] == self.model and accounting["reasoning_effort"] == self.reasoning_effort,
                "accounting model/effort mismatch")
        if self.reasoning_effort == "none":
            details = "completion_tokens_details" if deepseek else "output_tokens_details"
            reasoning_output = (any(c.get("message", {}).get("reasoning_content") for c in response.get("choices", []))
                                if deepseek else any(item.get("type") == "reasoning" for item in response["output"]))
            reasoning_tokens = ((response.get("usage") or {}).get(details) or {}).get("reasoning_tokens")
            valid_counter = reasoning_tokens in (None, 0) if deepseek else reasoning_tokens == 0
            require(valid_counter and not reasoning_output,
                    "non-thinking response contains reasoning or incompatible usage")
        require(accounting["status"] == "ok", "provider failure")
        if not deepseek and not openrouter:
            require(response["status"] == "completed", "provider failure")
        expected_format = ("function_tool" if issubclass(output_type, ExecutionAction) else "strict_tool") if deepseek else "json_schema"
        require(request["output_format"] == expected_format, "provider output format mismatch")
        require(request["input"] == request_input(instructions, messages),
                "recorded model input differs from public-only replay")
        require(request["output_schema"] == output_type.model_json_schema(), "output schema mismatch")
        if openrouter:
            wire = structured_request(self.model, self.reasoning_effort, request["input"],
                                      output_type, self.limits.max_output_tokens)
            require(request["endpoint"] == OPENROUTER_ENDPOINT and request["provider_request"] == wire,
                    "pinned OpenRouter request/history differs from replay")
            require(accounting["actual_provider"] == response["provider"]
                    and accounting["actual_service_tier"] == response["service_tier"], "routing accounting mismatch")
            parsed = parse_structured_output(response, output_type)
        elif deepseek:
            wire = strict_request(self.model, self.reasoning_effort, request["input"],
                                  output_type, self.limits.max_output_tokens)
            require(request["endpoint"] == ENDPOINT and request["provider_request"] == wire,
                    "strict provider request/history differs from replay")
            parsed = parse_tool_output(response, output_type)
        else:
            parsed = parse_output(response, output_type)
        require(response["usage"] == accounting["usage"], "provider accounting mismatch")
        self.calls += 1
        self.tokens += response["usage"]["total_tokens"]
        return parsed
