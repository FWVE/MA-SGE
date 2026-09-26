"""Pinned OpenRouter transport; the public agent contracts remain unchanged."""
from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel

from masge.agentic.models import ExecutionAction

T = TypeVar("T", bound=BaseModel)
MODEL = "google/gemini-3.8-flash"
ENDPOINT = "https://openrouter.ai/api/v1"
PROVIDER = "google-ai-studio/flex"


def routing() -> dict[str, Any]:
    return {"order": [PROVIDER], "only": [PROVIDER], "allow_fallbacks": False,
            "require_parameters": True}


def structured_request(model: str, effort: str, inputs: list[dict[str, Any]],
                       output_type: type[BaseModel], max_output_tokens: int) -> dict[str, Any]:
    from openai.lib._parsing._responses import type_to_text_format_param

    if model != MODEL or effort != "low":
        raise ValueError("OpenRouter campaign requires Gemini 3.8 Flash / low")
    if any(item["role"] not in {"system", "user"} for item in inputs):
        raise RuntimeError("stateless requests cannot include assistant/tool history")
    schema: dict[str, Any] = dict(type_to_text_format_param(output_type))
    if issubclass(output_type, ExecutionAction):
        schema.update(strict=False, schema=output_type.model_json_schema())
    schema.pop("type")
    # This route clips recursive structured-output arrays to zero elements.
    # Serialize only Formula as JSON text: no tree-depth limit or lost constraints.
    # The original recursive schema is supplied verbatim in the field description
    # and the decoded object must still pass the original Pydantic/runtime checks.
    definitions = schema["schema"].get("$defs", {})
    if "Formula" in definitions:
        formula_schema = {"$defs": {"Formula": definitions["Formula"]}, "$ref": "#/$defs/Formula"}
        definitions["Formula"] = {"type": "string", "description": (
            "JSON text encoding the full recursive Formula object, not a condition name. "
            "Decode this string to obtain the formula described in the agent instructions. "
            "All original fields are required, including children=[] on leaves; no depth truncation. "
            "Decoded JSON must satisfy this schema: " + json.dumps(formula_schema, ensure_ascii=False))}
    return {"model": model, "messages": inputs, "provider": routing(),
            "service_tier": "flex", "reasoning": {"effort": effort},
            "response_format": {"type": "json_schema", "json_schema": schema},
            "max_tokens": max_output_tokens, "stream": False, "store": False}


def validate_response(response: dict[str, Any]) -> None:
    """Check returned identity/tier; Chat does not promise an effort echo."""
    if response.get("model") != MODEL:
        raise ValueError("OpenRouter response model differs from the pinned model")
    if response.get("provider") != "Google AI Studio" or response.get("service_tier") != "flex":
        raise ValueError("OpenRouter response did not confirm Google AI Studio / flex")
    if response.get("reasoning") and response["reasoning"].get("effort") not in (None, "low"):
        raise ValueError("OpenRouter response reasoning effort contradicts low")


def parse_structured_output(response: dict[str, Any], output_type: type[T]) -> T:
    choices = response.get("choices") or []
    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise ValueError("structured output requires one completed choice")
    message = choices[0].get("message") or {}
    if message.get("tool_calls") or message.get("refusal") or not isinstance(message.get("content"), str):
        raise ValueError("structured output requires JSON content without tool calls or refusal")
    decoded = json.loads(message["content"])
    if isinstance(decoded, dict) and isinstance(decoded.get("plan"), dict):
        formula = decoded["plan"].get("formula")
        if not isinstance(formula, str):
            raise ValueError("OpenRouter Plan formula must be serialized JSON text")
        decoded["plan"]["formula"] = json.loads(formula)
    return output_type.model_validate(decoded)
