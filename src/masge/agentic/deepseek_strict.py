"""DeepSeek Beta strict function output for independent agent requests."""
from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from masge.agentic.models import ExecutionAction

T = TypeVar("T", bound=BaseModel)
ENDPOINT = "https://api.deepseek.com/beta"
TOOL_NAME = "submit_result"


def strict_schema(output_type: type[BaseModel]) -> dict[str, Any]:
    """Keep the contract intact in DeepSeek's documented reference dialect.

    Beta rejects an anyOf branch consisting only of a $ref (missing type).
    Inline that branch, while retaining recursive references elsewhere.
    """
    original = output_type.model_json_schema()
    definitions = original.get("$defs", {})

    def visit(value: Any) -> Any:
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {}
        for key, item in value.items():
            if key == "$ref":
                if not item.startswith("#/$defs/"):
                    raise ValueError("strict contract requires local definition references")
                result[key] = item.replace("#/$defs/", "#/$def/", 1)
            elif key == "anyOf":
                result[key] = [visit(definitions[branch["$ref"].removeprefix("#/$defs/")])
                               if set(branch) == {"$ref"} else visit(branch) for branch in item]
            else:
                result["$def" if key == "$defs" else key] = visit(item)
        return result

    result = dict(visit(original))
    rendered = result.pop("$def", {})

    def references(value: Any) -> set[str]:
        if isinstance(value, list):
            return set().union(*(references(item) for item in value))
        if not isinstance(value, dict):
            return set()
        found = {value["$ref"].removeprefix("#/$def/")} if "$ref" in value else set()
        return found.union(*(references(item) for item in value.values()))

    # Inlining anyOf branches makes some definitions unreachable (notably Plan).
    # Keep the transitive closure, including recursive Formula references.
    needed, pending = set(), references(result)
    while pending:
        name = pending.pop()
        if name not in needed:
            needed.add(name)
            pending.update(references(rendered[name]) - needed)
    if needed:
        result["$def"] = {name: definition for name, definition in rendered.items() if name in needed}
    return result


def tool_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    if len(choices) != 1 or choices[0].get("finish_reason") != "tool_calls":
        raise ValueError("strict output requires one completed tool-call choice")
    message = choices[0]["message"]
    calls = message.get("tool_calls") or []
    if (len(calls) != 1 or calls[0].get("type") != "function" or not calls[0].get("id")
            or calls[0].get("function", {}).get("name") != TOOL_NAME):
        raise ValueError("strict output requires exactly one submit_result function call")
    return dict(message)


def parse_tool_output(response: dict[str, Any], output_type: type[T]) -> T:
    message = tool_message(response)
    return output_type.model_validate_json(message["tool_calls"][0]["function"]["arguments"])


def strict_request(model: str, effort: str, inputs: list[dict[str, Any]],
                   output_type: type[BaseModel], max_output_tokens: int) -> dict[str, Any]:
    """Each decision is a new request; raw output history is audit data only."""
    if any(item["role"] not in {"system", "user"} for item in inputs):
        raise RuntimeError("stateless requests cannot include assistant/tool history")
    result = {"model": model, "messages": [{"role": item["role"], "content": item["content"]} for item in inputs],
              "tools": [{"type": "function", "function": {
                  "name": TOOL_NAME, "description": "Submit the single structured result for this agent turn.",
                  # Server strict mode requires every declared property, including
                  # unused action slots. Local typed validation still applies.
                  "strict": not issubclass(output_type, ExecutionAction), "parameters": strict_schema(output_type)}}],
              "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
              "max_tokens": max_output_tokens, "stream": False, "store": False,
              "thinking": {"type": "disabled" if effort == "none" else "enabled"}}
    if effort != "none":
        result["reasoning_effort"] = effort
    return result
