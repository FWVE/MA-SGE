"""OpenAI structured responses and DeepSeek strict tools with shared query budgets."""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from masge.agentic.batch import BatchGuard
from masge.agentic.deepseek_strict import ENDPOINT, parse_tool_output, strict_request
from masge.agentic.models import ExecutionAction, ExecutionLimits
from masge.agentic.openrouter import ENDPOINT as OPENROUTER_ENDPOINT
from masge.agentic.openrouter import MODEL as OPENROUTER_MODEL
from masge.agentic.openrouter import parse_structured_output, structured_request, validate_response
from masge.logging_utils import redact_text

T = TypeVar("T", bound=BaseModel)
DEEPSEEK_MODELS = ("deepseek-v4-flash", "deepseek-v4-pro")
SUPPORTED_MODELS = ("gpt-5.6-terra", "gpt-5.6-luna", *DEEPSEEK_MODELS, OPENROUTER_MODEL)


class ModelOutputError(RuntimeError):
    """Unusable model content fails this query without cancelling its peers."""


class QueryBudgetError(RuntimeError):
    """A query exhausted its authorized provider budget."""


class ProviderConfigurationError(RuntimeError):
    """The returned model or thinking configuration violates the frozen request."""


def validate_model_effort(model: str, effort: str) -> None:
    allowed = {"low"} if model == OPENROUTER_MODEL else {"high", "none"}
    if model not in SUPPORTED_MODELS or effort not in allowed:
        raise ValueError("unsupported native model or reasoning effort")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_output(response: dict[str, Any], output_type: type[T]) -> T:
    """Parse terminal messages only; commentary cannot issue runtime actions."""
    texts = [item["text"] for message in response["output"] if message["type"] == "message"
             and message.get("phase") in (None, "final_answer")
             for item in message["content"] if item["type"] == "output_text"]
    if not texts:
        raise ValueError("provider returned no structured output")
    outputs = [output_type.model_validate_json(text) for text in texts]
    if any(value != outputs[0] for value in outputs[1:]):
        raise ValueError("provider returned conflicting structured decisions")
    return outputs[0]


class Provider(Protocol):
    calls: int
    tokens: int

    async def request(self, role: str, instructions: str, messages: list[dict[str, Any]],
                      output_type: type[T]) -> T: ...


def request_input(instructions: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Canonical agent input; output contracts belong to the provider schema."""
    if any(item["role"] != "user" for item in messages):
        raise RuntimeError("stateless agents require freshly assembled user context")
    return [{"role": "system", "content": instructions}, *messages]


class NativeProvider:
    def __init__(self, model: str, limits: ExecutionLimits, output_dir: Path,
                 guard: BatchGuard | None = None, reasoning_effort: str = "high") -> None:
        from dotenv import load_dotenv
        from openai import AsyncOpenAI

        validate_model_effort(model, reasoning_effort)
        load_dotenv(Path.cwd() / ".env", override=False)
        if model == OPENROUTER_MODEL:
            api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY is required")
            self.client: Any = AsyncOpenAI(api_key=api_key, base_url=OPENROUTER_ENDPOINT,
                                           max_retries=0, timeout=limits.provider_seconds)
        elif model in DEEPSEEK_MODELS:
            api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
            if not api_key:
                raise ValueError(f"DEEPSEEK_API_KEY is required for {model}")
            self.client = AsyncOpenAI(api_key=api_key, base_url=ENDPOINT,
                                           max_retries=0, timeout=limits.provider_seconds)
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required")
            self.client = AsyncOpenAI(api_key=api_key, base_url="https://api.openai.com/v1",
                                      max_retries=0, timeout=limits.provider_seconds)
        self.model, self.limits, self.output_dir = model, limits, output_dir
        self.reasoning_effort = reasoning_effort
        self.guard = guard
        self.calls = 0
        self.tokens = 0
        self.records: list[dict[str, Any]] = []

    async def close(self) -> None:
        await self.client.close()

    async def request(self, role: str, instructions: str, messages: list[dict[str, Any]],
                      output_type: type[T]) -> T:
        from openai.lib._parsing._responses import type_to_text_format_param

        if self.guard is not None and self.guard.failure is not None:
            raise RuntimeError("batch stopped; no further provider request permitted")
        if ((self.limits.max_calls is not None and self.calls >= self.limits.max_calls)
                or (self.limits.max_tokens is not None and self.tokens >= self.limits.max_tokens)):
            raise QueryBudgetError("query provider budget exhausted")
        self.calls += 1
        call_dir = self.output_dir / "calls" / f"{self.calls:04d}_{role}"
        payload = request_input(instructions, messages)
        deepseek = self.model in DEEPSEEK_MODELS
        openrouter = self.model == OPENROUTER_MODEL
        chat = deepseek or openrouter
        wire = (strict_request(self.model, self.reasoning_effort, payload, output_type,
                               self.limits.max_output_tokens) if deepseek else None)
        if openrouter:
            wire = structured_request(self.model, self.reasoning_effort, payload, output_type,
                                      self.limits.max_output_tokens)
        response_format = type_to_text_format_param(output_type)
        sparse_action = issubclass(output_type, ExecutionAction)
        if sparse_action:
            response_format = {"type": "json_schema", "name": output_type.__name__,
                               "strict": False, "schema": output_type.model_json_schema()}
        request_record = {"model": self.model, "reasoning_effort": self.reasoning_effort,
                                               "store": False, "input": payload,
                                               "output_format": ("function_tool" if sparse_action else "strict_tool") if deepseek else response_format["type"],
                                               "output_schema": output_type.model_json_schema()}
        if wire is not None:
            request_record.update(endpoint=OPENROUTER_ENDPOINT if openrouter else ENDPOINT, provider_request=wire)
        write_json(call_dir / "request.json", request_record)
        start = time.monotonic()
        record: dict[str, Any] = {"call": self.calls, "role": role, "model": self.model,
                                  "reasoning_effort": self.reasoning_effort, "retry_count": 0}
        try:
            # Parse terminal output ourselves; never accept commentary as a decision.
            if openrouter and wire is not None:
                extra = {k: wire[k] for k in ("provider", "reasoning")}
                response = await self.client.chat.completions.create(
                    **{k: v for k, v in wire.items() if k not in extra}, extra_body=extra)
            elif wire is not None:
                response = await self.client.chat.completions.create(
                    **{k: v for k, v in wire.items() if k != "thinking"},
                    extra_body={"thinking": wire["thinking"]})
            else:
                response = await self.client.responses.create(
                    model=self.model, input=payload, reasoning={"effort": self.reasoning_effort},
                    text={"format": response_format},
                    max_output_tokens=self.limits.max_output_tokens, store=False,
                )
            raw = response.model_dump(mode="json", warnings=False)
            write_json(call_dir / "response.json", raw)
            usage = response.usage
            if usage is not None:
                self.tokens += int(usage.total_tokens)
                record["usage"] = usage.model_dump(mode="json")
            if raw.get("model") != self.model:
                raise ProviderConfigurationError("provider response model differs from the requested native model")
            if openrouter:
                try:
                    validate_response(raw)
                except ValueError as error:
                    raise ProviderConfigurationError(str(error)) from error
                record.update(actual_provider=raw["provider"], actual_service_tier=raw["service_tier"],
                              effort_verification="explicit frozen request; response echo not required")
            if not chat and (raw.get("reasoning") or {}).get("effort") != self.reasoning_effort:
                raise ProviderConfigurationError("provider response reasoning effort differs from the requested setting")
            if self.reasoning_effort == "none":
                details = "completion_tokens_details" if deepseek else "output_tokens_details"
                reasoning_tokens = ((raw.get("usage") or {}).get(details) or {}).get("reasoning_tokens")
                reasoning_output = (any(c.get("message", {}).get("reasoning_content") for c in raw.get("choices", []))
                                    if deepseek else any(item.get("type") == "reasoning" for item in raw.get("output", [])))
                # Chat may omit the reasoning-token breakdown entirely. Preserve
                # that unknown value; an absent counter is not evidence of thinking.
                valid_counter = reasoning_tokens in (None, 0) if deepseek else reasoning_tokens == 0
                if not valid_counter or reasoning_output:
                    raise ProviderConfigurationError("non-thinking response contains reasoning or incompatible usage")
            truncated = (any(c.get("finish_reason") == "length" for c in raw.get("choices", [])) if chat else
                         raw.get("status") == "incomplete" and (raw.get("incomplete_details") or {}).get("reason") == "max_output_tokens")
            if truncated:
                raise ModelOutputError("model output exhausted max_output_tokens before completion")
            if not chat and response.status != "completed":
                raise RuntimeError(f"provider did not complete the required output: {response.status}")
            try:
                if openrouter:
                    parsed = parse_structured_output(raw, output_type)
                else:
                    parsed = parse_tool_output(raw, output_type) if deepseek else parse_output(raw, output_type)
            except ValueError as error:
                raise ModelOutputError(str(error)) from error
            record["status"] = "ok"
            return parsed
        except asyncio.CancelledError as error:
            record.update(status="cancelled", error_type="CancelledError", error=str(error),
                          usage_unavailable=True)
            raise
        except Exception as error:
            query_failure = isinstance(error, ModelOutputError)
            record.update(status="failed", error_type=type(error).__name__, error=redact_text(str(error))[:2000],
                          failure_scope="query" if query_failure else "batch")
            if error.__cause__ is not None:
                record["cause_type"] = type(error.__cause__).__name__
            # Invalid content is never repaired or accepted here. Only transport,
            # configuration and unexpected provider failures cancel unrelated queries.
            if self.guard is not None and not query_failure:
                self.guard.stop(error)
                write_json(self.output_dir.parent / "batch_stop.json", {
                    **(self.guard.failure or {}), "trigger_run": self.output_dir.name,
                    "trigger_call": self.calls, "dispatch_stopped": True,
                    "peer_cancellation_requested": True})
            raise
        finally:
            record["seconds"] = time.monotonic() - start
            self.records.append(record)
            write_json(call_dir / "accounting.json", record)
            write_json(self.output_dir / "provider_usage.json", {"calls": self.calls, "tokens": self.tokens,
                                                                 "records": self.records})
