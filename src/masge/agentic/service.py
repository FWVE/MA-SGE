"""Persistent Coordinator decisions with fresh Inspector calls and exact feedback."""
from __future__ import annotations

import asyncio
import itertools
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from masge.agentic.context import (
    compact_feedback,
    coordinator_messages,
    history_entries,
    inspector_message,
    json_text,
    memory_records,
    page,
    update_memory,
)
from masge.agentic.graph import Graph, GraphLimitError
from masge.agentic.models import Action, ExecutionAction, ExecutionLimits, InspectionReport, Plan
from masge.agentic.prompts import GUIDES, coordinator_prompt, inspector_prompt
from masge.agentic.provider import ModelOutputError, Provider, QueryBudgetError, write_json
from masge.agentic.workspace import (
    QuoteValidationError,
    ReceiptFormatError,
    Workspace,
    digest,
    validate_plan,
)
from masge.logging_utils import redact_text


class InspectionStageError(ValueError):
    """A rejected unit reply with the exact progress of its ordered stage."""

    def __init__(self, error: Exception, progress: dict[str, Any]) -> None:
        super().__init__(str(error))
        self.progress = progress


class Solver:
    runtime_name = "core"

    def __init__(self, graph: Graph, provider: Provider, limits: ExecutionLimits,
                 output_dir: Path | None = None) -> None:
        self.graph, self.provider, self.limits = graph, provider, limits
        self.output_dir = output_dir
        self.workspace: Workspace | None = None
        self.events: list[dict[str, Any]] = []
        self.issues: list[dict[str, Any]] = []
        self.revisions = 0
        self._resolved_ids: set[int] = set()
        self._lookups: dict[str, dict[str, Any]] = {}
        self._memory: dict[str, Any] = {}
        self._plan_reason: str | None = None
        self._last_action: dict[str, Any] | None = None
        self._last_inspection: dict[str, Any] | None = None
        self._rejections: dict[str, dict[str, Any]] = {}
        self._revision_request: dict[str, str] | None = None
        self.context_routes: list[dict[str, Any]] = []
        self._block_state: str | None = None
        self._block_reads: dict[str, dict[str, Any]] = {}
        self._block_errors: dict[str, dict[str, Any]] = {}
        self._block_repeats = 0

    def _coordinator_input(self, public: dict[str, Any], current: dict[str, Any]) -> tuple[
        list[dict[str, Any]], str, list[str], type[BaseModel]
    ]:
        context = coordinator_messages(public, current,
            self.workspace.plan if self.workspace else None, self.workspace.version if self.workspace else None,
            self._lookups, self.issues, self._memory, self._plan_reason, self._last_action, self._last_inspection)
        planning = self.workspace is None or self._revision_request is not None
        instructions, modules = coordinator_prompt(planning=planning,
            revision=self._revision_request is not None, repair="action_error" in current,
            blocked=bool((current.get("workspace") or {}).get("blocked_items")))
        return context, instructions, modules, Action if planning else ExecutionAction

    def _action(self, reply: BaseModel) -> Action:
        return reply.command() if isinstance(reply, ExecutionAction) else Action.model_validate(reply.model_dump(mode="json"))

    def _build_workspace(self, plan: Plan, query: str, anchors: list[int], version: int) -> Workspace:
        validate_plan(plan, query, anchors, self.graph, self._resolved_ids)
        return Workspace(self.graph, plan, self.graph.enumerate(plan, anchors, self.limits), version, self.limits)

    def _guide(self, topic: str) -> str:
        return GUIDES[topic]

    def _inspection_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        return inspector_message(payload)

    def _block_key(self) -> str | None:
        if self.workspace is None or not self.workspace.state()["blocked_items"]:
            return None
        return digest([self.workspace.version, self.workspace.values, self.workspace.state()["blocked_items"],
                       list(self.workspace.clarifications)])

    def memory_snapshot(self) -> dict[str, Any]:
        return {"plan_version": self.workspace.version if self.workspace else None,
                "accepted_plan_reason": self._plan_reason, "notes": self._memory,
                "last_action": self._last_action, "last_inspection": self._last_inspection,
                "inspection_rejections": self._rejections, "revision_request": self._revision_request,
                **({"block_diagnostics": {"reads": list(self._block_reads.values()),
                     "failed_actions": list(self._block_errors.values()), "repeated_actions": self._block_repeats}}
                   if self._block_state else {})}

    def _save(self) -> None:
        if self.output_dir is not None:
            write_json(self.output_dir / "events.json", self.events)
            write_json(self.output_dir / "reasoning_memory.json", self.memory_snapshot())
            write_json(self.output_dir / "context_routes.json", self.context_routes)
            if self.workspace is not None:
                write_json(self.output_dir / "workspace.json", self.workspace.snapshot())

    async def run(self, query: str, anchors: list[int], schema: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        public: dict[str, Any] = {"query": query, "public_anchor_node_ids": anchors,
                                  "graph_schema": schema, "limits": self.limits.model_dump(mode="json")}
        observation: dict[str, Any] = {"instruction": "Interpret the public question and request a graph operation."}
        result: dict[str, Any]
        try:
            public["public_anchor_kinds"] = sorted({self.graph.kind(n) for n in anchors if 0 <= n < self.graph.node_count})
            if self.output_dir is not None:
                write_json(self.output_dir / "public_input.json", {**public, **observation})
            async with asyncio.timeout(self.limits.query_seconds):
                result = await self._loop(query, anchors, observation, public)
        except (Exception, asyncio.CancelledError) as error:
            # This application boundary settles cancelled attempts in the persistent ledger.
            # No further model request or answer synthesis follows cancellation.
            result = {"status": "unresolved", "answers": None, "reason": redact_text(str(error))[:2000],
                      "error_type": type(error).__name__,
                      "failure_scope": "cancelled" if isinstance(error, asyncio.CancelledError) else
                      "query" if isinstance(error, (ValueError, ModelOutputError, QueryBudgetError,
                                                   GraphLimitError, TimeoutError)) else "batch"}
            self.issues.append({"stage": "execution", "error_type": type(error).__name__,
                                "error": result["reason"]})
        result["runtime"] = self.runtime_name
        self._save()
        result["audit"] = {"protocol_valid": not self.issues,
                            "handoff_eligible": not self.issues and result["status"] == "complete",
                            "issues": self.issues, "provider_calls": self.provider.calls,
                            "provider_tokens": self.provider.tokens, "seconds": time.monotonic() - started,
                            "plan_revisions": self.revisions,
                            "gold_or_private_candidates_loaded": False}
        if self.output_dir is not None:
            write_json(self.output_dir / "result.json", result)
        return result

    async def _loop(self, query: str, anchors: list[int],
                    observation: dict[str, Any], public: dict[str, Any]) -> dict[str, Any]:
        consecutive_action_errors = 0
        for turn in itertools.count():
            if self.limits.max_actions is not None and turn >= self.limits.max_actions:
                break
            if self.workspace is not None and self.workspace.state()["completion_ready"]:
                return self.workspace.answer()
            observation["budget"] = {"calls_used": self.provider.calls, "tokens_used": self.provider.tokens,
                                      "max_calls": self.limits.max_calls, "max_tokens": self.limits.max_tokens}
            block_key = self._block_key()
            if block_key != self._block_state:
                self._block_state = block_key
                self._block_reads.clear()
                self._block_errors.clear()
                self._block_repeats = 0
            current = dict(observation)
            if block_key and (self._block_reads or self._block_errors):
                current["block_diagnostics"] = {
                    "reads": list(self._block_reads.values()), "failed_actions": list(self._block_errors.values())}
            if self._revision_request is not None:
                current["revision_request"] = self._revision_request
            if self.workspace is not None and current.get("workspace") is None:
                current["workspace"] = self.workspace.observe()
            context, instructions, modules, output_type = self._coordinator_input(public, current)
            self.context_routes.append({"call": self.provider.calls + 1, "role": "coordinator", "modules": modules})
            action: Action | None = None
            try:
                reply = await self.provider.request("coordinator", instructions, context, output_type)
                action = self._action(reply)
                self._last_action = action.model_dump(mode="json")
                self.events.append({"role": "coordinator", "action": action.model_dump(mode="json")})
                observation = await self._dispatch(action, query, anchors)
                consecutive_action_errors = 0
            except (ValueError, ModelOutputError) as error:
                # Inspector format repair is query-local and independently bounded.
                # Its exhaustion is feedback for the Coordinator, not a bad action.
                issue: dict[str, Any] = {"stage": "action_validation" if action else "coordinator_output",
                                         "error": str(error), "action": action.action if action else None}
                if action is None:
                    self._last_action = None
                self.issues.append(issue)
                self.events.append(issue)
                cause = error.__cause__ if isinstance(error, ModelOutputError) else error
                detail = "; ".join(f"{'.'.join(map(str, item['loc'])) or 'JSON'}: {item['msg']}"
                                  for item in cause.errors(include_input=False, include_url=False)) if isinstance(cause, ValidationError) else str(error)
                observation = {"action_error": detail[:600], "prior_valid_receipts_preserved": True,
                               "workspace": self.workspace.observe() if self.workspace else None}
                if isinstance(error, InspectionStageError):
                    consecutive_action_errors = 0
                    issue.update(stage="inspection", inspection_progress=error.progress)
                    observation["inspection_progress"] = error.progress
                else:
                    consecutive_action_errors += 1
                if consecutive_action_errors > 2:
                    raise ValueError("bounded action-repair limit exhausted") from error
            self.events.append({"role": "runtime", "feedback": observation})
            if "inspection_progress" in observation:
                compact = compact_feedback(observation)
                self._last_inspection = {"version": observation["inspection_progress"]["version"],
                                         "reason": action.reason if action else "",
                                         **{k: compact[k] for k in ("inspection_progress", "newly_confirmed", "newly_excluded") if k in compact}}
            if block_key and block_key == self._block_key() and self.workspace is not None:
                detail = observation.get("detail", {})
                readable = detail.get("kind") in {"unit", "dependencies", "receipt", "completion"}
                has_items = any(isinstance(v, dict) and v.get("items") for v in detail.values())
                read_key = digest(detail)
                fresh_read = readable and has_items and read_key not in self._block_reads
                if fresh_read:
                    self._block_reads[read_key] = detail
                if "action_error" in observation:
                    failed = {"action": action.action if action else None,
                              "unit_ids": action.unit_ids if action else [],
                              "reason": action.reason if action else "",
                              "error": observation["action_error"]}
                    self._block_errors[digest(failed)] = failed
                state = self.workspace.state()
                live = set(state["live_atoms"])
                only_blocked = all(u in state["blocked_items"] or not live.intersection(unit["atom_ids"])
                                   for u, unit in self.workspace.units.items())
                self._block_repeats = (self._block_repeats + 1 if only_blocked and not fresh_read else 0)
            self._save()
            if observation.get("unresolved"):
                return {"status": "unresolved", "answers": None, "reason": observation["unresolved"]}
            if self._block_repeats >= 6:
                return {"status": "unresolved", "answers": None,
                        "reason": "Blocked work made no progress in six actions; prior judgments preserved"}
        return {"status": "unresolved", "answers": None, "reason": "Coordinator action budget exhausted"}

    async def _dispatch(self, action: Action, query: str, anchors: list[int]) -> dict[str, Any]:
        reinterpreting = action.action in {"plan", "revise"}
        version = self.workspace.version if self.workspace else None
        if reinterpreting:
            version = (version or 0) + 1
        memory = {} if reinterpreting else self._memory
        memory_error = None
        try:
            memory = update_memory(memory, action.memory_updates, version,
                                   set(self.workspace.units) if self.workspace and not reinterpreting else set())
        except ValueError as error:
            memory_error = str(error)
        feedback = await self._execute_action(action, query, anchors)
        self._memory = memory
        if memory_error is not None:
            issue = {"stage": "memory_validation", "action": action.action, "error": memory_error}
            self.issues.append(issue)
            self.events.append(issue)
            feedback["memory_update_error"] = "Notes not saved; requested action executed. " + memory_error[:300]
        if action.action in {"inspect", "finish"}:
            self._revision_request = None
        return feedback

    async def _execute_action(self, action: Action, query: str, anchors: list[int]) -> dict[str, Any]:
        if action.action not in {"plan", "revise"} and action.plan is not None:
            raise ValueError("plan payload belongs only to plan/revise")
        if action.action != "inspect" and action.unit_ids:
            raise ValueError("unit IDs belong only to inspect")
        if action.action != "lookup" and action.lookup_text is not None:
            raise ValueError("lookup_text belongs only to lookup")
        if action.action not in {"view"} and action.offset != 0:
            raise ValueError("offset belongs only to view")
        if action.action not in {"revise", "request_revision"} and action.query_evidence is not None:
            raise ValueError("query_evidence belongs only to request_revision/revise")
        if action.action != "view" and (action.view_kind != "frontier" or action.view_unit_id is not None):
            raise ValueError("view parameters belong only to view")
        if action.action in {"plan", "revise", "request_revision"} and not action.reason.strip():
            raise ValueError("action requires a brief decision reason")
        if action.action == "view" and action.view_kind == "guide":
            if action.offset != 0 or action.view_unit_id not in GUIDES:
                raise ValueError("guide needs offset=0 and topic planning, memory or views in view_unit_id")
            return {"detail": {"kind": "guide", "topic": action.view_unit_id,
                                "instructions": self._guide(action.view_unit_id)}}
        if action.action == "view" and action.view_kind == "plan":
            if self.workspace is None or action.offset != 0 or action.view_unit_id is not None:
                raise ValueError("plan view needs an accepted Plan, offset=0 and no unit ID")
            return {"detail": {"kind": "plan", "accepted_plan": self.workspace.plan.model_dump(mode="json"),
                                "reason": self._plan_reason}}
        if action.action == "request_revision":
            if self.workspace is None or self.revisions >= self.limits.max_revisions:
                raise ValueError("revision unavailable or exhausted")
            if not action.query_evidence or action.query_evidence not in query:
                raise ValueError("revision needs exact public-query evidence of the interpretation issue")
            self._revision_request = {"reason": action.reason, "query_evidence": action.query_evidence}
            return {"revision_requested": True, "workspace": self.workspace.observe()}
        if action.action == "lookup":
            if self.workspace is not None and self._revision_request is None:
                raise ValueError("name lookup is available only during planning or a pending interpretation revision")
            if not action.lookup_text or action.lookup_text.casefold() not in query.casefold():
                raise ValueError("lookup must use a literal name span from the public question")
            matches = self.graph.lookup(action.lookup_text)
            self._resolved_ids.update(item["node_id"] for item in matches["matches"])
            self._lookups[action.lookup_text] = matches
            return matches
        if action.action in {"plan", "revise"}:
            if action.plan is None:
                raise ValueError("plan/revise needs a complete replacement plan")
            if action.action == "plan" and self.workspace is not None:
                raise ValueError("accepted plans are immutable; order changes use inspect, not plan")
            if action.action == "revise":
                if self.workspace is None or self.revisions >= self.limits.max_revisions:
                    raise ValueError("revision unavailable or exhausted")
                if not action.query_evidence or action.query_evidence not in query:
                    raise ValueError("revision needs exact public-query evidence of the interpretation issue")
                if action.plan == self.workspace.plan:
                    raise ValueError("inspection order changes do not justify a new plan version")
            version = 1 if self.workspace is None else self.workspace.version + 1
            replacement = self._build_workspace(action.plan, query, anchors, version)
            if self.workspace is not None:
                if self.output_dir is not None:
                    write_json(self.output_dir / f"superseded_workspace_v{self.workspace.version}.json",
                               self.workspace.snapshot())
                self.revisions += 1
            self.workspace = replacement
            self._last_inspection = None
            self._rejections = {}
            self._revision_request = None
            self._plan_reason = action.reason
            return {"plan_accepted": True, "version": version, "workspace": replacement.observe(),
                    "all_public_anchors_enumerated": list(anchors), "previous_receipts_reused": False}
        if action.action == "view" and action.view_kind in {"history", "memory"}:
            if action.view_unit_id is not None:
                raise ValueError("history/memory view does not take a unit ID")
            entries = (history_entries(self.events) if action.view_kind == "history" else
                       memory_records(self._memory))
            return {"workspace": self.workspace.observe() if self.workspace else None,
                    "detail": {"kind": action.view_kind, **page(entries, action.offset, self.limits.page_size)}}
        if self.workspace is None:
            raise ValueError("first request a graph operation with plan")
        workspace = self.workspace
        if action.action == "view":
            if action.view_kind == "frontier":
                if action.view_unit_id is not None:
                    raise ValueError("frontier view does not take a unit ID")
                return {"workspace": workspace.observe(action.offset)}
            return {"workspace": workspace.observe(),
                    "detail": workspace.detail(action.view_kind, action.view_unit_id, action.offset)}
        if action.action == "finish":
            if not workspace.state()["completion_ready"]:
                if workspace.state()["blocked_items"] and action.reason.strip():
                    return {"unresolved": action.reason, "workspace": workspace.observe()}
                raise ValueError("finish refused: membership or output obligations remain unresolved")
            return {"completion_ready": True}
        if not action.unit_ids or len(set(action.unit_ids)) != len(action.unit_ids):
            raise ValueError("inspect needs a nonempty ordered list of distinct unit IDs")
        if any(unit_id not in workspace.units for unit_id in action.unit_ids):
            raise ValueError("inspection contains an unknown or stale-version unit")
        if any(unit_id in workspace.blocked for unit_id in action.unit_ids):
            if len(action.unit_ids) != 1 or not action.reason.strip():
                raise ValueError("cannot rewrite blocked evidence: inspect ONE blocked unit with clarification in reason")
        before = workspace.state()
        unresolved_before = {a for a, value in workspace.values.items() if value is None}
        progress: dict[str, Any] = {"version": workspace.version, "requested_unit_ids": list(action.unit_ids),
                                    "accepted_unit_ids": [], "blocked_unit_ids": [],
                                    "skipped_units_with_reasons": [], "failed_unit_id": None,
                                    "not_started_unit_ids": list(action.unit_ids)}
        for index, unit_id in enumerate(action.unit_ids):
            progress["not_started_unit_ids"] = action.unit_ids[index + 1:]
            recovering = unit_id in workspace.blocked
            if unit_id in workspace.receipts and not recovering:
                progress["skipped_units_with_reasons"].append({"unit_id": unit_id, "reason": "accepted receipt reused; no new Inspector call"})
                continue
            live = set(workspace.state()["live_atoms"])
            if not live & set(workspace.units[unit_id]["atom_ids"]):
                progress["skipped_units_with_reasons"].append({"unit_id": unit_id, "reason": "no remaining membership/output dependency"})
                continue
            base_payload = (workspace.begin_clarification(unit_id, action.reason) if recovering
                            else workspace.inspection_input(unit_id))
            if recovering:
                self._rejections.pop(unit_id, None)
            for attempt in range(3):
                try:
                    payload = dict(base_payload)
                    if unit_id in self._rejections:
                        payload["prior_rejection"] = self._rejections[unit_id]
                    visible = self._inspection_message(payload)
                    instructions, modules = inspector_prompt(visible)
                    self.context_routes.append({"call": self.provider.calls + 1, "role": "inspector", "modules": modules})
                    report = await self.provider.request("inspector", instructions,
                                                        [{"role": "user", "content": json_text(visible)}],
                                                        InspectionReport)
                    reply = report.receipt(payload)
                    self.events.append({"role": "inspector", "input": payload, "output": reply.model_dump(mode="json")})
                    workspace.accept(reply, clarified=True) if recovering else workspace.accept(reply)
                    self._rejections.pop(unit_id, None)
                    break
                except (ReceiptFormatError, ModelOutputError) as error:
                    self._rejections[unit_id] = {"unit_id": unit_id, "failed_attempts": attempt + 1,
                        "error": str(error), **(error.details if isinstance(error, QuoteValidationError) else {})}
                    issue = {"stage": "inspector_output", "unit_id": unit_id,
                             "attempt": attempt + 1, "error": str(error)}
                    self.issues.append(issue)
                    self.events.append(issue)
                    if attempt == 2:
                        workspace.blocked[unit_id] = "Inspector output repair exhausted; no judgment accepted"
                        progress["failed_unit_id"] = unit_id
                        progress["blocked_unit_ids"].append(unit_id)
                        self._save()
                        raise InspectionStageError(error, progress) from error
                    self._save()
            if unit_id in workspace.receipts or (recovering and workspace.clarifications[unit_id].get("output", {}).get("judgments")):
                progress["accepted_unit_ids"].append(unit_id)
            if unit_id in workspace.blocked:
                progress["blocked_unit_ids"].append(unit_id)
            self._save()
        after = workspace.state()
        return {"newly_confirmed": sorted(set(after["certain"]) - set(before["certain"])),
                "newly_excluded": sorted(set(after["excluded"]) - set(before["excluded"])),
                "newly_resolved_conditions": sorted(a for a in unresolved_before if workspace.values[a] is not None),
                "inspection_progress": progress, "workspace": workspace.observe(),
                "output_obligations": after["output_obligations"], "blocked_items": after["blocked_items"],
                "completion_ready": after["completion_ready"]}
