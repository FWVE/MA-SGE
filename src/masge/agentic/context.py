"""Stateless model requests assembled from explicit, query-local execution memory."""
from __future__ import annotations

import json
from typing import Any

from masge.agentic.models import MemoryUpdate, Plan

SUMMARY_ITEMS = 3
MEMORY_LIMITS = {"notes": 12, "updates_per_action": 4, "key_characters": 48, "text_characters": 600,
                 "total_text_characters": 1800, "unit_references_per_note": 4}


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def page(items: list[Any], offset: int, size: int) -> dict[str, Any]:
    if offset < 0 or size < 1:
        raise ValueError("page requires nonnegative offset and positive size")
    return {"count": len(items), "offset": offset, "items": items[offset:offset + size],
            "next_offset": offset + size if offset + size < len(items) else None}


def summary(items: list[Any]) -> dict[str, Any]:
    return {"count": len(items), "sample": items[:SUMMARY_ITEMS]}


def update_memory(notes: dict[str, Any], updates: list[MemoryUpdate], version: int | None,
                  known_units: set[str]) -> dict[str, Any]:
    """Apply explicit keyed edits atomically; never evict a note by age or salience."""
    if len(updates) > MEMORY_LIMITS["updates_per_action"] or len({u.key for u in updates}) != len(updates):
        raise ValueError("memory updates need distinct keys within the stated per-action limit")
    result = dict(notes)
    for update in updates:
        if not update.key.strip() or len(update.key) > MEMORY_LIMITS["key_characters"]:
            raise ValueError("memory key is empty or exceeds the stated limit")
        if update.text is None:
            if update.unit_ids:
                raise ValueError("removing a memory key takes no unit references")
            result.pop(update.key, None)
            continue
        if not update.text.strip() or len(update.text) > MEMORY_LIMITS["text_characters"]:
            raise ValueError("memory text is empty or exceeds the stated limit")
        if (len(update.unit_ids) > MEMORY_LIMITS["unit_references_per_note"]
                or len(set(update.unit_ids)) != len(update.unit_ids)
                or not set(update.unit_ids) <= known_units):
            raise ValueError("memory unit references must be distinct current-version units within the stated limit")
        result[update.key] = {"category": update.category, "text": update.text,
                              "unit_ids": list(update.unit_ids), "plan_version": version}
    if len(result) > MEMORY_LIMITS["notes"]:
        raise ValueError("working memory is full; explicitly merge/update or remove completed notes")
    if sum(len(note["text"]) for note in result.values()) > MEMORY_LIMITS["total_text_characters"]:
        raise ValueError("working memory text budget exceeded; explicitly consolidate notes using history/receipt references")
    return result


def compact_feedback(observation: dict[str, Any]) -> dict[str, Any]:
    """Omit duplicate/obsolete display fields, never mutate the saved observation."""
    result = dict(observation)
    original = observation.get("workspace")
    if original is not None:
        workspace = dict(original)
        # Conditions are supplied through semantic_task or the accepted Plan.
        workspace.pop("condition_definitions", None)
        workspace["units"] = [{
            "unit_id": unit["unit_id"],
            "subjects": summary(unit["subjects"]), "context": summary(unit["context"]),
            "condition_ids": list(dict.fromkeys(c["condition_id"] for c in unit["conditions"])),
            "dependent_candidates": summary(unit["dependent_candidates"]),
            "estimated_text_characters": unit["estimated_text_characters"],
            **({k: unit[k] for k in ("blocked", "reason", "bindings", "clarification_used") if k in unit}
               if unit.get("blocked") else {}),
        } for unit in original["units"]]
        workspace["output_obligations"] = summary(original["output_obligations"])
        workspace["blocked_items"] = summary([
            {"unit_id": unit_id, "reason": reason} for unit_id, reason in original["blocked_items"].items()
        ])
        result["workspace"] = workspace
        for key in ("output_obligations", "blocked_items", "completion_ready"):
            result.pop(key, None)  # Already present in the current workspace.
    for key in ("newly_confirmed", "newly_excluded", "newly_resolved_conditions"):
        if key in result:
            result[key] = summary(result[key])
    return result


def select_memory(notes: dict[str, Any], focus_units: set[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Keep global/open reasoning; defer unrelated local notes without deleting them."""
    selected, deferred = {}, []
    for key, note in notes.items():
        if (not note["unit_ids"] or note["category"] in {"open_question", "hypothesis"}
                or set(note["unit_ids"]) & focus_units):
            selected[key] = note
        else:
            deferred.append({"key": key, "category": note["category"], "unit_ids": note["unit_ids"]})
    return selected, deferred


def memory_records(notes: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose the writable note shape; version metadata stays in runtime storage."""
    return [{"key": key, **{field: note[field] for field in ("category", "text", "unit_ids")}}
            for key, note in notes.items()]


def action_summary(action: dict[str, Any]) -> dict[str, Any]:
    """A decision and its references, without repeating its accepted Plan or note edits."""
    return {key: value for key, value in action.items()
            if key not in {"plan", "memory_updates"} and value is not None and value != []
            and (action["action"] == "view" or key not in {"view_kind", "view_unit_id", "offset"})}


def coordinator_messages(public: dict[str, Any], observation: dict[str, Any],
                         plan: Plan | None, version: int | None,
                         lookups: dict[str, dict[str, Any]], issues: list[dict[str, Any]],
                         notes: dict[str, Any] | None = None, plan_reason: str | None = None,
                         last_action: dict[str, Any] | None = None,
                         last_inspection: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Explicit role/stage allowlists. Audit/configuration objects never flow through."""
    planning = plan is None or "revision_request" in observation
    context: dict[str, Any] = {k: public[k] for k in ("query", "public_anchor_node_ids") if k in public}
    if planning:
        if "public_anchor_kinds" in public:
            context["public_anchor_kinds"] = public["public_anchor_kinds"]
        schema = public.get("graph_schema", {})
        context["graph_schema"] = {k: schema[k] for k in
            ("directed", "node_kinds", "text_fields", "text_semantics", "parallel_edges", "edge_direction", "text_projection")
            if k in schema and schema[k] is not None}
        if plan is not None:
            context["accepted_plan"] = plan.model_dump(mode="json")
            context["accepted_plan_reason"] = plan_reason
    else:
        assert plan is not None
        # Full topology stays in the immutable workspace and the explicit plan view.
        context["semantic_task"] = {"operation": plan.operation,
            "conditions": [c.model_dump(mode="json") for c in plan.conditions],
            "formula": plan.formula.model_dump(mode="json"),
            "output_all_assignments": plan.output_all_assignments}
        context["can_revise"] = (version or 1) - 1 < public.get("limits", {}).get("max_revisions", 1)
    if plan is not None and (observation.get("workspace") or {}).get("blocked_items"):
        context.pop("semantic_task", None)
        context["accepted_plan"] = plan.model_dump(mode="json")
    feedback = compact_feedback(observation)
    current: dict[str, Any] = {k: feedback[k] for k in
               ("workspace", "detail", "action_error", "memory_update_error", "inspection_progress", "revision_request",
                "newly_confirmed", "newly_excluded", "block_diagnostics") if k in feedback and feedback[k] is not None}
    if "accepted_plan" in context and current.get("detail", {}).get("kind") == "plan":
        current["detail"] = {k: v for k, v in current["detail"].items() if k != "accepted_plan"}
    if "workspace" in current:
        original = current["workspace"]
        current["workspace"] = {k: original[k] for k in
            ("certain_count", "possible_count", "units", "unit_count", "offset", "next_offset")}
        for key in ("output_obligations", "blocked_items"):
            if original[key]["count"]:
                current["workspace"][key] = original[key]
    # Budgets are enforced by Host. Only an imminent call shortage affects a decision.
    budget = observation.get("budget", {})
    cap = budget.get("max_calls", 160)
    if cap is not None:
        remaining = cap - budget.get("calls_used", 0)
        if remaining <= 6:
            current["calls_remaining"] = remaining
    if lookups and (planning or (last_action or {}).get("action") == "lookup"):
        current["name_lookups"] = lookups
    focus = {u["unit_id"] for u in (observation.get("workspace") or {}).get("units", [])}
    if last_inspection and last_inspection["version"] == version:
        focus.update(last_inspection["inspection_progress"].get("requested_unit_ids", []))
    if last_action:
        focus.update(last_action["unit_ids"])
        if last_action.get("view_unit_id"):
            focus.add(last_action["view_unit_id"])
        # An invalid proposal must remain intact so the next call can actually repair it.
        if observation.get("action_error"):
            current["last_action"] = last_action
        elif last_action["action"] != "plan":
            current["last_action"] = action_summary(last_action)
    if notes:
        selected, deferred = select_memory(notes, focus)
        if selected:
            current["working_memory"] = memory_records(selected)
        if deferred:
            current["deferred_memory"] = deferred
    if last_inspection and "inspection_progress" not in observation and last_inspection["version"] == version:
        current["last_inspection"] = last_inspection
    return [{"role": "user", "content": json_text(context)}, {"role": "user", "content": json_text(current)}]


def inspector_message(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep complete text and structural bindings; retain hashes only in receipts/audit."""
    def record(item: dict[str, Any]) -> dict[str, Any]:
        return {k: item[k] for k in ("object_ref", "text", "kind", "source", "target",
                                    "source_kind", "target_kind", "role") if k in item}

    result = {"unit_id": payload["unit_id"], "objects": [record(o) for o in payload["objects"]],
              "conditions": payload["conditions"]}
    if payload["context"]:
        result["context"] = [record(o) for o in payload["context"]]
    for key in ("prior_rejection", "clarification"):
        if key in payload:
            result[key] = payload[key]
    return result


def history_entries(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Index past decisions and outcomes without copying old frontier snapshots."""
    entries: list[dict[str, Any]] = []
    action: dict[str, Any] | None = None
    for event in events:
        if event.get("role") == "coordinator":
            action = event["action"]
        elif event.get("role") == "runtime" and action is not None:
            feedback = compact_feedback(event["feedback"])
            feedback.pop("detail", None)  # Never nest earlier history/detail pages.
            feedback.pop("budget", None)
            if feedback.get("workspace") is not None:
                feedback["workspace"] = {k: v for k, v in feedback["workspace"].items()
                                         if k not in {"units", "unit_count", "offset", "next_offset"}}
            entries.append({"step": len(entries) + 1, "action": action, "outcome": feedback})
            action = None
    return entries
