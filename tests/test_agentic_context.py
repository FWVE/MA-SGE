"""Context compression must preserve decisions, full access and runtime semantics."""
from __future__ import annotations

import json
from typing import Any

import pytest
from test_agentic_runtime import Scripted, action, graph, plan, receipt

from masge.agentic.context import (
    MEMORY_LIMITS,
    compact_feedback,
    coordinator_messages,
    history_entries,
    inspector_message,
    json_text,
    memory_records,
    select_memory,
    update_memory,
)
from masge.agentic.models import Action, ExecutionAction, ExecutionLimits, MemoryUpdate
from masge.agentic.prompts import COORDINATOR, PLANNING_COORDINATOR, inspector_prompt
from masge.agentic.service import Solver
from masge.agentic.workspace import Workspace


def test_compact_dependency_previews_preserve_complete_paginated_access() -> None:
    g, p = graph([(0, n) for n in range(1, 16)], nodes=16), plan()
    limits = ExecutionLimits(page_size=4)
    w = Workspace(g, p, g.enumerate(p, [0], limits), 1, limits)
    before = w.snapshot()
    original = {"workspace": w.observe()}
    compact = compact_feedback(original)["workspace"]
    assert "condition_definitions" not in compact
    seen_units = []
    offset = 0
    while True:
        observed = compact_feedback({"workspace": w.observe(offset)})["workspace"]
        for unit in observed["units"]:
            seen_units.append(unit["unit_id"])
            assert unit["condition_ids"] == ["p"]
            assert "atom_ids" not in unit and "conditions" not in unit
            assert unit["dependent_candidates"]["count"] == 14
            assert len(unit["dependent_candidates"]["sample"]) == 3
            restored = []
            cursor = 0
            while True:
                detail = w.detail("dependencies", unit["unit_id"], cursor)["dependent_candidates"]
                assert len(detail["items"]) <= limits.page_size
                restored.extend(detail["items"])
                if detail["next_offset"] is None:
                    break
                cursor = detail["next_offset"]
            atom = w.units[unit["unit_id"]]["atom_ids"][0]
            assert restored == sorted(w.atoms[atom]["candidate_ids"])
        if observed["next_offset"] is None:
            break
        offset = observed["next_offset"]
    assert seen_units == list(w.units) and w.snapshot() == before
    assert isinstance(original["workspace"]["units"][0]["dependent_candidates"], list)
    messages = coordinator_messages({}, original, p, 1, {}, [])
    assert json.loads(messages[-1]["content"])["workspace"]["units"] == compact["units"]


def test_stateless_context_keeps_meaning_repair_payload_and_investigation_progress() -> None:
    public = {"query": "supports claim", "public_anchor_node_ids": [0], "graph_schema": {"domain": "fixture"}}
    p = plan()
    lookups = {"paper 3": {"matches": [{"node_id": 3, "name": "paper 3", "kind": "paper"}], "truncated": False}}
    issues = [{"stage": "action_validation", "action": "plan", "error": "old binding error"}]
    notes = update_memory({}, [MemoryUpdate(key="compare", category="hypothesis", text="Compare both texts before deciding.", unit_ids=[])], 2, set())
    investigation = {"version": 2, "reason": "Investigate shared dependencies", "inspection_progress": {"failed_unit_id": "v2:u1"}}
    last = action("view", reason="Read older evidence").model_dump(mode="json")
    messages = coordinator_messages(public, {}, p, 2, lookups, issues, notes, "Keep both roles distinct.", last, investigation)
    pinned = json.loads(messages[0]["content"])
    current = json.loads(messages[-1]["content"])
    assert pinned["query"] == public["query"] and "graph_schema" not in pinned
    assert "inspect_batch_size" not in pinned and "inspect_batch_size" not in COORDINATOR
    assert pinned["semantic_task"]["conditions"] == p.model_dump(mode="json")["conditions"]
    assert "name_lookups" not in current and "action_issues" not in current and len(messages) == 2
    assert all(m["role"] == "user" for m in messages)
    assert current["working_memory"] == memory_records(notes)
    assert "accepted_plan_reason" not in pinned
    assert current["last_action"]["reason"] == last["reason"]
    assert current["last_inspection"] == investigation
    failed = action("plan", plan=p).model_dump(mode="json")
    repaired = coordinator_messages(public, {"action_error": "bad binding"}, None, None, {}, [], last_action=failed)
    assert json.loads(repaired[-1]["content"])["last_action"] == failed
    accepted = coordinator_messages(public, {}, p, 2, {}, [], last_action=failed)
    assert "last_action" not in json.loads(accepted[-1]["content"])
    planning = coordinator_messages(public, {}, None, None, lookups, issues)
    assert json.loads(planning[-1]["content"])["name_lookups"] == lookups


def test_working_memory_never_silently_evicts_or_accepts_stale_evidence_references() -> None:
    notes: dict[str, Any] = {}
    for index in range(MEMORY_LIMITS["notes"]):
        notes = update_memory(notes, [MemoryUpdate(key=f"n{index}", category="open_question", text=f"Question {index}", unit_ids=[])], 1, set())
    before = json_text(notes)
    with pytest.raises(ValueError, match="memory is full"):
        update_memory(notes, [MemoryUpdate(key="extra", category="strategy", text="New route", unit_ids=[])], 1, set())
    with pytest.raises(ValueError, match="current-version"):
        update_memory(notes, [MemoryUpdate(key="n0", category="hypothesis", text="Tentative", unit_ids=["v0:u1"])], 1, {"v1:u1"})
    assert json_text(notes) == before
    replaced = update_memory(notes, [MemoryUpdate(key="n0", category="open_question", text=None, unit_ids=[]),
                                     MemoryUpdate(key="extra", category="strategy", text="New route", unit_ids=["v1:u1"])], 1, {"v1:u1"})
    assert "n0" not in replaced and len(replaced) == MEMORY_LIMITS["notes"]
    assert replaced["extra"]["plan_version"] == 1 and replaced["extra"]["unit_ids"] == ["v1:u1"]
    with pytest.raises(ValueError, match="text budget"):
        update_memory({}, [MemoryUpdate(key=f"long{i}", category="hypothesis", text="x" * 600, unit_ids=[])
                           for i in range(4)], 1, set())


def test_selective_memory_keeps_open_questions_and_defers_local_notes_without_deleting() -> None:
    notes = update_memory({}, [
        MemoryUpdate(key="global", category="strategy", text="Global route", unit_ids=[]),
        MemoryUpdate(key="open", category="open_question", text="Still unresolved", unit_ids=["v1:u9"]),
        MemoryUpdate(key="local", category="binding_rationale", text="Local binding", unit_ids=["v1:u1"]),
        MemoryUpdate(key="distant", category="strategy", text="Later branch", unit_ids=["v1:u9"]),
    ], 1, {"v1:u1", "v1:u9"})
    before = json_text(notes)
    selected, deferred = select_memory(notes, {"v1:u1"})
    assert set(selected) == {"global", "open", "local"}
    assert deferred == [{"key": "distant", "category": "strategy", "unit_ids": ["v1:u9"]}]
    selected, _ = select_memory(notes, {"v1:u9"})
    assert "distant" in selected and json_text(notes) == before


@pytest.mark.asyncio
async def test_memory_read_and_write_shapes_match_without_losing_stored_version() -> None:
    solver = Solver(graph([(0, 1), (0, 2)]), Scripted([]), ExecutionLimits())
    note = MemoryUpdate(key="route", category="strategy", text="Compare support.", unit_ids=[])
    await solver._dispatch(action("plan", plan=plan(), memory_updates=[note]), "supports claim", [0])
    original = json_text(solver.memory_snapshot())
    messages = coordinator_messages({}, {}, plan(), 1, {}, [], solver._memory)
    visible = json.loads(messages[-1]["content"])["working_memory"]
    page = await solver._dispatch(action("view", view_kind="memory"), "supports claim", [0])
    assert visible == page["detail"]["items"] == [note.model_dump(mode="json")]
    assert MemoryUpdate.model_validate(visible[0]) == note
    assert solver._memory["route"]["plan_version"] == 1
    assert json_text(solver.memory_snapshot()) == original
    with pytest.raises(ValueError, match="Extra inputs"):
        MemoryUpdate.model_validate({**visible[0], "plan_version": 1})


@pytest.mark.asyncio
async def test_revision_opens_full_schema_without_mutating_receipts_until_accepted() -> None:
    g, p = graph([(0, 1), (0, 2)]), plan()
    w = Workspace(g, p, g.enumerate(p, [0], ExecutionLimits()), 1, ExecutionLimits())
    unit, other = list(w.units)
    changed = p.model_copy(update={"output_all_assignments": True})
    v2 = Workspace(g, changed, g.enumerate(changed, [0], ExecutionLimits()), 2, ExecutionLimits())
    observations = []

    class Observing(Scripted):
        async def request(self, role, instructions, messages, output_type):
            if role == "coordinator":
                observations.append((output_type, json.loads(messages[-1]["content"])))
            return await super().request(role, instructions, messages, output_type)

    provider = Observing([
        action("plan", plan=p), action("inspect", unit_ids=[unit]), receipt(w, unit, "established"),
        action("request_revision", query_evidence="supports claim", reason="Correct output binding"),
        action("view", view_kind="receipt", view_unit_id=unit),
        action("revise", plan=changed, query_evidence="supports claim"),
        action("inspect", unit_ids=list(v2.units)),
        *(receipt(v2, u, "established") for u in v2.units),
    ])
    solver = Solver(g, provider, ExecutionLimits(max_calls=9))
    result = await solver.run("supports claim", [0], {})
    assert result["audit"]["handoff_eligible"] and provider.calls == 9
    assert [t for t, _ in observations] == [Action, ExecutionAction, ExecutionAction,
                                           Action, Action, ExecutionAction]
    assert observations[3][1]["workspace"]["unit_count"] == 1
    assert observations[4][1]["detail"]["judgments"]["items"][0]["status"] == "established"
    assert observations[4][1]["revision_request"]["query_evidence"] == "supports claim"
    assert "revision_request" not in observations[-1][1]
    assert solver.workspace is not None and solver.workspace.version == 2
    assert other not in solver.workspace.units and solver.revisions == 1
    with pytest.raises(ValueError, match="exhausted"):
        await solver._dispatch(action("request_revision", query_evidence="supports claim"), "supports claim", [0])


def test_execution_schema_keeps_action_contract_but_rejects_plan_payload() -> None:
    schema = ExecutionAction.model_json_schema()
    assert set(schema["$defs"]) == {"MemoryUpdate"}
    assert schema["additionalProperties"] is False
    assert "plan" not in schema["properties"] and "lookup_text" not in schema["properties"]
    assert schema["required"] == ["action"]
    with pytest.raises(ValueError):
        ExecutionAction.model_validate(action("plan", plan=plan()).model_dump(mode="json"))
    with pytest.raises(ValueError):
        ExecutionAction.model_validate(action("inspect", plan=plan()).model_dump(mode="json"))
    with pytest.raises(ValueError):
        ExecutionAction.model_validate(action("lookup", lookup_text="claim").model_dump(mode="json"))
    original = action("inspect", unit_ids=["v1:u1"]).model_dump(mode="json")
    decision = ExecutionAction.model_validate({k: v for k, v in original.items() if k not in {"plan", "lookup_text"}})
    assert decision.command().model_dump(mode="json") == original


@pytest.mark.asyncio
async def test_inspect_needs_only_action_and_unit_ids_while_semantic_fields_stay_required() -> None:
    from test_agentic_runtime import strict_reply

    from masge.agentic.deepseek_strict import parse_tool_output, strict_request

    g, p = graph([(0, 1), (0, 2)]), plan()
    w = Workspace(g, p, g.enumerate(p, [0], ExecutionLimits()), 1, ExecutionLimits())
    units = list(w.units)
    raw = json.dumps({"action": "inspect", "unit_ids": units})
    choice = parse_tool_output(strict_reply(raw), ExecutionAction)
    solver = Solver(g, Scripted([receipt(w, units[0], "not_established")]), ExecutionLimits())
    await solver._dispatch(action("plan", plan=p), "supports claim", [0])
    feedback = await solver._dispatch(choice.command(), "supports claim", [0])
    assert feedback["completion_ready"] and choice.reason == ""
    assert solver.provider.calls == 1
    wire = strict_request("deepseek-v4-pro", "none", [], ExecutionAction, 100)
    assert wire["tools"][0]["function"]["strict"] is False
    assert wire["tools"][0]["function"]["parameters"]["required"] == ["action"]
    with pytest.raises(ValueError, match="view_kind"):
        ExecutionAction.model_validate({"action": "view"})
    with pytest.raises(ValueError, match="brief decision reason"):
        await solver._dispatch(ExecutionAction(action="request_revision").command(), "supports claim", [0])
    with pytest.raises(ValueError):
        ExecutionAction.model_validate({"action": "inspect", "unit_ids": [1]})


@pytest.mark.asyncio
async def test_paged_views_are_read_only_and_cannot_hide_completion_obligations() -> None:
    g = graph([(0, 1), (0, 2), (0, 3)])
    p = plan(output_all_assignments=True)
    solver = Solver(g, Scripted([]), ExecutionLimits(page_size=1))
    await solver._dispatch(action("plan", plan=p), "supports claim", [0])
    w = solver.workspace
    assert w is not None
    first = next(iter(w.units))
    w.accept(receipt(w, first, "uncertain"))
    before = w.snapshot()
    detail = await solver._dispatch(action("view", view_kind="completion"), "supports claim", [0])
    assert detail["detail"]["possible"]["count"] == 3
    assert detail["detail"]["possible"]["next_offset"] == 1
    assert detail["detail"]["blocked_items"]["items"][0]["unit_id"] == first
    assert not compact_feedback(detail)["workspace"]["completion_ready"]
    binding = await solver._dispatch(action("view", view_kind="unit", view_unit_id=first), "supports claim", [0])
    assert binding["detail"]["subjects"]["items"] == w.units[first]["subjects"]
    assert w.snapshot() == before and solver.provider.calls == 0
    with pytest.raises(ValueError, match="finish refused"):
        await solver._dispatch(action("finish", reason=""), "supports claim", [0])
    with pytest.raises(ValueError, match="current-version"):
        await solver._dispatch(action("view", view_kind="dependencies", view_unit_id="v0:u1"), "supports claim", [0])
    with pytest.raises(ValueError, match="view parameters"):
        await solver._dispatch(action("inspect", unit_ids=[first], view_kind="unit"), "supports claim", [0])


@pytest.mark.asyncio
async def test_solver_keeps_meaning_after_compaction_and_preserves_inspector_source_scope() -> None:
    g, p = graph([(0, 1), (0, 2)]), plan()
    w = Workspace(g, p, g.enumerate(p, [0], ExecutionLimits()), 1, ExecutionLimits())
    unit, other = list(w.units)
    expected_inspections = {u: w.inspection_input(u) for u in w.units}
    requests = []

    class Observing(Scripted):
        async def request(self, role, instructions, messages, output_type):
            if role == "coordinator":
                requests.append(messages)
                assert instructions == (PLANNING_COORDINATOR if len(requests) == 1 else COORDINATOR)
                if len(requests) > 2:
                    assert json.loads(messages[0]["content"])["semantic_task"]["conditions"] == p.model_dump(mode="json")["conditions"]
                assert len(messages) == 2 and all(m["role"] == "user" for m in messages)
                if len(requests) > 3:
                    current = json.loads(messages[-1]["content"])
                    assert current["last_inspection"]["reason"] == "Resolve shared evidence"
                    assert current["last_inspection"]["inspection_progress"]["accepted_unit_ids"] == [unit]
            else:
                payload = json.loads(messages[-1]["content"])
                assert payload == inspector_message(expected_inspections[payload["unit_id"]])
            return await super().request(role, instructions, messages, output_type)

    provider = Observing([action("plan", plan=p), action("inspect", unit_ids=[unit], reason="Resolve shared evidence"),
                          receipt(w, unit, "established"), action("view"), action("view"), action("view"),
                          action("inspect", unit_ids=[other]), receipt(w, other, "not_established")])
    solver = Solver(g, provider, ExecutionLimits())
    result = await solver.run("supports claim", [0], {"domain": "fixture"})
    assert result["status"] == "complete" and result["answers"] == [] and result["audit"]["handoff_eligible"]
    assert len([e for e in solver.events if e.get("role") == "coordinator"]) == 6
    assert "condition_definitions" in solver.events[1]["feedback"]["workspace"]
    assert not any(m["role"] == "assistant" for m in requests[-1])


@pytest.mark.asyncio
async def test_reasoning_notes_history_and_receipts_survive_window_but_not_reinterpretation() -> None:
    g, p = graph([(0, 1), (0, 2)]), plan()
    solver = Solver(g, Scripted([]), ExecutionLimits(page_size=1))
    note = MemoryUpdate(key="route", category="strategy", text="Resolve the shared condition first.", unit_ids=[])
    await solver._dispatch(action("plan", plan=p, memory_updates=[note]), "supports claim", [0])
    assert solver._memory["route"]["text"] == note.text
    memory_page = await solver._dispatch(action("view", view_kind="memory"), "supports claim", [0])
    assert memory_page["detail"]["items"][0]["text"] == note.text
    assert solver.workspace is not None
    w = solver.workspace
    unit = next(iter(w.units))
    judged = receipt(w, unit, "established")
    w.accept(judged)
    for index in range(7):
        chosen = action("view", reason=f"investigation {index}")
        solver.events.append({"role": "coordinator", "action": chosen.model_dump(mode="json")})
        feedback = await solver._dispatch(chosen, "supports claim", [0])
        solver.events.append({"role": "runtime", "feedback": feedback})
    assert solver._memory["route"]["text"] == note.text
    old = await solver._dispatch(action("view", view_kind="history"), "supports claim", [0])
    assert old["detail"]["count"] == 7 and old["detail"]["next_offset"] == 1
    assert old["detail"]["items"][0]["action"]["reason"] == "investigation 0"
    assert "units" not in old["detail"]["items"][0]["outcome"]["workspace"]
    evidence = await solver._dispatch(action("view", view_kind="receipt", view_unit_id=unit), "supports claim", [0])
    assert evidence["detail"]["judgments"]["items"] == judged.model_dump(mode="json")["judgments"]
    # A history lookup must not recursively duplicate its own earlier page.
    solver.events += [{"role": "coordinator", "action": action("view", view_kind="history").model_dump(mode="json")},
                      {"role": "runtime", "feedback": old}]
    assert "detail" not in history_entries(solver.events)[-1]["outcome"]
    changed = p.model_copy(update={"output_all_assignments": True})
    await solver._dispatch(action("revise", plan=changed, query_evidence="supports claim"), "supports claim", [0])
    assert not solver._memory and not solver.workspace.receipts
    assert solver.workspace.version == 2


@pytest.mark.asyncio
async def test_stage_allowlists_and_lazy_plan_preserve_audit_and_full_source() -> None:
    sentinel = "PRIVATE_SENTINEL"
    public = {"query": "supports claim", "public_anchor_node_ids": [0], "gold": sentinel,
              "limits": ExecutionLimits().model_dump(mode="json"),
              "graph_schema": {"source_sha256": sentinel, "node_count": 99,
                               "node_kinds": ["paper"], "directed": True, "gold": sentinel}}
    p, g = plan(), graph([(0, 1), (0, 2)])
    planning = coordinator_messages(public, {"gold": sentinel}, None, None, {}, [])
    assert sentinel not in json_text(planning) and '"limits"' not in json_text(planning)
    assert json.loads(planning[0]["content"])["graph_schema"] == {"directed": True, "node_kinds": ["paper"]}
    execution = coordinator_messages(public, {}, p, 1, {}, [])
    pinned = json.loads(execution[0]["content"])
    assert "graph_schema" not in pinned and "accepted_plan" not in pinned
    assert pinned["semantic_task"]["formula"] == p.formula.model_dump(mode="json")
    solver = Solver(g, Scripted([]), ExecutionLimits())
    guide = await solver._dispatch(action("view", view_kind="guide", view_unit_id="planning"), "supports claim", [0])
    assert guide["detail"]["instructions"] and solver.workspace is None
    await solver._dispatch(action("plan", plan=p), "supports claim", [0])
    assert solver.workspace is not None
    before = json_text(solver.workspace.snapshot())
    with pytest.raises(ValueError, match="only during planning"):
        await solver._dispatch(action("lookup", lookup_text="claim"), "supports claim", [0])
    detail = await solver._dispatch(action("view", view_kind="plan"), "supports claim", [0])
    assert detail["detail"]["accepted_plan"] == p.model_dump(mode="json")
    payload = solver.workspace.inspection_input(next(iter(solver.workspace.units)))
    visible = inspector_message(payload)
    assert "source_sha256" not in json_text(visible) and "graph_context" not in visible
    assert {x["object_ref"]: x["text"] for x in visible["objects"]} == {x["object_ref"]: x["text"] for x in payload["objects"]}
    assert all("source_sha256" in x for x in payload["objects"])
    assert visible["conditions"] == payload["conditions"]
    assert json_text(solver.workspace.snapshot()) == before


def test_inspector_activates_only_binding_context_and_error_modules() -> None:
    g, p = graph([(0, 1), (0, 2)]), plan()
    w = Workspace(g, p, g.enumerate(p, [0], ExecutionLimits()), 1, ExecutionLimits())
    payload = inspector_message(w.inspection_input(next(iter(w.units))))
    instructions, active = inspector_prompt(payload)
    assert active == ["evidence", "each"]
    assert "ONE target intentionally" in instructions and 'binding_issue="specific problem"' in instructions
    payload["conditions"][0]["target_contract"]["mode"] = "joint"
    payload["context"] = [{"object_ref": "node:0", "text": "supports claim", "role": "anchor"}]
    payload["prior_rejection"] = {"error": "invalid quote"}
    instructions, active = inspector_prompt(payload)
    assert active == ["evidence", "context", "joint", "quote_repair"]
    assert "COMPLETE ordered" in instructions and "BOTH" in instructions


def test_source_text_semantics_and_anchor_kinds_are_planning_only() -> None:
    semantics = {"node": {"user": "identity_name_only"}, "edge": "full_post_text_when_nonempty"}
    public = {"query": "supports claim", "public_anchor_node_ids": [0], "public_anchor_kinds": ["user"],
              "graph_schema": {"text_semantics": semantics, "source_sha256": "hidden-internal-hash"}}
    initial = json.loads(coordinator_messages(public, {}, None, None, {}, [])[0]["content"])
    assert initial["graph_schema"] == {"text_semantics": semantics}
    assert initial["public_anchor_kinds"] == ["user"]
    investigating = json.loads(coordinator_messages(public, {}, plan(), 1, {}, [])[0]["content"])
    assert "graph_schema" not in investigating and "public_anchor_kinds" not in investigating
