from __future__ import annotations

import json
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel
from test_agentic_runtime import Scripted, action, condition, graph, leaf, plan, receipt

from masge.agentic.context import compact_feedback
from masge.agentic.models import ExecutionLimits, Inspection
from masge.agentic.service import Solver
from masge.agentic.workspace import Workspace

T = TypeVar("T", bound=BaseModel)


def test_clarification_preserves_definite_atoms_and_original_receipt() -> None:
    g = graph([(0, 1), (0, 2)])
    p = plan(conditions=[condition("p", target={"node_roles": ["B", "C"]}),
                         condition("q", target={"node_roles": ["B", "C"]})],
             formula={"op": "all", "condition_id": None, "children": [leaf("p"), leaf("q")], "k": None})
    w = Workspace(g, p, g.enumerate(p, [0], ExecutionLimits()), 1, ExecutionLimits())
    first, second = list(w.units)
    mixed = receipt(w, first, "established")
    mixed.judgments[1].status = "uncertain"
    mixed.judgments[1].evidence = []
    w.accept(mixed)
    w.accept(receipt(w, second, "established"))
    original = json.dumps(w.receipts, sort_keys=True)
    definite = {a: v for a, v in w.values.items() if v is not None}
    shown = compact_feedback({"workspace": w.observe()})["workspace"]["units"]
    assert len(shown) == 1 and shown[0]["blocked"] is True
    assert shown[0]["bindings"][0]["role_bindings"] == [{"B": "node:1"}, {"C": "node:1"}]
    payload = w.begin_clarification(first, "Interpret the textual claim on its supplied subject.")
    assert [c["atom_id"] for c in payload["conditions"]] == [mixed.judgments[1].atom_id]
    valid = receipt(w, first, "established")
    valid.judgments = [j for j in valid.judgments if j.atom_id not in definite]
    invalid = valid.model_copy(deep=True)
    invalid.judgments[0].evidence[0].quote = "invented source words"
    with pytest.raises(ValueError, match="literal span"):
        w.accept(invalid, clarified=True)
    assert all(w.values[a] == v for a, v in definite.items())
    assert json.dumps(w.receipts, sort_keys=True) == original
    w.accept(valid, clarified=True)
    assert w.state()["completion_ready"] and w.state()["certain"]
    assert json.dumps(w.receipts, sort_keys=True) == original
    with pytest.raises(ValueError, match="previously reviewed"):
        w.accept(valid, clarified=True)


@pytest.mark.asyncio
async def test_blocked_inspect_recovers_and_retains_plan_and_read_diagnostics() -> None:
    g, p = graph([(0, 1), (0, 2)]), plan()
    w = Workspace(g, p, g.enumerate(p, [0], ExecutionLimits()), 1, ExecutionLimits())
    first, second = list(w.units)

    class Observing(Scripted):
        async def request(self, role: str, instructions: str, messages: list[dict[str, Any]],
                          output_type: type[T]) -> T:
            if role == "coordinator":
                pinned, current = [json.loads(m["content"]) for m in messages]
                units = current.get("workspace", {}).get("units", [])
                assert all("blocked" not in u for u in units if u["unit_id"] == second)
                if current.get("workspace", {}).get("blocked_items"):
                    assert pinned["accepted_plan"] == p.model_dump(mode="json")
                    assert "semantic_task" not in pinned
                    assert "One clarification delegation" in instructions
                    if current.get("last_action", {}).get("view_kind") == "frontier":
                        assert current["block_diagnostics"]["reads"][0]["kind"] == "dependencies"
            else:
                payload = json.loads(messages[0]["content"])
                if "clarification" in payload:
                    assert payload["clarification"]["interpretation"] == "Judge the claim on its bound subject."
                    assert payload["conditions"][0]["subjects"] == w.units[first]["subjects"]
            return await super().request(role, instructions, messages, output_type)

    provider = Observing([
        action("plan", plan=p), action("inspect", unit_ids=[first, second]),
        Inspection(unit_id=first, judgments=[], binding_issue="role binding unclear"),
        receipt(w, second, "established"),
        action("view", view_kind="dependencies", view_unit_id=first),
        action("view", view_kind="frontier"),
        action("inspect", unit_ids=[first], reason="Judge the claim on its bound subject."),
        receipt(w, first, "established")])
    solver = Solver(g, provider, ExecutionLimits())
    result = await solver.run("supports claim", [0], {})
    assert result["status"] == "complete" and len(result["answers"]) == 1
    assert solver.workspace is not None and first in solver.workspace.clarifications
    assert not solver.workspace.state()["blocked_items"]


@pytest.mark.asyncio
async def test_repeated_successful_views_cannot_keep_blocked_work_alive() -> None:
    g, p = graph([(0, 1), (0, 2)]), plan()
    w = Workspace(g, p, g.enumerate(p, [0], ExecutionLimits()), 1, ExecutionLimits())
    first, second = list(w.units)
    provider = Scripted([action("plan", plan=p), action("inspect", unit_ids=[first, second]),
                         Inspection(unit_id=first, judgments=[], binding_issue="ambiguous binding"),
                         receipt(w, second, "established"),
                         *[action("view", view_kind="plan" if i % 2 else "frontier") for i in range(6)]])
    solver = Solver(g, provider, ExecutionLimits())
    result = await solver.run("supports claim", [0], {})
    assert result["status"] == "unresolved" and result["answers"] is None
    assert "six actions" in result["reason"] and provider.calls == 10
    assert solver.workspace is not None and second in solver.workspace.receipts


@pytest.mark.asyncio
async def test_unresolved_clarification_is_one_attempt_and_never_false_empty() -> None:
    g, p = graph([(0, 1), (0, 2)]), plan()
    provider = Scripted([])
    solver = Solver(g, provider, ExecutionLimits())
    await solver._dispatch(action("plan", plan=p), "supports claim", [0])
    w = solver.workspace
    assert w is not None
    first, second = list(w.units)
    w.accept(receipt(w, first, "uncertain"))
    w.accept(receipt(w, second, "established"))
    provider.outputs = [receipt(w, first, "uncertain")]
    feedback = await solver._dispatch(action("inspect", unit_ids=[first], reason="Reassess the stated qualifier."),
                                      "supports claim", [0])
    assert feedback["workspace"]["units"][0]["clarification_used"] is True
    with pytest.raises(ValueError, match="already used"):
        await solver._dispatch(action("inspect", unit_ids=[first], reason="Try again."), "supports claim", [0])
    assert provider.calls == 1
    stopped = await solver._dispatch(action("finish", reason="Meaning remains unresolved."), "supports claim", [0])
    assert stopped["unresolved"] and not w.state()["completion_ready"]
