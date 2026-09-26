from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from masge.agentic.campaign import reserve
from masge.agentic.deepseek_strict import (
    ENDPOINT,
    TOOL_NAME,
    parse_tool_output,
    strict_request,
    strict_schema,
)
from masge.agentic.graph import Graph, GraphLimitError
from masge.agentic.models import Action, ExecutionLimits, Inspection, Plan
from masge.agentic.provider import ModelOutputError, parse_output, write_json
from masge.agentic.service import Solver
from masge.agentic.workspace import Expr, Workspace, validate_plan

T = TypeVar("T", bound=BaseModel)


def graph(edges: list[tuple[int, int]], nodes: int = 5) -> Graph:
    return Graph(node_count=nodes, sources=[s for s, _ in edges], targets=[t for _, t in edges],
                 text=lambda kind, n: f"{kind} {n} supports claim", kind=lambda n: "paper",
                 name=lambda n: f"paper {n}", domain="fixture", source_hash="fixture-sha")


def condition(name: str = "p", target: dict[str, Any] | None = None, **updates: Any) -> dict[str, Any]:
    return {"name": name, "phrase": "supports claim", "target": {"node_roles": ["B"]} if target is None else target,
            "mode": "each", "quantifier": "all",
            "k": None, "context_roles": [], **updates}


def leaf(name: str) -> dict[str, Any]:
    return {"op": "condition", "condition_id": name, "children": [], "k": None}


def plan(**updates: Any) -> Plan:
    value = {"operation": "pattern", "roles": [{"name": r, "kind": "paper"} for r in ["A", "B", "C"]],
             "relations": [{"name": "AB", "source": "A", "target": "B"},
                           {"name": "AC", "source": "A", "target": "C"}],
             "anchor_role": "A", "fixed_bindings": [], "symmetric_role_groups": [{"roles": ["B", "C"]}],
             "path_target_role": None, "hop_count": None, "direction": "directed", "core_mode": None,
             "core_k": None, "conditions": [condition(target={"node_roles": ["B", "C"]})],
             "formula": leaf("p"), "output_all_assignments": False, **updates}
    return Plan.model_validate(value)


def receipt(workspace: Workspace, unit_id: str, status: str) -> Inspection:
    data = workspace.inspection_input(unit_id)
    objects = {o["object_ref"]: o for o in data["objects"] + data["context"]}
    return Inspection.model_validate({"unit_id": unit_id, "binding_issue": None,
        "judgments": [{"atom_id": c["atom_id"], "status": status, "explanation": "fixture judgment",
                       "reviewed_objects": list(objects),
                       "evidence": [] if status in {"not_established", "uncertain"} else
                       [{"object_ref": next(iter(objects)), "quote": "supports claim"}]}
                      for c in data["conditions"]]})


def test_positive_formula_bounds_and_frontier_over_all_completions() -> None:
    a, b, c, d = (Expr(atom=name) for name in "abcd")
    formulas = [Expr(children=(Expr(children=(a, b), threshold=2), Expr(children=(c, d), threshold=2))),
                Expr(children=(a, b, c), threshold=2), Expr(children=(a, a, b), threshold=2)]
    for values_tuple in itertools.product([None, False, True], repeat=4):
        values = dict(zip("abcd", values_tuple, strict=True))
        unknown = [key for key, value in values.items() if value is None]
        completions = [{**values, **dict(zip(unknown, fill, strict=True))}
                       for fill in itertools.product([False, True], repeat=len(unknown))]
        for formula in formulas:
            outcomes = [formula.bounds(full)[0] for full in completions]
            assert formula.bounds(values) == (all(outcomes), any(outcomes))
            if len(set(outcomes)) > 1:
                assert formula.pending(values)


def test_shared_atoms_negative_support_and_complete_original_records() -> None:
    g = graph([(0, 1), (0, 1), (0, 2), (0, 3)])
    p = plan()
    validate_plan(p, "supports claim", [0], g, set())
    assignments = g.enumerate(p, [0], ExecutionLimits())
    w = Workspace(g, p, assignments, 1, ExecutionLimits())
    assert len(assignments) == 6 and len(w.candidates) == 3 and len(w.atoms) == 3
    for unit in list(w.units):
        target = w.units[unit]["subjects"][0]
        w.accept(receipt(w, unit, "not_established" if target == "node:3" else "established"))
    answer = w.answer()
    assert len(answer["answers"]) == 1
    assert answer["answers"][0]["edge_ids"] == [0, 1, 2]
    assert answer["answers"][0]["node_ids"] == [0, 1, 2]


def test_source_quote_validation_is_transactional() -> None:
    g, p = graph([(0, 1), (0, 2)]), plan()
    w = Workspace(g, p, g.enumerate(p, [0], ExecutionLimits()), 1, ExecutionLimits())
    unit = next(iter(w.units))
    value = receipt(w, unit, "established").model_dump(mode="json")
    value["judgments"][0]["evidence"][0]["quote"] = "fabricated quote"
    with pytest.raises(ValueError, match="literal span"):
        w.accept(Inspection.model_validate(value))
    assert all(value is None for value in w.values.values()) and not w.receipts


def test_inspector_report_omits_host_scope_without_relaxing_evidence_validation() -> None:
    from masge.agentic.models import InspectionReport

    g = graph([(0, 1), (0, 2)])
    p = plan(conditions=[condition(context_roles=["C"])])
    w = Workspace(g, p, g.enumerate(p, [0], ExecutionLimits()), 1, ExecutionLimits())
    unit = next(iter(w.units))
    original = receipt(w, unit, "established")
    report = scripted_reply(InspectionReport, original)
    schema = json.dumps(strict_schema(InspectionReport))
    assert "unit_id" not in schema and "reviewed_objects" not in schema
    assert report.receipt(w.inspection_input(unit)) == original
    with pytest.raises(ValueError, match="Extra inputs"):
        InspectionReport.model_validate({**report.model_dump(mode="json"), "unit_id": unit})
    report.judgments[0].evidence[0].quote = "invented source quote"
    with pytest.raises(ValueError, match="literal span"):
        w.accept(report.receipt(w.inspection_input(unit)))
    assert not w.receipts and all(value is None for value in w.values.values())


def test_joint_and_context_bound_judgments_do_not_share_incorrectly() -> None:
    g = graph([(0, 1), (0, 2), (0, 3)])
    p = plan(conditions=[condition(context_roles=["C"])])
    w = Workspace(g, p, g.enumerate(p, [0], ExecutionLimits()), 1, ExecutionLimits())
    assert len(w.atoms) == 6
    joint = plan(conditions=[condition(target={"node_roles": ["B", "C"]}, mode="joint")])
    j = Workspace(g, joint, g.enumerate(joint, [0], ExecutionLimits()), 1, ExecutionLimits())
    assert len(j.atoms) == 6 and all(len(u["subjects"]) == 2 for u in j.units.values())


def test_output_assignment_obligation_and_certified_role_witness() -> None:
    g = graph([(0, 1), (0, 2)])
    p = plan(conditions=[condition()], output_all_assignments=True)
    w = Workspace(g, p, g.enumerate(p, [0], ExecutionLimits()), 1, ExecutionLimits())
    positive = next(u for u in w.units if w.units[u]["subjects"] == ["node:2"])
    w.accept(receipt(w, positive, "established"))
    state = w.state()
    assert state["certain"] == state["possible"] and not state["completion_ready"]
    with pytest.raises(ValueError, match="completion refused"):
        w.answer()
    other = next(u for u in w.units if u != positive)
    w.accept(receipt(w, other, "not_established"))
    answers = w.answer()["answers"]
    assert answers[0]["assignments"][0]["role_node_ids"]["B"] == 2


def test_global_shortest_paths_are_not_semantically_recomputed() -> None:
    g = graph([(0, 1), (1, 3), (0, 2), (2, 3), (0, 4), (4, 1)])
    p = plan(operation="shortest_paths", relations=[{"name": "AB", "source": "A", "target": "B"},
                {"name": "BC", "source": "B", "target": "C"}],
             fixed_bindings=[{"role": "C", "node_id": 3}], symmetric_role_groups=[],
             path_target_role="C", hop_count=2, conditions=[condition()])
    validate_plan(p, "3 supports claim", [0], g, set())
    assignments = g.enumerate(p, [0], ExecutionLimits())
    assert {a.nodes for a in assignments} == {(0, 1, 3), (0, 2, 3)}
    w = Workspace(g, p, assignments, 1, ExecutionLimits())
    for unit in w.units:
        w.accept(receipt(w, unit, "not_established"))
    assert w.answer()["outcome"] == "no_solution"
    assert len(w.assignments) == 2

    dynamic = p.model_copy(update={"roles": [p.roles[0], p.roles[-1]], "relations": [], "hop_count": None,
                                  "conditions": plan(conditions=[condition(target={"scope": "all_nodes"})]).conditions})
    validate_plan(dynamic, "3 supports claim", [0], g, set())
    d = Workspace(g, dynamic, g.enumerate(dynamic, [0], ExecutionLimits()), 1, ExecutionLimits())
    assert len(d.candidates) == 2


def test_core_keeps_complete_component_parallel_edges_and_self_loops() -> None:
    g = graph([(0, 1), (1, 2), (2, 0), (0, 1), (1, 1), (2, 3)])
    p = plan(operation="anchored_core", roles=[{"name": "A", "kind": "paper"}], relations=[],
             symmetric_role_groups=[], direction="undirected", core_mode="fixed", core_k=2,
             conditions=[condition(target={"scope": "all_nodes"}, quantifier="any")])
    validate_plan(p, "supports claim", [0], g, set())
    assignment = g.enumerate(p, [0], ExecutionLimits())[0]
    assert assignment.nodes == (0, 1, 2) and assignment.edges == (0, 1, 2, 3, 4)
    w = Workspace(g, p, [assignment], 1, ExecutionLimits())
    w.accept(receipt(w, next(iter(w.units)), "established"))
    assert w.answer()["answers"][0]["edge_ids"] == [0, 1, 2, 3, 4]


def test_cap_and_uncertainty_cannot_certify_false_empty_result() -> None:
    g, p = graph([(0, 1), (0, 2), (0, 3)]), plan()
    with pytest.raises(GraphLimitError):
        g.enumerate(p, [0], ExecutionLimits(max_assignments=1))
    w = Workspace(g, p, g.enumerate(p, [0], ExecutionLimits()), 1, ExecutionLimits())
    w.accept(receipt(w, next(iter(w.units)), "uncertain"))
    assert not w.state()["completion_ready"]
    with pytest.raises(ValueError):
        w.answer()


def test_query_budget_counts_failed_and_interrupted_starts(tmp_path: Path) -> None:
    write_json(tmp_path / "authorization.json", {"query_execution_limit": 2})
    reserve(tmp_path, "Q0001", "debug", "hash")
    reserve(tmp_path, "Q0001", "debug", "hash")
    with pytest.raises(RuntimeError, match="limit reached"):
        reserve(tmp_path, "Q0002", "acceptance", "hash")
    ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert len(ledger["runs"]) == 2


def test_plan_types_and_fixed_anchor_cannot_silently_shrink_public_scope() -> None:
    g = graph([(0, 1), (0, 2)])
    p = plan(fixed_bindings=[{"role": "A", "node_id": 0}])
    validate_plan(p, "supports claim", [0], g, set())
    assert g.enumerate(p, [0], ExecutionLimits()) == g.enumerate(plan(), [0], ExecutionLimits())
    for anchors in ([1], [0, 1, 2]):
        with pytest.raises(ValueError, match="entire public anchor set"):
            validate_plan(p, "supports claim", anchors, g, set())
    invalid = plan(roles=[{"name": r, "kind": "unknown_kind"} for r in ["A", "B", "C"]])
    with pytest.raises(ValueError, match="declared graph node kinds"):
        validate_plan(invalid, "supports claim", [0], g, set())


def test_condition_phrases_allow_rewording_but_cannot_be_blank() -> None:
    g = graph([(0, 1), (0, 2)])
    p = plan(conditions=[condition(phrase="the paper supports the claim")])
    validate_plan(p, "the paper assigned to role B supports the claim", [0], g, set())
    p.conditions[0].phrase = "   "
    with pytest.raises(ValueError, match="nonempty phrase"):
        validate_plan(p, "the paper assigned to role B supports the claim", [0], g, set())


def test_condition_targets_are_exclusive_and_preserve_declared_binding_checks() -> None:
    g = graph([(0, 1), (0, 2)])
    for target in ({"node_roles": ["B"]}, {"relation_roles": ["AB"]},
                   {"scope": "all_nodes"}, {"scope": "non_anchor_nodes"}, {"scope": "all_edges"}):
        p = plan(conditions=[condition(target=target)])
        validate_plan(p, "supports claim", [0], g, set())
        assert p.conditions[0].model_dump(mode="json")["target"] == target
        assert parse_tool_output(strict_reply(action("plan", plan=p).model_dump_json()), Action).plan == p
    for target in ({}, {"node_roles": ["B"], "relation_roles": []},
                   {"scope": "all_edges", "node_roles": []}):
        with pytest.raises(ValueError):
            plan(conditions=[condition(target=target)])
    for field, name, wrong in (("node_roles", "B", "AB"), ("relation_roles", "AB", "B")):
        for names in ([], [name, name], [wrong]):
            with pytest.raises(ValueError, match=f"target.{field}"):
                validate_plan(plan(conditions=[condition(target={field: names})]), "supports claim", [0], g, set())
    with pytest.raises(ValueError, match="context binds an undeclared"):
        validate_plan(plan(conditions=[condition(context_roles=["missing"])]), "supports claim", [0], g, set())


def test_edge_targets_keep_parallel_witnesses_and_separate_relation_quantifiers() -> None:
    g = graph([(0, 1), (0, 1), (0, 2), (0, 2)])
    p = plan(fixed_bindings=[{"role": "B", "node_id": 1}, {"role": "C", "node_id": 2}],
             symmetric_role_groups=[], conditions=[
                 condition("left", target={"relation_roles": ["AB"]}, quantifier="any"),
                 condition("right", target={"relation_roles": ["AC"]}, quantifier="any")],
             formula={"op": "all", "condition_id": None, "children": [leaf("left"), leaf("right")], "k": None})
    validate_plan(p, "1 2 supports claim", [0], g, set())
    w = Workspace(g, p, g.enumerate(p, [0], ExecutionLimits()), 1, ExecutionLimits())
    atoms = {atom["subjects"][0]: atom_id for atom_id, atom in w.atoms.items()}
    assert set(atoms) == {"edge:0", "edge:1", "edge:2", "edge:3"}
    for values in itertools.product([False, True], repeat=4):
        assignment = {atoms[f"edge:{i}"]: value for i, value in enumerate(values)}
        expected = any(values[:2]) and any(values[2:])
        assert w.expressions[0].bounds(assignment) == (expected, expected)
    for unit_id in w.units:
        positive = w.units[unit_id]["subjects"][0] in {"edge:1", "edge:3"}
        w.accept(receipt(w, unit_id, "established" if positive else "not_established"))
    answer = w.answer()["answers"][0]
    assert answer["edge_ids"] == [0, 1, 2, 3]
    assert answer["assignments"][0]["relation_edge_ids"] == {"AB": [0, 1], "AC": [2, 3]}


def test_whole_targets_preserve_core_non_anchor_nodes_and_original_edges() -> None:
    g = graph([(0, 1), (1, 2), (2, 0), (0, 1), (1, 1), (2, 3)])
    p = plan(operation="anchored_core", roles=[{"name": "A", "kind": "paper"}], relations=[],
             symmetric_role_groups=[], direction="undirected", core_mode="fixed", core_k=2,
             conditions=[condition("nodes", target={"scope": "non_anchor_nodes"}),
                         condition("edges", target={"scope": "all_edges"}, quantifier="at_least", k=3)],
             formula={"op": "all", "condition_id": None, "children": [leaf("nodes"), leaf("edges")], "k": None})
    validate_plan(p, "supports claim", [0], g, set())
    w = Workspace(g, p, g.enumerate(p, [0], ExecutionLimits()), 1, ExecutionLimits())
    assert {tuple(atom["subjects"]) for atom in w.atoms.values()} == {
        ("node:1",), ("node:2",)} | {(f"edge:{i}",) for i in range(5)}
    for unit_id in w.units:
        w.accept(receipt(w, unit_id, "not_established" if w.units[unit_id]["subjects"] in [["edge:3"], ["edge:4"]]
                         else "established"))
    assert w.answer()["answers"][0]["edge_ids"] == [0, 1, 2, 3, 4]


def action(kind: str, **updates: Any) -> Action:
    return Action.model_validate({"action": kind, "plan": None, "unit_ids": [], "offset": 0,
                                  "lookup_text": None, "reason": "fixture decision", "query_evidence": None,
                                  "view_kind": "frontier", "view_unit_id": None,
                                  "memory_updates": [],
                                  **updates})


def strict_reply(arguments: str, model: str = "deepseek-v4-pro", reasoning: str = "") -> dict[str, Any]:
    return {"id": "chatcmpl_fixture", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "reasoning_content": reasoning,
                "tool_calls": [{"id": "call_fixture", "type": "function", "function": {
                    "name": TOOL_NAME, "arguments": arguments}}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                      "completion_tokens_details": {"reasoning_tokens": 1 if reasoning else 0}}}


def test_strict_tools_use_fresh_requests_and_reject_inherited_transcripts() -> None:
    decision = action("plan", plan=plan())
    response = strict_reply(decision.model_dump_json(), reasoning="fixture reasoning")
    inputs = [{"role": "system", "content": "fixture"}, {"role": "user", "content": "public query"}]
    wire = strict_request("deepseek-v4-pro", "high", inputs, Action, 10000)
    assert wire["tools"][0]["function"]["parameters"] == strict_schema(Action)
    assert wire["thinking"] == {"type": "enabled"} and wire["reasoning_effort"] == "high"
    parsed = parse_tool_output(response, Action)
    assert parsed == decision and wire["messages"] == inputs
    fresh = [{"role": "system", "content": "inspector"}, {"role": "user", "content": "single unit"}]
    inspection = strict_request("deepseek-v4-pro", "none", fresh, Inspection, 10000)
    assert inspection["messages"] == fresh and inspection["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in inspection
    from masge.agentic.provider import request_input
    for role in ("assistant", "tool"):
        with pytest.raises(RuntimeError, match="stateless"):
            strict_request("deepseek-v4-pro", "high", [*inputs, {"role": role, "content": "old turn"}], Action, 10000)
        with pytest.raises(RuntimeError, match="stateless"):
            request_input("fixture", [{"role": role, "content": "old turn"}])


def test_strict_tools_reject_extra_calls_and_do_not_parse_content_as_arguments() -> None:
    valid = action("view")
    response = strict_reply(valid.model_dump_json())
    message = response["choices"][0]["message"]
    message["content"] = action("finish").model_dump_json()
    assert parse_tool_output(response, Action) == valid
    message["tool_calls"] *= 2
    with pytest.raises(ValueError, match="exactly one"):
        parse_tool_output(response, Action)
    message["tool_calls"] = []
    with pytest.raises(ValueError, match="exactly one"):
        parse_tool_output(response, Action)


def test_strict_schema_retains_nullable_plan_recursion_and_every_contract_constraint() -> None:
    adapted = strict_schema(Action)
    assert set(adapted["required"]) == set(adapted["properties"])
    branches = adapted["properties"]["plan"]["anyOf"]
    assert "Plan" not in adapted["$def"] and branches[0]["type"] == "object"
    adapted["$def"]["Plan"] = branches[0]  # Restore the unreachable definition for equivalence below.
    assert branches[1] == {"type": "null"}
    assert adapted["$def"]["Formula"]["properties"]["children"]["items"] == {"$ref": "#/$def/Formula"}
    targets = adapted["$def"]["Condition"]["properties"]["target"]["anyOf"]
    for index, name in enumerate(["NodeTarget", "EdgeTarget", "WholeTarget"]):
        assert name not in adapted["$def"]
        assert targets[index]["additionalProperties"] is False
        adapted["$def"][name] = targets[index]
        targets[index] = {"$ref": f"#/$def/{name}"}
    # Undo only representation changes: the full schema must match byte-for-byte
    # under canonical JSON serialization, including required fields and enums.
    branches[0] = {"$ref": "#/$def/Plan"}
    restored = json.dumps(adapted, sort_keys=True).replace('"$def":', '"$defs":').replace('#/$def/', '#/$defs/')
    assert restored == json.dumps(Action.model_json_schema(), sort_keys=True)
    inspection = json.dumps(strict_schema(Inspection), sort_keys=True)
    assert inspection.replace('"$def":', '"$defs":').replace('#/$def/', '#/$defs/') == json.dumps(Inspection.model_json_schema(), sort_keys=True)


def test_strict_inspection_rejects_binding_issue_as_judgment_status() -> None:
    output = {"unit_id": "v1:u1", "binding_issue": None, "judgments": [{
        "atom_id": "v1:a1", "status": "binding_issue", "explanation": "missing binding",
        "reviewed_objects": ["node:1"], "evidence": []}]}
    response = strict_reply(json.dumps(output))
    with pytest.raises(ValueError, match="judgments.0.status"):
        parse_tool_output(response, Inspection)
    assert json.loads(response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]) == output
    output.update(binding_issue="missing binding", judgments=[])
    assert parse_tool_output(strict_reply(json.dumps(output)), Inspection).binding_issue == "missing binding"


class Scripted:
    def __init__(self, outputs: list[BaseModel]) -> None:
        self.outputs = outputs
        self.calls = 0
        self.tokens = 0

    async def request(self, role: str, instructions: str, messages: list[dict[str, Any]], output_type: type[T]) -> T:
        self.calls += 1
        return scripted_reply(output_type, self.outputs.pop(0))


def scripted_reply(output_type, value):
    from masge.agentic.models import ExecutionAction, InspectionReport
    data = value.model_dump(mode="json")
    if output_type is InspectionReport:
        data.pop("unit_id")
        for judgment in data["judgments"]:
            judgment.pop("reviewed_objects")
    if issubclass(output_type, ExecutionAction):
        assert data.pop("plan", None) is None and data.pop("lookup_text", None) is None
    return output_type.model_validate(data)


@pytest.mark.asyncio
async def test_coordinator_schema_failure_uses_compact_bounded_repair_without_fabricating_actions() -> None:
    from pydantic import ValidationError

    class InvalidCoordinator(Scripted):
        async def request(self, role, instructions, messages, output_type):
            self.calls += 1
            if self.calls > 1:
                visible = json.dumps(messages)
                assert "Field required" in visible and "DO_NOT_ECHO_RAW_RESPONSE" not in visible
                assert "complete schema-valid action" in instructions
                assert solver._last_action is None
            try:
                output_type.model_validate_json('{"reason":"DO_NOT_ECHO_RAW_RESPONSE"}')
            except ValidationError as error:
                raise ModelOutputError(str(error)) from error

    provider = InvalidCoordinator([])
    solver = Solver(graph([(0, 1)]), provider, ExecutionLimits())
    result = await solver.run("supports claim", [0], {})
    assert provider.calls == 3 and result["status"] == "unresolved"
    assert result["reason"] == "bounded action-repair limit exhausted"
    assert solver.workspace is None and not any(event.get("role") == "coordinator" for event in solver.events)
    assert [item["stage"] for item in result["audit"]["issues"]] == ["coordinator_output"] * 3 + ["execution"]
    assert "DO_NOT_ECHO_RAW_RESPONSE" in result["audit"]["issues"][0]["error"]


@pytest.mark.asyncio
async def test_invalid_inspector_output_is_reported_without_committing_or_hiding_failure() -> None:
    g, p = graph([(0, 1), (0, 2)]), plan()
    w = Workspace(g, p, g.enumerate(p, [0], ExecutionLimits()), 1, ExecutionLimits())
    units = list(w.units)

    class InvalidOnce(Scripted):
        failed = False

        async def request(self, role: str, instructions: str, messages: list[dict[str, Any]],
                          output_type: type[T]) -> T:
            if role == "inspector" and not self.failed:
                self.failed = True
                self.calls += 1
                raise ModelOutputError("invalid JSON: trailing characters")
            if role == "coordinator" and self.failed:
                feedback = json.loads(messages[-1]["content"])
                assert feedback["action_error"] == "invalid JSON: trailing characters"
                assert solver.workspace is not None and not solver.workspace.receipts
            return await super().request(role, instructions, messages, output_type)

    provider = InvalidOnce([action("plan", plan=p.model_dump(mode="json")),
                            action("inspect", unit_ids=units),
                            receipt(w, units[0], "not_established")])
    solver = Solver(g, provider, ExecutionLimits(max_calls=5))
    result = await solver.run("supports claim", [0], {})
    assert result["status"] == "complete" and result["answers"] == []
    assert result["audit"]["handoff_eligible"] is False and len(result["audit"]["issues"]) == 1
    assert provider.calls == 4 and solver.workspace is not None
    assert list(solver.workspace.receipts) == [units[0]]


@pytest.mark.asyncio
async def test_repaired_actions_do_not_consume_later_action_or_inspector_repairs() -> None:
    g, p = graph([(0, 1), (0, 2)]), plan()
    w = Workspace(g, p, g.enumerate(p, [0], ExecutionLimits()), 1, ExecutionLimits())
    units = list(w.units)

    class Failing(Scripted):
        async def request(self, role, instructions, messages, output_type):
            if isinstance(self.outputs[0], Exception):
                self.calls += 1
                raise self.outputs.pop(0)
            return await super().request(role, instructions, messages, output_type)

    malformed = ModelOutputError("malformed JSON")
    provider = Failing([action("plan", plan=p), malformed, malformed,
                       action("inspect", unit_ids=[units[0]]), malformed, malformed, malformed,
                       malformed, malformed, action("inspect", unit_ids=[units[1]]),
                       receipt(w, units[1], "not_established")])
    solver = Solver(g, provider, ExecutionLimits(max_calls=11))
    result = await solver.run("supports claim", [0], {})
    assert result["status"] == "complete" and result["answers"] == []
    assert provider.calls == 11 and len(result["audit"]["issues"]) == 8
    assert solver.workspace is not None and units[0] in solver.workspace.blocked
    assert units[0] not in solver.workspace.receipts
    assert not result["audit"]["protocol_valid"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["invalid_output", "invalid_quote"])
async def test_partial_stage_failure_reports_exact_progress_and_resumes_only_selected_units(
    tmp_path: Path, failure_kind: str,
) -> None:
    g, p = graph([(0, 1), (0, 2), (0, 3)]), plan()
    w = Workspace(g, p, g.enumerate(p, [0], ExecutionLimits()), 1, ExecutionLimits())
    units = list(w.units)
    inspected = []
    failures = []

    class FailSecondOnce(Scripted):
        async def request(self, role, instructions, messages, output_type):
            if role == "inspector":
                payload = json.loads(messages[-1]["content"])
                unit = payload["unit_id"]
                inspected.append(unit)
                if len(inspected) == 3:
                    assert "prior_rejection describes" in instructions
                    prior = payload.pop("prior_rejection")
                    assert prior["unit_id"] == units[1] and prior["failed_attempts"] == 1
                    if failure_kind == "invalid_quote":
                        assert prior["rejected_quote"] == "fabricated quote"
                    else:
                        assert prior["error"] == "invalid JSON: trailing characters"
                else:
                    assert "prior_rejection" not in payload
                    assert "prior_rejection describes" not in instructions
                from masge.agentic.context import inspector_message
                assert payload == inspector_message(w.inspection_input(unit))
                if len(inspected) == 2:
                    self.calls += 1
                    if failure_kind == "invalid_output":
                        raise ModelOutputError("invalid JSON: trailing characters")
                    invalid = receipt(w, unit, "established").model_dump(mode="json")
                    invalid["judgments"][0]["evidence"][0]["quote"] = "fabricated quote"
                    return scripted_reply(output_type, Inspection.model_validate(invalid))
            return await super().request(role, instructions, messages, output_type)

    provider = FailSecondOnce([action("plan", plan=p), action("inspect", unit_ids=units),
                               *(receipt(w, unit, "established") for unit in units)])
    solver = Solver(g, provider, ExecutionLimits(max_calls=7), tmp_path)
    result = await solver.run("supports claim", [0], {})
    assert result["status"] == "complete" and len(result["answers"]) == 3
    assert not result["audit"]["handoff_eligible"] and len(result["audit"]["issues"]) == 1
    assert inspected == [units[0], units[1], units[1], units[2]] and provider.calls == 6
    assert not provider.outputs and not failures
    assert not solver.memory_snapshot()["inspection_rejections"]
    saved = json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))
    error = next(event for event in saved if event.get("stage") == "inspector_output")
    assert error["unit_id"] == units[1] and error["attempt"] == 1
    progress = saved[-1]["feedback"]["inspection_progress"]
    assert progress["accepted_unit_ids"] == units and progress["failed_unit_id"] is None
    assert not progress["not_started_unit_ids"]


@pytest.mark.asyncio
async def test_stage_feedback_marks_blocked_units_without_authorizing_reinspection() -> None:
    g, p = graph([(0, 1), (0, 2)]), plan()
    provider = Scripted([])
    solver = Solver(g, provider, ExecutionLimits())
    await solver._dispatch(action("plan", plan=p), "supports claim", [0])
    assert solver.workspace is not None
    first, second = list(solver.workspace.units)
    provider.outputs = [Inspection(unit_id=first, binding_issue="missing required context", judgments=[]),
                        receipt(solver.workspace, second, "uncertain")]
    feedback = await solver._dispatch(action("inspect", unit_ids=[first, second]), "supports claim", [0])
    assert feedback["inspection_progress"]["accepted_unit_ids"] == [second]
    assert feedback["inspection_progress"]["blocked_unit_ids"] == [first, second]
    assert not feedback["completion_ready"]
    frontier = solver.workspace.observe()
    assert frontier["unit_count"] == 2
    assert all(u["blocked"] for u in frontier["units"])
    assert set(frontier["blocked_items"]) == {first, second}
    assert not frontier["completion_ready"] and frontier["unresolved_count"] == 1
    assert solver.workspace.detail("receipt", first, 0)["blocking_reason"] == "missing required context"
    assert solver.workspace.detail("dependencies", first, 0)["dependent_candidates"]["count"] == 1
    with pytest.raises(ValueError, match="cannot rewrite"):
        await solver._dispatch(action("inspect", unit_ids=[first], reason=""), "supports claim", [0])
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_coordinator_loop_autocompletes_on_last_call_and_safely_skips() -> None:
    g, p = graph([(0, 1), (0, 2)]), plan()
    w = Workspace(g, p, g.enumerate(p, [0], ExecutionLimits()), 1, ExecutionLimits())
    units = list(w.units)
    provider = Scripted([action("plan", plan=p.model_dump(mode="json")), action("inspect", unit_ids=units),
                         receipt(w, units[0], "not_established")])
    solver = Solver(g, provider, ExecutionLimits(max_calls=3))
    result = await solver.run("supports claim", [0], {})
    assert result["status"] == "complete" and result["outcome"] == "no_solution"
    assert result["audit"]["handoff_eligible"] and provider.calls == 3
    progress = solver.events[-1]["feedback"]["inspection_progress"]
    assert progress["skipped_units_with_reasons"][0]["unit_id"] == units[1]
    assert progress["accepted_unit_ids"] == [units[0]] and not progress["not_started_unit_ids"]
    assert progress["failed_unit_id"] is None


@pytest.mark.asyncio
async def test_coordinator_selects_more_than_four_units_without_merging_inspector_contexts() -> None:
    g, p = graph([(0, n) for n in range(1, 7)], nodes=7), plan()
    limits = ExecutionLimits(max_calls=8)
    w = Workspace(g, p, g.enumerate(p, [0], limits), 1, limits)
    units = list(w.units)
    provider = Scripted([action("plan", plan=p), action("inspect", unit_ids=units),
                         *(receipt(w, unit, "established") for unit in units)])
    solver = Solver(g, provider, limits)
    result = await solver.run("supports claim", [0], {})
    assert result["status"] == "complete" and result["audit"]["handoff_eligible"]
    assert provider.calls == 8 and not provider.outputs
    assert [e["input"]["unit_id"] for e in solver.events if e.get("role") == "inspector"] == units
    assert "max_units_per_action" not in limits.model_dump()


@pytest.mark.asyncio
async def test_repeated_resolved_unit_reuses_receipt_without_call_or_mutation() -> None:
    g, p = graph([(0, 1), (0, 2)]), plan()
    provider = Scripted([])
    solver = Solver(g, provider, ExecutionLimits())
    await solver._dispatch(action("plan", plan=p), "supports claim", [0])
    w = solver.workspace
    assert w is not None
    first, second = list(w.units)
    provider.outputs = [receipt(w, first, "established"), receipt(w, second, "established")]
    await solver._dispatch(action("inspect", unit_ids=[first]), "supports claim", [0])
    saved = json.dumps(w.receipts[first], sort_keys=True)
    feedback = await solver._dispatch(action("inspect", unit_ids=[first, second]), "supports claim", [0])
    assert provider.calls == 2 and not provider.outputs
    assert json.dumps(w.receipts[first], sort_keys=True) == saved and not solver.issues
    assert feedback["inspection_progress"]["accepted_unit_ids"] == [second]
    assert feedback["inspection_progress"]["skipped_units_with_reasons"] == [
        {"unit_id": first, "reason": "accepted receipt reused; no new Inspector call"}]
    assert feedback["completion_ready"]


@pytest.mark.asyncio
async def test_invalid_optional_notes_preserve_actions_and_do_not_consume_action_repairs() -> None:
    from masge.agentic.context import coordinator_messages

    g, p = graph([(0, 1), (0, 2), (0, 3)]), plan()
    w = Workspace(g, p, g.enumerate(p, [0], ExecutionLimits()), 1, ExecutionLimits())
    bad_note = {"key": "coverage", "category": "strategy", "text": "Tentative coverage", "unit_ids": ["v1:c1"]}
    outputs = [action("plan", plan=p)]
    for index, unit in enumerate(w.units):
        if index == 2:
            outputs.append(action("view", view_kind="unit", view_unit_id="v0:u1"))
        outputs.extend([action("inspect", unit_ids=[unit], memory_updates=[bad_note]), receipt(w, unit, "established")])
    provider = Scripted(outputs)
    solver = Solver(g, provider, ExecutionLimits())
    result = await solver.run("supports claim", [0], {})
    assert result["status"] == "complete" and provider.calls == 8 and not provider.outputs
    assert not result["audit"]["handoff_eligible"] and not solver._memory
    assert [i["stage"] for i in solver.issues].count("memory_validation") == 3
    assert [i["stage"] for i in solver.issues].count("action_validation") == 1
    feedback = solver.events[-1]["feedback"]
    assert "Notes not saved; requested action executed" in feedback["memory_update_error"]
    visible = coordinator_messages({}, feedback, p, 1, {}, solver.issues)
    assert "memory_update_error" in json.loads(visible[-1]["content"])


@pytest.mark.asyncio
async def test_revision_invalidates_all_old_receipts_and_ids() -> None:
    g, p = graph([(0, 1), (0, 2)]), plan()
    solver = Solver(g, Scripted([]), ExecutionLimits())
    await solver._dispatch(action("plan", plan=p.model_dump(mode="json")), "supports claim", [0])
    assert solver.workspace is not None
    old_unit = next(iter(solver.workspace.units))
    solver.workspace.accept(receipt(solver.workspace, old_unit, "established"))
    changed = p.model_copy(update={"output_all_assignments": True})
    await solver._dispatch(action("revise", plan=changed.model_dump(mode="json"), query_evidence="supports claim"),
                           "supports claim", [0])
    assert solver.workspace.version == 2 and not solver.workspace.receipts
    assert all(value is None for value in solver.workspace.values.values())
    with pytest.raises(ValueError, match="stale"):
        await solver._dispatch(action("inspect", unit_ids=[old_unit]), "supports claim", [0])


@pytest.mark.asyncio
async def test_inspector_cannot_commit_another_units_valid_receipt() -> None:
    g, p = graph([(0, 1), (0, 2)]), plan()
    w = Workspace(g, p, g.enumerate(p, [0], ExecutionLimits()), 1, ExecutionLimits())
    requested, other = list(w.units)
    provider = Scripted([receipt(w, other, "established")] * 3)
    solver = Solver(g, provider, ExecutionLimits())
    await solver._dispatch(action("plan", plan=p.model_dump(mode="json")), "supports claim", [0])
    with pytest.raises(ValueError, match="each fixed unit atom"):
        await solver._dispatch(action("inspect", unit_ids=[requested]), "supports claim", [0])
    assert solver.workspace is not None and not solver.workspace.receipts
    assert all(value is None for value in solver.workspace.values.values())
    assert requested in solver.workspace.blocked and provider.calls == 3
    with pytest.raises(ValueError, match="blocked"):
        await solver._dispatch(action("inspect", unit_ids=[requested], reason=""), "supports claim", [0])
    assert provider.calls == 3


def test_transport_repetition_is_idempotent_but_conflicting_actions_are_rejected() -> None:
    first, different = action("view"), action("finish")
    def response(values):
        return {"output": [{"type": "message", "content": [{"type": "output_text", "text": v.model_dump_json()}]}
                            for v in values]}
    assert parse_output(response([first, first]), Action) == first
    with pytest.raises(ValueError, match="conflicting structured decisions"):
        parse_output(response([first, different]), Action)


def test_transport_commentary_cannot_issue_actions_or_replace_a_terminal_answer() -> None:
    first, different = action("view"), action("finish")
    def message(phase, text):
        return {"type": "message", "phase": phase, "content": [{"type": "output_text", "text": text}]}
    commentary = [message("commentary", "I will inspect the graph."),
                  message("commentary", different.model_dump_json())]
    terminal = message("final_answer", first.model_dump_json())
    assert parse_output({"output": [*commentary, terminal]}, Action) == first
    with pytest.raises(ValueError, match="no structured output"):
        parse_output({"output": commentary}, Action)
    with pytest.raises(ValueError, match="conflicting structured decisions"):
        parse_output({"output": [terminal, message("final_answer", different.model_dump_json())]}, Action)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-5.6-luna", "gpt-5.6-terra"])
async def test_native_transport_preserves_raw_commentary_and_parses_only_final(tmp_path: Path, model: str) -> None:
    import httpx
    from openai import AsyncOpenAI

    from masge.agentic.provider import NativeProvider

    final = action("view")
    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == model and payload["reasoning"]["effort"] == "high"
        assert payload["text"]["format"]["strict"] is True
        assert payload["text"]["format"]["schema"]["additionalProperties"] is False
        return httpx.Response(200, json={"id": "resp_fixture", "object": "response", "created_at": 0,
            "model": model, "reasoning": {"effort": "high"}, "status": "completed", "output": [
                {"id": f"msg_{index}", "type": "message", "role": "assistant", "status": "completed",
                 "phase": phase, "content": [{"type": "output_text", "text": text, "annotations": []}]}
                for index, (phase, text) in enumerate([
                    ("commentary", "I will inspect the graph."), ("final_answer", final.model_dump_json())])],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}})

    provider = NativeProvider.__new__(NativeProvider)
    provider.client = AsyncOpenAI(api_key="test-only", max_retries=0,
                                 http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    provider.model, provider.limits, provider.output_dir = model, ExecutionLimits(), tmp_path
    provider.calls, provider.tokens, provider.records = 0, 0, []
    provider.guard = None
    provider.reasoning_effort = "high"
    try:
        assert await provider.request("coordinator", "fixture", [], Action) == final
    finally:
        await provider.close()
    assert provider.calls == 1 and provider.tokens == 15
    saved = json.loads((tmp_path / "calls/0001_coordinator/response.json").read_text(encoding="utf-8"))
    assert saved["output"][0]["phase"] == "commentary"
    assert provider.records[0]["status"] == "ok"


@pytest.mark.parametrize("model", ["deepseek-v4-flash", "deepseek-v4-pro"])
def test_deepseek_uses_its_own_credential_and_never_falls_back_to_openai(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str,
) -> None:
    from masge.agentic.provider import NativeProvider

    captured: dict[str, Any] = {}
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-native-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setattr("openai.AsyncOpenAI", lambda **kwargs: captured.update(kwargs))
    NativeProvider(model, ExecutionLimits(), tmp_path)
    assert captured == {"api_key": "test-deepseek-key", "base_url": ENDPOINT,
                        "max_retries": 0, "timeout": 180.0}
    captured.clear()
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY is required"):
        NativeProvider(model, ExecutionLimits(), tmp_path)
    assert not captured


@pytest.mark.asyncio
@pytest.mark.parametrize("model,effort", [("deepseek-v4-flash", "high"),
                                         ("deepseek-v4-flash", "none"), ("deepseek-v4-pro", "none"),
                                         ("gpt-5.6-terra", "none"), ("gpt-5.6-luna", "none")])
async def test_native_output_contract_replays_and_rejects_invalid_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str, effort: str,
) -> None:
    import httpx
    from openai import AsyncOpenAI

    from masge.agentic.provider import ModelOutputError, NativeProvider
    from masge.agentic.replay import ReplayProvider

    final = action("view")
    replies = [final.model_dump_json(), '{"action":"unsupported"}']
    deepseek = model.startswith("deepseek-")
    base_url = ENDPOINT if deepseek else "https://api.openai.com/v1"

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == model
        if deepseek:
            assert str(request.url) == base_url + "/chat/completions"
            assert payload["thinking"] == {"type": "disabled" if effort == "none" else "enabled"}
            assert payload["tools"][0]["function"]["strict"] is True
            assert payload["tools"][0]["function"]["parameters"] == strict_schema(Action)
            assert payload["tool_choice"] == {"type": "function", "function": {"name": TOOL_NAME}}
            response = strict_reply(replies.pop(0), model)
            if model == "deepseek-v4-pro" and effort == "none":
                response["usage"]["completion_tokens_details"] = None  # Actual Beta Chat response.
                response["choices"][0]["message"].pop("reasoning_content")
            return httpx.Response(200, json=response)
        assert str(request.url) == base_url + "/responses"
        assert payload["text"]["format"]["type"] == "json_schema" and payload["text"]["format"]["strict"] is True
        assert payload["reasoning"] == {"effort": effort}
        return httpx.Response(200, json={"id": "resp_fixture", "object": "response", "created_at": 0,
            "model": model, "reasoning": {"effort": effort}, "status": "completed",
            "output": [{"id": "msg_fixture", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": replies.pop(0), "annotations": []}]}],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                      "output_tokens_details": {"reasoning_tokens": 0}}})

    provider = NativeProvider.__new__(NativeProvider)
    provider.client = AsyncOpenAI(api_key="test-only", base_url=base_url, max_retries=0,
                                 http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    provider.model, provider.limits, provider.output_dir = model, ExecutionLimits(), tmp_path
    provider.calls, provider.tokens, provider.records = 0, 0, []
    provider.guard = None
    provider.reasoning_effort = effort
    try:
        assert await provider.request("coordinator", "fixture", [], Action) == final
        if deepseek and model == "deepseek-v4-pro" and effort == "none":
            assert provider.records[0]["usage"]["completion_tokens_details"] is None
        replay = ReplayProvider(tmp_path, model, effort)
        assert await replay.request("coordinator", "fixture", [], Action) == final
        with pytest.raises(ModelOutputError) as invalid:
            await provider.request("coordinator", "fixture", [
                {"role": "user", "content": "next feedback"}], Action)
        assert type(invalid.value.__cause__).__name__ == "ValidationError"
    finally:
        await provider.close()
    assert provider.calls == 2 and provider.tokens == 30 and not replies
    assert provider.records[-1]["status"] == "failed" and provider.records[-1]["retry_count"] == 0
    assert (tmp_path / "calls/0002_coordinator/response.json").is_file()
    if effort == "none":
        with pytest.raises(ValueError, match="request settings mismatch"):
            await ReplayProvider(tmp_path, model, "high").request("coordinator", "fixture", [], Action)
        path = tmp_path / "calls/0001_coordinator/response.json"
        changed = json.loads(path.read_text(encoding="utf-8"))
        changed["usage"]["completion_tokens_details" if deepseek else "output_tokens_details"] = {"reasoning_tokens": 1}
        write_json(path, changed)
        with pytest.raises(ValueError, match="non-thinking response"):
            await ReplayProvider(tmp_path, model, effort).request("coordinator", "fixture", [], Action)






@pytest.mark.asyncio
async def test_batch_does_not_consult_gold_or_stop_for_a_complete_answer() -> None:
    import asyncio

    from masge.agentic.batch import BatchGuard, run_bounded

    guard = BatchGuard()
    ids = [str(i) for i in range(18)]

    async def execute(qid):
        await asyncio.sleep(0)
        return {"query_id": qid, "status": "complete", "answers": []}

    results = await run_bounded(ids, execute, 9, guard)
    assert {row["query_id"] for row in results} == set(ids)
    assert guard.failure is None


@pytest.mark.asyncio
async def test_cancelling_a_batch_settles_active_workers_without_starting_queued_queries() -> None:
    import asyncio

    from masge.agentic.batch import BatchGuard, run_bounded

    guard = BatchGuard()
    ready, never = asyncio.Event(), asyncio.Event()
    started, settled = [], []

    async def execute(qid):
        started.append(qid)
        if len(started) == 9:
            ready.set()
        try:
            await never.wait()
        except asyncio.CancelledError:
            settled.append(qid)
        return {"query_id": qid, "status": "unresolved"}

    task = asyncio.create_task(run_bounded([str(i) for i in range(18)], execute, 9, guard))
    async with asyncio.timeout(10):
        await ready.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(started) == len(settled) == 9 and set(started) == set(settled)
    assert guard.failure is not None and not guard.tasks


def test_inspector_receives_grounded_typed_edge_roles_without_candidate_answers() -> None:
    g = Graph(node_count=3, sources=[0, 0], targets=[1, 2],
              text=lambda kind, n: "supports claim", kind=lambda n: "user" if n == 0 else "subreddit",
              name=lambda n: str(n), domain="reddit", source_hash="fixture",
              node_kinds=("user", "subreddit"))
    p = plan(roles=[{"name": "A", "kind": "user"}, {"name": "B", "kind": "subreddit"},
                   {"name": "C", "kind": "subreddit"}],
             conditions=[condition(target={"relation_roles": ["AB"]})])
    w = Workspace(g, p, g.enumerate(p, [0], ExecutionLimits()), 1, ExecutionLimits())
    payload = w.inspection_input(next(iter(w.units)))
    assert payload["graph_context"] == {"domain": "reddit", "operation": "pattern",
                                        "structural_bindings_established_by_runtime": True}
    assert payload["objects"][0]["source_kind"] == "user"
    assert payload["objects"][0]["target_kind"] == "subreddit"
    assert payload["conditions"][0]["target_contract"]["relation_roles"] == [
        {"name": "AB", "source": "A", "target": "B"}]
    assert "candidate_ids" not in str(payload) and "certain" not in payload
    assert len(w.atoms) == 2  # Role permutations still share the same bound semantic predicate.


@pytest.mark.parametrize("mismatch", ["anchor", "fixed"])
def test_incompatible_bound_node_types_cannot_silently_complete_as_empty(mismatch: str) -> None:
    g = Graph(node_count=3, sources=[0, 0], targets=[1, 2], text=lambda kind, n: "supports claim",
              kind=lambda n: "user" if n == 0 else "item", name=lambda n: str(n),
              domain="fixture", source_hash="fixture", node_kinds=("user", "item"))
    p = plan(roles=[{"name": "A", "kind": "item" if mismatch == "anchor" else "user"},
                    {"name": "B", "kind": "item"}, {"name": "C", "kind": "item"}],
             symmetric_role_groups=[], fixed_bindings=[{"role": "B", "node_id": 0}] if mismatch == "fixed" else [])
    with pytest.raises(ValueError, match="kind conflicts"):
        validate_plan(p, "node 0", [0], g, set())




def test_compact_frontier_resolves_every_atom_to_full_condition_and_dependency() -> None:
    g, p = graph([(0, 1), (0, 2), (0, 3)]), plan()
    w = Workspace(g, p, g.enumerate(p, [0], ExecutionLimits()), 1, ExecutionLimits())
    observation = w.observe()
    definitions = {c["name"]: c for c in observation["condition_definitions"]}
    assert definitions["p"] == p.conditions[0].model_dump(mode="json")
    for unit in observation["units"]:
        assert unit["dependent_candidates"] and unit["subjects"]
        assert {c["atom_id"] for c in unit["conditions"]} == set(unit["atom_ids"])
        for ref in unit["conditions"]:
            atom = w.atoms[ref["atom_id"]]
            assert definitions[ref["condition_id"]]["phrase"] == atom["phrase"]
            assert definitions[ref["condition_id"]]["quantifier"] == "all"
