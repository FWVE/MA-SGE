"""Query-version-local dependency propagation and complete output construction."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from masge.agentic.context import page
from masge.agentic.graph import Assignment, Graph
from masge.agentic.models import (
    Condition,
    EdgeTarget,
    ExecutionLimits,
    Formula,
    Inspection,
    NodeTarget,
    Plan,
    WholeTarget,
)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


class ReceiptFormatError(ValueError):
    """Malformed model evidence, distinct from a reported binding/meaning problem."""


class QuoteValidationError(ReceiptFormatError):
    """Rejected citation details, without inventing a replacement or semantic judgment."""

    def __init__(self, atom_id: str, object_ref: str, quote: str) -> None:
        super().__init__("evidence quote is not a literal span of an inspected source field")
        self.details = {"atom_id": atom_id, "object_ref": object_ref, "rejected_quote": quote}


@dataclass(frozen=True)
class Expr:
    atom: str | None = None
    children: tuple[Expr, ...] = ()
    threshold: int = 1

    def bounds(self, values: dict[str, bool | None]) -> tuple[bool, bool]:
        if self.atom is not None:
            value = values[self.atom]
            return value is True, value is not False
        bounds = [child.bounds(values) for child in self.children]
        return (sum(low for low, _ in bounds) >= self.threshold,
                sum(high for _, high in bounds) >= self.threshold)

    def pending(self, values: dict[str, bool | None]) -> set[str]:
        low, high = self.bounds(values)
        if low == high:
            return set()
        if self.atom is not None:
            return {self.atom}
        return set().union(*(child.pending(values) for child in self.children))


def validate_plan(plan: Plan, query: str, anchors: list[int], graph: Graph,
                  resolved_ids: set[int]) -> None:
    role_names = [r.name for r in plan.roles]
    role_set = set(role_names)
    relation_names = [r.name for r in plan.relations]
    relation_set = set(relation_names)
    condition_names = [c.name for c in plan.conditions]
    if not role_names or len(role_set) != len(role_names) or plan.anchor_role not in role_set:
        raise ValueError("roles must be unique and include the public anchor role")
    if any(role.kind is not None and role.kind not in graph.node_kinds for role in plan.roles):
        raise ValueError("role kind must be null or one of the declared graph node kinds")
    if len(relation_set) != len(relation_names) or len(set(condition_names)) != len(condition_names):
        raise ValueError("relation and condition names must be unique")
    if not anchors or len(set(anchors)) != len(anchors):
        raise ValueError("public anchors must be nonempty and unique")
    if any(not 0 <= n < graph.node_count for n in anchors):
        raise ValueError("public anchor outside source graph")
    kinds = {role.name: role.kind for role in plan.roles}
    if kinds[plan.anchor_role] is not None and any(graph.kind(n) != kinds[plan.anchor_role] for n in anchors):
        raise ValueError("anchor_role kind conflicts with public anchor kinds; correct the role, not the public anchors")
    seen_bindings = set()
    for binding in plan.fixed_bindings:
        if binding.role not in role_set or binding.role in seen_bindings:
            raise ValueError("invalid or duplicate fixed role")
        seen_bindings.add(binding.role)
        if binding.role == plan.anchor_role:
            if anchors != [binding.node_id]:
                raise ValueError("fixed anchor must agree with the entire public anchor set")
            continue
        if not 0 <= binding.node_id < graph.node_count:
            raise ValueError("fixed node outside source graph")
        if binding.node_id not in resolved_ids and not re.search(rf"(?<!\d){binding.node_id}(?!\d)", query):
            raise ValueError("fixed node must occur in public question or an explicit name lookup")
        if kinds[binding.role] is not None and graph.kind(binding.node_id) != kinds[binding.role]:
            raise ValueError(f"fixed role {binding.role} kind conflicts with bound node kind {graph.kind(binding.node_id)}")
    if plan.operation == "anchored_core" and (
        len(role_names) != 1 or plan.relations or seen_bindings - {plan.anchor_role} or plan.symmetric_role_groups
    ):
        raise ValueError("core uses only the anchor role, relations=[], fixed_bindings=[], symmetric_role_groups=[]; conditions use component scope")
    edge_pairs = {(r.source, r.target) for r in plan.relations}
    if len(edge_pairs) != len(plan.relations):
        raise ValueError("use one relation for each endpoint pair; parallel records are retained automatically")
    if any(r.source not in role_set or r.target not in role_set or r.source == r.target for r in plan.relations):
        raise ValueError("pattern relation must connect distinct declared roles")
    grouped: set[str] = set()
    for group in plan.symmetric_role_groups:
        if len(group.roles) < 2 or len(set(group.roles)) != len(group.roles):
            raise ValueError("symmetry groups need distinct roles")
        if not set(group.roles) <= role_set or set(group.roles) & (grouped | seen_bindings | {plan.anchor_role}):
            raise ValueError("symmetry cannot change anchor/fixed roles or overlap groups")
        grouped.update(group.roles)
        for left, right in zip(group.roles[:-1], group.roles[1:], strict=True):
            swap = {left: right, right: left}
            transformed = {(swap.get(s, s), swap.get(t, t)) for s, t in edge_pairs}
            if transformed != edge_pairs or kinds[left] != kinds[right]:
                raise ValueError("declared symmetry is not a typed structural automorphism")
    if plan.operation == "anchored_core":
        if plan.core_mode is None or (plan.core_mode == "fixed" and (plan.core_k is None or plan.core_k < 1)):
            raise ValueError("core needs a valid mode/k")
        if plan.direction != "undirected":
            raise ValueError("core uses the weak undirected simple projection")
    else:
        if plan.core_mode is not None or plan.core_k is not None:
            raise ValueError("core settings are exclusive to anchored_core")
        dynamic_path = plan.operation == "shortest_paths" and len(role_names) == 2 and not plan.relations
        if not plan.relations and not dynamic_path:
            raise ValueError("pattern/path needs relations")
        reached = {plan.anchor_role} | seen_bindings
        while True:
            enlarged = reached | {x for pair in edge_pairs if set(pair) & reached for x in pair}
            if enlarged == reached:
                break
            reached = enlarged
        if reached != role_set and not dynamic_path:
            raise ValueError("every role must be connected to a public/fixed binding")
    if plan.operation == "shortest_paths":
        if plan.path_target_role not in seen_bindings or role_names[0] != plan.anchor_role or role_names[-1] != plan.path_target_role:
            raise ValueError("shortest_paths needs a non-null path_target_role matching its fixed target; roles are in source-to-target order")
        if seen_bindings - {plan.anchor_role, plan.path_target_role}:
            raise ValueError("shortest_paths binds endpoints only; use pattern for fixed intermediate roles")
        if ((edge_pairs != set(zip(role_names[:-1], role_names[1:], strict=True)) and not dynamic_path)
                or plan.symmetric_role_groups):
            raise ValueError("shortest_paths requires exactly the consecutive path relations")
    elif plan.path_target_role is not None or plan.hop_count is not None:
        raise ValueError("path target/hops are exclusive to shortest_paths; fixed-hop chains use pattern")
    if plan.hop_count is not None and plan.hop_count < 1:
        raise ValueError("hop count must be positive")
    for condition in plan.conditions:
        if not condition.name or not condition.phrase.strip():
            raise ValueError("each condition must have a name and a nonempty phrase")
        if not set(condition.context_roles) <= role_set:
            raise ValueError("condition context binds an undeclared node role")
        target = condition.target
        if isinstance(target, (NodeTarget, EdgeTarget)):
            names, declared = ((target.node_roles, role_set) if isinstance(target, NodeTarget)
                               else (target.relation_roles, relation_set))
            field = "node_roles" if isinstance(target, NodeTarget) else "relation_roles"
            if not names or len(set(names)) != len(names):
                raise ValueError(f"condition {condition.name}.target.{field} needs nonempty distinct names")
            if not set(names) <= declared:
                raise ValueError(f"condition {condition.name}.target.{field} references undeclared names; available: {sorted(declared)}")
        if condition.quantifier == "at_least":
            if condition.k is None or condition.k < 1:
                raise ValueError("at_least needs a positive k")
        elif condition.k is not None:
            raise ValueError("k is exclusive to at_least")
        if condition.mode == "joint" and (condition.quantifier != "all" or condition.k is not None):
            raise ValueError("joint condition is one grouped judgment, use all with k=null")
    used: set[str] = set()

    def visit(formula: Formula) -> None:
        if formula.op == "condition":
            if formula.condition_id not in condition_names:
                raise ValueError("formula references an undeclared condition")
            used.add(str(formula.condition_id))
        for child in formula.children:
            visit(child)

    visit(plan.formula)
    if used != set(condition_names):
        raise ValueError("every declared condition must be used in the formula")


class Workspace:
    def __init__(self, graph: Graph, plan: Plan, assignments: list[Assignment], version: int,
                 limits: ExecutionLimits) -> None:
        self.graph, self.plan, self.version, self.limits = graph, plan, version, limits
        self.assignments = assignments
        self.values: dict[str, bool | None] = {}
        self.atoms: dict[str, dict[str, Any]] = {}
        self.units: dict[str, dict[str, Any]] = {}
        self.receipts: dict[str, dict[str, Any]] = {}
        self.blocked: dict[str, str] = {}
        self.clarifications: dict[str, dict[str, Any]] = {}
        self._atom_keys: dict[str, str] = {}
        self._unit_keys: dict[str, str] = {}
        self._role_bindings: dict[str, list[dict[str, Any]]] = {}
        self.expressions: list[Expr] = []
        self.candidates: dict[str, list[int]] = {}
        for index, assignment in enumerate(assignments):
            expressions = {c.name: self._condition(c, assignment) for c in plan.conditions}
            self.expressions.append(self._formula(plan.formula, expressions))
            key = self._candidate_key(assignment)
            self.candidates.setdefault(key, []).append(index)
        self.candidate_ids = {key: f"v{version}:c{index + 1}" for index, key in enumerate(self.candidates)}
        for atom in self.atoms.values():
            atom["candidate_ids"] = []
        for key, indices in self.candidates.items():
            needed = set().union(*(self.expressions[index].pending(self.values) for index in indices))
            for atom_id in needed:
                self.atoms[atom_id]["candidate_ids"].append(self.candidate_ids[key])

    def _candidate_key(self, assignment: Assignment) -> str:
        if self.plan.operation in {"anchored_core", "shortest_paths"}:
            return digest([assignment.anchor, assignment.nodes])
        mapping: dict[str, Any] = dict(assignment.roles)
        for group in self.plan.symmetric_role_groups:
            ids = sorted(mapping[role] for role in group.roles)
            for role, value in zip(sorted(group.roles), ids, strict=True):
                mapping[role] = value
        return digest([assignment.anchor, mapping])

    def _condition(self, condition: Condition, assignment: Assignment) -> Expr:
        target = condition.target
        if isinstance(target, NodeTarget):
            identifiers = [assignment.roles[role] for role in target.node_roles]
        elif isinstance(target, EdgeTarget):
            identifiers = sorted({e for relation in target.relation_roles for e in assignment.relations[relation]})
        elif target.scope == "all_nodes":
            identifiers = list(assignment.nodes)
        elif target.scope == "non_anchor_nodes":
            identifiers = [n for n in assignment.nodes if n != assignment.anchor]
        else:
            identifiers = list(assignment.edges)
        target_kind = self._target_kind(condition)
        context = [(role, assignment.roles[role]) for role in condition.context_roles]
        groups = [identifiers] if condition.mode == "joint" and identifiers else [[n] for n in identifiers]
        children = []
        for identifiers_group in groups:
            subjects = [f"{target_kind}:{n}" for n in identifiers_group]
            # Equality includes query condition, binding order, context, source field and source snapshot.
            identity = [condition.name, condition.phrase, subjects, context, "text", self.graph.source_hash]
            key = digest(identity)
            if key not in self._atom_keys:
                atom_id = f"v{self.version}:a{len(self.atoms) + 1}"
                self._atom_keys[key] = atom_id
                self.atoms[atom_id] = {"atom_id": atom_id, "condition_id": condition.name,
                                       "phrase": condition.phrase, "subjects": subjects,
                                       "context": context, "identity_sha256": key}
                self.values[atom_id] = None
                unit_key = digest([subjects, context, "text", self.graph.source_hash])
                if unit_key not in self._unit_keys:
                    unit_id = f"v{self.version}:u{len(self.units) + 1}"
                    self._unit_keys[unit_key] = unit_id
                    self.units[unit_id] = {"unit_id": unit_id, "subjects": subjects,
                                          "context": context, "atom_ids": []}
                self.units[self._unit_keys[unit_key]]["atom_ids"].append(atom_id)
            atom_id = self._atom_keys[key]
            binding: dict[str, Any]
            if isinstance(target, NodeTarget):
                binding = {role: f"node:{assignment.roles[role]}" for role in target.node_roles
                           if assignment.roles[role] in identifiers_group}
            elif isinstance(target, EdgeTarget):
                binding = {role: [f"edge:{e}" for e in assignment.relations[role] if e in identifiers_group]
                           for role in target.relation_roles
                           if any(e in identifiers_group for e in assignment.relations[role])}
            else:
                binding = {"scope": target.scope, "subjects": subjects}
            alternatives = self._role_bindings.setdefault(atom_id, [])
            if binding not in alternatives:
                alternatives.append(binding)
            children.append(Expr(atom=atom_id))
        threshold = (len(children) if condition.quantifier == "all" else
                     1 if condition.quantifier == "any" else int(condition.k or 1))
        return Expr(children=tuple(children), threshold=threshold)

    def _formula(self, formula: Formula, conditions: dict[str, Expr]) -> Expr:
        if formula.op == "condition":
            return conditions[str(formula.condition_id)]
        children = tuple(self._formula(child, conditions) for child in formula.children)
        threshold = len(children) if formula.op == "all" else 1 if formula.op == "any" else int(formula.k or 1)
        return Expr(children=children, threshold=threshold)

    def state(self) -> dict[str, Any]:
        certain, possible, excluded, output_pending = [], [], [], []
        live: set[str] = set()
        for key, indices in self.candidates.items():
            candidate_id = self.candidate_ids[key]
            expr = Expr(children=tuple(self.expressions[index] for index in indices))
            low, high = expr.bounds(self.values)
            if low:
                certain.append(candidate_id)
            if high:
                possible.append(candidate_id)
            else:
                excluded.append(candidate_id)
            live.update(expr.pending(self.values))
            if low and self.plan.output_all_assignments:
                for index in indices:
                    pending = self.expressions[index].pending(self.values)
                    if pending:
                        output_pending.append({"candidate_id": candidate_id, "assignment": index})
                        live.update(pending)
        relevant_blocks = {u: reason for u, reason in self.blocked.items()
                           if set(self.units[u]["atom_ids"]) & live}
        return {"structural_complete": True, "certain": certain, "possible": possible,
                "excluded": excluded, "live_atoms": sorted(live), "output_obligations": output_pending,
                "blocked_items": relevant_blocks,
                "completion_ready": certain == possible and not output_pending and not relevant_blocks}

    def observe(self, offset: int = 0) -> dict[str, Any]:
        state = self.state()
        live = set(state["live_atoms"])
        units = []
        for unit in self.units.values():
            if unit["unit_id"] in self.receipts and unit["unit_id"] not in self.blocked:
                continue
            pending = set(unit["atom_ids"]) & live
            if not pending:
                continue
            candidate_ids = sorted({c for atom in pending for c in self.atoms[atom]["candidate_ids"]})
            units.append({**unit, "conditions": [{"atom_id": a, "condition_id": self.atoms[a]["condition_id"]}
                                                 for a in unit["atom_ids"]
                                                 if unit["unit_id"] not in state["blocked_items"] or a in pending],
                          "dependent_candidates": candidate_ids,
                          "estimated_text_characters": sum(len(self._record(ref)["text"]) for ref in unit["subjects"]),
                          **({"blocked": True, "reason": state["blocked_items"][unit["unit_id"]],
                              "bindings": [{"atom_id": a, "role_bindings": self._role_bindings[a]}
                                           for a in unit["atom_ids"] if a in pending],
                              **({"clarification_used": True} if unit["unit_id"] in self.clarifications else {})}
                             if unit["unit_id"] in state["blocked_items"] else {})})
        return {"version": self.version, "condition_definitions": [c.model_dump(mode="json") for c in self.plan.conditions],
                "candidate_count": len(self.candidates),
                "assignment_count": len(self.assignments), "certain_count": len(state["certain"]),
                "possible_count": len(state["possible"]), "unresolved_count": len(state["possible"]) - len(state["certain"]),
                "resolved_atoms": sum(value is not None for value in self.values.values()),
                "unit_count": len(units), "offset": offset,
                "next_offset": offset + self.limits.page_size if offset + self.limits.page_size < len(units) else None,
                "units": units[offset:offset + self.limits.page_size],
                "output_obligations": state["output_obligations"], "blocked_items": state["blocked_items"],
                "completion_ready": state["completion_ready"]}

    def detail(self, kind: str, unit_id: str | None, offset: int) -> dict[str, Any]:
        """Read-only, complete pagination behind model-facing count/sample summaries."""
        state = self.state()
        size = self.limits.page_size
        if kind == "completion":
            if unit_id is not None:
                raise ValueError("completion view does not take a unit ID")
            return {"kind": kind, "version": self.version,
                    "certain": page(state["certain"], offset, size),
                    "possible": page(state["possible"], offset, size),
                    "excluded": page(state["excluded"], offset, size),
                    "output_obligations": page(state["output_obligations"], offset, size),
                    "blocked_items": page([{"unit_id": u, "reason": reason}
                                           for u, reason in state["blocked_items"].items()], offset, size)}
        if kind not in {"unit", "dependencies", "receipt"} or unit_id is None or unit_id not in self.units:
            raise ValueError("detail view needs a current-version unit ID and supported view kind")
        unit = self.units[unit_id]
        if kind == "receipt":
            receipt = self.receipts.get(unit_id)
            return {"kind": kind, "version": self.version, "unit_id": unit_id,
                    "receipt_accepted": receipt is not None, "blocking_reason": self.blocked.get(unit_id),
                    "judgments": page(receipt["output"]["judgments"] if receipt else [], offset, size),
                    **({"clarification": {k: v for k, v in self.clarifications[unit_id].items() if k != "input"}}
                       if unit_id in self.clarifications else {})}
        if kind == "dependencies":
            pending = set(unit["atom_ids"]) & set(state["live_atoms"])
            candidates = sorted({c for a in pending for c in self.atoms[a]["candidate_ids"]})
            return {"kind": kind, "version": self.version, "unit_id": unit_id,
                    "dependent_candidates": page(candidates, offset, size)}
        return {"kind": kind, "version": self.version, "unit_id": unit_id,
                "subjects": page(unit["subjects"], offset, size),
                "context": page(unit["context"], offset, size),
                "conditions": page([{"atom_id": a, "condition_id": self.atoms[a]["condition_id"]}
                                    for a in unit["atom_ids"]], offset, size)}

    def _record(self, ref: str) -> dict[str, Any]:
        kind, identifier = ref.split(":")
        return self.graph.evidence(kind, int(identifier))

    @staticmethod
    def _target_kind(condition: Condition) -> str:
        target = condition.target
        return "edge" if (isinstance(target, EdgeTarget)
                          or isinstance(target, WholeTarget) and target.scope == "all_edges") else "node"

    def inspection_input(self, unit_id: str) -> dict[str, Any]:
        unit = self.units[unit_id]
        objects = [self._record(ref) for ref in unit["subjects"]]
        context = [{"role": role, **self._record(f"node:{node}")} for role, node in unit["context"]]
        contracts = {c.name: {"target_kind": self._target_kind(c), "target": c.target.model_dump(mode="json"),
                             "relation_roles": [r.model_dump(mode="json") for r in self.plan.relations
                                                if isinstance(c.target, EdgeTarget) and r.name in c.target.relation_roles],
                             "mode": c.mode}
                     for c in self.plan.conditions}
        return {"version": self.version, "unit_id": unit_id, "objects": objects, "context": context,
                "graph_context": {"domain": self.graph.domain, "operation": self.plan.operation,
                                  "structural_bindings_established_by_runtime": True},
                "conditions": [{"atom_id": a, "phrase": self.atoms[a]["phrase"],
                                "subjects": self.atoms[a]["subjects"],
                                "target_contract": contracts[self.atoms[a]["condition_id"]]}
                               for a in unit["atom_ids"]]}

    def begin_clarification(self, unit_id: str, reason: str) -> dict[str, Any]:
        """Reserve one fresh delegation for live unknown atoms; retain all prior receipts."""
        if unit_id not in self.state()["blocked_items"]:
            raise ValueError("clarification requires a relevant blocked unit")
        if unit_id in self.clarifications:
            raise ValueError("blocked unit already used its one clarification delegation")
        if not reason.strip():
            raise ValueError("blocked inspect requires a clarification in reason")
        payload = self.inspection_input(unit_id)
        live = set(self.state()["live_atoms"])
        payload["conditions"] = [c for c in payload["conditions"]
                                 if c["atom_id"] in live and self.values[c["atom_id"]] is None]
        for atom in payload["conditions"]:
            atom["role_bindings"] = self._role_bindings[atom["atom_id"]]
        payload["clarification"] = {"interpretation": reason, "previous_issue": self.blocked[unit_id]}
        self.clarifications[unit_id] = {"input": payload, **payload["clarification"]}
        return payload

    def accept(self, output: Inspection, *, clarified: bool = False) -> None:
        recovery = self.clarifications.get(output.unit_id) if clarified else None
        if (output.unit_id not in self.units
                or (clarified and (recovery is None or "output" in recovery))
                or (not clarified and output.unit_id in self.receipts)):
            raise ValueError("unknown, stale or previously reviewed unit")
        input_data = recovery["input"] if recovery is not None else self.inspection_input(output.unit_id)
        if output.binding_issue:
            if output.judgments:
                raise ReceiptFormatError("binding issue must not commit semantic judgments")
            self.blocked[output.unit_id] = output.binding_issue
            if recovery is not None:
                recovery["output"] = output.model_dump(mode="json")
            return
        expected = {c["atom_id"] for c in input_data["conditions"]}
        actual = [j.atom_id for j in output.judgments]
        if set(actual) != expected or len(actual) != len(expected):
            raise ReceiptFormatError("Inspector must return each fixed unit atom exactly once")
        records = {item["object_ref"]: item for item in input_data["objects"] + input_data["context"]}
        for judgment in output.judgments:
            if set(judgment.reviewed_objects) != set(records) or len(judgment.reviewed_objects) != len(records):
                raise ValueError("receipt provenance must match the complete materialized source scope")
            if not judgment.explanation.strip():
                raise ReceiptFormatError("receipt requires a judgment explanation")
            if judgment.status in {"established", "explicitly_false"} and not judgment.evidence:
                raise ReceiptFormatError("established/refuted claims need literal source quotes")
            for item in judgment.evidence:
                if item.object_ref not in records or not item.quote or item.quote not in records[item.object_ref]["text"]:
                    raise QuoteValidationError(judgment.atom_id, item.object_ref, item.quote)
        # Validate the entire response before committing any state.
        if recovery is not None:
            recovery["output"] = output.model_dump(mode="json")
            self.blocked.pop(output.unit_id, None)
        else:
            self.receipts[output.unit_id] = {"input": input_data, "output": output.model_dump(mode="json")}
        for judgment in output.judgments:
            self.values[judgment.atom_id] = (True if judgment.status == "established" else
                                            None if judgment.status == "uncertain" else False)
        if any(j.status == "uncertain" for j in output.judgments):
            self.blocked[output.unit_id] = "Inspector uncertainty; no truth value committed for uncertain atoms"

    def answer(self) -> dict[str, Any]:
        state = self.state()
        if not state["completion_ready"]:
            raise ValueError("completion refused: unresolved membership, output obligations or blocking evidence")
        answers = []
        for key, indices in self.candidates.items():
            accepted = [i for i in indices if self.expressions[i].bounds(self.values)[0]]
            if not accepted:
                continue
            chosen = accepted if self.plan.output_all_assignments else accepted[:1]
            witnesses = []
            for index in chosen:
                assignment = self.assignments[index]
                witnesses.append({"role_node_ids": dict(assignment.roles),
                                  "relation_edge_ids": {r: [int(self.graph.original_ids[e]) for e in edges]
                                                        for r, edges in assignment.relations.items()}})
            assignment = self.assignments[accepted[0]]
            answers.append({"candidate_id": self.candidate_ids[key], "anchor_node_id": assignment.anchor,
                            "node_ids": list(assignment.nodes),
                            "edge_ids": sorted(int(self.graph.original_ids[e]) for e in assignment.edges),
                            "assignments": witnesses})
        return {"status": "complete", "outcome": "answers" if answers else "no_solution", "answers": answers,
                "certificate": {"version": self.version, "structural_complete": True,
                                "membership_complete": True, "output_ready": True, "no_blocking_conflict": True,
                                "plan_sha256": digest(self.plan.model_dump(mode="json")),
                                "source_sha256": self.graph.source_hash,
                                "correctness_is_conditional_on_plan_grounding_and_receipts": True}}

    def snapshot(self) -> dict[str, Any]:
        return {"version": self.version, "plan": self.plan.model_dump(mode="json"),
                "assignments": [{"anchor": a.anchor, "roles": a.roles, "relations": a.relations,
                                  "nodes": a.nodes, "edges": a.edges} for a in self.assignments],
                "candidate_groups": {self.candidate_ids[k]: v for k, v in self.candidates.items()},
                "atoms": self.atoms, "units": self.units, "values": self.values,
                "receipts": self.receipts, "state": self.state(),
                **({"clarifications": self.clarifications} if self.clarifications else {})}
