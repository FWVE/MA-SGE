"""Small public graph-operation and agent-message contracts.

These describe model-selected operations, never benchmark labels or gold.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _omit_schema_titles(schema: dict[str, Any]) -> None:
    """Generated display labels repeat property names; constraints remain intact."""
    schema.pop("title", None)
    for value in schema.get("properties", {}).values():
        value.pop("title", None)


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, json_schema_extra=_omit_schema_titles)


class Role(Record):
    name: str
    kind: str | None


class Relation(Record):
    name: str
    source: str
    target: str


class Binding(Record):
    role: str
    node_id: int


class RoleGroup(Record):
    roles: list[str]


class Formula(Record):
    op: Literal["condition", "all", "any", "at_least"]
    condition_id: str | None
    children: list[Formula]
    k: int | None

    @model_validator(mode="after")
    def shape(self) -> Formula:
        if self.op == "condition":
            if not self.condition_id or self.children or self.k is not None:
                raise ValueError("condition leaf needs only condition_id")
        elif self.condition_id is not None or (not self.children and self.op != "all"):
            raise ValueError("Boolean operator needs nonempty children and no condition_id")
        if self.op == "at_least":
            if self.k is None or not 1 <= self.k <= len(self.children):
                raise ValueError("at_least needs k within child count")
        elif self.k is not None:
            raise ValueError("k is only for at_least")
        return self


class NodeTarget(Record):
    node_roles: list[str] = Field(description="Nonempty distinct node role names from this Plan, in binding order.")


class EdgeTarget(Record):
    relation_roles: list[str] = Field(description="Nonempty distinct relation names from this Plan; includes every original parallel record.")


class WholeTarget(Record):
    scope: Literal["all_nodes", "non_anchor_nodes", "all_edges"]


class Condition(Record):
    name: str
    phrase: str
    target: NodeTarget | EdgeTarget | WholeTarget = Field(description="Choose exactly one: node_roles, relation_roles, or a whole-result scope. Do not combine branches.")
    mode: Literal["each", "joint"]
    quantifier: Literal["all", "any", "at_least"]
    k: int | None
    context_roles: list[str]


class Plan(Record):
    operation: Literal["pattern", "shortest_paths", "anchored_core"]
    roles: list[Role]
    relations: list[Relation]
    anchor_role: str
    fixed_bindings: list[Binding] = Field(description="Fixed non-anchor roles only; the runtime binds all public anchors.")
    symmetric_role_groups: list[RoleGroup]
    path_target_role: str | None = Field(description="REQUIRED target role name for shortest_paths, matching a fixed binding; null for other operations.")
    hop_count: int | None
    direction: Literal["directed", "undirected"]
    core_mode: Literal["fixed", "anchor_innermost"] | None
    core_k: int | None
    conditions: list[Condition]
    formula: Formula
    output_all_assignments: bool


class MemoryUpdate(Record):
    key: str = Field(min_length=1, max_length=48)
    category: Literal["strategy", "hypothesis", "open_question", "binding_rationale"]
    text: str | None = Field(max_length=600, description="Upsert note; null removes it. Not evidence.")
    unit_ids: list[str] = Field(max_length=4)


class Action(Record):
    action: Literal["plan", "inspect", "view", "request_revision", "revise", "finish", "lookup"]
    plan: Plan | None
    unit_ids: list[str]
    offset: int = Field(ge=0)
    lookup_text: str | None
    reason: str
    query_evidence: str | None
    view_kind: Literal["frontier", "unit", "dependencies", "completion", "receipt", "history", "memory", "plan", "guide"]
    view_unit_id: str | None
    memory_updates: list[MemoryUpdate] = Field(max_length=4)


class ExecutionAction(Record):
    """Only relevant fields are written; neutral runtime slots have fixed defaults."""

    action: Literal["inspect", "view", "request_revision", "finish"]
    unit_ids: list[str] = Field(default_factory=list)
    offset: int = Field(default=0, ge=0)
    reason: str = ""
    query_evidence: str | None = None
    view_kind: Literal["frontier", "unit", "dependencies", "completion", "receipt", "history", "memory", "plan", "guide"] = "frontier"
    view_unit_id: str | None = None
    memory_updates: list[MemoryUpdate] = Field(default_factory=list, max_length=4)

    @model_validator(mode="before")
    @classmethod
    def explicit_view(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("action") == "view" and "view_kind" not in value:
            raise ValueError("view requires an explicit view_kind")
        return value

    def command(self) -> Action:
        """Construct a runtime command from a fully validated phase-specific decision."""
        return Action(plan=None, lookup_text=None, **self.model_dump(mode="json"))


class Evidence(Record):
    object_ref: str
    quote: str


class Judgment(Record):
    atom_id: str
    status: Literal["established", "explicitly_false", "not_established", "uncertain"]
    explanation: str
    reviewed_objects: list[str]
    evidence: list[Evidence]


class Inspection(Record):
    unit_id: str
    judgments: list[Judgment]
    binding_issue: str | None


class EvidenceJudgment(Record):
    atom_id: str
    status: Literal["established", "explicitly_false", "not_established", "uncertain"]
    explanation: str
    evidence: list[Evidence]


class InspectionReport(Record):
    """Model judgments only; the host already knows the delegation and source scope."""

    judgments: list[EvidenceJudgment]
    binding_issue: str | None

    def receipt(self, payload: dict[str, Any]) -> Inspection:
        supplied = list(dict.fromkeys(item["object_ref"] for item in payload["objects"] + payload["context"]))
        return Inspection(unit_id=payload["unit_id"], binding_issue=self.binding_issue,
            judgments=[Judgment(**item.model_dump(mode="json"), reviewed_objects=supplied)
                       for item in self.judgments])


class ExecutionLimits(Record):
    max_calls: int | None = Field(default=160, ge=1)
    max_tokens: int | None = Field(default=400_000, ge=1)
    provider_seconds: int = Field(default=180, ge=1)
    query_seconds: int = Field(default=1800, ge=1)
    graph_seconds: int = Field(default=180, ge=1)
    max_assignments: int = Field(default=100_000, ge=1)
    max_revisions: int = Field(default=1, ge=0)
    page_size: int = Field(default=40, ge=1)
    max_actions: int | None = Field(default=120, ge=1)
    max_output_tokens: int = Field(default=10_000, ge=1)
