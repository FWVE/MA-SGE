"""Exact public graph operations with complete-domain or explicit failure results."""
from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import networkx as nx  # type: ignore[import-untyped]
import numpy as np

from masge.agentic.models import ExecutionLimits, Plan


class GraphLimitError(RuntimeError):
    """An interrupted enumeration is never an empty or complete answer."""


@dataclass(frozen=True)
class Assignment:
    anchor: int
    roles: dict[str, int]
    relations: dict[str, tuple[int, ...]]
    nodes: tuple[int, ...]
    edges: tuple[int, ...]


class Graph:
    def __init__(
        self, *, node_count: int, sources: Any, targets: Any,
        text: Callable[[str, int], str], kind: Callable[[int], str],
        name: Callable[[int], str], domain: str, source_hash: str,
        original_edge_ids: Any = None, node_kinds: tuple[str, ...] = ("paper",),
    ) -> None:
        self.node_count = node_count
        self.sources = np.asarray(sources, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.int64)
        self.edge_count = len(self.sources)
        self.original_ids = (np.arange(self.edge_count) if original_edge_ids is None
                             else np.asarray(original_edge_ids, dtype=np.int64))
        self._text, self.kind, self.name = text, kind, name
        self.domain, self.source_hash = domain, source_hash
        self.node_kinds = node_kinds
        self._out_order = np.argsort(self.sources, kind="stable")
        self._in_order = np.argsort(self.targets, kind="stable")
        self._out_ptr = np.r_[0, np.cumsum(np.bincount(self.sources, minlength=node_count))]
        self._in_ptr = np.r_[0, np.cumsum(np.bincount(self.targets, minlength=node_count))]
        self._core_graph: Any = None
        self._core_numbers: dict[int, int] | None = None

    def incident(self, node: int, incoming: bool = False) -> Any:
        order, ptr = (self._in_order, self._in_ptr) if incoming else (self._out_order, self._out_ptr)
        return order[ptr[node]:ptr[node + 1]]

    def neighbors(self, node: int, incoming: bool = False, undirected: bool = False) -> set[int]:
        values = self.sources if incoming else self.targets
        result = set(map(int, values[self.incident(node, incoming)]))
        if undirected:
            result.update(self.neighbors(node, not incoming))
        return result

    def between(self, source: int, target: int, undirected: bool = False) -> tuple[int, ...]:
        ids = self.incident(source)
        result = list(map(int, ids[self.targets[ids] == target]))
        if undirected and source != target:
            result.extend(self.between(target, source))
        return tuple(sorted(result))

    def evidence(self, kind: str, identifier: int) -> dict[str, Any]:
        if kind not in {"node", "edge"} or not 0 <= identifier < (
            self.node_count if kind == "node" else self.edge_count
        ):
            raise ValueError("evidence object outside graph")
        text = self._text(kind, identifier)
        value: dict[str, Any] = {"object_ref": f"{kind}:{identifier}", "field": "text",
                                 "text": text, "source_sha256": self.source_hash}
        if kind == "node":
            value["kind"] = self.kind(identifier)
        else:
            source, target = int(self.sources[identifier]), int(self.targets[identifier])
            value.update(source=source, target=target, source_kind=self.kind(source), target_kind=self.kind(target))
        return value

    def lookup(self, text: str, limit: int = 40) -> dict[str, Any]:
        if len(text.strip()) < 3:
            raise ValueError("lookup needs at least three literal name characters")
        matches = []
        for node in range(self.node_count):
            label = self.name(node)
            if text.casefold() in label.casefold():
                matches.append({"node_id": node, "kind": self.kind(node), "name": label})
                if len(matches) > limit:
                    return {"matches": matches[:limit], "truncated": True}
        return {"matches": matches, "truncated": False}

    def enumerate(self, plan: Plan, anchors: list[int], limits: ExecutionLimits) -> list[Assignment]:
        start = time.monotonic()

        def check() -> None:
            if time.monotonic() - start > limits.graph_seconds:
                raise GraphLimitError("structural time limit; domain is not complete")

        iterator = (self._patterns(plan, anchors, check) if plan.operation == "pattern" else
                    self._paths(plan, anchors, check) if plan.operation == "shortest_paths" else
                    self._cores(plan, anchors, check))
        result = []
        for count, assignment in enumerate(iterator, 1):
            check()
            if count > limits.max_assignments:
                raise GraphLimitError("assignment cap reached; domain is not complete")
            result.append(assignment)
        check()
        return result

    def _patterns(self, plan: Plan, anchors: list[int], check: Callable[[], None]) -> Iterator[Assignment]:
        kinds = {r.name: r.kind for r in plan.roles}
        undirected = plan.direction == "undirected"
        fixed = {b.role: b.node_id for b in plan.fixed_bindings}
        relations = plan.relations

        def recurse(bound: dict[str, int]) -> Iterator[dict[str, int]]:
            check()
            for role, node in bound.items():
                if kinds[role] and self.kind(node) != kinds[role]:
                    return
            if len(set(bound.values())) != len(bound):
                return
            for relation in relations:
                if relation.source in bound and relation.target in bound:
                    if not self.between(bound[relation.source], bound[relation.target], undirected):
                        return
            if len(bound) == len(kinds):
                yield dict(bound)
                return
            domains = []
            for role in kinds.keys() - bound.keys():
                options: set[int] | None = None
                for relation in relations:
                    nearby = None
                    if relation.source == role and relation.target in bound:
                        nearby = self.neighbors(bound[relation.target], True, undirected)
                    if relation.target == role and relation.source in bound:
                        nearby = self.neighbors(bound[relation.source], False, undirected)
                    if nearby is not None:
                        options = nearby if options is None else options & nearby
                if options is not None:
                    domains.append((len(options), role, options))
            if not domains:
                raise ValueError("pattern has an unbound disconnected role")
            _, role, options = min(domains, key=lambda value: (value[0], value[1]))
            for node in sorted(options - set(bound.values())):
                yield from recurse({**bound, role: node})

        for anchor in anchors:
            for roles in recurse({**fixed, plan.anchor_role: anchor}):
                relation_edges = {r.name: self.between(roles[r.source], roles[r.target], undirected)
                                  for r in relations}
                edges = tuple(sorted({e for values in relation_edges.values() for e in values}))
                yield Assignment(anchor, roles, relation_edges, tuple(sorted(roles.values())), edges)

    def _paths(self, plan: Plan, anchors: list[int], check: Callable[[], None]) -> Iterator[Assignment]:
        target = next(b.node_id for b in plan.fixed_bindings if b.role == plan.path_target_role)
        undirected = plan.direction == "undirected"
        for anchor in anchors:
            distance = {anchor: 0}
            predecessors: dict[int, list[int]] = {}
            queue = deque([anchor])
            while queue:
                check()
                node = queue.popleft()
                if target in distance and distance[node] >= distance[target]:
                    continue
                for neighbor in self.neighbors(node, undirected=undirected):
                    value = distance[node] + 1
                    if neighbor not in distance:
                        distance[neighbor] = value
                        queue.append(neighbor)
                    if distance[neighbor] == value:
                        predecessors.setdefault(neighbor, []).append(node)
            if target not in distance or (plan.hop_count is not None and distance[target] != plan.hop_count):
                continue
            dynamic = len(plan.roles) == 2 and not plan.relations
            if not dynamic and len(plan.roles) != distance[target] + 1:
                raise ValueError("shortest path role count does not match actual global distance")
            stack = [(target, [target])]
            while stack:
                check()
                node, backwards = stack.pop()
                if node != anchor:
                    stack.extend((p, [*backwards, p]) for p in predecessors.get(node, []))
                    continue
                nodes = tuple(reversed(backwards))
                if dynamic:
                    roles = {plan.anchor_role: anchor, str(plan.path_target_role): target}
                    rels = {f"path_step_{i}": self.between(s, t, undirected)
                            for i, (s, t) in enumerate(zip(nodes[:-1], nodes[1:], strict=True), 1)}
                else:
                    roles = dict(zip([r.name for r in plan.roles], nodes, strict=True))
                    rels = {r.name: self.between(roles[r.source], roles[r.target], undirected) for r in plan.relations}
                if any(role.kind is not None and self.kind(roles[role.name]) != role.kind for role in plan.roles):
                    continue
                edges = tuple(sorted({e for values in rels.values() for e in values}))
                yield Assignment(anchor, roles, rels, nodes, edges)

    def _cores(self, plan: Plan, anchors: list[int], check: Callable[[], None]) -> Iterator[Assignment]:
        if self._core_numbers is None:
            graph = nx.Graph()
            graph.add_nodes_from(range(self.node_count))
            for index, (source, target) in enumerate(zip(self.sources, self.targets, strict=True)):
                if index % 10000 == 0:
                    check()
                if source != target:
                    graph.add_edge(int(source), int(target))
            self._core_graph = graph
            self._core_numbers = nx.core_number(graph)
            check()
        numbers = self._core_numbers
        assert numbers is not None
        for anchor in anchors:
            if plan.roles[0].kind is not None and self.kind(anchor) != plan.roles[0].kind:
                continue
            k = numbers[anchor] if plan.core_mode == "anchor_innermost" else plan.core_k
            if k is None or k < 1:
                continue
            if plan.core_mode == "anchor_innermost" and plan.core_k is not None and k != plan.core_k:
                continue
            if numbers[anchor] < k:
                continue
            seen, queue = {anchor}, deque([anchor])
            while queue:
                check()
                node = queue.popleft()
                for neighbor in self._core_graph[node]:
                    if numbers[neighbor] >= k and neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
            edges = tuple(int(e) for node in sorted(seen) for e in self.incident(node)
                          if int(self.targets[e]) in seen)
            yield Assignment(anchor, {plan.anchor_role: anchor}, {}, tuple(sorted(seen)), tuple(sorted(edges)))
