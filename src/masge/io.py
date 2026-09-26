"""Public text-graph JSON input, independent of datasets and evaluators."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from masge.agentic.graph import Graph
from masge.agentic.models import Record


class Node(Record):
    id: int = Field(ge=0)
    kind: str = Field(min_length=1)
    name: str
    text: str


class Edge(Record):
    id: int = Field(ge=0, le=2**63 - 1)
    source: int = Field(ge=0)
    target: int = Field(ge=0)
    text: str


class GraphDocument(Record):
    domain: str = Field(min_length=1)
    directed: bool
    nodes: list[Node] = Field(min_length=1)
    edges: list[Edge]

    @model_validator(mode="after")
    def validate_ids(self) -> GraphDocument:
        if sorted(node.id for node in self.nodes) != list(range(len(self.nodes))):
            raise ValueError("node IDs must be unique and dense from 0 to node_count - 1")
        if len({edge.id for edge in self.edges}) != len(self.edges):
            raise ValueError("edge IDs must be unique original record IDs")
        if any(max(edge.source, edge.target) >= len(self.nodes) for edge in self.edges):
            raise ValueError("edge endpoint is outside the declared nodes")
        return self


def load_graph(path: Path) -> tuple[Graph, dict[str, Any]]:
    """Validate a public graph and preserve its text, topology and original edge IDs.

    Node IDs are dense integers. Edges retain file order internally; external IDs
    are restored in final answers. Parallel records and self-loops are preserved.
    """
    raw = path.read_bytes()
    document = GraphDocument.model_validate_json(raw)
    nodes = sorted(document.nodes, key=lambda node: node.id)
    edges = document.edges
    source_hash = hashlib.sha256(raw).hexdigest()
    kinds = tuple(sorted({node.kind for node in nodes}))
    graph = Graph(
        node_count=len(nodes), sources=[edge.source for edge in edges],
        targets=[edge.target for edge in edges],
        text=lambda kind, identifier: (nodes[identifier].text if kind == "node"
                                       else edges[identifier].text),
        kind=lambda identifier: nodes[identifier].kind,
        name=lambda identifier: nodes[identifier].name,
        domain=document.domain, source_hash=source_hash,
        original_edge_ids=[edge.id for edge in edges], node_kinds=kinds,
    )
    schema: dict[str, Any] = {
        "domain": document.domain, "directed": document.directed, "weighted": False,
        "node_count": len(nodes), "edge_count": len(edges), "node_kinds": list(kinds),
        "text_fields": ["text"], "parallel_edges": "distinct original records",
        "edge_direction": "stored source to target; original ID restored only at output",
        "source_sha256": source_hash,
        "identity_lookup": "literal substring search over public node names; not semantic evidence",
    }
    return graph, schema
