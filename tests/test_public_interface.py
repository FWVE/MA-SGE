"""Public graph ingestion and CLI integration, without external model calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from test_agentic_runtime import Scripted, action, plan, receipt

from masge import ExecutionLimits, load_graph
from masge.agentic.workspace import Workspace
from masge.cli import main
from masge.io import GraphDocument


def document() -> dict[str, Any]:
    return {
        "domain": "public-test", "directed": True,
        "nodes": [{"id": i, "kind": "paper", "name": f"Paper {i}",
                   "text": "supports claim"} for i in [2, 0, 1]],
        "edges": [{"id": edge_id, "source": 0, "target": target, "text": "supports claim"}
                  for edge_id, target in [(90, 1), (12, 1), (51, 2)]],
    }


def save_graph(tmp_path: Path) -> Path:
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(document()), encoding="utf-8")
    return path


@pytest.mark.parametrize("fault", ["duplicate_node", "duplicate_edge", "endpoint", "gold", "boolean_id"])
def test_public_input_rejects_ambiguous_or_nonpublic_fields(fault: str) -> None:
    value = document()
    if fault == "duplicate_node":
        value["nodes"][0]["id"] = 0
    elif fault == "duplicate_edge":
        value["edges"][0]["id"] = 12
    elif fault == "endpoint":
        value["edges"][0]["target"] = 3
    elif fault == "gold":
        value["gold"] = []
    else:
        value["nodes"][0]["id"] = True
    with pytest.raises(ValidationError):
        GraphDocument.model_validate_json(json.dumps(value))


def test_loader_preserves_identity_text_parallel_edges_and_self_loops(tmp_path: Path) -> None:
    value = document()
    value["edges"].append({"id": 100, "source": 1, "target": 1, "text": "loop text"})
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    graph, schema = load_graph(path)
    assert graph.name(2) == "Paper 2"
    assert graph.between(0, 1) == (0, 1)
    assert graph.between(1, 1) == (3,)
    assert graph.original_ids.tolist() == [90, 12, 51, 100]
    assert graph.evidence("edge", 3)["text"] == "loop text"
    assert schema["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert graph.evidence("node", 0)["source_sha256"] == schema["source_sha256"]


def test_cli_runs_full_semantic_loop_and_preserves_original_edge_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = save_graph(tmp_path)
    graph, _ = load_graph(path)
    request = plan()
    limits = ExecutionLimits()
    workspace = Workspace(graph, request, graph.enumerate(request, [0], limits), 1, limits)
    units = list(workspace.units)

    class LocalProvider(Scripted):
        closed = False

        async def close(self) -> None:
            self.closed = True

    provider = LocalProvider([action("plan", plan=request), action("inspect", unit_ids=units),
                              *[receipt(workspace, unit, "established") for unit in units]])
    monkeypatch.setattr("masge.cli.NativeProvider", lambda *args, **kwargs: provider)
    output = tmp_path / "run"
    args = ["run", "--graph", str(path), "--query", "supports claim", "--anchor", "0",
            "--model", "deepseek-v4-pro", "--effort", "none", "--output", str(output)]
    assert main(args) == 2
    assert provider.calls == 0 and not output.exists()
    assert main([*args, "--allow-api"]) == 0
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    assert result["answers"][0]["edge_ids"] == [12, 51, 90]
    assert provider.closed and not provider.outputs
    assert (output / "workspace.json").is_file()
    # Existing runs must remain intact; neither a second provider nor a query may start.
    calls = provider.calls
    assert main([*args, "--allow-api"]) == 2
    assert provider.calls == calls


def test_validate_is_offline_and_reports_graph_schema(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", "--graph", str(save_graph(tmp_path))]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["node_count"] == 3 and schema["edge_count"] == 3


def test_limits_reject_nonpositive_execution_bounds() -> None:
    with pytest.raises(ValidationError):
        ExecutionLimits(page_size=0)
    with pytest.raises(ValidationError):
        ExecutionLimits(query_seconds=-1)
