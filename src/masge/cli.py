"""Run a single public query or validate its graph without network access."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from masge.agentic.models import ExecutionLimits
from masge.agentic.provider import (
    SUPPORTED_MODELS,
    NativeProvider,
    validate_model_effort,
    write_json,
)
from masge.agentic.service import Solver
from masge.io import load_graph
from masge.logging_utils import redact_text


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate a public JSON graph; no API calls")
    validate.add_argument("--graph", type=Path, required=True)
    run = commands.add_parser("run", help="Solve a query and save its complete execution record")
    run.add_argument("--graph", type=Path, required=True)
    run.add_argument("--query", required=True)
    run.add_argument("--anchor", type=int, action="append", required=True,
                     help="Public anchor node ID; repeat for multiple anchors")
    run.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    run.add_argument("--effort", choices=("high", "none", "low"), required=True)
    run.add_argument("--output", type=Path, required=True, help="New run directory")
    run.add_argument("--limits", type=Path, help="JSON object overriding ExecutionLimits defaults")
    run.add_argument("--allow-api", action="store_true", help="Authorize model calls for this query")
    return root


async def execute(args: argparse.Namespace) -> int:
    if args.command == "run" and not args.allow_api:
        raise ValueError("run requires --allow-api to authorize external model calls")
    graph, schema = load_graph(args.graph)
    if args.command == "validate":
        print(json.dumps(schema, ensure_ascii=False, indent=2))
        return 0
    if not args.query.strip():
        raise ValueError("query must not be empty")
    anchors = list(dict.fromkeys(args.anchor))
    if any(anchor < 0 or anchor >= graph.node_count for anchor in anchors):
        raise ValueError("anchor is outside the declared nodes")
    validate_model_effort(args.model, args.effort)
    limits = (ExecutionLimits.model_validate_json(args.limits.read_bytes())
              if args.limits else ExecutionLimits())
    args.output.mkdir(parents=True, exist_ok=False)
    provider = NativeProvider(args.model, limits, args.output, reasoning_effort=args.effort)
    try:
        write_json(args.output / "run_configuration.json", {
            "runtime": "core", "model": args.model, "reasoning_effort": args.effort,
            "source": schema, "limits": limits.model_dump(mode="json"),
        })
        result = await Solver(graph, provider, limits, args.output).run(args.query, anchors, schema)
        write_json(args.output / "result.json", result)
    finally:
        await provider.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "complete" else 1


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return asyncio.run(execute(args))
    except (ValueError, OSError) as error:
        print(redact_text(f"{type(error).__name__}: {error}"), file=sys.stderr)
        return 2
