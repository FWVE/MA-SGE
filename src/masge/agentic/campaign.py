"""Persistent query-execution accounting, not a generation scheduler."""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from masge.agentic.provider import write_json


@contextmanager
def ledger_lock(root: Path) -> Iterator[None]:
    path = root / ".ledger.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        yield
    finally:
        os.close(descriptor)
        path.unlink()


def reserve(root: Path, query_id: str, phase: str, source_hash: str) -> tuple[int, Path]:
    with ledger_lock(root):
        authorization = json.loads((root / "authorization.json").read_text(encoding="utf-8"))
        path = root / "ledger.json"
        ledger = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": []}
        if len(ledger["runs"]) >= authorization["query_execution_limit"]:
            raise RuntimeError("authorized query execution limit reached; no provider call started")
        index = len(ledger["runs"]) + 1
        destination = root / f"run_{index:02d}_{query_id}"
        destination.mkdir(exist_ok=False)
        ledger["runs"].append({"run": index, "query_id": query_id, "phase": phase,
                                "source_hash": source_hash, "path": destination.name, "status": "started",
                                "started_at": datetime.now(UTC).isoformat()})
        write_json(path, ledger)
        return index, destination


def complete(root: Path, index: int, result: dict[str, Any]) -> None:
    with ledger_lock(root):
        path = root / "ledger.json"
        ledger = json.loads(path.read_text(encoding="utf-8"))
        row = ledger["runs"][index - 1]
        row.update(status=result["status"], audit=result["audit"], ended_at=datetime.now(UTC).isoformat())
        write_json(path, ledger)
