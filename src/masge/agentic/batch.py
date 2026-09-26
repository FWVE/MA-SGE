"""Bounded workers; transport, configuration and unexpected host failures stop peers."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from masge.logging_utils import redact_text


class BatchGuard:
    def __init__(self) -> None:
        self.failure: dict[str, str] | None = None
        self.tasks: set[asyncio.Task[Any]] = set()

    def stop(self, error: BaseException) -> None:
        if self.failure is not None:
            return
        self.failure = {"error_type": type(error).__name__, "error": redact_text(str(error))[:2000]}
        current = asyncio.current_task()
        for task in self.tasks:
            if task is not current and not task.done():
                task.cancel("batch stopped after transport, configuration or host failure")


async def run_bounded(query_ids: list[str], operation: Callable[[str], Awaitable[dict[str, Any]]],
                      concurrency: int, guard: BatchGuard) -> list[dict[str, Any]]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    pending = iter(query_ids)
    results: list[dict[str, Any]] = []

    async def worker() -> None:
        while guard.failure is None:
            query_id = next(pending, None)
            if query_id is None:
                return
            try:
                results.append(await operation(query_id))
            except Exception as error:
                guard.stop(error)
                raise

    tasks = {asyncio.create_task(worker()) for _ in range(min(concurrency, len(query_ids)))}
    guard.tasks.update(tasks)
    try:
        group = asyncio.gather(*tasks, return_exceptions=True)
        try:
            outcomes = await asyncio.shield(group)
        except asyncio.CancelledError as error:
            guard.stop(error)
            await group
            raise
        for outcome in outcomes:
            if isinstance(outcome, Exception):
                raise outcome
    finally:
        guard.tasks.difference_update(tasks)
    return results
