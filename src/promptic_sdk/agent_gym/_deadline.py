"""Bound trace finalization even when a synchronous exporter ignores its timeout."""

from __future__ import annotations

import asyncio
import contextvars
import time
from collections.abc import Callable
from concurrent.futures import Future
from threading import BoundedSemaphore, Thread
from typing import TypeVar

T = TypeVar("T")
# A broken third-party exporter cannot be forcibly killed in Python. Bound the
# number of abandoned calls and use daemon threads, not an executor whose shutdown
# waits for them. No background call may mutate submission/prediction state.
_slots = BoundedSemaphore(8)


def remaining(deadline: float) -> float:
    """Return the remaining monotonic budget, or fail before starting more work."""
    budget = deadline - time.monotonic()
    if budget <= 0:
        raise TimeoutError("Trace finalization deadline exceeded")
    return budget


def _start(call: Callable[[], T], slots: BoundedSemaphore, deadline: float) -> Future[T]:
    """Start work with an acquired slot, releasing it on every exit path."""
    result: Future[T] = Future()
    context = contextvars.copy_context()

    def run() -> None:
        try:
            if result.set_running_or_notify_cancel():
                try:
                    result.set_result(context.run(call))
                except BaseException as error:
                    result.set_exception(error)
        finally:
            slots.release()

    try:
        remaining(deadline)
        Thread(target=run, name="promptic-trace-finalization", daemon=True).start()
    except BaseException:
        slots.release()
        raise
    return result


def bounded_call(call: Callable[[], T], deadline: float) -> T:
    """Bound a read/export call without waiting for a non-cooperative worker."""
    slots = _slots
    if not slots.acquire(timeout=remaining(deadline)):
        raise TimeoutError("Trace finalization deadline exceeded while waiting for a worker")
    return _start(call, slots, deadline).result(timeout=remaining(deadline))


async def bounded_call_async(call: Callable[[], T], deadline: float) -> T:
    """Run synchronous export outside the event loop with the same deadline."""
    slots = _slots
    while True:
        budget = remaining(deadline)
        if slots.acquire(blocking=False):
            break
        # Wait without blocking the event loop or allocating another worker.
        await asyncio.sleep(min(0.01, budget))
    return await asyncio.wait_for(
        asyncio.wrap_future(_start(call, slots, deadline)), remaining(deadline)
    )
