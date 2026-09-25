"""Wall-clock regression tests for trace evidence and finalization budgets."""

import asyncio
import time
from threading import BoundedSemaphore, Event, Timer
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import httpx
import pytest

from promptic_sdk import AgentGymClient, AsyncAgentGymClient, UnresolvedTraceError
from promptic_sdk.agent_gym import _deadline
from promptic_sdk.agent_gym._http import RequestSpec
from promptic_sdk.agent_gym.client import AsyncExternalSubmissionSession, ExternalSubmissionSession
from promptic_sdk.agent_gym.models import ExternalSubmissionManifest
from promptic_sdk.agent_gym.runner import _flush_traces
from promptic_sdk.agent_gym.submissions import SubmissionPredictionBuilder
from tests.test_agent_gym_runner import (
    BENCHMARK_ID,
    CASE_ID,
    RAW_TRACE_ID,
    SUBMISSION_ID,
    TRACE_DB_ID,
    _manifest,
)


def builder(raw=True):
    value = SubmissionPredictionBuilder()
    value.set_manifest(cast(ExternalSubmissionManifest, _manifest()))
    value.stage(
        CASE_ID,
        {"dataset_case_id": CASE_ID, "status": "succeeded", "output": "ok"},
        raw_trace_ids=[RAW_TRACE_ID] if raw else [],
    )
    return value


def test_required_policy_rejects_empty_ids_per_successful_case():
    value = builder(raw=False)
    with pytest.raises(UnresolvedTraceError) as failure:
        value.validate_trace_coverage("required")
    assert failure.value.code == "required_trace_missing"
    assert failure.value.case_ids == [CASE_ID]
    assert f"successful case(s) have no trace evidence: {[CASE_ID]}" in str(failure.value)
    assert "were not resolved" not in str(failure.value)
    value.validate_trace_coverage("best_effort")
    value.predictions[CASE_ID]["execution_refs"] = {"trace_ids": [TRACE_DB_ID]}
    value.validate_trace_coverage("required")


def test_unresolved_trace_error_preserves_resolution_message():
    assert str(UnresolvedTraceError([RAW_TRACE_ID])) == (
        "trace_resolution_timeout: 1 trace ID(s) were not resolved"
    )


@pytest.mark.parametrize("timeout", [None, 0.25])
def test_sync_transport_preserves_default_timeout_or_applies_override(timeout):
    def handle(request):
        assert (
            request.extensions["timeout"]
            == httpx.Timeout(7 if timeout is None else timeout).as_dict()
        )
        return httpx.Response(200, json={})

    with AgentGymClient(api_key="ptc_test") as client:
        client._transport.api.close()
        client._transport.api = httpx.Client(
            base_url="https://promptic.test", timeout=7, transport=httpx.MockTransport(handle)
        )
        assert client._transport.request(RequestSpec("GET", "/test", timeout=timeout)) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("request_timeout", [None, 0.25])
async def test_async_transport_preserves_default_timeout_or_applies_override(request_timeout):
    def handle(request):
        assert (
            request.extensions["timeout"]
            == httpx.Timeout(7 if request_timeout is None else request_timeout).as_dict()
        )
        return httpx.Response(200, json={})

    async with AsyncAgentGymClient(api_key="ptc_test") as client:
        await client._transport.api.aclose()
        client._transport.api = httpx.AsyncClient(
            base_url="https://promptic.test", timeout=7, transport=httpx.MockTransport(handle)
        )
        assert (
            await client._transport.request(RequestSpec("GET", "/test", timeout=request_timeout))
            == {}
        )


def test_flush_false_is_an_export_failure(monkeypatch):
    monkeypatch.setattr(
        "opentelemetry.trace.get_tracer_provider",
        lambda: SimpleNamespace(force_flush=lambda **kwargs: False),
    )
    with pytest.raises(RuntimeError, match="did not complete"):
        _flush_traces(10)
    with AgentGymClient(api_key="ptc_test") as client:
        session = ExternalSubmissionSession(client, BENCHMARK_ID, SUBMISSION_ID)
        session._builder = builder()
        with pytest.raises(UnresolvedTraceError) as failure:
            session._resolve_staged_traces("required", max_wait=0.5, poll_interval=0.01)
        assert failure.value.code == "trace_export_failed"
        with pytest.warns(RuntimeWarning, match="trace_export_failed"):
            assert (
                session._resolve_staged_traces("best_effort", max_wait=0.5, poll_interval=0.01)
                == {}
            )


@pytest.mark.parametrize("phase", ["flush", "resolution"])
def test_sync_blocking_call_cannot_exceed_combined_budget(monkeypatch, phase):
    release = Event()
    with AgentGymClient(api_key="ptc_test") as client:
        session = ExternalSubmissionSession(client, BENCHMARK_ID, SUBMISSION_ID)
        session._builder = builder()
        monkeypatch.setattr(
            "promptic_sdk.agent_gym.runner._flush_traces",
            lambda *args: release.wait(1) if phase == "flush" else None,
        )
        monkeypatch.setattr(client, "resolve_traces", lambda *args, **kw: release.wait(1))
        started = time.monotonic()
        try:
            with pytest.raises(UnresolvedTraceError) as failure:
                session.submit(
                    idempotency_key="bounded-submit",
                    trace_policy="required",
                    trace_max_wait=0.05,
                    trace_poll_interval=0.01,
                )
            assert time.monotonic() - started < 0.2
            assert failure.value.phase == phase
        finally:
            release.set()


def test_slow_flush_reduces_resolution_budget(monkeypatch):
    monkeypatch.setattr(
        "promptic_sdk.agent_gym.runner._flush_traces", lambda *args: time.sleep(0.04)
    )
    with AgentGymClient(api_key="ptc_test") as client:
        session = ExternalSubmissionSession(client, BENCHMARK_ID, SUBMISSION_ID)
        session._builder = builder()
        resolver = Mock(return_value=[TRACE_DB_ID])
        monkeypatch.setattr(session, "wait_for_resolved_traces", resolver)
        assert session._resolve_staged_traces("required", max_wait=0.2, poll_interval=0.01) == {
            RAW_TRACE_ID: TRACE_DB_ID
        }
        assert 0 < resolver.call_args.kwargs["max_wait"] < 0.18


@pytest.mark.asyncio
async def test_async_flush_is_bounded_without_blocking_event_loop(monkeypatch):
    release = Event()
    monkeypatch.setattr(
        "promptic_sdk.agent_gym.runner._flush_traces", lambda *args: release.wait(1)
    )
    async with AsyncAgentGymClient(api_key="ptc_test") as client:
        session = AsyncExternalSubmissionSession(client, BENCHMARK_ID, SUBMISSION_ID)
        session._builder = builder()
        started = time.monotonic()
        task = asyncio.create_task(
            session.submit(
                idempotency_key="bounded-submit",
                trace_policy="required",
                trace_max_wait=0.05,
                trace_poll_interval=0.01,
            )
        )
        try:
            await asyncio.sleep(0.01)
            assert not task.done()
            with pytest.raises(UnresolvedTraceError) as failure:
                await task
            assert failure.value.code == "trace_flush_timeout"
            assert time.monotonic() - started < 0.2
        finally:
            release.set()


@pytest.mark.asyncio
async def test_async_resolution_request_obeys_remaining_budget(monkeypatch):
    async with AsyncAgentGymClient(api_key="ptc_test") as client:

        async def blocked(*args, **kwargs):
            assert 0 < kwargs["timeout"] <= 0.05
            await asyncio.sleep(1)

        monkeypatch.setattr(client, "resolve_traces", blocked)
        started = time.monotonic()
        with pytest.raises(UnresolvedTraceError):
            await client.wait_for_resolved_traces(
                BENCHMARK_ID, SUBMISSION_ID, [RAW_TRACE_ID], max_wait=0.05
            )
        assert time.monotonic() - started < 0.2


def test_sync_resolution_request_and_polling_obey_budget(monkeypatch):
    with AgentGymClient(api_key="ptc_test") as client:

        def unresolved(*args, **kwargs):
            assert 0 < kwargs["timeout"] <= 0.05
            return {"data": [{"trace_id": RAW_TRACE_ID, "trace_db_id": None}]}

        monkeypatch.setattr(client, "resolve_traces", unresolved)
        started = time.monotonic()
        with pytest.raises(UnresolvedTraceError):
            client.wait_for_resolved_traces(
                BENCHMARK_ID, SUBMISSION_ID, [RAW_TRACE_ID], max_wait=0.05, poll_interval=10
            )
        assert time.monotonic() - started < 0.2


def test_sync_finalization_waits_for_worker_capacity(monkeypatch):
    slots = BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(_deadline, "_slots", slots)
    release = Timer(0.03, slots.release)
    release.start()
    try:
        assert _deadline.bounded_call(lambda: "done", time.monotonic() + 1) == "done"
    finally:
        release.join()


def test_sync_capacity_wait_obeys_deadline_without_starting_work(monkeypatch):
    slots = Mock(spec=BoundedSemaphore)
    slots.acquire.return_value = False
    monkeypatch.setattr(_deadline, "_slots", slots)
    monkeypatch.setattr(_deadline.time, "monotonic", lambda: 100.0)
    call = Mock()
    with pytest.raises(TimeoutError):
        _deadline.bounded_call(call, 100.03)
    slots.acquire.assert_called_once_with(timeout=pytest.approx(0.03))
    call.assert_not_called()
    slots.release.assert_not_called()


@pytest.mark.asyncio
async def test_async_finalization_waits_without_blocking_event_loop(monkeypatch):
    slots = BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(_deadline, "_slots", slots)
    task = asyncio.create_task(_deadline.bounded_call_async(lambda: "done", time.monotonic() + 1))
    await asyncio.sleep(0.03)
    assert not task.done()
    slots.release()
    assert await task == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_async_capacity_wait_deadline_and_cancellation_do_not_start_work(monkeypatch, cancel):
    slots = BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(_deadline, "_slots", slots)
    call = Mock()
    task = asyncio.create_task(_deadline.bounded_call_async(call, time.monotonic() + 0.05))
    if cancel:
        await asyncio.sleep(0.01)
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await task
    call.assert_not_called()
    slots.release()
    assert await _deadline.bounded_call_async(lambda: "next", time.monotonic() + 1) == "next"
