"""Pre-submit trace aborts are reported without masking errors or failing submitted runs."""

import asyncio
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from promptic_sdk import AgentGymClient, AsyncAgentGymClient, UnresolvedTraceError
from promptic_sdk.agent_gym.client import (
    _FAILURE_REPORT_TASKS,
    AsyncExternalSubmissionSession,
    ExternalSubmissionSession,
)
from tests.test_agent_gym_runner import BENCHMARK_ID, SUBMISSION_ID
from tests.test_trace_finalization import builder


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "code",
    [
        "required_trace_missing",
        "trace_flush_timeout",
        "trace_export_failed",
        "trace_resolution_timeout",
    ],
)
@pytest.mark.parametrize("report_result", ["accepted", "late", "offline", "old_server", "invalid"])
@pytest.mark.asyncio
async def test_reports_safe_code_and_preserves_original_error(
    monkeypatch, async_mode, code, report_result
):
    requests = []

    def handle(request):
        requests.append(request)
        assert request.url.path.endswith(f"/submissions/{SUBMISSION_ID}/failure")
        assert request.method == "POST"
        assert request.content == ('{"code":"' + code + '"}').encode()
        assert request.extensions["timeout"] == httpx.Timeout(1).as_dict()
        if report_result == "offline":
            raise httpx.ReadTimeout("private network detail")
        if report_result == "old_server":
            return httpx.Response(404, json={"error": "not_found"})
        if report_result == "invalid":
            return httpx.Response(200, json=None)
        return httpx.Response(
            200,
            json={
                "status": "failed" if report_result == "accepted" else "queued",
                "recorded": report_result == "accepted",
            },
        )

    original = UnresolvedTraceError(["sensitive-raw-trace"], code=code)
    if async_mode:
        async with AsyncAgentGymClient(api_key="ptc_test") as client:
            await client._transport.api.aclose()
            client._transport.api = httpx.AsyncClient(
                base_url="https://promptic.test/api/v1", transport=httpx.MockTransport(handle)
            )
            session = AsyncExternalSubmissionSession(client, BENCHMARK_ID, SUBMISSION_ID)
            session._builder = builder()
            monkeypatch.setattr(session, "_resolve_staged_traces", AsyncMock(side_effect=original))
            with pytest.raises(UnresolvedTraceError) as caught:
                await session.submit(idempotency_key="submit", trace_policy="required")
    else:
        with AgentGymClient(api_key="ptc_test") as client:
            client._transport.api.close()
            client._transport.api = httpx.Client(
                base_url="https://promptic.test/api/v1", transport=httpx.MockTransport(handle)
            )
            session = ExternalSubmissionSession(client, BENCHMARK_ID, SUBMISSION_ID)
            session._builder = builder()
            monkeypatch.setattr(session, "_resolve_staged_traces", Mock(side_effect=original))
            with pytest.raises(UnresolvedTraceError) as caught:
                session.submit(idempotency_key="submit", trace_policy="required")
    assert caught.value is original
    assert original.submission_failure_reported is (report_result == "accepted")
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_async_reporting_timeout_preserves_trace_error(monkeypatch):
    async with AsyncAgentGymClient(api_key="ptc_test") as client:
        session = AsyncExternalSubmissionSession(client, BENCHMARK_ID, SUBMISSION_ID)
        session._builder = builder()
        original = UnresolvedTraceError([])
        monkeypatch.setattr(session, "_resolve_staged_traces", AsyncMock(side_effect=original))
        stopped = asyncio.Event()

        async def blocked(spec):
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        monkeypatch.setattr(client._transport, "request", blocked)
        with pytest.raises(UnresolvedTraceError) as caught:
            await asyncio.wait_for(session.submit(idempotency_key="submit"), timeout=2)
        assert caught.value is original
        await asyncio.wait_for(stopped.wait(), timeout=0.5)
        assert original.submission_failure_reported is False


@pytest.mark.parametrize("cleanup_result", ["cancelled", "error", "accepted"])
@pytest.mark.asyncio
async def test_async_reporting_does_not_wait_for_cancellation_cleanup(monkeypatch, cleanup_result):
    async with AsyncAgentGymClient(api_key="ptc_test") as client:
        session = AsyncExternalSubmissionSession(client, BENCHMARK_ID, SUBMISSION_ID)
        session._builder = builder()
        original = UnresolvedTraceError([])
        monkeypatch.setattr(session, "_resolve_staged_traces", AsyncMock(side_effect=original))
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()
        request_tasks = []

        async def slow_cleanup(spec):
            request_tasks.append(asyncio.current_task())
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup_started.set()
                await release_cleanup.wait()
                if cleanup_result == "error":
                    raise RuntimeError("cleanup failed") from None
                if cleanup_result == "accepted":
                    return {"recorded": True}
                raise

        monkeypatch.setattr(client._transport, "request", slow_cleanup)
        submission = asyncio.create_task(session.submit(idempotency_key="submit"))
        try:
            done, _ = await asyncio.wait({submission}, timeout=1.5)
            assert submission in done, "Reporting must not wait for transport cancellation cleanup"
            with pytest.raises(UnresolvedTraceError) as caught:
                submission.result()
            assert caught.value is original
            await asyncio.wait_for(cleanup_started.wait(), timeout=0.5)
            assert original.submission_failure_reported is False
            assert len(request_tasks) == 1
            assert request_tasks[0] in _FAILURE_REPORT_TASKS
        finally:
            release_cleanup.set()
            await asyncio.gather(submission, *request_tasks, return_exceptions=True)
        # A late response must not change the error's already-returned reporting status.
        assert original.submission_failure_reported is False
        assert request_tasks[0] not in _FAILURE_REPORT_TASKS


@pytest.mark.asyncio
async def test_cancelling_submission_cancels_failure_report_without_waiting_for_cleanup(
    monkeypatch,
):
    async with AsyncAgentGymClient(api_key="ptc_test") as client:
        session = AsyncExternalSubmissionSession(client, BENCHMARK_ID, SUBMISSION_ID)
        session._builder = builder()
        original = UnresolvedTraceError([])
        monkeypatch.setattr(session, "_resolve_staged_traces", AsyncMock(side_effect=original))
        started = asyncio.Event()
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()
        request_tasks = []

        async def slow_cleanup(spec):
            request_tasks.append(asyncio.current_task())
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                await release_cleanup.wait()

        monkeypatch.setattr(client._transport, "request", slow_cleanup)
        submission = asyncio.create_task(session.submit(idempotency_key="submit"))
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            submission.cancel()
            done, _ = await asyncio.wait({submission}, timeout=0.5)
            assert submission in done
            with pytest.raises(asyncio.CancelledError):
                submission.result()
            await asyncio.wait_for(cleanup_started.wait(), timeout=0.5)
            assert original.submission_failure_reported is False
        finally:
            release_cleanup.set()
            await asyncio.gather(submission, *request_tasks, return_exceptions=True)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.asyncio
async def test_ambiguous_final_submit_does_not_send_failure_report(async_mode):
    requests = []

    def handle(request):
        requests.append(request)
        raise httpx.ReadTimeout("The server may have already accepted this submit")

    if async_mode:
        async with AsyncAgentGymClient(api_key="ptc_test") as client:
            await client._transport.api.aclose()
            client._transport.api = httpx.AsyncClient(
                base_url="https://promptic.test", transport=httpx.MockTransport(handle)
            )
            session = AsyncExternalSubmissionSession(client, BENCHMARK_ID, SUBMISSION_ID)
            with pytest.raises(httpx.ReadTimeout):
                await session.submit({}, idempotency_key="submit")
    else:
        with AgentGymClient(api_key="ptc_test") as client:
            client._transport.api.close()
            client._transport.api = httpx.Client(
                base_url="https://promptic.test", transport=httpx.MockTransport(handle)
            )
            session = ExternalSubmissionSession(client, BENCHMARK_ID, SUBMISSION_ID)
            with pytest.raises(httpx.ReadTimeout):
                session.submit({}, idempotency_key="submit")
    assert len(requests) == 1
    assert requests[0].url.path.endswith("/submit")
