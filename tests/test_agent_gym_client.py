"""Tests for Agent Gym authentication and result inspection."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from promptic_sdk import AgentGymAPIError, AgentGymClient, AsyncAgentGymClient

BENCHMARK_ID = "00000000-0000-4000-8000-000000000001"
RUN_ID = "00000000-0000-4000-8000-000000000002"
EVALUATION_RUN_ID = "00000000-0000-4000-8000-000000000008"
SUBMISSION_ID = "00000000-0000-4000-8000-000000000006"
CASE_ID = 3
PARENT_RUN_ID = "00000000-0000-4000-8000-000000000004"
CANDIDATE_RUN_ID = "00000000-0000-4000-8000-000000000005"


def _replace_sync_api(client: AgentGymClient, handler: Any) -> None:
    client._transport.api.close()
    client._transport.api = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://promptic.test/api/v1",
        headers=client._client.headers,
    )
    client._client = client._transport.api


async def _replace_async_api(client: AsyncAgentGymClient, handler: Any) -> None:
    await client._transport.api.aclose()
    client._transport.api = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://promptic.test/api/v1",
        headers=client._client.headers,
    )
    client._client = client._transport.api


def _case(case_id: int) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "prediction_id": "00000000-0000-4000-8000-000000000003",
        "architecture": {"name": "agent", "version": "1"},
        "status": "succeeded",
        "output": {"kind": "text", "value": "ok"},
        "overall_score": 0.5,
        "scores": [],
        "judgements": [],
        "artifacts": [],
        "traces": [],
        "trace_artifact_ids": [],
        "metrics": {"latency_ms": 10, "token_usage": None},
        "execution": {
            "runtime_location": "external",
            "implementation_reference_id": None,
            "executor_id": None,
            "executor_version": None,
        },
        "error": None,
        "diagnostics": None,
        "insights": [],
    }


def _dispatch_failed_status() -> dict[str, Any]:
    return {
        "submission_id": SUBMISSION_ID,
        "revision_id": "00000000-0000-4000-8000-000000000007",
        "status": "dispatch_failed",
        "expires_at": "2026-08-20T12:00:00Z",
        "prediction_count": 1,
        "expected_prediction_count": 1,
        "submitted_at": "2026-08-19T12:00:00Z",
        "queued_at": "2026-08-19T12:00:00Z",
        "completed_at": None,
        "validation_error": None,
        "submit_metadata": {"runtime": "external"},
        "dispatch": {
            "status": "failed",
            "attempts": 3,
            "retry_available_at": "2026-08-19T12:05:00Z",
            "last_error": "service unavailable",
        },
        "run": {
            "id": RUN_ID,
            "status": "queued",
            "scoring_status": "pending",
            "eligibility_status": "pending",
            "eligibility_reasons": [],
            "scored_at": None,
            "error": None,
        },
    }


class TestAuthentication:
    def test_api_key_uses_bearer_without_application_header(self, monkeypatch):
        monkeypatch.delenv("PROMPTIC_ACCESS_TOKEN", raising=False)
        with AgentGymClient(api_key="ptc_test", endpoint="https://promptic.test") as client:
            assert client._client.headers["Authorization"] == "Bearer ptc_test"
            assert "X-AI-Application-Id" not in client._client.headers

    def test_explicit_api_key_suppresses_environment_access_token(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_ACCESS_TOKEN", "session-token")
        with AgentGymClient(api_key="ptc_test", endpoint="https://promptic.test") as client:
            assert client._client.headers["Authorization"] == "Bearer ptc_test"
            assert "X-AI-Application-Id" not in client._client.headers

    def test_access_token_prefers_canonical_application_header(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "ptc_ignored")
        monkeypatch.setenv("PROMPTIC_ACCESS_TOKEN", "session-token")
        with AgentGymClient(
            ai_application_id="app-1",
        ) as client:
            assert client._client.headers["Authorization"] == "Bearer session-token"
            assert client._client.headers["X-AI-Application-Id"] == "app-1"

    def test_access_token_requires_application_scope(self, monkeypatch):
        monkeypatch.delenv("PROMPTIC_AI_APPLICATION_ID", raising=False)
        monkeypatch.delenv("PROMPTIC_WORKSPACE_ID", raising=False)
        monkeypatch.setenv("PROMPTIC_ACCESS_TOKEN", "session-token")
        monkeypatch.delenv("PROMPTIC_API_KEY", raising=False)
        monkeypatch.setattr("promptic_sdk.cli.config.load_config", lambda: None)
        with pytest.raises(ValueError, match="ai_application_id is required"):
            AgentGymClient()


class TestResults:
    @pytest.mark.parametrize(
        "token_usage",
        [
            {"prompt": 1, "completion": 2},
            {"prompt": 1, "completion": 2, "total": None},
            {"prompt": 1, "completion": 2, "total": "3"},
            {"prompt": 1, "completion": 2, "total": True},
        ],
    )
    def test_upload_predictions_rejects_invalid_token_usage(self, token_usage):
        with (
            AgentGymClient(api_key="ptc_test") as client,
            pytest.raises(ValueError, match=r"token_usage\.(prompt|completion|total)"),
        ):
            client.upload_predictions(
                BENCHMARK_ID,
                SUBMISSION_ID,
                [
                    {
                        "dataset_case_id": 1,
                        "status": "succeeded",
                        "output": "ok",
                        "token_usage": token_usage,
                    }
                ],
            )

    def test_upload_predictions_uses_the_staged_batch_endpoint(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "PUT"
            assert request.url.path == (
                f"/api/v1/benchmarks/{BENCHMARK_ID}/submissions/{SUBMISSION_ID}/predictions"
            )
            assert request.read()
            return httpx.Response(200, json={"accepted": 1, "stored": 1})

        with AgentGymClient(api_key="ptc_test") as client:
            _replace_sync_api(client, handler)
            result = client.upload_predictions(
                BENCHMARK_ID,
                SUBMISSION_ID,
                [{"dataset_case_id": 1, "status": "succeeded", "output": "ok"}],
            )

        assert result == {"accepted": 1, "stored": 1}

    def test_upload_predictions_retries_transient_failures_idempotently(self, monkeypatch):
        attempts = 0
        monkeypatch.setattr("promptic_sdk.agent_gym.client.time.sleep", lambda _seconds: None)

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                return httpx.Response(503, json={"error": "temporarily_unavailable"})
            return httpx.Response(200, json={"accepted": 1, "stored": 1})

        with AgentGymClient(api_key="ptc_test") as client:
            _replace_sync_api(client, handler)
            result = client.upload_predictions(
                BENCHMARK_ID,
                SUBMISSION_ID,
                [{"dataset_case_id": 1, "status": "succeeded", "output": "ok"}],
            )

        assert attempts == 3
        assert result == {"accepted": 1, "stored": 1}

    @pytest.mark.asyncio
    async def test_async_upload_predictions_retries_transient_failures(self, monkeypatch):
        attempts = 0

        async def no_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr("promptic_sdk.agent_gym.client.asyncio.sleep", no_sleep)

        async def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(429, json={"error": "rate_limited"})
            return httpx.Response(200, json={"accepted": 1, "stored": 1})

        async with AsyncAgentGymClient(api_key="ptc_test") as client:
            await _replace_async_api(client, handler)
            result = await client.upload_predictions(
                BENCHMARK_ID,
                SUBMISSION_ID,
                [{"dataset_case_id": 1, "status": "succeeded", "output": "ok"}],
            )

        assert attempts == 2
        assert result == {"accepted": 1, "stored": 1}

    def test_submit_exposes_recoverable_dispatch_failure(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == (
                f"/api/v1/benchmarks/{BENCHMARK_ID}/submissions/{SUBMISSION_ID}/submit"
            )
            return httpx.Response(
                503,
                json={
                    "error": "scoring_dispatch_failed",
                    "message": "Retry this idempotent submit request.",
                    "submission_id": SUBMISSION_ID,
                    "run_id": RUN_ID,
                    "variant_id": "00000000-0000-4000-8000-000000000009",
                    "status": "dispatch_failed",
                    "dispatch_status": "failed",
                    "dispatch_attempts": 3,
                    "created": True,
                },
            )

        with AgentGymClient(api_key="ptc_test") as client:
            _replace_sync_api(client, handler)
            with pytest.raises(AgentGymAPIError) as captured:
                client.submit_submission(
                    BENCHMARK_ID,
                    SUBMISSION_ID,
                    {
                        "variant_identity": {"name": "candidate", "version": "1"},
                    },
                    idempotency_key="submit-dispatch-failure",
                )

        assert captured.value.code == "scoring_dispatch_failed"
        assert captured.value.details is not None
        assert captured.value.details["status"] == "dispatch_failed"
        assert captured.value.details["dispatch_attempts"] == 3

    def test_wait_returns_recoverable_dispatch_failure_without_polling(self):
        requests = 0

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal requests
            requests += 1
            return httpx.Response(200, json=_dispatch_failed_status())

        with AgentGymClient(api_key="ptc_test") as client:
            _replace_sync_api(client, handler)
            status = client.wait_for_submission(BENCHMARK_ID, SUBMISSION_ID, max_wait=1)

        assert status["status"] == "dispatch_failed"
        assert status["dispatch"] == {
            "status": "failed",
            "attempts": 3,
            "retry_available_at": "2026-08-19T12:05:00Z",
            "last_error": "service unavailable",
        }
        assert requests == 1

    @pytest.mark.asyncio
    async def test_async_wait_returns_recoverable_dispatch_failure_without_polling(self):
        requests = 0

        async def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal requests
            requests += 1
            return httpx.Response(200, json=_dispatch_failed_status())

        async with AsyncAgentGymClient(api_key="ptc_test") as client:
            await _replace_async_api(client, handler)
            status = await client.wait_for_submission(BENCHMARK_ID, SUBMISSION_ID, max_wait=1)

        assert status["status"] == "dispatch_failed"
        assert status["run"] is not None
        assert status["run"]["id"] == RUN_ID
        assert requests == 1

    def test_retry_scoring_is_idempotent_for_the_same_run_and_maps_errors(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if len(requests) == 3:
                return httpx.Response(409, json={"error": "official_score_exists"})
            return httpx.Response(
                202,
                json={
                    "run_id": RUN_ID,
                    "scoring_status": "pending",
                    "dispatch_status": "dispatched",
                    "dispatch_attempts": 1,
                },
            )

        with AgentGymClient(api_key="ptc_test") as client:
            _replace_sync_api(client, handler)
            first = client.retry_scoring(BENCHMARK_ID, RUN_ID)
            second = client.retry_scoring(BENCHMARK_ID, RUN_ID)
            with pytest.raises(AgentGymAPIError) as raised:
                client.retry_scoring(BENCHMARK_ID, RUN_ID)

        assert first["run_id"] == second["run_id"] == RUN_ID
        assert {request.url.path for request in requests} == {
            f"/api/v1/benchmarks/{BENCHMARK_ID}/runs/{RUN_ID}/retry-scoring"
        }
        assert raised.value.status_code == 409
        assert raised.value.code == "official_score_exists"

    @pytest.mark.asyncio
    async def test_async_retry_scoring_reuses_the_existing_run(self):
        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                202,
                json={
                    "run_id": RUN_ID,
                    "scoring_status": "pending",
                    "dispatch_status": "dispatched",
                    "dispatch_attempts": 1,
                },
            )

        async with AsyncAgentGymClient(api_key="ptc_test") as client:
            await _replace_async_api(client, handler)
            response = await client.retry_scoring(BENCHMARK_ID, RUN_ID)

        assert response["run_id"] == RUN_ID

    def test_routes_queries_sort_and_pagination(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.path.endswith("/results"):
                return httpx.Response(200, json={"run_id": RUN_ID, "aggregates": []})
            if request.url.path.endswith("/reevaluate"):
                return httpx.Response(
                    202,
                    json={
                        "run_id": RUN_ID,
                        "evaluation_run_id": EVALUATION_RUN_ID,
                        "status": "queued",
                    },
                )
            if request.url.path.endswith("/case-results"):
                cursor = request.url.params.get("cursor")
                return httpx.Response(
                    200,
                    json={
                        "run_id": RUN_ID,
                        "data": [_case(CASE_ID)] if cursor is None else [],
                        "total": 1,
                        "next_cursor": "next" if cursor is None else None,
                    },
                )
            if "/case-results/" in request.url.path:
                return httpx.Response(200, json=_case(CASE_ID))
            if request.url.path.endswith("/runs/compare"):
                return httpx.Response(200, json={"benchmark_id": BENCHMARK_ID, "cases": []})
            return httpx.Response(404, json={"error": "not_found"})

        with AgentGymClient(api_key="ptc_test") as client:
            _replace_sync_api(client, handler)
            assert client.get_run_results(BENCHMARK_ID, RUN_ID)["run_id"] == RUN_ID
            reevaluation = client.reevaluate_run(BENCHMARK_ID, RUN_ID)
            assert reevaluation == {
                "run_id": RUN_ID,
                "evaluation_run_id": EVALUATION_RUN_ID,
                "status": "queued",
            }
            page = client.list_case_results(
                BENCHMARK_ID, RUN_ID, cursor="cursor-1", limit=17, sort="latency"
            )
            assert page["total"] == 1
            assert client.get_case_result(BENCHMARK_ID, RUN_ID, CASE_ID)["case_id"] == CASE_ID
            client.compare_runs(
                BENCHMARK_ID,
                parent_run_id=PARENT_RUN_ID,
                candidate_run_id=CANDIDATE_RUN_ID,
            )
            assert [row["case_id"] for row in client.iter_case_results(BENCHMARK_ID, RUN_ID)] == [
                CASE_ID
            ]

        page_request = requests[2]
        assert (
            page_request.url.path == f"/api/v1/benchmarks/{BENCHMARK_ID}/runs/{RUN_ID}/case-results"
        )
        assert dict(page_request.url.params) == {
            "limit": "17",
            "sort": "latency",
            "cursor": "cursor-1",
        }
        comparison = requests[4]
        assert dict(comparison.url.params) == {
            "parent_run_id": PARENT_RUN_ID,
            "candidate_run_id": CANDIDATE_RUN_ID,
        }

    @pytest.mark.asyncio
    async def test_async_reevaluation_exposes_immutable_operation_id(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == (
                f"/api/v1/benchmarks/{BENCHMARK_ID}/runs/{RUN_ID}/reevaluate"
            )
            return httpx.Response(
                202,
                json={
                    "run_id": RUN_ID,
                    "evaluation_run_id": EVALUATION_RUN_ID,
                    "status": "queued",
                },
            )

        async with AsyncAgentGymClient(api_key="ptc_test") as client:
            await _replace_async_api(client, handler)
            reevaluation = await client.reevaluate_run(BENCHMARK_ID, RUN_ID)

        assert reevaluation["run_id"] == RUN_ID
        assert reevaluation["evaluation_run_id"] == EVALUATION_RUN_ID
        assert reevaluation["status"] == "queued"

    def test_repeated_result_cursor_is_rejected(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"run_id": RUN_ID, "data": [], "total": 0, "next_cursor": "same"},
            )

        with AgentGymClient(api_key="ptc_test") as client:
            _replace_sync_api(client, handler)
            with pytest.raises(RuntimeError, match="repeated cursor"):
                list(client.iter_case_results(BENCHMARK_ID, RUN_ID))

    def test_structured_error_mapping_preserves_comparison_context(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                409,
                json={
                    "error": "runs_not_comparable",
                    "comparable_context": {"same_evaluator": False},
                },
            )

        with AgentGymClient(api_key="ptc_test") as client:
            _replace_sync_api(client, handler)
            with pytest.raises(AgentGymAPIError) as raised:
                client.compare_runs(
                    BENCHMARK_ID,
                    parent_run_id=PARENT_RUN_ID,
                    candidate_run_id=CANDIDATE_RUN_ID,
                )
        assert raised.value.status_code == 409
        assert raised.value.code == "runs_not_comparable"
        assert raised.value.details == {"comparable_context": {"same_evaluator": False}}

    @pytest.mark.asyncio
    async def test_sync_async_request_parity(self):
        sync_requests: list[tuple[str, str, str]] = []
        async_requests: list[tuple[str, str, str]] = []

        def sync_handler(request: httpx.Request) -> httpx.Response:
            sync_requests.append((request.method, request.url.path, request.url.query.decode()))
            return httpx.Response(
                200,
                json={"run_id": RUN_ID, "data": [], "total": 0, "next_cursor": None},
            )

        async def async_handler(request: httpx.Request) -> httpx.Response:
            async_requests.append((request.method, request.url.path, request.url.query.decode()))
            return httpx.Response(
                200,
                json={"run_id": RUN_ID, "data": [], "total": 0, "next_cursor": None},
            )

        with AgentGymClient(api_key="ptc_test") as sync_client:
            _replace_sync_api(sync_client, sync_handler)
            sync_client.list_case_results(
                BENCHMARK_ID, RUN_ID, cursor="opaque", limit=23, sort="score_desc"
            )
        async with AsyncAgentGymClient(api_key="ptc_test") as async_client:
            await _replace_async_api(async_client, async_handler)
            await async_client.list_case_results(
                BENCHMARK_ID, RUN_ID, cursor="opaque", limit=23, sort="score_desc"
            )

        assert async_requests == sync_requests
