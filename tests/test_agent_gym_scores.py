"""Canonical score responses through both public Agent Gym clients."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest

from promptic_sdk import (
    AgentGymClient,
    AsyncAgentGymClient,
    BenchmarkEvaluationCoverage,
    BenchmarkRunComparison,
    BenchmarkRunResults,
)
from promptic_sdk.agent_gym.models import BenchmarkAggregate

BENCHMARK_ID = "00000000-0000-4000-8000-000000000001"
PARENT_ID = "00000000-0000-4000-8000-000000000002"
CANDIDATE_ID = "00000000-0000-4000-8000-000000000003"

COVERAGE: BenchmarkEvaluationCoverage = {
    "evaluations": {
        "expected": 6,
        "succeeded": 5,
        "failed": 1,
        "skipped": 0,
        "insufficient_evidence": 0,
        "missing": 0,
    },
    "cases": {"total": 3, "fully_evaluated": 2},
}


def _aggregate(
    score: float | None, coverage: BenchmarkEvaluationCoverage | None = COVERAGE
) -> BenchmarkAggregate:
    return {
        "architecture": {"name": "agent", "version": "1"},
        "overall_score": score,
        "evaluation_coverage": coverage,
        "mean_per_field_scores": {"answer": 1.0},
        "mean_per_field_scores_basis": "succeeded_only",
        "success_rate": 1.0,
        "mean_latency_ms": 10,
        "total_token_usage": None,
        "case_count": 3,
    }


@asynccontextmanager
async def _client(
    use_async: bool, payload: BenchmarkRunResults | BenchmarkRunComparison, path: str
) -> AsyncIterator[AgentGymClient | AsyncAgentGymClient]:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == path
        return httpx.Response(200, json=payload)

    if use_async:
        async with AsyncAgentGymClient(api_key="ptc_test") as client:
            await client._transport.api.aclose()
            client._transport.api = httpx.AsyncClient(
                transport=httpx.MockTransport(handler), base_url="https://promptic.test/api/v1"
            )
            yield client
    else:
        with AgentGymClient(api_key="ptc_test") as client:
            client._transport.api.close()
            client._transport.api = httpx.Client(
                transport=httpx.MockTransport(handler), base_url="https://promptic.test/api/v1"
            )
            yield client


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("score,coverage", [(5 / 6, COVERAGE), (0.0, COVERAGE), (None, None)])
async def test_run_scores_preserve_partial_zero_and_unavailable_results(
    use_async: bool, score: float | None, coverage: BenchmarkEvaluationCoverage | None
):
    payload: BenchmarkRunResults = {
        "benchmark_id": BENCHMARK_ID,
        "run_id": CANDIDATE_ID,
        "revision_id": None,
        "revision_fingerprint": None,
        "scorer_contract_version": None,
        "evaluator_fingerprint": None,
        "runtime_location": "external",
        "status": "succeeded",
        "scoring_status": "succeeded",
        "evaluation": {
            "id": "evaluation-1",
            "trigger": "initial",
            "status": "succeeded",
            "evaluators": [
                {
                    "id": "evaluator-1",
                    "source_evaluator_id": "evaluator-1",
                    "metric_key": None,
                    "type": "fieldLevelJudge",
                    "name": "Answer quality",
                    "weight": 1.0,
                    "architecture": {"name": "agent", "version": "1"},
                    "score": score,
                    "mean_per_field_scores": None,
                    "mean_per_field_scores_basis": "succeeded_only",
                    "case_count": 3,
                    "error": None,
                }
            ],
            "created_at": "2026-10-01T12:00:00Z",
            "completed_at": "2026-10-01T12:01:00Z",
        },
        "eligibility": {"status": "eligible", "reasons": []},
        "architectures": [],
        "aggregates": [_aggregate(score, coverage)],
        "progress": {"terminal": 3, "total": 3, "succeeded": 3, "failed": 0},
        "score_status_counts": {"succeeded": 5, "failed": 1},
        "timestamps": {
            "created_at": "2026-10-01T12:00:00Z",
            "started_at": None,
            "finished_at": None,
            "scored_at": None,
        },
        "error": None,
        "insights": [],
        "links": {},
    }
    async with _client(
        use_async, payload, f"/api/v1/benchmarks/{BENCHMARK_ID}/runs/{CANDIDATE_ID}/results"
    ) as client:
        result = (
            await client.get_run_results(BENCHMARK_ID, CANDIDATE_ID)
            if isinstance(client, AsyncAgentGymClient)
            else client.get_run_results(BENCHMARK_ID, CANDIDATE_ID)
        )
    assert result == payload
    assert result["aggregates"][0]["overall_score"] == score
    assert result["aggregates"][0]["evaluation_coverage"] == coverage
    assert "mean_score" not in result["aggregates"][0]
    assert result["evaluation"] is not None
    assert "composite_score" not in result["evaluation"]


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("delta", [0.15, -0.15, 0.0, None])
async def test_comparison_preserves_server_overall_delta(use_async: bool, delta: float | None):
    payload: BenchmarkRunComparison = {
        "benchmark_id": BENCHMARK_ID,
        "comparable_context": {
            "same_revision": True,
            "same_scorer_contract": True,
            "same_evaluator": True,
            "same_case_set": True,
        },
        "parent": {"run_id": PARENT_ID, "architecture": None, "aggregate": _aggregate(0.5)},
        "candidate": {
            "run_id": CANDIDATE_ID,
            "architecture": None,
            "aggregate": _aggregate(None if delta is None else 0.5 + delta),
        },
        "summary": {
            "overall_score_delta": delta,
            "success_rate_delta": 0,
            "mean_latency_delta_ms": 0,
            "improved_cases": 0,
            "regressed_cases": 0,
            "unchanged_cases": 0,
            "incomparable_cases": 3,
        },
        "cases": [],
    }
    async with _client(
        use_async, payload, f"/api/v1/benchmarks/{BENCHMARK_ID}/runs/compare"
    ) as client:
        result = (
            await client.compare_runs(
                BENCHMARK_ID, parent_run_id=PARENT_ID, candidate_run_id=CANDIDATE_ID
            )
            if isinstance(client, AsyncAgentGymClient)
            else client.compare_runs(
                BENCHMARK_ID, parent_run_id=PARENT_ID, candidate_run_id=CANDIDATE_ID
            )
        )
    assert result == payload
    assert result["summary"]["overall_score_delta"] == delta
    assert "mean_score_delta" not in result["summary"]
