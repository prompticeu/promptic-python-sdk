"""Shared Agent Gym result-inspection request construction."""

from __future__ import annotations

from promptic_sdk.agent_gym._http import RequestSpec
from promptic_sdk.agent_gym.models import CaseResultSort, PredictionArtifact
from promptic_sdk.agent_gym.submissions import require_uuid, validate_artifact_path, validate_sha256


def run_results_request(benchmark_id: str, run_id: str) -> RequestSpec:
    """Build an aggregate run-results request."""
    require_uuid(benchmark_id, "benchmark_id")
    require_uuid(run_id, "run_id")
    return RequestSpec("GET", f"/benchmarks/{benchmark_id}/runs/{run_id}/results")


def case_results_request(
    benchmark_id: str,
    run_id: str,
    *,
    cursor: str | None,
    limit: int,
    sort: CaseResultSort,
) -> RequestSpec:
    """Build one cursor-paginated case-results request."""
    require_uuid(benchmark_id, "benchmark_id")
    require_uuid(run_id, "run_id")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
        raise ValueError("limit must be an integer between 1 and 500")
    if sort not in {"score", "score_desc", "latency", "case"}:
        raise ValueError("sort must be one of: score, score_desc, latency, case")
    params: dict[str, str | int] = {"limit": limit, "sort": sort}
    if cursor is not None:
        params["cursor"] = cursor
    return RequestSpec(
        "GET",
        f"/benchmarks/{benchmark_id}/runs/{run_id}/case-results",
        params=params,
    )


def case_result_request(benchmark_id: str, run_id: str, case_id: int | str) -> RequestSpec:
    """Build an individual case-result request."""
    require_uuid(benchmark_id, "benchmark_id")
    require_uuid(run_id, "run_id")
    if isinstance(case_id, str):
        require_uuid(case_id, "case_id")
    elif isinstance(case_id, bool) or not isinstance(case_id, int) or case_id <= 0:
        raise ValueError("case_id must be a positive integer or prediction UUID")
    return RequestSpec(
        "GET",
        f"/benchmarks/{benchmark_id}/runs/{run_id}/case-results/{case_id}",
    )


def reevaluate_run_request(benchmark_id: str, run_id: str) -> RequestSpec:
    """Build a request to score an execution-current run with active evaluators."""
    require_uuid(benchmark_id, "benchmark_id")
    require_uuid(run_id, "run_id")
    return RequestSpec("POST", f"/benchmarks/{benchmark_id}/runs/{run_id}/reevaluate", json={})


def retry_scoring_request(benchmark_id: str, run_id: str) -> RequestSpec:
    """Build an idempotent request to restore scoring delivery for the same run."""
    require_uuid(benchmark_id, "benchmark_id")
    require_uuid(run_id, "run_id")
    return RequestSpec("POST", f"/benchmarks/{benchmark_id}/runs/{run_id}/retry-scoring", json={})


def compare_runs_request(
    benchmark_id: str, *, parent_run_id: str, candidate_run_id: str
) -> RequestSpec:
    """Build a paired benchmark-run comparison request."""
    require_uuid(benchmark_id, "benchmark_id")
    require_uuid(parent_run_id, "parent_run_id")
    require_uuid(candidate_run_id, "candidate_run_id")
    if parent_run_id == candidate_run_id:
        raise ValueError("parent_run_id and candidate_run_id must be distinct")
    return RequestSpec(
        "GET",
        f"/benchmarks/{benchmark_id}/runs/compare",
        params={"parent_run_id": parent_run_id, "candidate_run_id": candidate_run_id},
    )


def prediction_artifact_request(artifact: PredictionArtifact) -> RequestSpec:
    """Build a download request from trusted artifact identity, not its supplied URL."""
    storage_object_id = require_uuid(artifact["storage_object_id"], "storage_object_id")
    path = artifact.get("path")
    if path is not None:
        validate_artifact_path(path)
    size = artifact.get("size_bytes")
    if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
        raise ValueError("artifact size_bytes must be a non-negative integer or None")
    digest = artifact.get("sha256")
    if digest is not None:
        validate_sha256(digest)
    return RequestSpec("GET", f"/storage-objects/{storage_object_id}/content")
