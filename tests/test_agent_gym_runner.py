"""Tests for the high-level Agent Gym candidate runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from promptic_sdk import (
    AgentGymCase,
    AgentGymCaseResult,
    AgentGymRunResult,
    MaterializedManifest,
)
from promptic_sdk.agent_gym import AgentGymClient, AsyncAgentGymClient
from promptic_sdk.agent_gym_models import (
    BundleIdentity,
    ExternalPrediction,
    ExternalSubmissionManifest,
    SubmissionStatus,
)
from promptic_sdk.agent_gym_runner import submit_benchmark, submit_benchmark_async

BENCHMARK_ID = "11111111-1111-4111-8111-111111111111"
SUBMISSION_ID = "22222222-2222-4222-8222-222222222222"
REVISION_ID = "33333333-3333-4333-8333-333333333333"
RUN_ID = "44444444-4444-4444-8444-444444444444"
BUNDLE_ID = "55555555-5555-4555-8555-555555555555"
CASE_IDS = (
    "66666666-6666-4666-8666-666666666666",
    "77777777-7777-4777-8777-777777777777",
)
RAW_TRACE_ID = "0123456789abcdef0123456789abcdef"
TRACE_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


def _manifest() -> ExternalSubmissionManifest:
    return {
        "submission_id": SUBMISSION_ID,
        "revision": {
            "id": REVISION_ID,
            "version": 1,
            "fingerprint": "fingerprint",
            "case_count": 2,
        },
        "task": {
            "taskId": None,
            "name": "Report task",
            "description": "Create a report.",
            "inputContract": {},
            "outputContract": {"kind": "artifact"},
            "publicSuccessCriteria": None,
        },
        "data": [
            {
                "case_id": CASE_IDS[0],
                "ordinal": 0,
                "input_payload": {"title": "First"},
                "input_files": [],
            },
            {
                "case_id": CASE_IDS[1],
                "ordinal": 1,
                "input_payload": {"title": "Second"},
                "input_files": [],
            },
        ],
        "next_cursor": None,
    }


def _status() -> SubmissionStatus:
    return {
        "submission_id": SUBMISSION_ID,
        "revision_id": REVISION_ID,
        "status": "succeeded",
        "expires_at": "2026-07-30T12:00:00Z",
        "finalized_at": "2026-07-29T12:00:00Z",
        "queued_at": "2026-07-29T12:00:01Z",
        "completed_at": "2026-07-29T12:01:00Z",
        "validation_error": None,
        "run": {
            "id": RUN_ID,
            "status": "succeeded",
            "scoring_status": "succeeded",
            "eligibility_status": "eligible",
            "eligibility_reasons": [],
            "scored_at": "2026-07-29T12:01:00Z",
            "error": None,
        },
    }


class _SyncSession:
    revision_id = REVISION_ID

    def __init__(self) -> None:
        self.predictions: list[ExternalPrediction] = []
        self.identity: BundleIdentity | None = None
        self.uploaded_paths: list[str] = []
        self.cancelled = False

    def materialize_manifest(self, destination: Path) -> MaterializedManifest:
        destination.mkdir(parents=True, exist_ok=True)
        return MaterializedManifest(
            root=destination,
            manifest_path=destination / "manifest.json",
            files=(),
            manifest=_manifest(),
        )

    def upload_artifact_file(
        self,
        source: Path,
        *,
        path: str,
        mime_type: str | None,
        role: str,
    ) -> dict[str, str]:
        assert source.is_file()
        assert mime_type is None
        assert role == "output"
        self.uploaded_paths.append(path)
        return {
            "artifact_id": "88888888-8888-4888-8888-888888888888",
            "storage_object_id": "99999999-9999-4999-8999-999999999999",
            "path": path,
            "status": "verified",
        }

    def finalize(
        self,
        *,
        bundle_identity: BundleIdentity,
        predictions: list[ExternalPrediction],
        idempotency_key: str,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        assert idempotency_key == "run-1:finalize"
        assert metadata == {"source": "test"}
        self.identity = bundle_identity
        self.predictions = predictions
        return {
            "submission_id": SUBMISSION_ID,
            "run_id": RUN_ID,
            "bundle_id": BUNDLE_ID,
            "status": "queued",
            "dispatch_status": "dispatched",
            "created": True,
        }

    def wait(self, *, max_wait: float, poll_interval: float) -> SubmissionStatus:
        assert max_wait == 600
        assert poll_interval == 2
        return _status()

    def status(self) -> SubmissionStatus:
        return _status()

    def wait_for_resolved_traces(
        self,
        raw_trace_ids: list[str],
        *,
        max_wait: float,
        poll_interval: float,
    ) -> list[str]:
        assert raw_trace_ids == [RAW_TRACE_ID]
        assert max_wait == 30
        assert poll_interval == 0.5
        return [TRACE_ID]

    def cancel(self) -> None:
        self.cancelled = True


class _SyncClient:
    def __init__(self, session: _SyncSession) -> None:
        self.session = session

    def start_submission(
        self,
        benchmark_id: str,
        *,
        revision_id: str | None,
        idempotency_key: str,
    ) -> _SyncSession:
        assert benchmark_id == BENCHMARK_ID
        assert revision_id is None
        assert idempotency_key == "run-1:create"
        return self.session


def test_submit_runs_cases_uploads_artifacts_and_records_failures(tmp_path: Path):
    session = _SyncSession()

    def candidate(case: AgentGymCase) -> AgentGymCaseResult:
        if case.ordinal == 1:
            raise RuntimeError("candidate failed")
        report = tmp_path / "report.html"
        report.write_text("<h1>Report</h1>")
        return AgentGymCaseResult.artifact(
            report,
            summary="Report generated",
            raw_trace_ids=(RAW_TRACE_ID,),
        )

    result = submit_benchmark(
        cast(AgentGymClient, _SyncClient(session)),
        BENCHMARK_ID,
        candidate,
        name="report-agent",
        version="1.0.0",
        architecture_description="Generates an HTML report.",
        metadata={"source": "test"},
        workdir=tmp_path / "work",
        idempotency_key="run-1",
    )

    assert isinstance(result, AgentGymRunResult)
    assert result.run_id == RUN_ID
    assert session.identity == {
        "name": "report-agent",
        "version": "1.0.0",
        "architecture_description": "Generates an HTML report.",
    }
    assert session.uploaded_paths == [f"cases/{CASE_IDS[0]}/report.html"]
    assert session.predictions[0]["output"] == {
        "kind": "artifact",
        "value": {"summary": "Report generated"},
    }
    assert session.predictions[0]["execution_refs"] == {"trace_ids": [TRACE_ID]}
    assert session.predictions[1]["status"] == "failed"
    assert session.predictions[1]["error_code"] == "candidate_exception"
    assert session.predictions[1]["error"] == "candidate failed"
    assert result.status["run"] == {
        "id": RUN_ID,
        "status": "succeeded",
        "scoring_status": "succeeded",
        "eligibility_status": "eligible",
        "eligibility_reasons": [],
        "scored_at": "2026-07-29T12:01:00Z",
        "error": None,
    }


class _AsyncSession:
    revision_id = REVISION_ID

    def __init__(self) -> None:
        self._sync = _SyncSession()

    @property
    def predictions(self) -> list[ExternalPrediction]:
        return self._sync.predictions

    async def materialize_manifest(self, destination: Path) -> MaterializedManifest:
        return self._sync.materialize_manifest(destination)

    async def upload_artifact_file(
        self,
        source: Path,
        *,
        path: str,
        mime_type: str | None,
        role: str,
    ) -> dict[str, str]:
        return self._sync.upload_artifact_file(
            source,
            path=path,
            mime_type=mime_type,
            role=role,
        )

    async def finalize(
        self,
        *,
        bundle_identity: BundleIdentity,
        predictions: list[ExternalPrediction],
        idempotency_key: str,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return self._sync.finalize(
            bundle_identity=bundle_identity,
            predictions=predictions,
            idempotency_key=idempotency_key,
            metadata=metadata,
        )

    async def wait(self, *, max_wait: float, poll_interval: float) -> SubmissionStatus:
        return self._sync.wait(max_wait=max_wait, poll_interval=poll_interval)

    async def status(self) -> SubmissionStatus:
        return self._sync.status()

    async def cancel(self) -> None:
        self._sync.cancel()


class _AsyncClient:
    def __init__(self, session: _AsyncSession) -> None:
        self.session = session

    async def start_submission(
        self,
        benchmark_id: str,
        *,
        revision_id: str | None,
        idempotency_key: str,
    ) -> _AsyncSession:
        assert benchmark_id == BENCHMARK_ID
        assert revision_id is None
        assert idempotency_key == "run-1:create"
        return self.session


@pytest.mark.asyncio
async def test_async_submit_accepts_a_simple_mapping_result(tmp_path: Path):
    session = _AsyncSession()

    async def candidate(case: AgentGymCase) -> dict[str, Any]:
        return {"title": cast(str, case.input["title"])}

    result = await submit_benchmark_async(
        cast(AsyncAgentGymClient, _AsyncClient(session)),
        BENCHMARK_ID,
        candidate,
        name="structured-agent",
        version="1.0.0",
        metadata={"source": "test"},
        workdir=tmp_path,
        idempotency_key="run-1",
        wait=False,
    )

    assert result.run_id == RUN_ID
    assert [prediction["output"]["kind"] for prediction in session.predictions] == [
        "structured",
        "structured",
    ]
