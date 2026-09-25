"""End-to-end mocked tests for the trusted Agent Gym callback runner."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypedDict

import httpx
import pytest

from promptic_sdk import (
    AgentGymCase,
    AgentGymCaseResult,
    AgentGymClient,
    AgentGymOutputArtifact,
    AsyncAgentGymClient,
    AsyncExternalSubmissionSession,
    ExternalSubmissionSession,
    MaterializedInputFile,
    UnresolvedTraceError,
    schema_for_model,
)
from promptic_sdk.agent_gym import ExternalPrediction
from promptic_sdk.agent_gym.submissions import prediction_upload_batches

BENCHMARK_ID = "00000000-0000-4000-8000-000000000020"
SUBMISSION_ID = "00000000-0000-4000-8000-000000000021"
REVISION_ID = "00000000-0000-4000-8000-000000000022"
CASE_ID = 23
ARTIFACT_ID = "00000000-0000-4000-8000-000000000024"
STORAGE_ID = "00000000-0000-4000-8000-000000000025"
TRACE_DB_ID = "00000000-0000-4000-8000-000000000026"
TRACE_ARTIFACT_ID = "00000000-0000-4000-8000-000000000029"
RUN_ID = "00000000-0000-4000-8000-000000000027"
BUNDLE_ID = "00000000-0000-4000-8000-000000000028"
RAW_TRACE_ID = "1234567890abcdef1234567890abcdef"
RAW_TRACE_ID_2 = "abcdef1234567890abcdef1234567890"
REPORT = b"<html><body><h1>Case report</h1></body></html>"


@dataclass(frozen=True)
class ReportInput:
    topic: str


class PaperQAInput(TypedDict):
    question: str
    paper: list[MaterializedInputFile]


@dataclass(frozen=True)
class PaperQAOutput:
    answer: str


TASK = {
    "taskId": "task-1",
    "name": "Build a report",
    "description": "Build an HTML or PDF-style report.",
    "inputContract": {},
    "outputContract": {"target_schema": {"type": "object"}},
    "publicSuccessCriteria": None,
}


def _created() -> dict[str, Any]:
    return {
        "submission_id": SUBMISSION_ID,
        "run_id": RUN_ID,
        "variant_id": BUNDLE_ID,
        "revision": {
            "id": REVISION_ID,
            "version": 1,
            "fingerprint": "fingerprint",
            "case_count": 1,
            "scorer_contract_version": "v1",
        },
        "status": "created",
        "expires_at": "2026-08-05T00:00:00.000Z",
        "task": TASK,
        "links": {
            "manifest": f"/submissions/{SUBMISSION_ID}/manifest",
            "artifacts": f"/submissions/{SUBMISSION_ID}/artifacts",
            "predictions": f"/submissions/{SUBMISSION_ID}/predictions",
            "submit": f"/submissions/{SUBMISSION_ID}/submit",
            "status": f"/submissions/{SUBMISSION_ID}",
        },
        "created": True,
    }


def _manifest() -> dict[str, Any]:
    return {
        "submission_id": SUBMISSION_ID,
        "revision": {
            "id": REVISION_ID,
            "version": 1,
            "fingerprint": "fingerprint",
            "case_count": 1,
        },
        "task": TASK,
        "data": [
            {
                "dataset_case_id": CASE_ID,
                "ordinal": 0,
                "input_payload": {"topic": "Agent Gym"},
                "input_files": [],
            }
        ],
        "next_cursor": None,
    }


def _status() -> dict[str, Any]:
    return {
        "submission_id": SUBMISSION_ID,
        "revision_id": REVISION_ID,
        "status": "succeeded",
        "expires_at": "2026-08-05T00:00:00.000Z",
        "prediction_count": 1,
        "expected_prediction_count": 1,
        "submitted_at": "2026-08-04T00:00:00.000Z",
        "queued_at": "2026-08-04T00:00:00.000Z",
        "completed_at": "2026-08-04T00:00:01.000Z",
        "validation_error": None,
        "run": {
            "id": RUN_ID,
            "status": "succeeded",
            "scoring_status": "succeeded",
            "eligibility_status": "eligible",
            "eligibility_reasons": [],
            "scored_at": "2026-08-04T00:00:01.000Z",
            "error": None,
        },
    }


def _case_result() -> dict[str, Any]:
    return {
        "dataset_case_id": CASE_ID,
        "prediction_id": CASE_ID,
        "architecture": {"name": "html-agent", "version": "1.0.0"},
        "status": "succeeded",
        "output": {"report": ["promptic-artifact://00000000-0000-4000-8000-000000000024"]},
        "overall_score": 0.4,
        "scores": [],
        "judgements": [],
        "artifacts": [
            {
                "storage_object_id": STORAGE_ID,
                "path": "report.html",
                "mime_type": "text/html",
                "size_bytes": len(REPORT),
                "role": "output",
                "content_url": f"/api/v1/storage-objects/{STORAGE_ID}/content",
                "download_url": f"/api/v1/storage-objects/{STORAGE_ID}/content?disposition=attachment",
            }
        ],
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


def test_full_trusted_callback_submit_and_inspect_workflow(tmp_path, monkeypatch):
    requests: list[httpx.Request] = []
    submitted_body: dict[str, Any] = {}
    uploaded_predictions: list[dict[str, Any]] = []

    def api_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if request.method == "POST" and path.endswith(f"/benchmarks/{BENCHMARK_ID}/submissions"):
            assert request.headers["Idempotency-Key"] == "workflow-session"
            created_body = json.loads(request.content)
            assert created_body["variant_identity"]["name"] == "html-agent"
            assert created_body["variant_identity"]["architecture_description"]
            return httpx.Response(201, json=_created())
        if request.method == "GET" and path.endswith(f"/{SUBMISSION_ID}/manifest"):
            return httpx.Response(200, json=_manifest())
        if request.method == "POST" and path.endswith(f"/{SUBMISSION_ID}/artifacts"):
            body = json.loads(request.content)
            assert body["path"] == "report.html"
            assert body["size_bytes"] == len(REPORT)
            return httpx.Response(
                201,
                json={
                    "artifact_id": ARTIFACT_ID,
                    "storage_object_id": STORAGE_ID,
                    "path": "report.html",
                    "status": "reserved",
                    "upload": {
                        "strategy": "url",
                        "provider": "s3",
                        "uploadUrl": "https://storage.test/upload",
                        "finalUrl": "https://storage.test/final",
                        "method": "PUT",
                        "headers": {"Content-Type": "text/html"},
                        "maxSizeBytes": len(REPORT),
                        "expiresAt": "2026-08-04T00:15:00.000Z",
                    },
                },
            )
        if request.method == "POST" and path.endswith(f"/{ARTIFACT_ID}/complete"):
            return httpx.Response(200, json={"artifact_id": ARTIFACT_ID, "status": "verified"})
        if request.method == "GET" and path.endswith(f"/{SUBMISSION_ID}/traces"):
            assert request.url.params.get_list("trace_id") == [RAW_TRACE_ID]
            return httpx.Response(
                200,
                json={"data": [{"trace_id": RAW_TRACE_ID, "trace_db_id": TRACE_DB_ID}]},
            )
        if request.method == "PUT" and path.endswith(f"/{SUBMISSION_ID}/predictions"):
            uploaded_predictions.extend(json.loads(request.content)["predictions"])
            return httpx.Response(200, json={"accepted": 1, "stored": 1})
        if request.method == "POST" and path.endswith(f"/{SUBMISSION_ID}/submit"):
            submitted_body.update(json.loads(request.content))
            assert request.headers["Idempotency-Key"] == "workflow-submit"
            return httpx.Response(
                202,
                json={
                    "submission_id": SUBMISSION_ID,
                    "run_id": RUN_ID,
                    "variant_id": BUNDLE_ID,
                    "status": "queued",
                    "dispatch_status": "dispatched",
                    "created": True,
                },
            )
        if request.method == "GET" and path.endswith(f"/{SUBMISSION_ID}"):
            return httpx.Response(200, json=_status())
        if request.method == "GET" and path.endswith(f"/{RUN_ID}/results"):
            return httpx.Response(200, json={"run_id": RUN_ID, "aggregates": []})
        if request.method == "GET" and path.endswith(f"/{RUN_ID}/case-results"):
            return httpx.Response(
                200,
                json={"run_id": RUN_ID, "data": [_case_result()], "total": 1, "next_cursor": None},
            )
        if request.method == "GET" and path.endswith(f"/storage-objects/{STORAGE_ID}/content"):
            return httpx.Response(307, headers={"Location": "https://storage.test/download"})
        return httpx.Response(404, json={"error": "unexpected", "message": path})

    def direct_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert "authorization" not in request.headers
        if request.method == "PUT":
            assert request.content == REPORT
            return httpx.Response(200)
        return httpx.Response(200, content=REPORT, headers={"Content-Length": str(len(REPORT))})

    def candidate(case: AgentGymCase[ReportInput]) -> AgentGymCaseResult[Any]:
        assert case.input == ReportInput(topic="Agent Gym")
        report_path = tmp_path / "report.html"
        report_path.write_bytes(REPORT)
        return AgentGymCaseResult.artifact(
            AgentGymOutputArtifact(report_path, "report", mime_type="text/html"),
        )

    @contextmanager
    def traced_case(*_args: Any, **_kwargs: Any):
        yield RAW_TRACE_ID

    monkeypatch.setattr("promptic_sdk.agent_gym.runner._case_trace", traced_case)

    with AgentGymClient(api_key="ptc_test") as client:
        client._transport.api.close()
        client._transport.direct.close()
        client._transport.api = httpx.Client(
            transport=httpx.MockTransport(api_handler),
            base_url="https://promptic.test/api/v1",
            headers={"Authorization": "Bearer ptc_test"},
        )
        client._transport.direct = httpx.Client(
            transport=httpx.MockTransport(direct_handler), follow_redirects=True
        )
        client._client = client._transport.api
        client._direct_client = client._transport.direct

        submitted = client.run_and_submit(
            BENCHMARK_ID,
            candidate,
            name="html-agent",
            version="1.0.0",
            architecture_description="Builds and validates an HTML report.",
            repository_url="https://github.com/acme/html-report-agent",
            commit_hash="6f1ed002ab5595859014ebf0951522d9d5f25a73",
            idempotency_key="workflow",
            workdir=tmp_path / "run",
            trace_cases=True,
        )
        assert submitted.run_id == RUN_ID
        assert client.get_run_results(BENCHMARK_ID, RUN_ID)["run_id"] == RUN_ID
        weakest = client.list_case_results(BENCHMARK_ID, RUN_ID, sort="score", limit=5)
        destination = tmp_path / "review" / "report.html"
        client.download_prediction_artifact(weakest["data"][0]["artifacts"][0], destination)

    prediction = uploaded_predictions[-1]
    assert prediction["artifact_ids"] == [ARTIFACT_ID]
    assert prediction["execution_refs"] == {"trace_ids": [TRACE_DB_ID]}
    assert submitted_body["variant_identity"]["architecture_description"]
    assert submitted_body["variant_identity"]["repository_url"] == (
        "https://github.com/acme/html-report-agent"
    )
    assert submitted_body["variant_identity"]["commit_hash"] == (
        "6f1ed002ab5595859014ebf0951522d9d5f25a73"
    )
    assert destination.read_bytes() == REPORT


def test_typed_case_schema_and_structured_dataclass_output():
    assert schema_for_model(PaperQAInput) == {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "paper": {"type": "array", "x-promptic-type": "file"},
        },
        "required": ["question", "paper"],
        "additionalProperties": False,
    }
    result = AgentGymCaseResult.succeeded(PaperQAOutput(answer="The Transformer"))
    assert result.output == {"answer": "The Transformer"}


@pytest.mark.asyncio
async def test_async_submit_captures_case_exception_and_matches_protocol(tmp_path):
    submitted: dict[str, Any] = {}
    uploaded_predictions: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path.endswith(f"/benchmarks/{BENCHMARK_ID}/submissions"):
            created_body = json.loads(request.content)
            assert created_body["variant_identity"]["name"] == "async-agent"
            return httpx.Response(201, json=_created())
        if request.method == "GET" and path.endswith("/manifest"):
            return httpx.Response(200, json=_manifest())
        if request.method == "PUT" and path.endswith("/predictions"):
            uploaded_predictions.extend(json.loads(request.content)["predictions"])
            return httpx.Response(200, json={"accepted": 1, "stored": 1})
        if request.method == "POST" and path.endswith("/submit"):
            submitted.update(json.loads(request.content))
            return httpx.Response(
                202,
                json={
                    "submission_id": SUBMISSION_ID,
                    "run_id": RUN_ID,
                    "variant_id": BUNDLE_ID,
                    "status": "queued",
                    "dispatch_status": "dispatched",
                    "created": True,
                },
            )
        if request.method == "GET" and path.endswith(f"/{SUBMISSION_ID}"):
            return httpx.Response(200, json=_status())
        return httpx.Response(404, json={"error": "unexpected"})

    async def candidate(_case: AgentGymCase) -> AgentGymCaseResult:
        raise RuntimeError("candidate boom")

    async with AsyncAgentGymClient(api_key="ptc_test") as client:
        await client._transport.api.aclose()
        client._transport.api = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://promptic.test/api/v1",
            headers={"Authorization": "Bearer ptc_test"},
        )
        client._client = client._transport.api
        result = await client.run_and_submit(
            BENCHMARK_ID,
            candidate,
            name="async-agent",
            version="1.0.0",
            architecture_description="Async trusted callback.",
            repository_url="https://github.com/acme/async-agent",
            commit_hash="async-agent-commit",
            idempotency_key="async-workflow",
            workdir=tmp_path,
        )

    assert result.run_id == RUN_ID
    prediction = uploaded_predictions[0]
    assert prediction["status"] == "failed"
    assert prediction["error_code"] == "candidate_exception"
    assert prediction["error"] == "candidate boom"
    assert submitted["variant_identity"]["repository_url"] == (
        "https://github.com/acme/async-agent"
    )
    assert submitted["variant_identity"]["commit_hash"] == "async-agent-commit"


def test_session_prediction_builder_uploads_resolves_and_validates(tmp_path, monkeypatch):
    submitted: dict[str, Any] = {}
    uploaded_predictions: list[dict[str, Any]] = []
    trace_waits: list[list[str]] = []
    report_path = tmp_path / "builder-report.html"
    report_path.write_bytes(REPORT)

    with AgentGymClient(api_key="ptc_test") as client:
        session = ExternalSubmissionSession(client, BENCHMARK_ID, SUBMISSION_ID, REVISION_ID)
        monkeypatch.setattr(session, "get_manifest", lambda **_kwargs: _manifest())
        monkeypatch.setattr(
            session,
            "upload_artifact_file",
            lambda *_args, **_kwargs: {
                "artifact_id": ARTIFACT_ID,
                "storage_object_id": STORAGE_ID,
                "path": "builder-report.html",
                "status": "verified",
            },
        )

        def wait_for_resolved_traces(trace_ids: list[str], **_kwargs: Any) -> list[str]:
            trace_waits.append(trace_ids)
            return [TRACE_DB_ID] if trace_ids == [RAW_TRACE_ID] else []

        monkeypatch.setattr(session, "wait_for_resolved_traces", wait_for_resolved_traces)

        def upload_predictions(
            _benchmark_id: str,
            _submission_id: str,
            predictions: list[dict[str, Any]],
        ) -> dict[str, Any]:
            uploaded_predictions.extend(predictions)
            return {"accepted": len(predictions), "stored": len(predictions)}

        def submit_submission(
            _benchmark_id: str,
            _submission_id: str,
            body: dict[str, Any],
            *,
            idempotency_key: str,
        ) -> dict[str, Any]:
            assert idempotency_key == "builder-submit"
            submitted.update(body)
            return {
                "submission_id": SUBMISSION_ID,
                "run_id": RUN_ID,
                "variant_id": BUNDLE_ID,
                "status": "queued",
                "dispatch_status": "dispatched",
                "created": True,
            }

        monkeypatch.setattr(client, "upload_predictions", upload_predictions)
        monkeypatch.setattr(client, "submit_submission", submit_submission)
        result = AgentGymCaseResult.artifact(
            AgentGymOutputArtifact(report_path, "report", path="builder-report.html"),
            raw_trace_ids=[RAW_TRACE_ID],
        )
        with session as active:
            prediction = active.add_prediction(CASE_ID, result)
            assert trace_waits == []
            with pytest.raises(ValueError, match="already has"):
                active.add_prediction(CASE_ID, result)
            response = active.submit(
                identity={
                    "name": "external-agent",
                    "version": "1.0.0",
                    "repository_url": "https://github.com/acme/external-agent",
                    "commit_hash": "external-agent-commit",
                },
                metadata={"runtime": "isolated"},
                idempotency_key="builder-submit",
            )

    assert response["run_id"] == RUN_ID
    assert prediction["artifact_ids"] == [ARTIFACT_ID]
    assert "execution_refs" not in prediction
    assert trace_waits == [[RAW_TRACE_ID]]
    assert uploaded_predictions[-1]["execution_refs"] == {"trace_ids": [TRACE_DB_ID]}
    assert submitted["metadata"] == {"runtime": "isolated"}
    assert submitted["variant_identity"]["repository_url"] == (
        "https://github.com/acme/external-agent"
    )
    assert submitted["variant_identity"]["commit_hash"] == "external-agent-commit"


def test_session_persists_each_ready_prediction_immediately(monkeypatch):
    uploaded_batches: list[list[dict[str, Any]]] = []
    manifest = _manifest()
    manifest["revision"]["case_count"] = 3
    manifest["data"] = [
        {**manifest["data"][0], "dataset_case_id": case_id, "ordinal": case_id - 1}
        for case_id in range(1, 4)
    ]

    with AgentGymClient(api_key="ptc_test") as client:
        session = ExternalSubmissionSession(client, BENCHMARK_ID, SUBMISSION_ID, REVISION_ID)
        monkeypatch.setattr(session, "get_manifest", lambda **_kwargs: manifest)

        def upload_predictions(
            _benchmark_id: str,
            _submission_id: str,
            predictions: list[dict[str, Any]],
        ) -> dict[str, Any]:
            uploaded_batches.append(predictions)
            return {"accepted": len(predictions), "stored": len(predictions)}

        monkeypatch.setattr(client, "upload_predictions", upload_predictions)
        for case_id in range(1, 4):
            session.add_prediction(case_id, AgentGymCaseResult.succeeded(f"answer-{case_id}"))

    assert [len(batch) for batch in uploaded_batches] == [1, 1, 1]
    assert [batch[0]["dataset_case_id"] for batch in uploaded_batches] == [1, 2, 3]


def test_prediction_batches_split_before_the_request_body_becomes_large():
    predictions: list[ExternalPrediction] = [
        {
            "dataset_case_id": case_id,
            "status": "succeeded",
            "output": "x" * 600_000,
        }
        for case_id in (1, 2)
    ]

    batches = prediction_upload_batches(predictions)

    assert [len(batch) for batch in batches] == [1, 1]


@pytest.mark.parametrize(
    ("identity", "message"),
    [
        (
            {
                "name": "external-agent",
                "version": "1.0.0",
                "repository_url": "not-a-url",
            },
            "repository_url must be a valid HTTPS URL without credentials",
        ),
        *[
            (
                {
                    "name": "external-agent",
                    "version": "1.0.0",
                    "repository_url": repository_url,
                },
                "repository_url must be a valid HTTPS URL without credentials",
            )
            for repository_url in (
                "http://github.com/acme/agent",
                "javascript:alert(1)",
                "data:text/plain,agent",
                "file:///tmp/agent",
                "https://user:secret@github.com/acme/agent",
                "https://github.com/acme/agent path",
                "https://",
                "https://example.com:not-a-port/repo",
                "https://example.com:99999/repo",
            )
        ],
        (
            {
                "name": "external-agent",
                "version": "1.0.0",
                "commit_hash": " ",
            },
            "commit_hash must contain 1-128 characters",
        ),
        (
            {
                "name": "external-agent",
                "version": "1.0.0",
                "repository_url": f"https://example.com/{'a' * 2_000}",
            },
            "repository_url must contain 1-2000 characters",
        ),
        (
            {
                "name": "external-agent",
                "version": "1.0.0",
                "commit_hash": "a" * 129,
            },
            "commit_hash must contain 1-128 characters",
        ),
    ],
)
def test_session_submit_validates_variant_provenance(identity, message, monkeypatch):
    with AgentGymClient(api_key="ptc_test") as client:
        session = ExternalSubmissionSession(client, BENCHMARK_ID, SUBMISSION_ID, REVISION_ID)
        monkeypatch.setattr(session, "get_manifest", lambda **_kwargs: _manifest())
        monkeypatch.setattr(
            client,
            "submit_submission",
            lambda *_args, **_kwargs: pytest.fail("invalid provenance must not be sent"),
        )
        monkeypatch.setattr(
            client,
            "upload_predictions",
            lambda *_args, **_kwargs: {"accepted": 1, "stored": 1},
        )
        session.add_prediction(CASE_ID, AgentGymCaseResult.succeeded("answer"))

        with pytest.raises(ValueError, match=message):
            session.submit(
                identity=identity,
                idempotency_key="invalid-provenance-submit",
            )


def test_best_effort_trace_policy_submits_without_unresolved_traces(monkeypatch):
    uploaded_predictions: list[dict[str, Any]] = []
    with AgentGymClient(api_key="ptc_test") as client:
        session = ExternalSubmissionSession(client, BENCHMARK_ID, SUBMISSION_ID, REVISION_ID)
        monkeypatch.setattr(session, "get_manifest", lambda **_kwargs: _manifest())
        monkeypatch.setattr(
            session,
            "wait_for_resolved_traces",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                UnresolvedTraceError(
                    [RAW_TRACE_ID_2],
                    resolved={RAW_TRACE_ID: TRACE_DB_ID},
                )
            ),
        )

        def upload_predictions(
            _benchmark_id: str,
            _submission_id: str,
            predictions: list[dict[str, Any]],
        ) -> dict[str, Any]:
            uploaded_predictions.extend(predictions)
            return {"accepted": len(predictions), "stored": len(predictions)}

        def submit_submission(
            _benchmark_id: str,
            _submission_id: str,
            body: dict[str, Any],
            *,
            idempotency_key: str,
        ) -> dict[str, Any]:
            return {
                "submission_id": SUBMISSION_ID,
                "run_id": RUN_ID,
                "variant_id": BUNDLE_ID,
                "status": "queued",
                "dispatch_status": "dispatched",
                "created": True,
            }

        monkeypatch.setattr(client, "upload_predictions", upload_predictions)
        monkeypatch.setattr(client, "submit_submission", submit_submission)
        result = AgentGymCaseResult.succeeded(
            "answer",
            raw_trace_ids=[RAW_TRACE_ID, RAW_TRACE_ID_2],
        )
        result.trace_artifact_ids = (TRACE_ARTIFACT_ID,)
        session.add_prediction(CASE_ID, result)
        with pytest.warns(RuntimeWarning, match="Omitted 1 unresolved trace"):
            response = session.submit(
                identity={"name": "external-agent", "version": "1.0.0"},
                idempotency_key="best-effort-submit",
                trace_policy="best_effort",
                trace_max_wait=0.5,
            )

    assert response["run_id"] == RUN_ID
    assert uploaded_predictions[-1]["execution_refs"] == {
        "trace_ids": [TRACE_DB_ID],
        "trace_artifact_ids": [TRACE_ARTIFACT_ID],
    }


def test_required_trace_policy_blocks_submission(monkeypatch):
    with AgentGymClient(api_key="ptc_test") as client:
        session = ExternalSubmissionSession(client, BENCHMARK_ID, SUBMISSION_ID, REVISION_ID)
        monkeypatch.setattr(session, "get_manifest", lambda **_kwargs: _manifest())
        monkeypatch.setattr(
            session,
            "wait_for_resolved_traces",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(UnresolvedTraceError([RAW_TRACE_ID])),
        )
        monkeypatch.setattr(
            client,
            "submit_submission",
            lambda *_args, **_kwargs: pytest.fail("required traces must block submission"),
        )
        monkeypatch.setattr(
            client,
            "upload_predictions",
            lambda *_args, **_kwargs: {"accepted": 1, "stored": 1},
        )
        session.add_prediction(
            CASE_ID,
            AgentGymCaseResult.succeeded("answer", raw_trace_ids=[RAW_TRACE_ID]),
        )

        with pytest.raises(UnresolvedTraceError) as captured:
            session.submit(
                identity={"name": "external-agent", "version": "1.0.0"},
                idempotency_key="required-submit",
                trace_policy="required",
                trace_max_wait=0,
            )

    assert captured.value.trace_ids == [RAW_TRACE_ID]


def test_disabled_trace_policy_skips_all_trace_linkage(monkeypatch):
    uploaded_predictions: list[dict[str, Any]] = []
    with AgentGymClient(api_key="ptc_test") as client:
        session = ExternalSubmissionSession(client, BENCHMARK_ID, SUBMISSION_ID, REVISION_ID)
        monkeypatch.setattr(session, "get_manifest", lambda **_kwargs: _manifest())
        monkeypatch.setattr(
            session,
            "wait_for_resolved_traces",
            lambda *_args, **_kwargs: pytest.fail("disabled tracing must not poll"),
        )

        def upload_predictions(
            _benchmark_id: str,
            _submission_id: str,
            predictions: list[dict[str, Any]],
        ) -> dict[str, Any]:
            uploaded_predictions.extend(predictions)
            return {"accepted": len(predictions), "stored": len(predictions)}

        def submit_submission(
            _benchmark_id: str,
            _submission_id: str,
            body: dict[str, Any],
            **_kwargs: Any,
        ) -> dict[str, Any]:
            return {
                "submission_id": SUBMISSION_ID,
                "run_id": RUN_ID,
                "variant_id": BUNDLE_ID,
                "status": "queued",
                "dispatch_status": "dispatched",
                "created": True,
            }

        monkeypatch.setattr(client, "upload_predictions", upload_predictions)
        monkeypatch.setattr(client, "submit_submission", submit_submission)
        result = AgentGymCaseResult.succeeded("answer", raw_trace_ids=[RAW_TRACE_ID])
        result.trace_artifact_ids = (TRACE_ARTIFACT_ID,)
        session.add_prediction(CASE_ID, result)
        session.submit(
            identity={"name": "external-agent", "version": "1.0.0"},
            idempotency_key="disabled-submit",
            trace_policy="disabled",
        )

    assert "execution_refs" not in uploaded_predictions[-1]


def test_session_prediction_builder_rejects_unknown_and_incomplete_cases(tmp_path, monkeypatch):
    with AgentGymClient(api_key="ptc_test") as client:
        session = ExternalSubmissionSession(client, BENCHMARK_ID, SUBMISSION_ID)
        monkeypatch.setattr(session, "get_manifest", lambda **_kwargs: _manifest())
        monkeypatch.setattr(
            session,
            "upload_artifact_file",
            lambda *_args, **_kwargs: pytest.fail("invalid traces must fail before uploads"),
        )
        with pytest.raises(ValueError, match="does not belong"):
            session.add_prediction(
                99,
                AgentGymCaseResult.succeeded("unknown"),
            )
        artifact = tmp_path / "unused.txt"
        artifact.write_text("unused")
        invalid_trace_result = AgentGymCaseResult.succeeded(
            "invalid",
            artifacts=[AgentGymOutputArtifact(artifact, "report")],
        )
        invalid_trace_result.raw_trace_ids = ("not-a-trace-id",)
        with pytest.raises(ValueError, match="32 hexadecimal"):
            session.add_prediction(CASE_ID, invalid_trace_result)
        with pytest.raises(ValueError, match="cover every manifest case"):
            session.submit(
                identity={"name": "external-agent", "version": "1.0.0"},
                idempotency_key="incomplete-submit",
            )


@pytest.mark.asyncio
async def test_async_session_prediction_builder_matches_sync_protocol(monkeypatch):
    submitted: dict[str, Any] = {}
    uploaded_predictions: list[dict[str, Any]] = []
    trace_waits: list[list[str]] = []
    async with AsyncAgentGymClient(api_key="ptc_test") as client:
        session = AsyncExternalSubmissionSession(client, BENCHMARK_ID, SUBMISSION_ID, REVISION_ID)

        async def get_manifest(**_kwargs: Any) -> dict[str, Any]:
            return _manifest()

        async def upload_predictions(
            _benchmark_id: str,
            _submission_id: str,
            predictions: list[dict[str, Any]],
        ) -> dict[str, Any]:
            uploaded_predictions.extend(predictions)
            return {"accepted": len(predictions), "stored": len(predictions)}

        async def submit_submission(
            _benchmark_id: str,
            _submission_id: str,
            body: dict[str, Any],
            *,
            idempotency_key: str,
        ) -> dict[str, Any]:
            assert idempotency_key == "async-builder-submit"
            submitted.update(body)
            return {
                "submission_id": SUBMISSION_ID,
                "run_id": RUN_ID,
                "variant_id": BUNDLE_ID,
                "status": "queued",
                "dispatch_status": "dispatched",
                "created": True,
            }

        async def wait_for_resolved_traces(trace_ids: list[str], **_kwargs: Any) -> list[str]:
            trace_waits.append(trace_ids)
            return [TRACE_DB_ID]

        monkeypatch.setattr(session, "get_manifest", get_manifest)
        monkeypatch.setattr(session, "wait_for_resolved_traces", wait_for_resolved_traces)
        monkeypatch.setattr(client, "upload_predictions", upload_predictions)
        monkeypatch.setattr(client, "submit_submission", submit_submission)
        async with session as active:
            prediction = await active.add_prediction(
                CASE_ID,
                AgentGymCaseResult.succeeded(
                    {"answer": 42},
                    raw_trace_ids=[RAW_TRACE_ID],
                ),
            )
            assert trace_waits == []
            response = await active.submit(
                identity={
                    "name": "external-agent",
                    "version": "1.0.0",
                    "repository_url": "https://github.com/acme/async-external-agent",
                    "commit_hash": "async-external-commit",
                },
                idempotency_key="async-builder-submit",
            )

    assert response["run_id"] == RUN_ID
    assert "execution_refs" not in prediction
    assert trace_waits == [[RAW_TRACE_ID]]
    assert uploaded_predictions[-1]["execution_refs"] == {"trace_ids": [TRACE_DB_ID]}
    assert prediction["output"] == {"answer": 42}
    assert submitted["variant_identity"]["repository_url"] == (
        "https://github.com/acme/async-external-agent"
    )
    assert submitted["variant_identity"]["commit_hash"] == "async-external-commit"


@pytest.mark.asyncio
async def test_async_best_effort_trace_policy_submits_after_resolution_failure(monkeypatch):
    uploaded_predictions: list[dict[str, Any]] = []
    async with AsyncAgentGymClient(api_key="ptc_test") as client:
        session = AsyncExternalSubmissionSession(client, BENCHMARK_ID, SUBMISSION_ID, REVISION_ID)

        async def get_manifest(**_kwargs: Any) -> dict[str, Any]:
            return _manifest()

        async def wait_for_resolved_traces(*_args: Any, **_kwargs: Any) -> list[str]:
            raise RuntimeError("trace ingest unavailable")

        async def upload_predictions(
            _benchmark_id: str,
            _submission_id: str,
            predictions: list[dict[str, Any]],
        ) -> dict[str, Any]:
            uploaded_predictions.extend(predictions)
            return {"accepted": len(predictions), "stored": len(predictions)}

        async def submit_submission(
            _benchmark_id: str,
            _submission_id: str,
            body: dict[str, Any],
            **_kwargs: Any,
        ) -> dict[str, Any]:
            return {
                "submission_id": SUBMISSION_ID,
                "run_id": RUN_ID,
                "variant_id": BUNDLE_ID,
                "status": "queued",
                "dispatch_status": "dispatched",
                "created": True,
            }

        monkeypatch.setattr(session, "get_manifest", get_manifest)
        monkeypatch.setattr(session, "wait_for_resolved_traces", wait_for_resolved_traces)
        monkeypatch.setattr(client, "upload_predictions", upload_predictions)
        monkeypatch.setattr(client, "submit_submission", submit_submission)
        await session.add_prediction(
            CASE_ID,
            AgentGymCaseResult.succeeded("answer", raw_trace_ids=[RAW_TRACE_ID]),
        )
        with pytest.warns(RuntimeWarning, match="trace ingest unavailable"):
            response = await session.submit(
                identity={"name": "external-agent", "version": "1.0.0"},
                idempotency_key="async-best-effort-submit",
            )

    assert response["run_id"] == RUN_ID
    assert "execution_refs" not in uploaded_predictions[0]


def test_unreleased_submit_compatibility_alias_is_not_exposed():
    with AgentGymClient(api_key="ptc_test") as client:
        assert not hasattr(client, "submit")
