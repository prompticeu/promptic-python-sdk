"""Tests for Agent Gym external submission clients."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from promptic_sdk import (
    AgentGymAPIError,
    AgentGymClient,
    ArtifactUploadError,
    AsyncAgentGymClient,
    ExternalOutputKind,
    ExternalPrediction,
)

BENCHMARK_ID = "11111111-1111-4111-8111-111111111111"
SUBMISSION_ID = "22222222-2222-4222-8222-222222222222"
REVISION_ID = "33333333-3333-4333-8333-333333333333"
CASE_ONE_ID = "44444444-4444-4444-8444-444444444444"
CASE_TWO_ID = "55555555-5555-4555-8555-555555555555"
ARTIFACT_ID = "66666666-6666-4666-8666-666666666666"
STORAGE_OBJECT_ID = "77777777-7777-4777-8777-777777777777"
INPUT_ARTIFACT_ID = "88888888-8888-4888-8888-888888888888"
INPUT_STORAGE_ID = "99999999-9999-4999-8999-999999999999"
TRACE_DB_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
RUN_ID = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
BUNDLE_ID = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
TRACE_ARTIFACT_ID = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
RAW_TRACE_ID = "ABCDEF0123456789ABCDEF0123456789"
API_KEY = "promptic_test_api_key"  # noqa: S105 - synthetic credential

TASK = {
    "taskId": None,
    "name": "Document report",
    "description": "Produce an HTML report.",
    "inputContract": {"kind": "files"},
    "outputContract": {"kind": "artifact"},
    "publicSuccessCriteria": {"required": ["report.html"]},
}
REVISION = {
    "id": REVISION_ID,
    "version": 3,
    "fingerprint": "revision-fingerprint",
    "case_count": 2,
}


def _created_response() -> dict:
    return {
        "submission_id": SUBMISSION_ID,
        "revision": {
            **REVISION,
            "scorer_contract_version": "benchmark-v1",
        },
        "status": "created",
        "expires_at": "2026-07-30T12:00:00.000Z",
        "task": TASK,
        "links": {
            "manifest": f"/api/v1/benchmarks/{BENCHMARK_ID}/submissions/{SUBMISSION_ID}/manifest",
            "artifacts": f"/api/v1/benchmarks/{BENCHMARK_ID}/submissions/{SUBMISSION_ID}/artifacts",
            "finalize": f"/api/v1/benchmarks/{BENCHMARK_ID}/submissions/{SUBMISSION_ID}/finalize",
            "status": f"/api/v1/benchmarks/{BENCHMARK_ID}/submissions/{SUBMISSION_ID}",
        },
        "created": True,
    }


def _manifest_page(*, second: bool = False) -> dict:
    input_content = b"original document"
    case = (
        {
            "case_id": CASE_TWO_ID,
            "ordinal": 1,
            "input_payload": {"variables": {"title": "Second"}},
            "input_files": [],
        }
        if second
        else {
            "case_id": CASE_ONE_ID,
            "ordinal": 0,
            "input_payload": {"variables": {"title": "First"}},
            "input_files": [
                {
                    "artifact_id": INPUT_ARTIFACT_ID,
                    "storage_object_id": INPUT_STORAGE_ID,
                    "path": "documents/original.pdf",
                    "mime_type": "application/pdf",
                    "size_bytes": len(input_content),
                    "sha256": hashlib.sha256(input_content).hexdigest(),
                    "download_url": "https://storage.example/input?signature=download-secret",
                    "expires_at": "2026-07-29T12:05:00.000Z",
                }
            ],
        }
    )
    return {
        "submission_id": SUBMISSION_ID,
        "revision": REVISION,
        "task": TASK,
        "data": [case],
        "next_cursor": None if second else "MA",
    }


def _status_response(*, status: str = "succeeded") -> dict:
    return {
        "submission_id": SUBMISSION_ID,
        "revision_id": REVISION_ID,
        "status": status,
        "expires_at": "2026-07-30T12:00:00.000Z",
        "finalized_at": "2026-07-29T12:02:00.000Z",
        "queued_at": "2026-07-29T12:02:01.000Z",
        "completed_at": "2026-07-29T12:03:00.000Z",
        "validation_error": None,
        "run": {
            "id": RUN_ID,
            "status": "succeeded",
            "scoring_status": "succeeded",
            "eligibility_status": "eligible",
            "eligibility_reasons": [],
            "scored_at": "2026-07-29T12:03:00.000Z",
            "error": None,
        },
    }


def _sync_client(
    api_handler: httpx.MockTransport,
    direct_handler: httpx.MockTransport | None = None,
) -> AgentGymClient:
    client = AgentGymClient(
        api_key=API_KEY,
        endpoint="https://promptic.example",
    )
    client._client.close()
    client._direct_client.close()
    client._client = httpx.Client(
        transport=api_handler,
        base_url="https://promptic.example/api/v1",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    client._direct_client = httpx.Client(
        transport=direct_handler or httpx.MockTransport(lambda _: httpx.Response(404)),
        follow_redirects=True,
    )
    return client


def _async_client(
    api_handler: httpx.MockTransport,
    direct_handler: httpx.MockTransport | None = None,
) -> AsyncAgentGymClient:
    client = AsyncAgentGymClient(
        api_key=API_KEY,
        endpoint="https://promptic.example",
    )
    client._client = httpx.AsyncClient(
        transport=api_handler,
        base_url="https://promptic.example/api/v1",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    client._direct_client = httpx.AsyncClient(
        transport=direct_handler or httpx.MockTransport(lambda _: httpx.Response(404)),
        follow_redirects=True,
    )
    return client


class TestAgentGymClient:
    def test_normal_api_key_is_the_default_authentication_path(self):
        client = AgentGymClient(
            api_key="promptic-api-key",
            endpoint="https://promptic.example",
        )
        try:
            assert client._client.headers["authorization"] == "Bearer promptic-api-key"
            assert "x-ai-application-id" not in client._client.headers
        finally:
            client.close()

    def test_login_token_includes_the_selected_ai_application(self):
        client = AgentGymClient(
            access_token="login-token",  # noqa: S106 - synthetic credential
            ai_application_id="application-1",
            endpoint="https://promptic.example",
        )
        try:
            assert client._client.headers["authorization"] == "Bearer login-token"
            assert client._client.headers["x-ai-application-id"] == "application-1"
        finally:
            client.close()

    def test_full_external_submission_happy_path(self, tmp_path: Path):
        api_requests: list[httpx.Request] = []
        direct_requests: list[httpx.Request] = []
        trace_resolution_attempts = 0
        report = b"<html><body>Verified report</body></html>"

        def api_handler(request: httpx.Request) -> httpx.Response:
            nonlocal trace_resolution_attempts
            api_requests.append(request)
            assert request.headers["authorization"] == f"Bearer {API_KEY}"
            path = request.url.path

            if request.method == "POST" and path.endswith("/submissions"):
                assert request.headers["idempotency-key"] == "create-key"
                assert json.loads(request.content) == {"ttl_seconds": 3600}
                return httpx.Response(201, json=_created_response())

            if request.method == "GET" and path.endswith("/manifest"):
                assert request.url.params["limit"] == "1"
                return httpx.Response(
                    200,
                    json=_manifest_page(second=request.url.params.get("cursor") == "MA"),
                )

            if request.method == "POST" and path.endswith("/artifacts"):
                body = json.loads(request.content)
                assert body == {
                    "path": "reports/report.html",
                    "role": "output",
                    "mime_type": "text/html",
                    "size_bytes": len(report),
                    "sha256": hashlib.sha256(report).hexdigest(),
                }
                return httpx.Response(
                    201,
                    json={
                        "artifact_id": ARTIFACT_ID,
                        "storage_object_id": STORAGE_OBJECT_ID,
                        "path": body["path"],
                        "status": "reserved",
                        "upload": {
                            "strategy": "url",
                            "provider": "azure",
                            "uploadUrl": "https://storage.example/output?signature=upload-secret",
                            "finalUrl": "https://storage.example/read?signature=read-secret",
                            "method": "PUT",
                            "headers": {"x-ms-blob-type": "BlockBlob"},
                            "maxSizeBytes": len(report),
                            "expiresAt": "2026-07-29T12:15:00.000Z",
                        },
                    },
                )

            if request.method == "POST" and path.endswith(f"/{ARTIFACT_ID}/complete"):
                assert request.content == b""
                return httpx.Response(
                    200,
                    json={"artifact_id": ARTIFACT_ID, "status": "verified"},
                )

            if request.method == "GET" and path.endswith("/traces"):
                assert request.url.params.get_list("trace_id") == [RAW_TRACE_ID.lower()]
                trace_resolution_attempts += 1
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "trace_id": RAW_TRACE_ID.lower(),
                                "trace_db_id": (
                                    TRACE_DB_ID if trace_resolution_attempts > 1 else None
                                ),
                            }
                        ]
                    },
                )

            if request.method == "POST" and path.endswith("/finalize"):
                assert request.headers["idempotency-key"] == "finalize-key"
                body = json.loads(request.content)
                assert body["bundle_identity"] == {
                    "name": "document-reporter",
                    "version": "1.0.0",
                    "architecture_description": "Compares two PDFs and renders an HTML report.",
                }
                assert body["predictions"] == [
                    {
                        "revision_case_id": CASE_ONE_ID,
                        "status": "succeeded",
                        "output": {"kind": "artifact", "value": {"primary": ARTIFACT_ID}},
                        "artifact_ids": [ARTIFACT_ID],
                        "execution_refs": {
                            "trace_ids": [TRACE_DB_ID],
                            "trace_artifact_ids": [TRACE_ARTIFACT_ID],
                        },
                        "executor_id": "local-runner",
                        "executor_version": "abc123",
                        "token_usage": {"prompt": 100, "completion": 50, "total": 150},
                        "latency_ms": 1250,
                        "started_at": "2026-07-29T12:00:00Z",
                        "completed_at": "2026-07-29T12:00:01Z",
                        "diagnostics": {"attempt": 1},
                    },
                    {
                        "revision_case_id": CASE_TWO_ID,
                        "status": "failed",
                        "error_code": "agent_error",
                        "error_category": "runtime",
                        "retryable": False,
                        "error": "The model declined the input.",
                        "artifact_ids": [],
                    },
                ]
                return httpx.Response(
                    202,
                    json={
                        "submission_id": SUBMISSION_ID,
                        "run_id": RUN_ID,
                        "bundle_id": BUNDLE_ID,
                        "status": "queued",
                        "dispatch_status": "dispatched",
                        "created": True,
                    },
                )

            if request.method == "GET" and path.endswith(f"/submissions/{SUBMISSION_ID}"):
                return httpx.Response(200, json=_status_response())

            return httpx.Response(500, json={"error": "unexpected_test_request"})

        def direct_handler(request: httpx.Request) -> httpx.Response:
            direct_requests.append(request)
            assert "authorization" not in request.headers
            if request.method == "GET":
                assert "download-secret" in str(request.url)
                return httpx.Response(200, content=b"original document")
            assert request.method == "PUT"
            assert request.headers["x-ms-blob-type"] == "BlockBlob"
            assert request.content == report
            return httpx.Response(201)

        with _sync_client(
            httpx.MockTransport(api_handler),
            httpx.MockTransport(direct_handler),
        ) as client:
            session = client.start_submission(
                BENCHMARK_ID,
                idempotency_key=" create-key ",
                ttl_seconds=3600,
            )
            assert session.revision_id == REVISION_ID
            assert session.created_response is not None

            materialized = session.materialize_manifest(tmp_path, page_size=1)
            assert [case["case_id"] for case in materialized.manifest["data"]] == [
                CASE_ONE_ID,
                CASE_TWO_ID,
            ]
            assert materialized.files[0].local_path.read_bytes() == b"original document"
            saved_manifest = json.loads(materialized.manifest_path.read_text())
            saved_file = saved_manifest["data"][0]["input_files"][0]
            assert "download_url" not in saved_file
            assert saved_file["local_path"] == "case-000000/inputs/documents/original.pdf"

            artifact = session.upload_artifact_bytes(
                report,
                path="reports/report.html",
                mime_type="text/html",
            )
            assert artifact["artifact_id"] == ARTIFACT_ID
            assert artifact["status"] == "verified"

            trace_db_ids = session.wait_for_resolved_traces(
                [RAW_TRACE_ID],
                max_wait=1,
                poll_interval=0,
            )
            assert trace_db_ids == [TRACE_DB_ID]

            predictions: list[ExternalPrediction] = [
                {
                    "revision_case_id": CASE_ONE_ID,
                    "status": "succeeded",
                    "output": {"kind": "artifact", "value": {"primary": ARTIFACT_ID}},
                    "artifact_ids": [ARTIFACT_ID],
                    "execution_refs": {
                        "trace_ids": trace_db_ids,
                        "trace_artifact_ids": [TRACE_ARTIFACT_ID],
                    },
                    "executor_id": "local-runner",
                    "executor_version": "abc123",
                    "token_usage": {"prompt": 100, "completion": 50, "total": 150},
                    "latency_ms": 1250,
                    "started_at": "2026-07-29T12:00:00Z",
                    "completed_at": "2026-07-29T12:00:01Z",
                    "diagnostics": {"attempt": 1},
                },
                {
                    "revision_case_id": CASE_TWO_ID,
                    "status": "failed",
                    "error_code": "agent_error",
                    "error_category": "runtime",
                    "retryable": False,
                    "error": "The model declined the input.",
                },
            ]
            finalized = session.finalize(
                bundle_identity={
                    "name": "document-reporter",
                    "version": "1.0.0",
                    "architecture_description": ("Compares two PDFs and renders an HTML report."),
                },
                predictions=predictions,
                idempotency_key="finalize-key",
            )
            assert finalized["run_id"] == RUN_ID
            status = session.wait(max_wait=0)
            run = status["run"]
            assert run is not None
            assert run["id"] == RUN_ID
            assert run["eligibility_status"] == "eligible"

        assert (
            len([request for request in api_requests if request.url.path.endswith("/traces")]) == 2
        )
        assert [request.method for request in direct_requests] == ["GET", "PUT"]

    @pytest.mark.parametrize(
        "logical_path",
        [
            "../report.pdf",
            "/tmp/report.pdf",
            "reports\\report.pdf",
            "reports//report.pdf",
            "reports/./report.pdf",
            "reports/../report.pdf",
        ],
    )
    def test_rejects_unsafe_artifact_paths_before_request(self, logical_path: str):
        requests = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal requests
            requests += 1
            return httpx.Response(500)

        with (
            _sync_client(httpx.MockTransport(handler)) as client,
            pytest.raises(ValueError, match="relative normalized POSIX path"),
        ):
            client.reserve_artifact(
                BENCHMARK_ID,
                SUBMISSION_ID,
                path=logical_path,
                mime_type="application/pdf",
                size_bytes=1,
                sha256="0" * 64,
            )
        assert requests == 0

    def test_requires_explicit_idempotency_and_lowercase_digest(self):
        with _sync_client(httpx.MockTransport(lambda _: httpx.Response(500))) as client:
            with pytest.raises(ValueError, match="idempotency_key"):
                client.create_submission(BENCHMARK_ID, idempotency_key=" ")
            with pytest.raises(ValueError, match="lowercase"):
                client.reserve_artifact(
                    BENCHMARK_ID,
                    SUBMISSION_ID,
                    path="report.pdf",
                    mime_type="application/pdf",
                    size_bytes=1,
                    sha256="A" * 64,
                )

    def test_finalize_surfaces_incomplete_or_foreign_artifact_details(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["idempotency-key"] == "finalize"
            return httpx.Response(
                400,
                json={
                    "error": "artifact_not_verified",
                    "message": (
                        "Finalization references artifacts not verified for this submission"
                    ),
                    "details": {"artifact_ids": [ARTIFACT_ID]},
                },
            )

        with (
            _sync_client(httpx.MockTransport(handler)) as client,
            pytest.raises(AgentGymAPIError) as raised,
        ):
            client.finalize_submission(
                BENCHMARK_ID,
                SUBMISSION_ID,
                {
                    "bundle_identity": {
                        "name": "candidate",
                        "version": "1",
                        "architecture_description": "A test architecture.",
                    },
                    "predictions": [
                        {
                            "revision_case_id": CASE_ONE_ID,
                            "status": "succeeded",
                            "output": {"kind": "artifact"},
                            "artifact_ids": [ARTIFACT_ID],
                        }
                    ],
                },
                idempotency_key="finalize",
            )
        error = raised.value
        assert error.code == "artifact_not_verified"
        assert error.details == {"artifact_ids": [ARTIFACT_ID]}
        assert API_KEY not in str(error)

    @pytest.mark.parametrize(
        "kind",
        ["structured", "text", "artifact", "side_effect", "none", "custom"],
    )
    def test_finalize_preserves_every_server_output_kind(self, kind: ExternalOutputKind):
        def handler(request: httpx.Request) -> httpx.Response:
            assert json.loads(request.content)["predictions"][0]["output"] == {
                "kind": kind,
                "value": {"example": True},
            }
            return httpx.Response(
                202,
                json={
                    "submission_id": SUBMISSION_ID,
                    "run_id": RUN_ID,
                    "bundle_id": BUNDLE_ID,
                    "status": "queued",
                    "dispatch_status": "pending",
                    "created": True,
                },
            )

        with _sync_client(httpx.MockTransport(handler)) as client:
            result = client.finalize_submission(
                BENCHMARK_ID,
                SUBMISSION_ID,
                {
                    "bundle_identity": {
                        "name": "candidate",
                        "version": "1",
                        "architecture_description": "A test architecture.",
                    },
                    "predictions": [
                        {
                            "revision_case_id": CASE_ONE_ID,
                            "status": "succeeded",
                            "output": {"kind": kind, "value": {"example": True}},
                        }
                    ],
                },
                idempotency_key="finalize",
            )
        assert result["run_id"] == RUN_ID

    def test_completion_has_no_json_body_and_cancellation_is_exposed(self):
        methods: list[tuple[str, bytes]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            methods.append((request.method, request.content))
            if request.method == "POST":
                return httpx.Response(
                    200,
                    json={"artifact_id": ARTIFACT_ID, "status": "verified"},
                )
            return httpx.Response(
                200,
                json={"submission_id": SUBMISSION_ID, "status": "cancelled"},
            )

        with _sync_client(httpx.MockTransport(handler)) as client:
            client.complete_artifact(BENCHMARK_ID, SUBMISSION_ID, ARTIFACT_ID)
            result = client.cancel_submission(BENCHMARK_ID, SUBMISSION_ID)
            assert result["status"] == "cancelled"
        assert methods == [("POST", b""), ("DELETE", b"")]

    def test_upload_error_and_client_repr_do_not_leak_credentials(self):
        def api_handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                201,
                json={
                    "artifact_id": ARTIFACT_ID,
                    "storage_object_id": STORAGE_OBJECT_ID,
                    "path": "report.pdf",
                    "status": "reserved",
                    "upload": {
                        "strategy": "url",
                        "provider": "azure",
                        "uploadUrl": "https://storage.example/upload?signature=super-secret",
                        "finalUrl": "https://storage.example/read?signature=also-secret",
                        "method": "PUT",
                        "headers": {},
                        "maxSizeBytes": 3,
                        "expiresAt": "2026-07-29T12:15:00.000Z",
                    },
                },
            )

        with _sync_client(
            httpx.MockTransport(api_handler),
            httpx.MockTransport(lambda _: httpx.Response(403, text="signed secret")),
        ) as client:
            assert API_KEY not in repr(client)
            reservation = client.reserve_artifact(
                BENCHMARK_ID,
                SUBMISSION_ID,
                path="report.pdf",
                mime_type="application/pdf",
                size_bytes=3,
                sha256=hashlib.sha256(b"pdf").hexdigest(),
            )
            with pytest.raises(ArtifactUploadError) as raised:
                client.upload_reserved_artifact(
                    reservation,
                    b"pdf",
                    mime_type="application/pdf",
                )
        assert "super-secret" not in str(raised.value)
        assert "signed secret" not in str(raised.value)

    def test_manifest_rejects_server_path_traversal(self, tmp_path: Path):
        page = _manifest_page()
        page["data"][0]["input_files"][0]["path"] = "../../outside.pdf"
        page["next_cursor"] = None
        page["revision"] = {**page["revision"], "case_count": 1}

        with (
            _sync_client(httpx.MockTransport(lambda _: httpx.Response(200, json=page))) as client,
            pytest.raises(ValueError, match="relative normalized POSIX path"),
        ):
            client.materialize_manifest(BENCHMARK_ID, SUBMISSION_ID, tmp_path)
        assert not (tmp_path.parent / "outside.pdf").exists()


class TestAsyncAgentGymClient:
    async def test_async_flow_supports_post_upload_finalize_status_and_cancel(self):
        report = b"PDF"
        requests: list[httpx.Request] = []

        def api_handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            path = request.url.path
            if request.method == "POST" and path.endswith("/submissions"):
                return httpx.Response(201, json=_created_response())
            if request.method == "GET" and path.endswith("/manifest"):
                page = _manifest_page()
                page["data"][0]["input_files"] = []
                page["next_cursor"] = None
                page["revision"] = {**page["revision"], "case_count": 1}
                return httpx.Response(200, json=page)
            if request.method == "POST" and path.endswith("/artifacts"):
                return httpx.Response(
                    201,
                    json={
                        "artifact_id": ARTIFACT_ID,
                        "storage_object_id": STORAGE_OBJECT_ID,
                        "path": "deck.pdf",
                        "status": "reserved",
                        "upload": {
                            "strategy": "url",
                            "provider": "s3",
                            "uploadUrl": "https://storage.example/post?signature=secret",
                            "finalUrl": "https://storage.example/read?signature=secret",
                            "method": "POST",
                            "fields": {"key": "private/deck.pdf", "policy": "signed-policy"},
                            "maxSizeBytes": len(report),
                            "expiresAt": "2026-07-29T12:15:00.000Z",
                        },
                    },
                )
            if request.method == "POST" and path.endswith(f"/{ARTIFACT_ID}/complete"):
                assert request.content == b""
                return httpx.Response(200, json={"artifact_id": ARTIFACT_ID, "status": "verified"})
            if request.method == "GET" and path.endswith("/traces"):
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "trace_id": RAW_TRACE_ID.lower(),
                                "trace_db_id": TRACE_DB_ID,
                            }
                        ]
                    },
                )
            if request.method == "POST" and path.endswith("/finalize"):
                body = json.loads(request.content)
                assert body["predictions"][0]["execution_refs"]["trace_ids"] == [TRACE_DB_ID]
                assert body["predictions"][0]["artifact_ids"] == [ARTIFACT_ID]
                return httpx.Response(
                    202,
                    json={
                        "submission_id": SUBMISSION_ID,
                        "run_id": RUN_ID,
                        "bundle_id": BUNDLE_ID,
                        "status": "queued",
                        "dispatch_status": "pending",
                        "created": True,
                    },
                )
            if request.method == "GET":
                return httpx.Response(200, json=_status_response())
            if request.method == "DELETE":
                return httpx.Response(
                    200,
                    json={"submission_id": SUBMISSION_ID, "status": "cancelled"},
                )
            return httpx.Response(500, json={"error": "unexpected_test_request"})

        def direct_handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "POST"
            assert "authorization" not in request.headers
            assert b"signed-policy" in request.content
            assert report in request.content
            return httpx.Response(204)

        client = _async_client(
            httpx.MockTransport(api_handler),
            httpx.MockTransport(direct_handler),
        )
        assert API_KEY not in repr(client)
        try:
            session = await client.start_submission(BENCHMARK_ID, idempotency_key="async-create")
            assert (await session.get_manifest())["data"][0]["case_id"] == CASE_ONE_ID
            artifact = await session.upload_artifact_bytes(
                report,
                path="deck.pdf",
                mime_type="application/pdf",
            )
            trace_ids = await session.wait_for_resolved_traces([RAW_TRACE_ID], max_wait=0)
            finalized = await session.finalize(
                bundle_identity={
                    "name": "deck-builder",
                    "version": "1.0.0",
                    "architecture_description": "Builds a PDF pitch deck.",
                },
                predictions=[
                    {
                        "revision_case_id": CASE_ONE_ID,
                        "status": "succeeded",
                        "output": {"kind": "artifact"},
                        "artifact_ids": [artifact["artifact_id"]],
                        "execution_refs": {"trace_ids": trace_ids},
                    }
                ],
                idempotency_key="async-finalize",
            )
            assert finalized["run_id"] == RUN_ID
            run = (await session.status())["run"]
            assert run is not None
            assert run["id"] == RUN_ID
            assert (await session.cancel())["status"] == "cancelled"
        finally:
            await client.close()

        completion = [
            request for request in requests if request.url.path.endswith(f"/{ARTIFACT_ID}/complete")
        ]
        assert len(completion) == 1
