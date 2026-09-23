"""Tests for Agent Gym artifact integrity and transfer isolation."""

from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from promptic_sdk import AgentGymClient, ArtifactIntegrityError

STORAGE_ID = "00000000-0000-4000-8000-000000000010"
BENCHMARK_ID = "00000000-0000-4000-8000-000000000011"
SUBMISSION_ID = "00000000-0000-4000-8000-000000000012"
REVISION_ID = "00000000-0000-4000-8000-000000000013"
CASE_ID = 14
INPUT_ARTIFACT_ID = "00000000-0000-4000-8000-000000000015"
CONTENT = b"<html><body>report</body></html>"


def _artifact(*, size: int | None = None, digest: str | None = None):
    value = {
        "storage_object_id": STORAGE_ID,
        "path": "reports/result.html",
        "mime_type": "text/html",
        "size_bytes": len(CONTENT) if size is None else size,
        "role": "output",
        "content_url": f"/api/v1/storage-objects/{STORAGE_ID}/content",
        "download_url": f"/api/v1/storage-objects/{STORAGE_ID}/content?disposition=attachment",
    }
    if digest is not None:
        value["sha256"] = digest
    return value


def _configure_download(client: AgentGymClient, content: bytes = CONTENT) -> list[httpx.Request]:
    requests: list[httpx.Request] = []

    def api_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["authorization"] == "Bearer ptc_test"
        return httpx.Response(307, headers={"Location": "https://storage.test/result"})

    def direct_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert "authorization" not in request.headers
        return httpx.Response(200, content=content, headers={"Content-Length": str(len(content))})

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
    return requests


def test_download_reconstructs_safe_route_and_does_not_leak_auth(tmp_path):
    target = tmp_path / "result.html"
    with AgentGymClient(api_key="ptc_test") as client:
        requests = _configure_download(client)
        client.download_prediction_artifact(
            _artifact(),
            target,
            expected_sha256=hashlib.sha256(CONTENT).hexdigest(),
        )

    assert target.read_bytes() == CONTENT
    assert requests[0].url.path == f"/api/v1/storage-objects/{STORAGE_ID}/content"
    assert str(requests[1].url) == "https://storage.test/result"


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        (_artifact(size=len(CONTENT) + 1), "size does not match"),
        (_artifact(digest="0" * 64), "digest does not match"),
    ],
)
def test_integrity_failure_preserves_existing_file_and_cleans_temp(tmp_path, artifact, message):
    target = tmp_path / "result.html"
    target.write_bytes(b"existing")
    with AgentGymClient(api_key="ptc_test") as client:
        _configure_download(client)
        with pytest.raises(ArtifactIntegrityError, match=message):
            client.download_prediction_artifact(artifact, target, overwrite=True)

    assert target.read_bytes() == b"existing"
    assert not list(tmp_path.glob(".result.html.*"))


def test_download_rejects_untrusted_result_path_before_network(tmp_path):
    artifact = _artifact()
    artifact["path"] = "../escape.html"
    with (
        AgentGymClient(api_key="ptc_test") as client,
        pytest.raises(ValueError, match="relative normalized POSIX"),
    ):
        client.download_prediction_artifact(artifact, tmp_path / "result.html")


def test_download_limit_is_enforced_before_write(tmp_path):
    with AgentGymClient(api_key="ptc_test") as client:
        _configure_download(client)
        with pytest.raises(ArtifactIntegrityError, match="download limit"):
            client.download_prediction_artifact(
                _artifact(), tmp_path / "result.html", max_bytes=len(CONTENT) - 1
            )
    assert not (tmp_path / "result.html").exists()


def test_manifest_materialization_checks_hash_and_removes_signed_urls(tmp_path):
    input_bytes = b"source evidence"

    def api_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "submission_id": SUBMISSION_ID,
                "revision": {
                    "id": REVISION_ID,
                    "version": 1,
                    "fingerprint": "fingerprint",
                    "case_count": 1,
                },
                "task": {
                    "taskId": "task-1",
                    "name": "Review evidence",
                    "description": "Read the supplied evidence.",
                    "inputContract": {},
                    "outputContract": {},
                    "publicSuccessCriteria": None,
                },
                "data": [
                    {
                        "dataset_case_id": CASE_ID,
                        "ordinal": 0,
                        "input_payload": {},
                        "input_files": [
                            {
                                "artifact_id": INPUT_ARTIFACT_ID,
                                "storage_object_id": STORAGE_ID,
                                "path": "evidence/source.txt",
                                "field_path": "evidence",
                                "mime_type": "text/plain",
                                "size_bytes": len(input_bytes),
                                "sha256": hashlib.sha256(input_bytes).hexdigest(),
                                "download_url": "https://storage.test/input",
                                "expires_at": "2026-08-04T00:05:00.000Z",
                            }
                        ],
                    }
                ],
                "next_cursor": None,
            },
        )

    def direct_handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(200, content=input_bytes)

    with AgentGymClient(api_key="ptc_test") as client:
        client._transport.api.close()
        client._transport.direct.close()
        client._transport.api = httpx.Client(
            transport=httpx.MockTransport(api_handler),
            base_url="https://promptic.test/api/v1",
            headers={"Authorization": "Bearer ptc_test"},
        )
        client._transport.direct = httpx.Client(transport=httpx.MockTransport(direct_handler))
        materialized = client.materialize_manifest(
            BENCHMARK_ID, SUBMISSION_ID, tmp_path / "manifest"
        )

    assert materialized.files[0].local_path.read_bytes() == input_bytes
    serialized = json.loads(materialized.manifest_path.read_text())
    assert serialized["data"][0]["input_files"][0]["local_path"]
    assert "download_url" not in serialized["data"][0]["input_files"][0]


def test_download_dataset_materializes_public_inputs_without_creating_run(tmp_path):
    input_bytes = b"architecture discovery input"
    requests: list[httpx.Request] = []

    def api_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET" and request.url.path.endswith(
            f"/revisions/{REVISION_ID}/manifest"
        ):
            return httpx.Response(
                200,
                json={
                    "revision": {
                        "id": REVISION_ID,
                        "version": 1,
                        "fingerprint": "fingerprint",
                        "case_count": 1,
                    },
                    "task": {
                        "taskId": "task-1",
                        "name": "Review evidence",
                        "description": "Inspect the public input.",
                        "inputContract": {},
                        "outputContract": {},
                        "publicSuccessCriteria": None,
                    },
                    "data": [
                        {
                            "dataset_case_id": CASE_ID,
                            "ordinal": 0,
                            "input_payload": {
                                "question": "What is shown?",
                                "paper": [f"promptic-artifact://{INPUT_ARTIFACT_ID}"],
                            },
                            "input_files": [
                                {
                                    "artifact_id": INPUT_ARTIFACT_ID,
                                    "storage_object_id": STORAGE_ID,
                                    "path": "paper.txt",
                                    "field_path": "paper",
                                    "mime_type": "text/plain",
                                    "size_bytes": len(input_bytes),
                                    "sha256": hashlib.sha256(input_bytes).hexdigest(),
                                    "download_url": "https://storage.test/input",
                                    "expires_at": "2026-08-04T00:05:00.000Z",
                                }
                            ],
                        }
                    ],
                    "next_cursor": None,
                },
            )
        return httpx.Response(404)

    def direct_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=input_bytes)

    with AgentGymClient(api_key="ptc_test") as client:
        client._transport.api.close()
        client._transport.direct.close()
        client._transport.api = httpx.Client(
            transport=httpx.MockTransport(api_handler),
            base_url="https://promptic.test/api/v1",
            headers={"Authorization": "Bearer ptc_test"},
        )
        client._transport.direct = httpx.Client(transport=httpx.MockTransport(direct_handler))
        dataset = client.download_dataset(
            BENCHMARK_ID,
            tmp_path / "dataset",
            revision_id=REVISION_ID,
        )
        replayed = client.download_dataset(
            BENCHMARK_ID,
            tmp_path / "dataset",
            revision_id=REVISION_ID,
        )

    paper = dataset.cases[0].input["paper"][0]
    assert paper.local_path.read_bytes() == input_bytes
    assert paper.field_path == "paper"
    assert dataset.revision["id"] == REVISION_ID
    assert replayed.cases[0].input["paper"][0].local_path == paper.local_path
    serialized = json.loads(dataset.manifest_path.read_text())
    assert "download_url" not in dataset.manifest_path.read_text()
    assert "expectedOutput" not in dataset.manifest_path.read_text()
    assert set(serialized) == {"revision", "task", "data"}
    assert not any(request.method == "POST" for request in requests)
    assert sum(request.url.host == "storage.test" for request in requests) == 1
