"""Tests for the platform client."""

import json

import httpx
import pytest

from promptic_sdk.client import AsyncPrompticClient, PrompticClient
from promptic_sdk.models import DatasetCaseCreate, Iteration, ToolSelectionTool


def _canonical_dataset_payload(*, include_cases: bool = False) -> dict:
    payload = {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "name": "Regression",
        "description": None,
        "aiComponentId": "550e8400-e29b-41d4-a716-446655440001",
        "aiApplicationId": "550e8400-e29b-41d4-a716-446655440002",
        "caseCount": 1,
        "createdAt": "2026-07-21T10:00:00Z",
        "updatedAt": "2026-07-21T10:00:00Z",
    }
    if include_cases:
        payload["cases"] = [
            {
                "id": 7,
                "datasetId": payload["id"],
                "idx": 0,
                "inputPayload": {"question": "hello"},
                "expectedPayload": {"answer": "world"},
                "split": "eval",
                "metadata": {},
                "createdAt": "2026-07-21T10:00:00Z",
                "updatedAt": "2026-07-21T10:00:00Z",
            }
        ]
    return payload


class TestPrompticClient:
    def test_requires_api_key(self):
        with pytest.raises(ValueError, match="Authentication required"):
            PrompticClient(api_key=None)

    def test_reads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test_key")
        client = PrompticClient()
        assert client.api_key == "pk_test_key"
        client.close()

    def test_custom_endpoint(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        client = PrompticClient(endpoint="https://custom.example.com")
        assert client.endpoint == "https://custom.example.com"
        # httpx normalizes base_url with trailing slash
        assert str(client._client.base_url).rstrip("/") == "https://custom.example.com/api/v1"
        client.close()

    def test_default_endpoint(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        client = PrompticClient()
        assert client.endpoint == "https://promptic.eu"
        client.close()

    def test_context_manager(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        with PrompticClient() as client:
            assert client.api_key == "pk_test"

    def test_list_traces(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        response_data = {"traces": [{"traceId": "abc123"}], "total": 1}

        with PrompticClient() as client:

            def handler(request: httpx.Request) -> httpx.Response:
                assert "/traces" in str(request.url)
                assert request.headers["authorization"] == "Bearer pk_test"
                return httpx.Response(200, json=response_data)

            client._client = httpx.Client(
                transport=httpx.MockTransport(handler),
                base_url="https://promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            result = client.list_traces(limit=10)
            assert result == response_data

    def test_get_trace(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        response_data = {"traceId": "abc123", "spans": []}

        with PrompticClient() as client:

            def handler(request: httpx.Request) -> httpx.Response:
                assert "/traces/abc123" in str(request.url)
                return httpx.Response(200, json=response_data)

            client._client = httpx.Client(
                transport=httpx.MockTransport(handler),
                base_url="https://promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            result = client.get_trace("abc123")
            assert result == response_data

    def test_dataset_methods_return_canonical_payloads(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        dataset = _canonical_dataset_payload()
        dataset_with_cases = _canonical_dataset_payload(include_cases=True)

        with PrompticClient() as client:

            def handler(request: httpx.Request) -> httpx.Response:
                if request.method == "POST":
                    assert request.read() == b'{"name":"Regression"}'
                    return httpx.Response(201, json=dataset)
                if request.url.path.endswith(f"/datasets/{dataset['id']}"):
                    return httpx.Response(200, json=dataset_with_cases)
                return httpx.Response(200, json={"data": [dataset]})

            client._client = httpx.Client(
                transport=httpx.MockTransport(handler),
                base_url="https://promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            assert client.create_dataset(dataset["aiComponentId"], "Regression") == dataset
            assert client.list_datasets(dataset["aiComponentId"]) == {"data": [dataset]}
            assert client.get_dataset(dataset["aiComponentId"], dataset["id"]) == dataset_with_cases

    def test_get_artifact_content_follows_redirect(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        requests: list[httpx.Request] = []

        with PrompticClient() as client:

            def handler(request: httpx.Request) -> httpx.Response:
                requests.append(request)
                if request.url.path == "/api/v1/artifacts/artifact-id/content":
                    return httpx.Response(
                        307,
                        headers={"Location": "https://storage.example/artifact.png"},
                    )
                if str(request.url) == "https://storage.example/artifact.png":
                    return httpx.Response(200, content=b"artifact-bytes")
                return httpx.Response(404, json={"error": "not found"})

            client._client = httpx.Client(
                transport=httpx.MockTransport(handler),
                base_url="https://promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            assert client.get_artifact_content("artifact-id") == b"artifact-bytes"
            assert [str(request.url) for request in requests] == [
                "https://promptic.eu/api/v1/artifacts/artifact-id/content",
                "https://storage.example/artifact.png",
            ]

    def test_download_artifact_does_not_truncate_existing_file_on_failure(
        self, monkeypatch, tmp_path
    ):
        target = tmp_path / "artifact.bin"
        target.write_bytes(b"existing")

        client = PrompticClient(api_key="pk_test")

        def fail_get_content(_artifact_id: str) -> bytes:
            msg = "download failed"
            raise RuntimeError(msg)

        monkeypatch.setattr(client, "get_artifact_content", fail_get_content)

        with pytest.raises(RuntimeError, match="download failed"):
            client.download_artifact("artifact-id", target)

        assert target.read_bytes() == b"existing"
        client.close()

    def test_download_artifact_does_not_truncate_existing_file_on_write_failure(
        self, monkeypatch, tmp_path
    ):
        target = tmp_path / "artifact.bin"
        target.write_bytes(b"existing")
        client = PrompticClient(api_key="pk_test")
        monkeypatch.setattr(client, "get_artifact_content", lambda _artifact_id: b"new")

        def fail_fsync(_fd: int) -> None:
            msg = "disk full"
            raise OSError(msg)

        monkeypatch.setattr("promptic_sdk.client.os.fsync", fail_fsync)

        with pytest.raises(OSError, match="disk full"):
            client.download_artifact("artifact-id", target)

        assert target.read_bytes() == b"existing"
        assert not list(tmp_path.glob(".artifact.bin.*"))
        client.close()

    def test_get_stats(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        response_data = {
            "totalTraces": 100,
            "totalTokens": 50000,
            "totalCostUsd": 1.23,
            "errorRate": 0.05,
        }

        with PrompticClient() as client:

            def handler(request: httpx.Request) -> httpx.Response:
                assert "/traces/stats" in str(request.url)
                return httpx.Response(200, json=response_data)

            client._client = httpx.Client(
                transport=httpx.MockTransport(handler),
                base_url="https://promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            result = client.get_stats(days_back=7)
            assert result == response_data

    def test_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        client = PrompticClient(endpoint="https://example.com/")
        assert client.endpoint == "https://example.com"
        client.close()

    def test_dataset_case_crud_uses_canonical_routes_and_payloads(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        component_id = "component-123"
        dataset_id = "dataset-123"
        dataset_case = _canonical_dataset_payload(include_cases=True)["cases"][0]
        requests: list[tuple[str, str, bytes]] = []

        with PrompticClient() as client:

            def handler(request: httpx.Request) -> httpx.Response:
                requests.append((request.method, request.url.path, request.read()))
                if request.method == "GET":
                    return httpx.Response(
                        200,
                        json=dataset_case
                        if request.url.path.endswith("/7")
                        else {"data": [dataset_case]},
                    )
                if request.method == "POST":
                    return httpx.Response(201, json={"data": [dataset_case]})
                if request.method == "PATCH":
                    return httpx.Response(200, json=dataset_case)
                return httpx.Response(204)

            client._client = httpx.Client(
                transport=httpx.MockTransport(handler),
                base_url="https://promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            assert client.list_dataset_cases(component_id, dataset_id) == {"data": [dataset_case]}
            assert client.get_dataset_case(component_id, dataset_id, 7) == dataset_case
            payload: DatasetCaseCreate = {
                "inputPayload": {"question": "hello"},
                "expectedPayload": {"answer": "world"},
            }
            assert client.create_dataset_cases(component_id, dataset_id, [payload]) == {
                "data": [dataset_case]
            }
            assert (
                client.update_dataset_case(
                    component_id, dataset_id, 7, expectedPayload={"answer": "updated"}
                )
                == dataset_case
            )
            client.delete_dataset_case(component_id, dataset_id, 7)

        base_path = f"/api/v1/components/{component_id}/datasets/{dataset_id}/cases"
        assert [entry[:2] for entry in requests] == [
            ("GET", base_path),
            ("GET", f"{base_path}/7"),
            ("POST", base_path),
            ("PATCH", f"{base_path}/7"),
            ("DELETE", f"{base_path}/7"),
        ]
        assert requests[2][2] == (
            b'[{"inputPayload":{"question":"hello"},"expectedPayload":{"answer":"world"}}]'
        )

    def test_duplicate_experiment_default(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        with PrompticClient() as client:
            captured: dict = {}

            def handler(request: httpx.Request) -> httpx.Response:
                captured["url"] = str(request.url)
                captured["body"] = request.read()
                return httpx.Response(201, json={"id": "new-exp", "name": "Run 2"})

            client._client = httpx.Client(
                transport=httpx.MockTransport(handler),
                base_url="https://promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            result = client.duplicate_experiment("src-exp-id")
            assert result["id"] == "new-exp"
            assert "/experiments/src-exp-id/duplicate" in captured["url"]
            # No flags → empty body, no continueFromOptimized.
            assert b"continueFromOptimized" not in captured["body"]
            assert b"initialPromptOverride" not in captured["body"]

    def test_duplicate_experiment_continue_flow(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        with PrompticClient() as client:
            captured: dict = {}

            def handler(request: httpx.Request) -> httpx.Response:
                captured["body"] = request.read()
                return httpx.Response(201, json={"id": "new-exp"})

            client._client = httpx.Client(
                transport=httpx.MockTransport(handler),
                base_url="https://promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            client.duplicate_experiment("src", continue_from_optimized=True)
            assert b'"continueFromOptimized":true' in captured["body"]

    def test_duplicate_experiment_with_prompt_override(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        with PrompticClient() as client:
            captured: dict = {}

            def handler(request: httpx.Request) -> httpx.Response:
                captured["body"] = request.read()
                return httpx.Response(201, json={"id": "new-exp"})

            client._client = httpx.Client(
                transport=httpx.MockTransport(handler),
                base_url="https://promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            client.duplicate_experiment("src", initial_prompt_override="hello world")
            assert b'"initialPromptOverride":"hello world"' in captured["body"]

    def test_create_tool_selection_experiment(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        captured: dict = {}

        with PrompticClient() as client:

            def handler(request: httpx.Request) -> httpx.Response:
                assert request.method == "POST"
                assert str(request.url).endswith("/experiments/tool-selection")
                captured["body"] = json.loads(request.content)
                return httpx.Response(201, json={"id": "exp_1", "taskType": "toolSelection"})

            client._client = httpx.Client(
                transport=httpx.MockTransport(handler),
                base_url="https://promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            result = client.create_tool_selection_experiment(
                "comp_1",
                tools=[
                    {"name": "search", "description": "find", "input_schema": {"type": "object"}}
                ],
                test_cases=[
                    {"query": "find x", "expected_tool": "search"},
                    {"query": "chit chat", "expectedTool": ""},
                ],
                system_prompt="be precise",
                train_split_ratio=0.8,
            )

        assert result == {"id": "exp_1", "taskType": "toolSelection"}
        body = captured["body"]
        # snake_case inputs normalized to the API's camelCase shape
        assert body["tools"] == [
            {"name": "search", "description": "find", "inputSchema": {"type": "object"}}
        ]
        assert body["testCases"] == [
            {"query": "find x", "expectedTool": "search"},
            {"query": "chit chat", "expectedTool": ""},
        ]
        # defaults + passthrough
        assert body["toolSource"] == "manual"
        assert body["optimizeSystemPrompt"] is False
        assert body["systemPrompt"] == "be precise"
        assert body["trainSplitRatio"] == 0.8
        # target_model omitted so the platform applies its default
        assert "targetModel" not in body

    def test_tool_selection_input_schema_is_optional_at_runtime(self):
        assert "input_schema" in ToolSelectionTool.__optional_keys__
        assert "input_schema" not in ToolSelectionTool.__required_keys__

    def test_tool_selection_iteration_fields_are_optional_at_runtime(self):
        assert "toolDescriptions" in Iteration.__optional_keys__
        assert "selectionSystemPrompt" in Iteration.__optional_keys__
        assert "toolDescriptions" not in Iteration.__required_keys__
        assert "selectionSystemPrompt" not in Iteration.__required_keys__

    def test_tool_selection_requires_expected_tool(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        with (
            PrompticClient() as client,
            pytest.raises(ValueError, match="expected_tool or expectedTool"),
        ):
            client.create_tool_selection_experiment(
                "comp_1",
                tools=[{"name": "search", "description": "find"}],
                test_cases=[{"query": "find x"}],  # type: ignore[typeddict-item]
            )


class TestAsyncPrompticClient:
    def test_requires_api_key(self):
        with pytest.raises(ValueError, match="Authentication required"):
            AsyncPrompticClient(api_key=None)

    def test_reads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test_key")
        client = AsyncPrompticClient()
        assert client.api_key == "pk_test_key"

    def test_custom_endpoint(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        client = AsyncPrompticClient(endpoint="https://custom.example.com")
        assert client.endpoint == "https://custom.example.com"
        assert str(client._client.base_url).rstrip("/") == "https://custom.example.com/api/v1"

    def test_default_endpoint(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        client = AsyncPrompticClient()
        assert client.endpoint == "https://promptic.eu"

    async def test_context_manager(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        async with AsyncPrompticClient() as client:
            assert client.api_key == "pk_test"

    async def test_list_traces(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        response_data = {"traces": [{"traceId": "abc123"}], "total": 1}

        async with AsyncPrompticClient() as client:

            async def handler(request: httpx.Request) -> httpx.Response:
                assert "/traces" in str(request.url)
                assert request.headers["authorization"] == "Bearer pk_test"
                return httpx.Response(200, json=response_data)

            client._client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler),
                base_url="https://promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            result = await client.list_traces(limit=10)
            assert result == response_data

    async def test_get_trace(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        response_data = {"traceId": "abc123", "spans": []}

        async with AsyncPrompticClient() as client:

            async def handler(request: httpx.Request) -> httpx.Response:
                assert "/traces/abc123" in str(request.url)
                return httpx.Response(200, json=response_data)

            client._client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler),
                base_url="https://promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            result = await client.get_trace("abc123")
            assert result == response_data

    async def test_get_artifact_content_follows_redirect(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        requests: list[httpx.Request] = []

        async with AsyncPrompticClient() as client:

            async def handler(request: httpx.Request) -> httpx.Response:
                requests.append(request)
                if request.url.path == "/api/v1/artifacts/artifact-id/content":
                    return httpx.Response(
                        307,
                        headers={"Location": "https://storage.example/artifact.png"},
                    )
                if str(request.url) == "https://storage.example/artifact.png":
                    return httpx.Response(200, content=b"artifact-bytes")
                return httpx.Response(404, json={"error": "not found"})

            client._client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler),
                base_url="https://promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            assert await client.get_artifact_content("artifact-id") == b"artifact-bytes"
            assert [str(request.url) for request in requests] == [
                "https://promptic.eu/api/v1/artifacts/artifact-id/content",
                "https://storage.example/artifact.png",
            ]

    async def test_get_stats(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        response_data = {
            "totalTraces": 100,
            "totalTokens": 50000,
            "totalCostUsd": 1.23,
            "errorRate": 0.05,
        }

        async with AsyncPrompticClient() as client:

            async def handler(request: httpx.Request) -> httpx.Response:
                assert "/traces/stats" in str(request.url)
                return httpx.Response(200, json=response_data)

            client._client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler),
                base_url="https://promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            result = await client.get_stats(days_back=7)
            assert result == response_data

    async def test_get_dataset_returns_canonical_payload(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        dataset = _canonical_dataset_payload(include_cases=True)

        async with AsyncPrompticClient() as client:

            async def handler(request: httpx.Request) -> httpx.Response:
                assert request.url.path.endswith(f"/datasets/{dataset['id']}")
                return httpx.Response(200, json=dataset)

            client._client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler),
                base_url="https://promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            assert await client.get_dataset(dataset["aiComponentId"], dataset["id"]) == dataset

    def test_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        client = AsyncPrompticClient(endpoint="https://example.com/")
        assert client.endpoint == "https://example.com"

    async def test_create_dataset_cases_uses_canonical_payload(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        payload: DatasetCaseCreate = {
            "inputPayload": {"input": "in"},
            "expectedPayload": {"value": "out"},
        }

        async with AsyncPrompticClient() as client:

            async def handler(request: httpx.Request) -> httpx.Response:
                assert request.url.path == (
                    "/api/v1/components/component-123/datasets/dataset-123/cases"
                )
                assert request.read() == (
                    b'[{"inputPayload":{"input":"in"},"expectedPayload":{"value":"out"}}]'
                )
                return httpx.Response(201, json={"data": []})

            client._client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler),
                base_url="https://promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            result = await client.create_dataset_cases("component-123", "dataset-123", [payload])

        assert result == {"data": []}

    @pytest.mark.asyncio
    async def test_duplicate_experiment_continue_flow(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        async with AsyncPrompticClient() as client:
            captured: dict = {}

            async def handler(request: httpx.Request) -> httpx.Response:
                captured["url"] = str(request.url)
                captured["body"] = request.read()
                return httpx.Response(201, json={"id": "new-exp"})

            client._client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler),
                base_url="https://promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            result = await client.duplicate_experiment("src", continue_from_optimized=True)

        assert result["id"] == "new-exp"
        assert "/experiments/src/duplicate" in captured["url"]
        assert b'"continueFromOptimized":true' in captured["body"]


class TestAIApplicationScope:
    """AI Application scope resolution, header, and endpoint."""

    def test_ai_application_id_sets_header(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_ACCESS_TOKEN", "tok")
        monkeypatch.delenv("PROMPTIC_WORKSPACE_ID", raising=False)
        monkeypatch.delenv("PROMPTIC_AI_APPLICATION_ID", raising=False)
        client = PrompticClient(ai_application_id="app-1")
        assert client._client.headers["X-AI-Application-Id"] == "app-1"
        assert client.ai_application_id == "app-1"
        # Deprecated attribute stays in sync.
        assert client.workspace_id == "app-1"
        client.close()

    def test_deprecated_workspace_id_argument(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_ACCESS_TOKEN", "tok")
        monkeypatch.delenv("PROMPTIC_WORKSPACE_ID", raising=False)
        monkeypatch.delenv("PROMPTIC_AI_APPLICATION_ID", raising=False)
        client = PrompticClient(workspace_id="legacy-1")
        assert client._client.headers["X-AI-Application-Id"] == "legacy-1"
        assert client.ai_application_id == "legacy-1"
        client.close()

    def test_env_fallbacks(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_ACCESS_TOKEN", "tok")
        monkeypatch.delenv("PROMPTIC_AI_APPLICATION_ID", raising=False)
        monkeypatch.setenv("PROMPTIC_WORKSPACE_ID", "env-legacy")
        client = PrompticClient()
        assert client.ai_application_id == "env-legacy"
        client.close()

        monkeypatch.setenv("PROMPTIC_AI_APPLICATION_ID", "env-new")
        client = PrompticClient()
        # New env var wins over the legacy one.
        assert client.ai_application_id == "env-new"
        client.close()

    def test_get_ai_application(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        with PrompticClient() as client:

            def handler(request: httpx.Request) -> httpx.Response:
                assert str(request.url).endswith("/ai-application")
                return httpx.Response(200, json={"id": "app-1", "name": "App"})

            client._client = httpx.Client(
                transport=httpx.MockTransport(handler),
                base_url="https://promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            assert client.get_ai_application()["id"] == "app-1"
            # Deprecated alias hits the same endpoint.
            assert client.get_workspace()["id"] == "app-1"


class TestAsyncAIApplicationScope:
    """Async AI Application scope resolution, header, and endpoint."""

    async def test_ai_application_id_sets_header(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_ACCESS_TOKEN", "tok")
        monkeypatch.delenv("PROMPTIC_WORKSPACE_ID", raising=False)
        monkeypatch.delenv("PROMPTIC_AI_APPLICATION_ID", raising=False)

        async with AsyncPrompticClient(ai_application_id="app-1") as client:
            assert client._client.headers["X-AI-Application-Id"] == "app-1"
            assert client.ai_application_id == "app-1"
            assert client.workspace_id == "app-1"

    async def test_get_ai_application_and_deprecated_alias(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        requests: list[httpx.Request] = []

        async with AsyncPrompticClient() as client:

            async def handler(request: httpx.Request) -> httpx.Response:
                requests.append(request)
                return httpx.Response(200, json={"id": "app-1", "name": "App"})

            client._client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler),
                base_url="https://promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            assert (await client.get_ai_application())["id"] == "app-1"
            assert (await client.get_workspace())["id"] == "app-1"

        assert [request.url.path for request in requests] == [
            "/api/v1/ai-application",
            "/api/v1/ai-application",
        ]
