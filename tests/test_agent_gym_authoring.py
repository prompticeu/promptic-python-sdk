"""Focused tests for canonical Agent component authoring."""

from __future__ import annotations

import json
from typing import Any, cast

import httpx
import pytest

import promptic_sdk
from promptic_sdk import (
    AgentGymAPIError,
    AgentGymClient,
    AsyncAgentGymClient,
    BenchmarkCase,
    BenchmarkFile,
    BulkCaseProgress,
    BulkCaseUploadError,
    ClassificationF1,
    EvidenceKind,
    EvidencePolicy,
    ExpectedBehaviorJudge,
    FieldLevelJudge,
    FieldScoring,
    InvestigationBudget,
    MetricBinding,
    VerifierAgent,
    VerifierMetric,
)

WORKSPACE_ID = "00000000-0000-4000-8000-000000000040"
BENCHMARK_ID = "00000000-0000-4000-8000-000000000041"
STORAGE_ID = "00000000-0000-4000-8000-000000000042"


def _benchmark(**overrides: Any) -> dict[str, Any]:
    value = {
        "id": BENCHMARK_ID,
        "workspaceId": WORKSPACE_ID,
        "name": "Invoice agent",
        "description": "Extract invoices.",
        "inputSchema": {"type": "object", "properties": {}},
        "targetSchema": {"type": "object", "properties": {}},
        "configuration": {"readyForSubmission": True},
    }
    value.update(overrides)
    return value


def _case() -> dict[str, Any]:
    return {
        "id": 1,
        "datasetId": BENCHMARK_ID,
        "inputFiles": [],
        "inputPayload": {},
        "expectedOutput": {"label": "invoice"},
        "expectedBehavior": None,
        "outputFiles": [],
        "createdAt": "2026-08-04T00:00:00.000Z",
        "updatedAt": "2026-08-04T00:00:00.000Z",
    }


def _presign(filename: str) -> dict[str, Any]:
    return {
        "storageObjectId": STORAGE_ID,
        "access": "private",
        "strategy": "url",
        "provider": "s3",
        "uploadUrl": f"https://storage.test/{filename}",
        "finalUrl": f"https://storage.test/private/{filename}",
        "method": "PUT",
        "headers": {"Content-Type": "application/pdf"},
        "maxSizeBytes": 25_000_000,
        "expiresAt": "2026-08-04T00:15:00.000Z",
    }


def _replace_sync(client: AgentGymClient, handler: Any, direct: Any) -> None:
    client._transport.api.close()
    client._transport.direct.close()
    client._transport.api = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://promptic.test/api/v1",
        headers={"Authorization": "Bearer ptc_test"},
    )
    client._transport.direct = httpx.Client(transport=httpx.MockTransport(direct))


async def _replace_async(client: AsyncAgentGymClient, handler: Any, direct: Any) -> None:
    await client._transport.api.aclose()
    await client._transport.direct.aclose()
    client._transport.api = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://promptic.test/api/v1"
    )
    client._transport.direct = httpx.AsyncClient(transport=httpx.MockTransport(direct))


def test_create_configures_schemas_evaluators_and_nested_files(tmp_path):
    bodies: list[dict[str, Any]] = []
    source = tmp_path / "invoice.pdf"
    second_source = tmp_path / "invoice-2.pdf"
    source.write_bytes(b"pdf")
    second_source.write_bytes(b"pdf-2")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/storage-objects/presign"):
            return httpx.Response(200, json=_presign("invoice.pdf"))
        if request.method == "POST":
            bodies.append(json.loads(request.content))
            return httpx.Response(
                201, json=_case() if request.url.path.endswith("/cases") else _benchmark()
            )
        return httpx.Response(404)

    with AgentGymClient(api_key="ptc_test", workspace_id=WORKSPACE_ID) as client:
        _replace_sync(client, handler, lambda request: httpx.Response(200))
        agent = client.benchmarks.create(
            name="Invoice agent",
            goal="Extract the invoice classification.",
            input_schema={
                "type": "object",
                "properties": {"document": {"type": "array", "x-promptic-type": "file"}},
            },
            output_schema={
                "type": "object",
                "properties": {"label": {"type": "string", "enum": ["invoice", "other"]}},
            },
            evaluators=[ClassificationF1(("label",), weight=2)],
        )
        agent.cases.add(
            input={"document": [BenchmarkFile(source), BenchmarkFile(second_source)]},
            output={"label": "invoice"},
            expected_behavior="Read the document before classifying it.",
        )

    assert bodies[0]["taskDescription"] == "Extract the invoice classification."
    assert bodies[0]["evaluators"] == [
        {"kind": "f1", "weight": 2, "required": False, "fieldPaths": ["label"]}
    ]
    assert bodies[1]["input"] == {"document": ["invoice.pdf", "invoice-2.pdf"]}
    assert bodies[1]["output"] == {"label": "invoice"}
    assert len(bodies[1]["uploads"]) == 2
    assert all(upload["storageObjectId"] == STORAGE_ID for upload in bodies[1]["uploads"])
    assert {upload["fieldPath"] for upload in bodies[1]["uploads"]} == {"document"}
    assert agent.ready_for_submission is True


def test_add_rejects_files_over_the_combined_case_limit_before_upload(tmp_path):
    sources = []
    for index in range(5):
        source = tmp_path / f"document-{index}.pdf"
        with source.open("wb") as stream:
            stream.truncate(20_000_001)
        sources.append(source)

    requests: list[httpx.Request] = []
    schema = {
        "type": "object",
        "properties": {"documents": {"type": "array", "x-promptic-type": "file"}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_benchmark(inputSchema=schema))

    with AgentGymClient(api_key="ptc_test", workspace_id=WORKSPACE_ID) as client:
        _replace_sync(client, handler, lambda request: httpx.Response(500))
        benchmark = client.benchmarks.get(BENCHMARK_ID)
        with pytest.raises(ValueError, match="100000000 bytes in total"):
            benchmark.cases.add(input={"documents": [BenchmarkFile(source) for source in sources]})

    assert len(requests) == 1


def test_verifier_agent_serializes_finalized_contract_exactly():
    verifier = VerifierAgent(
        instructions="Inspect the selected evidence and score factual completeness.",
        model="gpt-4.1-nano",
        name="Evidence verifier",
        description="Checks the complete submission.",
        required=True,
        threshold=0.7,
        evidence=EvidencePolicy(
            selected=(
                EvidenceKind.CASE_INPUT,
                EvidenceKind.SUBMITTED_OUTPUT,
                EvidenceKind.EXPECTED_OUTPUT,
                EvidenceKind.EXECUTION_TRACE,
            )
        ),
        budget=InvestigationBudget(max_steps=200),
        metrics=(
            VerifierMetric("correctness", "Correctness", "Check facts.", weight=2, threshold=0.8),
            VerifierMetric("completeness", "Completeness", "Check coverage."),
        ),
    )

    assert verifier.as_request() == {
        "kind": "verifier_agent",
        "required": True,
        "name": "Evidence verifier",
        "description": "Checks the complete submission.",
        "threshold": 0.7,
        "instructions": "Inspect the selected evidence and score factual completeness.",
        "model": "gpt-4.1-nano",
        "evidence": {
            "caseInputs": True,
            "submittedOutput": True,
            "expectedBehavior": False,
            "expectedOutput": True,
            "executionTrace": True,
        },
        "metrics": [
            {"key": "correctness", "name": "Correctness", "instructions": "Check facts."},
            {"key": "completeness", "name": "Completeness", "instructions": "Check coverage."},
        ],
        "metricBindings": {
            "correctness": {"weight": 2.0, "threshold": 0.8},
        },
        "budget": {"maxSteps": 200},
    }
    assert "rubric" not in verifier.as_request()
    assert "tools" not in verifier.as_request()
    assert "required" not in verifier.as_request()["evidence"]


@pytest.mark.parametrize("metric_count", [0, 9])
def test_verifier_agent_rejects_metric_counts_outside_fixed_contract(metric_count):
    metrics = tuple(
        VerifierMetric(f"metric_{index}", f"Metric {index}", "Check it.")
        for index in range(metric_count)
    )
    with pytest.raises(ValueError, match="between 1 and 8 metrics"):
        VerifierAgent(
            instructions="Evaluate.",
            metrics=metrics,
        )


def test_verifier_agent_rejects_duplicate_and_invalid_metric_keys():
    duplicate = VerifierMetric("correctness", "Correctness", "Check facts.")
    with pytest.raises(ValueError, match="must be unique"):
        VerifierAgent(
            instructions="Evaluate.",
            metrics=(duplicate, duplicate),
        )
    for key in ("Correctness", "correct-ness", "_correctness", "1_correctness"):
        with pytest.raises(ValueError, match="snake_case"):
            VerifierMetric(key, "Correctness", "Check facts.")


def test_verifier_agent_legacy_rubric_maps_only_to_instructions():
    with pytest.warns(DeprecationWarning, match="rubric"):
        request = VerifierAgent(
            rubric="Evaluate the result.",
            metrics=(VerifierMetric("overall", "Overall", "Assess the complete result."),),
        ).as_request()

    assert request["instructions"] == "Evaluate the result."
    assert "rubric" not in request


@pytest.mark.parametrize("max_steps", [0, True, 1.5])
def test_verifier_agent_rejects_invalid_investigation_budget(max_steps):
    with pytest.raises(ValueError, match="at least 1"):
        InvestigationBudget(max_steps=max_steps)


def test_verifier_agent_accepts_investigation_budget_above_twenty():
    assert InvestigationBudget(max_steps=200).as_request() == {"maxSteps": 200}


def test_configure_and_add_many_use_canonical_contract():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=_benchmark())
        if request.method == "PATCH":
            return httpx.Response(200, json=_benchmark())
        return httpx.Response(
            201,
            json={
                "created": 1,
                "failed": 0,
                "results": [
                    {
                        "label": "One",
                        "status": "created",
                        "datasetCaseId": 1,
                        "hasExpectedOutput": True,
                        "hasExpectedBehavior": False,
                    }
                ],
            },
        )

    with AgentGymClient(api_key="ptc_test", workspace_id=WORKSPACE_ID) as client:
        _replace_sync(client, handler, lambda request: httpx.Response(500))
        agent = client.benchmarks.get(BENCHMARK_ID)
        agent.configure(
            goal="Updated goal",
            evaluators=[ExpectedBehaviorJudge()],
        )
        result = agent.cases.add_many(
            [BenchmarkCase(title="One", input={"task": "one"}, output={"answer": "1"})]
        )

    patch = json.loads(requests[1].content)
    assert patch["taskDescription"] == "Updated goal"
    assert patch["evaluators"][0] == {
        "kind": "expected_behavior_judge",
        "required": False,
    }
    assert requests[2].url.path.endswith(f"/benchmarks/{BENCHMARK_ID}/cases/bulk")
    assert json.loads(requests[2].content) == {
        "items": [{"title": "One", "input": {"task": "one"}, "output": {"answer": "1"}}]
    }
    assert result["created"] == 1


def test_published_configure_keeps_active_identity_and_applies_staged_draft_view():
    requests: list[httpx.Request] = []
    active_revision_id = "00000000-0000-4000-8000-000000000043"
    next_schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=_benchmark(activeRevisionId=active_revision_id))
        return httpx.Response(
            202,
            json={
                "staged": True,
                "message": "Benchmark changes staged",
                "draft": {
                    "id": "00000000-0000-4000-8000-000000000044",
                    "datasetId": BENCHMARK_ID,
                    "baseRevisionId": active_revision_id,
                    "workingDatasetId": "00000000-0000-4000-8000-000000000045",
                    "status": "ready",
                    "taskSnapshot": {"inputContract": {"text": {"schema": {"type": "object"}}}},
                    "targetSchema": next_schema,
                    "evaluatorSnapshot": {"evaluatorGraph": {"nodes": []}},
                    "conflicts": [],
                    "resolutions": {},
                    "lockVersion": 1,
                },
                "conflicts": [],
            },
        )

    with AgentGymClient(api_key="ptc_test", workspace_id=WORKSPACE_ID) as client:
        _replace_sync(client, handler, lambda request: httpx.Response(500))
        benchmark = client.benchmarks.get(BENCHMARK_ID)
        benchmark.configure(output_schema=next_schema)

    assert requests[1].method == "PATCH"
    assert benchmark.data["activeRevisionId"] == active_revision_id
    assert benchmark.data["targetSchema"] == next_schema
    assert benchmark.data["evaluatorGraph"] == {"nodes": []}
    assert benchmark.staged_message == "Benchmark changes staged"
    assert benchmark.staged_conflicts == []


def test_published_configure_preserves_cached_values_for_omitted_draft_snapshots():
    conflict = {
        "direction": "output",
        "fieldPath": "answer",
        "issue": "required_field_added",
        "affectedCaseIds": [1],
        "affectedCaseCount": 1,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                json=_benchmark(
                    targetSchema={"type": "object"},
                    evaluatorGraph={"nodes": [{"id": "judge"}]},
                ),
            )
        return httpx.Response(
            202,
            json={
                "staged": True,
                "message": "Resolve schema conflicts",
                "draft": {"status": "conflicted", "conflicts": [conflict]},
                "conflicts": [conflict],
            },
        )

    with AgentGymClient(api_key="ptc_test", workspace_id=WORKSPACE_ID) as client:
        _replace_sync(client, handler, lambda request: httpx.Response(500))
        benchmark = client.benchmarks.get(BENCHMARK_ID)
        original_input = benchmark.data["inputSchema"]
        original_target = benchmark.data["targetSchema"]
        original_graph = benchmark.data["evaluatorGraph"]
        benchmark.configure(goal="Updated goal")

    assert benchmark.data["inputSchema"] == original_input
    assert benchmark.data["targetSchema"] == original_target
    assert benchmark.data["evaluatorGraph"] == original_graph
    assert benchmark.staged_message == "Resolve schema conflicts"
    assert benchmark.staged_conflicts == [conflict]


def test_published_case_list_does_not_create_a_revision_draft():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/revisions/draft"):
            return httpx.Response(200, json={"draft": None})
        if request.url.path.endswith("/cases"):
            return httpx.Response(200, json={"data": [_case()]})
        return httpx.Response(200, json=_benchmark(activeRevisionId="revision-1"))

    with AgentGymClient(api_key="ptc_test", workspace_id=WORKSPACE_ID) as client:
        _replace_sync(client, handler, lambda request: httpx.Response(500))
        benchmark = client.benchmarks.get(BENCHMARK_ID)
        cases = benchmark.cases.list()

    assert cases == [_case()]
    assert [request.method for request in requests] == ["GET", "GET", "GET"]
    assert requests[-1].url.query == b""


def test_published_case_write_creates_and_targets_a_revision_draft():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/revisions/draft") and request.method == "GET":
            return httpx.Response(200, json={"draft": None})
        if request.url.path.endswith("/revisions/draft") and request.method == "POST":
            return httpx.Response(
                201,
                json={
                    "draft": {
                        "id": "draft-1",
                        "datasetId": BENCHMARK_ID,
                        "baseRevisionId": "revision-1",
                        "workingDatasetId": "working-1",
                        "status": "ready",
                    },
                    "created": True,
                },
            )
        if request.url.path.endswith("/cases"):
            return httpx.Response(201, json=_case())
        return httpx.Response(200, json=_benchmark(activeRevisionId="revision-1"))

    with AgentGymClient(api_key="ptc_test", workspace_id=WORKSPACE_ID) as client:
        _replace_sync(client, handler, lambda request: httpx.Response(500))
        benchmark = client.benchmarks.get(BENCHMARK_ID)
        created = benchmark.cases.add(input={"message": "hello"})

    assert created == _case()
    assert [request.method for request in requests] == ["GET", "GET", "POST", "POST"]
    assert requests[-1].url.query == b"revision=draft"


def test_add_many_reuses_one_presigned_file_binding_for_nested_files(tmp_path):
    requests: list[httpx.Request] = []
    source = tmp_path / "invoice.pdf"
    source.write_bytes(b"pdf")
    schema = {
        "type": "object",
        "properties": {"document": {"type": "array", "x-promptic-type": "file"}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=_benchmark(inputSchema=schema))
        if request.url.path.endswith("/storage-objects/presign"):
            return httpx.Response(200, json=_presign("invoice.pdf"))
        return httpx.Response(
            201,
            json={
                "created": 2,
                "failed": 0,
                "results": [],
                "draftCaseCount": 2,
                "published": False,
            },
        )

    with AgentGymClient(api_key="ptc_test", workspace_id=WORKSPACE_ID) as client:
        _replace_sync(client, handler, lambda request: httpx.Response(200))
        benchmark = client.benchmarks.get(BENCHMARK_ID)
        result = benchmark.cases.add_many(
            [
                BenchmarkCase(
                    title="One",
                    input={"document": [BenchmarkFile(source, path="shared/invoice.pdf")]},
                    output={"label": "invoice"},
                ),
                BenchmarkCase(
                    title="Two",
                    input={"document": [BenchmarkFile(source, path="shared/invoice.pdf")]},
                    output={"label": "invoice"},
                ),
            ]
        )

    assert result["created"] == 2
    assert result["draftCaseCount"] == 2
    assert len(requests) == 3
    assert sum(request.url.path.endswith("/storage-objects/presign") for request in requests) == 1
    bulk = requests[2]
    assert bulk.url.path.endswith(f"/benchmarks/{BENCHMARK_ID}/cases/bulk")
    body = json.loads(bulk.content)
    assert len(body["items"]) == 2
    assert all(item["uploads"][0]["storageObjectId"] == STORAGE_ID for item in body["items"])
    assert all(item["uploads"][0]["uploadPath"] == "shared/invoice.pdf" for item in body["items"])


def test_update_creates_draft_and_replaces_case():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/revisions/draft") and request.method == "GET":
            return httpx.Response(200, json={"draft": None})
        if request.url.path.endswith("/revisions/draft") and request.method == "POST":
            return httpx.Response(
                201,
                json={
                    "draft": {
                        "id": "draft-1",
                        "datasetId": BENCHMARK_ID,
                        "baseRevisionId": "revision-1",
                        "workingDatasetId": "working-1",
                        "status": "ready",
                    },
                    "created": True,
                },
            )
        if request.method == "PATCH" and request.url.path.endswith("/cases"):
            return httpx.Response(200, json=_case())
        return httpx.Response(200, json=_benchmark(activeRevisionId="revision-1"))

    with AgentGymClient(api_key="ptc_test", workspace_id=WORKSPACE_ID) as client:
        _replace_sync(client, handler, lambda request: httpx.Response(500))
        benchmark = client.benchmarks.get(BENCHMARK_ID)
        updated = benchmark.cases.update(
            7,
            title="Updated",
            input={"message": "changed"},
            output={"label": "invoice"},
            expected_behavior="Route it.",
        )

    assert updated == _case()
    patch_request = requests[-1]
    assert patch_request.method == "PATCH"
    assert patch_request.url.query == b"revision=draft&caseId=7"
    assert json.loads(patch_request.content) == {
        "title": "Updated",
        "input": {"message": "changed"},
        "output": {"label": "invoice"},
        "expectedBehavior": "Route it.",
        "uploads": [],
    }


@pytest.mark.asyncio
async def test_async_update_replaces_unpublished_case_without_draft():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "PATCH":
            return httpx.Response(200, json=_case())
        return httpx.Response(200, json=_benchmark(activeRevisionId=None))

    async with AsyncAgentGymClient(api_key="ptc_test", workspace_id=WORKSPACE_ID) as client:
        await _replace_async(client, handler, lambda request: httpx.Response(500))
        benchmark = await client.benchmarks.get(BENCHMARK_ID)
        updated = await benchmark.cases.update(3, input={"message": "changed"})

    assert updated == _case()
    assert requests[-1].method == "PATCH"
    assert requests[-1].url.query == b"caseId=3"


def test_add_many_rejects_conflicting_content_at_the_same_upload_path(tmp_path):
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    schema = {
        "type": "object",
        "properties": {"document": {"type": "array", "x-promptic-type": "file"}},
    }

    with AgentGymClient(api_key="ptc_test", workspace_id=WORKSPACE_ID) as client:
        _replace_sync(
            client,
            lambda request: httpx.Response(200, json=_benchmark(inputSchema=schema)),
            lambda request: httpx.Response(500),
        )
        benchmark = client.benchmarks.get(BENCHMARK_ID)
        with pytest.raises(ValueError, match="different files with the same path"):
            benchmark.cases.add_many(
                [
                    BenchmarkCase(
                        input={"document": [BenchmarkFile(first, path="document.pdf")]},
                        output={"label": "a"},
                    ),
                    BenchmarkCase(
                        input={"document": [BenchmarkFile(second, path="document.pdf")]},
                        output={"label": "b"},
                    ),
                ]
            )


@pytest.mark.asyncio
async def test_async_add_many_uses_the_bulk_endpoint():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=_benchmark())
        return httpx.Response(201, json={"created": 1, "failed": 0, "results": []})

    async with AsyncAgentGymClient(api_key="ptc_test", workspace_id=WORKSPACE_ID) as client:
        await _replace_async(client, handler, lambda request: httpx.Response(500))
        benchmark = await client.benchmarks.get(BENCHMARK_ID)
        result = await benchmark.cases.add_many(
            [BenchmarkCase(input={"task": "one"}, output={"answer": "1"})]
        )

    assert result["created"] == 1
    assert requests[1].url.path.endswith(f"/benchmarks/{BENCHMARK_ID}/cases/bulk")


def test_configure_explicit_field_evaluator_uses_output_schema():
    requests: list[httpx.Request] = []
    output_schema = {
        "type": "object",
        "properties": {
            "acceptance_criteria_value": {"type": "string"},
            "test_method": {"type": "string"},
            "comment": {"type": "string"},
            "confidence": {"type": "number"},
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_benchmark(targetSchema=output_schema))

    with AgentGymClient(api_key="ptc_test", workspace_id=WORKSPACE_ID) as client:
        _replace_sync(client, handler, lambda request: httpx.Response(500))
        agent = client.benchmarks.get(BENCHMARK_ID)
        agent.configure(
            output_schema=output_schema,
            evaluators=[
                FieldLevelJudge(
                    fields={
                        "acceptance_criteria_value": FieldScoring("exact"),
                        "test_method": FieldScoring("embedding"),
                        "comment": FieldScoring("exact", include=False),
                    }
                ),
                ExpectedBehaviorJudge(),
            ],
        )

    patch = json.loads(requests[1].content)
    expected_fields = {
        "acceptance_criteria_value": {"include": True, "method": "exact", "weight": 1.0},
        "test_method": {"include": True, "method": "embedding", "weight": 1.0},
        "comment": {"include": False, "method": "exact", "weight": 1.0},
    }
    assert patch["targetSchema"] == output_schema
    assert patch["evaluators"][0] == {
        "kind": "field_level_judge",
        "weight": 1.0,
        "required": False,
        "fields": expected_fields,
    }
    assert patch["evaluators"][1]["kind"] == "expected_behavior_judge"
    assert "evaluatorGraph" not in patch


def test_create_accepts_output_schema_with_explicit_field_evaluator():
    bodies: list[dict[str, Any]] = []
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(201, json=_benchmark(targetSchema=schema))

    with AgentGymClient(api_key="ptc_test", workspace_id=WORKSPACE_ID) as client:
        _replace_sync(client, handler, lambda request: httpx.Response(500))
        client.benchmarks.create(
            name="Structured benchmark",
            goal="Extract the answer.",
            output_schema=schema,
            evaluators=[FieldLevelJudge(fields={"answer": "exact"})],
        )

    assert bodies[0]["targetSchema"] == schema
    assert bodies[0]["aiApplicationId"] == WORKSPACE_ID
    assert "workspaceId" not in bodies[0]
    assert bodies[0]["evaluators"][0]["fields"] == {
        "answer": {"include": True, "method": "exact", "weight": 1.0}
    }


def test_scoped_api_key_can_create_and_list_without_application_id(monkeypatch):
    monkeypatch.delenv("PROMPTIC_AI_APPLICATION_ID", raising=False)
    monkeypatch.delenv("PROMPTIC_WORKSPACE_ID", raising=False)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(201, json=_benchmark())
        return httpx.Response(200, json={"data": [_benchmark()]})

    with AgentGymClient(api_key="ptc_scoped") as client:
        _replace_sync(client, handler, lambda request: httpx.Response(500))
        client.benchmarks.create(name="Invoice agent", goal="Extract invoices.")
        client.benchmarks.list()

    assert "aiApplicationId" not in json.loads(requests[0].content)
    assert "aiApplicationId" not in requests[1].url.params


def test_field_level_evaluator_serializes_explicit_typed_fields():
    request = FieldLevelJudge(
        name="Mixed structured scoring",
        fields={
            "answer": "embedding",
            "comment": FieldScoring("exact", include=False),
        },
    ).as_request()

    assert request["fields"] == {
        "answer": {"include": True, "method": "embedding", "weight": 1.0},
        "comment": {"include": False, "method": "exact", "weight": 1.0},
    }


def test_verifier_agent_serializes_configurable_investigation_in_sync_request():
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(201, json=_benchmark())

    with AgentGymClient(api_key="ptc_test", workspace_id=WORKSPACE_ID) as client:
        _replace_sync(client, handler, lambda request: httpx.Response(500))
        client.benchmarks.create(
            name="Artifact verifier",
            goal="Verify the submitted analysis.",
            evaluators=[
                VerifierAgent(
                    instructions="Check the evidence and reproduce the result.",
                    model="verifier-model",
                    evidence=EvidencePolicy(
                        selected=(
                            EvidenceKind.CASE_INPUT,
                            EvidenceKind.SUBMITTED_OUTPUT,
                            EvidenceKind.EXECUTION_TRACE,
                        ),
                    ),
                    metrics=(
                        VerifierMetric(
                            "correctness", "Correctness", "Reproduce and check the result."
                        ),
                    ),
                    budget=InvestigationBudget(max_steps=21),
                )
            ],
        )

    assert bodies[0]["evaluators"] == [
        {
            "kind": "verifier_agent",
            "required": False,
            "instructions": "Check the evidence and reproduce the result.",
            "model": "verifier-model",
            "evidence": {
                "caseInputs": True,
                "submittedOutput": True,
                "expectedBehavior": False,
                "expectedOutput": False,
                "executionTrace": True,
            },
            "metrics": [
                {
                    "key": "correctness",
                    "name": "Correctness",
                    "instructions": "Reproduce and check the result.",
                }
            ],
            "budget": {"maxSteps": 21},
        }
    ]


def test_default_verifier_agent_request_uses_finalized_defaults():
    request = VerifierAgent(
        "Check correctness.",
        metrics=(VerifierMetric("correctness", "Correctness", "Check the result."),),
    ).as_request()

    assert request == {
        "kind": "verifier_agent",
        "required": False,
        "instructions": "Check correctness.",
        "evidence": {
            "caseInputs": True,
            "submittedOutput": True,
            "expectedBehavior": True,
            "expectedOutput": True,
            "executionTrace": False,
        },
        "metrics": [
            {"key": "correctness", "name": "Correctness", "instructions": "Check the result."}
        ],
    }
    assert "budget" not in request
    assert "tools" not in request


def test_verifier_agent_rejects_evaluator_weight_and_unknown_metric_binding():
    with pytest.raises(ValueError, match="metric_bindings"):
        VerifierAgent(
            instructions="Check correctness.",
            weight=2,
            metrics=(VerifierMetric("correctness", "Correctness", "Check the result."),),
        )


def test_verifier_metric_validates_scoring_and_rejects_duplicate_configuration():
    with pytest.raises(ValueError, match="at most 10"):
        VerifierMetric("correctness", "Correctness", "Check it.", weight=11)
    with pytest.raises(ValueError, match="between 0 and 1"):
        VerifierMetric("correctness", "Correctness", "Check it.", threshold=1.1)
    with pytest.raises(ValueError, match="inline or through metric_bindings"):
        VerifierAgent(
            instructions="Check correctness.",
            metrics=(VerifierMetric("correctness", "Correctness", "Check it.", weight=2),),
            metric_bindings={"correctness": MetricBinding(weight=3)},
        )
    with pytest.raises(ValueError, match="unknown verifier metric"):
        VerifierAgent(
            instructions="Check correctness.",
            metrics=(VerifierMetric("correctness", "Correctness", "Check the result."),),
            metric_bindings={"other": MetricBinding()},
        )


def test_verifier_agent_configuration_validates_evidence_and_budget():
    with pytest.raises(ValueError, match="EvidenceKind values"):
        EvidencePolicy(selected=("submitted_output",))  # type: ignore[arg-type]

    for max_steps in (0, True):
        with pytest.raises(ValueError, match="at least 1"):
            InvestigationBudget(max_steps=max_steps)


def test_expected_behavior_judge_is_locked_and_trajectory_judge_is_unavailable():
    assert ExpectedBehaviorJudge(
        model="gpt-4.1-nano",
        metric_bindings={"behavior_compliance": MetricBinding(weight=2, threshold=0.8)},
        name="Expected behavior",
        description="Uses the platform preset.",
        required=True,
        threshold=0.8,
    ).as_request() == {
        "kind": "expected_behavior_judge",
        "required": True,
        "name": "Expected behavior",
        "description": "Uses the platform preset.",
        "threshold": 0.8,
        "model": "gpt-4.1-nano",
        "metricBindings": {"behavior_compliance": {"weight": 2.0, "threshold": 0.8}},
    }

    with pytest.raises(TypeError):
        ExpectedBehaviorJudge(instructions="Override")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        ExpectedBehaviorJudge(evidence=EvidencePolicy())  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        ExpectedBehaviorJudge(
            metrics=(VerifierMetric("overall", "Overall", "Override."),)  # type: ignore[call-arg]
        )
    with pytest.raises(TypeError):
        ExpectedBehaviorJudge(budget=InvestigationBudget())  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="behavior_compliance"):
        ExpectedBehaviorJudge(metric_bindings={"other": MetricBinding()})

    with pytest.raises(ValueError, match="behavior_compliance"):
        ExpectedBehaviorJudge(weight=2)

    assert not hasattr(promptic_sdk, "TrajectoryJudge")


def test_array_judge_accepts_per_field_guidance():
    request = FieldLevelJudge(
        fields={
            "citations": FieldScoring(
                method="array_judge", judge_instructions="Match equivalent sources."
            )
        }
    ).as_request()

    assert request["fields"]["citations"] == {
        "include": True,
        "method": "array_judge",
        "judge_instructions": "Match equivalent sources.",
        "weight": 1.0,
    }
    assert "rubric" not in request


def test_field_scoring_accepts_all_methods():
    for method in (
        "exact",
        "embedding",
        "contains",
        "judge",
        "array_exact",
        "array_similarity",
        "array_judge",
    ):
        assert FieldScoring(method=method).as_request()["method"] == method


@pytest.mark.parametrize(
    "configuration",
    [
        {"method": "array_similarity", "judge_instructions": "Check it."},
        {"method": "semantic"},
    ],
)
def test_field_scoring_rejects_invalid_configuration(configuration):
    with pytest.raises(ValueError):
        FieldScoring(**configuration)


@pytest.mark.parametrize("legacy_key", ["strategy", "array_strategy"])
def test_field_scoring_rejects_legacy_keys(legacy_key):
    with pytest.raises(TypeError):
        cast(Any, FieldScoring)(**{legacy_key: "exact"})


def test_bulk_cases_with_distinct_logical_paths_share_one_request(tmp_path):
    source = tmp_path / "shared.pdf"
    source.write_bytes(b"shared-pdf")
    requests: list[httpx.Request] = []
    schema = {
        "type": "object",
        "properties": {"files": {"type": "array", "x-promptic-type": "file"}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=_benchmark(inputSchema=schema))
        if request.url.path.endswith("/storage-objects/presign"):
            return httpx.Response(200, json=_presign("shared.pdf"))
        return httpx.Response(201, json={"created": 2, "failed": 0, "results": []})

    with AgentGymClient(api_key="ptc_test", workspace_id=WORKSPACE_ID) as client:
        _replace_sync(client, handler, lambda request: httpx.Response(200))
        agent = client.benchmarks.get(BENCHMARK_ID)
        result = agent.cases.add_many(
            [
                BenchmarkCase(input={"case": 1, "files": [BenchmarkFile(source)]}),
                BenchmarkCase(
                    input={
                        "case": 2,
                        "files": [BenchmarkFile(source, path="alternate-name.pdf")],
                    },
                ),
            ]
        )

    assert result["created"] == 2
    assert len(requests) == 3
    assert sum(request.url.path.endswith("/storage-objects/presign") for request in requests) == 1
    assert requests[2].url.path.endswith(f"/benchmarks/{BENCHMARK_ID}/cases/bulk")
    items = json.loads(requests[2].content)["items"]
    assert items[0]["uploads"][0]["path"] == "shared.pdf"
    assert items[1]["uploads"][0]["path"] == "alternate-name.pdf"


def test_add_many_batches_reports_confirmed_progress():
    requests: list[httpx.Request] = []
    progress: list[BulkCaseProgress] = []
    persisted = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal persisted
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=_benchmark())
        items = json.loads(request.content)["items"]
        persisted += len(items)
        return httpx.Response(
            201,
            json={
                "created": len(items),
                "failed": 0,
                "results": [
                    {"label": item["title"], "status": "created", "datasetCaseId": persisted}
                    for item in items
                ],
                "draftCaseCount": persisted,
                "published": False,
            },
        )

    cases = [
        BenchmarkCase(title=f"Case {index}", input={"task": str(index)}, output={"answer": "ok"})
        for index in range(5)
    ]
    with AgentGymClient(api_key="ptc_test", workspace_id=WORKSPACE_ID) as client:
        _replace_sync(client, handler, lambda request: httpx.Response(500))
        benchmark = client.benchmarks.get(BENCHMARK_ID)
        result = benchmark.cases.add_many(cases, batch_size=2, on_progress=progress.append)

    assert [len(json.loads(request.content)["items"]) for request in requests[1:]] == [2, 2, 1]
    assert result == {
        "total": 5,
        "processed": 5,
        "created": 5,
        "failed": 0,
        "batchesCompleted": 3,
        "draftCaseCount": 5,
        "published": False,
        "results": result["results"],
    }
    assert [item["processed"] for item in progress] == [2, 4, 5]
    assert all(item["totalBatches"] == 3 for item in progress)


def test_add_many_failure_exposes_only_confirmed_batches():
    batch_requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal batch_requests
        if request.method == "GET":
            return httpx.Response(200, json=_benchmark())
        batch_requests += 1
        if batch_requests == 2:
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(
            201,
            json={"created": 2, "failed": 0, "results": [], "draftCaseCount": 2},
        )

    cases = [
        BenchmarkCase(input={"task": str(index)}, output={"answer": "ok"}) for index in range(4)
    ]
    with AgentGymClient(api_key="ptc_test", workspace_id=WORKSPACE_ID) as client:
        _replace_sync(client, handler, lambda request: httpx.Response(500))
        benchmark = client.benchmarks.get(BENCHMARK_ID)
        with pytest.raises(BulkCaseUploadError) as captured:
            benchmark.cases.add_many(cases, batch_size=2)

    assert captured.value.batch_index == 1
    assert captured.value.result["processed"] == 2
    assert captured.value.result["created"] == 2
    assert captured.value.result["draftCaseCount"] == 2


@pytest.mark.asyncio
async def test_async_authoring_uses_same_contract():
    bodies: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(201, json=_benchmark())

    async with AsyncAgentGymClient(api_key="ptc_test", workspace_id=WORKSPACE_ID) as client:
        await _replace_async(client, handler, lambda request: httpx.Response(200))
        await client.benchmarks.create(name="Agent", goal="Do the work.")

    assert bodies[0]["taskDescription"] == "Do the work."
    assert bodies[0]["evaluators"] == []


@pytest.mark.asyncio
async def test_async_configure_explicit_field_evaluator_matches_sync_contract():
    requests: list[httpx.Request] = []
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "rationale": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "string"}},
        },
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_benchmark(targetSchema=schema))

    async with AsyncAgentGymClient(api_key="ptc_test", workspace_id=WORKSPACE_ID) as client:
        await _replace_async(client, handler, lambda request: httpx.Response(500))
        benchmark = await client.benchmarks.get(BENCHMARK_ID)
        await benchmark.configure(
            output_schema=schema,
            evaluators=[
                FieldLevelJudge(
                    fields={
                        "rationale": FieldScoring("judge", judge_instructions="Check evidence."),
                        "citations": FieldScoring(method="array_similarity"),
                    }
                )
            ],
        )

    patch = json.loads(requests[1].content)
    assert patch["targetSchema"] == schema
    assert patch["evaluators"][0]["fields"] == {
        "rationale": {
            "include": True,
            "method": "judge",
            "weight": 1.0,
            "judge_instructions": "Check evidence.",
        },
        "citations": {"include": True, "method": "array_similarity", "weight": 1.0},
    }


def test_authoring_admin_gate_maps_to_agent_gym_error():
    with AgentGymClient(api_key="ptc_test", workspace_id=WORKSPACE_ID) as client:
        _replace_sync(
            client,
            lambda request: httpx.Response(403, json={"error": "Forbidden"}),
            lambda request: httpx.Response(500),
        )
        with pytest.raises(AgentGymAPIError):
            client.benchmarks.list()


def test_revision_draft_lifecycle_uses_grouped_conflict_contract():
    requests: list[httpx.Request] = []
    conflict = {
        "direction": "output",
        "fieldPath": "name",
        "issue": "Required field is missing",
        "affectedCaseIds": [1, 2],
        "affectedCaseCount": 2,
    }
    draft = {"id": STORAGE_ID, "status": "conflicted", "conflicts": [conflict]}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET" and request.url.path.endswith("/revisions/draft"):
            return httpx.Response(200, json={"draft": draft})
        if request.method == "PATCH":
            return httpx.Response(200, json={"draft": {**draft, "status": "ready"}})
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json=_benchmark())

    with AgentGymClient(api_key="ptc_test", workspace_id=WORKSPACE_ID) as client:
        _replace_sync(client, handler, lambda request: httpx.Response(500))
        benchmark = client.benchmarks.get(BENCHMARK_ID)
        loaded = benchmark.revision_draft()
        assert loaded is not None
        assert loaded["conflicts"] == [conflict]
        resolved = benchmark.resolve_revision_draft(
            "output.name", kind="default_value", value="unknown"
        )
        benchmark.abandon_revision_draft()

    assert resolved["status"] == "ready"
    assert json.loads(requests[2].content) == {
        "field": "output.name",
        "resolution": {"kind": "default_value", "value": "unknown"},
    }
    assert requests[3].method == "DELETE"


def test_revision_draft_resolution_validates_kind_and_default_value():
    with AgentGymClient(api_key="ptc_test", workspace_id=WORKSPACE_ID) as client:
        _replace_sync(
            client,
            lambda request: httpx.Response(200, json=_benchmark()),
            lambda request: httpx.Response(500),
        )
        benchmark = client.benchmarks.get(BENCHMARK_ID)
        with pytest.raises(ValueError, match="value is required"):
            benchmark.resolve_revision_draft("output.name", kind="default_value")
        with pytest.raises(ValueError, match="kind must be"):
            benchmark.resolve_revision_draft("output.name", kind="replace")


@pytest.mark.asyncio
async def test_async_revision_draft_resolution_matches_sync_contract():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "PATCH":
            return httpx.Response(200, json={"draft": {"status": "ready", "conflicts": []}})
        return httpx.Response(200, json=_benchmark())

    async with AsyncAgentGymClient(api_key="ptc_test", workspace_id=WORKSPACE_ID) as client:
        await _replace_async(client, handler, lambda request: httpx.Response(500))
        benchmark = await client.benchmarks.get(BENCHMARK_ID)
        await benchmark.resolve_revision_draft("input.locale", kind="make_optional")

    assert json.loads(requests[1].content) == {
        "field": "input.locale",
        "resolution": {"kind": "make_optional"},
    }
