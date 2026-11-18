"""Guidance transport, optimistic versioning, and sync/async parity."""

import json

import httpx
import pytest

from promptic_sdk import AsyncPrompticClient, PrompticAPIError, PrompticClient
from promptic_sdk.models import Component, Experiment, OptimizationBrief

BRIEF: OptimizationBrief = {
    "rules": ["Focus on confusion between **X** and **Y**."],
}


def handler(request):
    body = json.loads(request.content)
    if request.url.path.endswith("/components/component"):
        assert body == {"optimizationRules": {"version": 2, "brief": BRIEF}}
        return httpx.Response(200, json={"optimizationRules": {"version": 3, "brief": BRIEF}})
    if request.method == "PATCH":
        assert body == {"optimizationGuidance": {}}
    else:
        assert body["optimizationGuidance"] == BRIEF
        assert "optimization_guidance" not in body
    return httpx.Response(200, json={"id": "experiment", **body})


def test_sync_guidance_create_tool_update_and_component():
    with PrompticClient(api_key="test") as client:
        client._client.close()
        client._client = httpx.Client(
            transport=httpx.MockTransport(handler), base_url="https://test/api/v1"
        )
        result = client.create_experiment("component", "model", optimization_guidance=BRIEF)
        assert result["optimizationGuidance"] == BRIEF
        client.create_tool_selection_experiment(
            "component",
            tools=[{"name": "search", "description": "Find"}],
            test_cases=[{"query": "Hi", "expected_tool": "search"}],
            optimization_guidance=BRIEF,
        )
        client.update_experiment("experiment", optimization_guidance={})
        rules = client.update_component_rules("component", BRIEF, expected_version=2).get(
            "optimizationRules"
        )
        assert rules is not None
        assert rules["version"] == 3


@pytest.mark.asyncio
async def test_async_guidance_create_tool_update_and_component():
    async with AsyncPrompticClient(api_key="test") as client:
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://test/api/v1"
        )
        result = await client.create_experiment("component", "model", optimization_guidance=BRIEF)
        assert result["optimizationGuidance"] == BRIEF
        await client.create_tool_selection_experiment(
            "component",
            tools=[{"name": "search", "description": "Find"}],
            test_cases=[{"query": "Hi", "expected_tool": "search"}],
            optimization_guidance=BRIEF,
        )
        await client.update_experiment("experiment", optimization_guidance={})
        rules = (await client.update_component_rules("component", BRIEF, expected_version=2)).get(
            "optimizationRules"
        )
        assert rules is not None
        assert rules["version"] == 3


def test_stale_component_version_is_not_silently_retried():
    calls: list[httpx.Request] = []

    def conflict(request: httpx.Request):
        calls.append(request)
        return httpx.Response(409, json={"error": "Component rules changed"})

    with PrompticClient(api_key="test") as client:
        client._client.close()
        client._client = httpx.Client(
            transport=httpx.MockTransport(conflict), base_url="https://test/api/v1"
        )
        with pytest.raises(PrompticAPIError) as exc:
            client.update_component_rules("component", BRIEF, expected_version=2)
        assert exc.value.status_code == 409
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_async_stale_component_version_is_not_silently_retried():
    calls: list[httpx.Request] = []

    def conflict(request: httpx.Request):
        calls.append(request)
        return httpx.Response(409, json={"error": "Component rules changed"})

    async with AsyncPrompticClient(api_key="test") as client:
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(conflict), base_url="https://test/api/v1"
        )
        with pytest.raises(PrompticAPIError) as exc:
            await client.update_component_rules("component", BRIEF, expected_version=2)
        assert exc.value.status_code == 409
        assert len(calls) == 1


def test_guidance_fields_remain_optional_for_older_servers():
    assert "optimizationRules" in Component.__optional_keys__
    assert "optimizationGuidance" in Experiment.__optional_keys__
