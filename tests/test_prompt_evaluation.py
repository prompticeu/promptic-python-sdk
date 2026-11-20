"""Evaluation-only request contracts for both SDK clients."""

import json

import httpx
import pytest

from promptic_sdk import AsyncPrompticClient, PrompticClient
from promptic_sdk.models import PromptMessage


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_evaluation_only_preserves_messages_and_settings(monkeypatch, asynchronous):
    monkeypatch.setenv("PROMPTIC_API_KEY", "test-key")
    messages: list[PromptMessage] = [
        {"role": "system", "content": "Keep the candidate unchanged."},
        {"role": "user", "content": "{message}"},
    ]
    hyperparameters = {"trainSplitRatio": 0.8, "numFewShots": 2}
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(201, json={"id": "candidate"})

    if asynchronous:
        async with AsyncPrompticClient() as client:
            client._client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler), base_url="https://promptic.eu/api/v1"
            )
            await client.create_experiment(
                "component",
                "gpt-5.6-luna",
                evaluation_only=True,
                prompt_format="multi_message",
                initial_prompt_messages=messages,
                hyperparameters=hyperparameters,
            )
    else:
        with PrompticClient() as client:
            client._client = httpx.Client(
                transport=httpx.MockTransport(handler), base_url="https://promptic.eu/api/v1"
            )
            client.create_experiment(
                "component",
                "gpt-5.6-luna",
                evaluation_only=True,
                prompt_format="multi_message",
                initial_prompt_messages=messages,
                hyperparameters=hyperparameters,
            )
    body = json.loads(requests[0].content)
    assert body["hyperparameters"] == {"epochs": 0, "numFewShots": 2, "trainSplitRatio": 0.8}
    assert body["promptFormat"] == "multi_message"
    assert body["initialPromptMessages"] == messages
    assert hyperparameters == {"trainSplitRatio": 0.8, "numFewShots": 2}
    assert "evaluationOnly" not in body  # server contract remains epochs=0


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_evaluation_only_rejects_conflicting_optimization_budget(monkeypatch, asynchronous):
    monkeypatch.setenv("PROMPTIC_API_KEY", "test-key")
    with pytest.raises(ValueError, match="epochs"):
        if asynchronous:
            async with AsyncPrompticClient() as client:
                await client.create_experiment(
                    "component", "gpt-5.6-luna", evaluation_only=True, hyperparameters={"epochs": 3}
                )
        else:
            with PrompticClient() as client:
                client.create_experiment(
                    "component", "gpt-5.6-luna", evaluation_only=True, hyperparameters={"epochs": 3}
                )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_case_results_keep_iteration_and_pagination_scope(monkeypatch, asynchronous):
    monkeypatch.setenv("PROMPTIC_API_KEY", "test-key")
    page = {
        "data": [{"id": "7", "expected": "yes", "prediction": "no", "score": 0}],
        "nextCursor": "next",
        "total": 38,
    }

    def handler(request):
        assert request.url.path == "/api/v1/experiments/candidate/cases"
        assert dict(request.url.params) == {
            "iterationId": "42",
            "split": "all",
            "limit": "10",
            "cursor": "previous",
            "sort": "score",
            "descending": "false",
        }
        return httpx.Response(200, json=page)

    if asynchronous:
        async with AsyncPrompticClient() as client:
            client._client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler), base_url="https://promptic.eu/api/v1"
            )
            result = await client.list_experiment_cases(
                "candidate", iteration_id=42, limit=10, cursor="previous", sort="score"
            )
    else:
        with PrompticClient() as client:
            client._client = httpx.Client(
                transport=httpx.MockTransport(handler), base_url="https://promptic.eu/api/v1"
            )
            result = client.list_experiment_cases(
                "candidate", iteration_id=42, limit=10, cursor="previous", sort="score"
            )
    assert result == page
