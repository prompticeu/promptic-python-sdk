"""Cancellation contracts shared by the sync/async clients and CLI."""

import json
from typing import get_args, get_type_hints
from unittest.mock import MagicMock, patch

import httpx
import pytest
from typer.testing import CliRunner

from promptic_sdk import Experiment, ExperimentCancellation, ExperimentStatus
from promptic_sdk.cli.main import app
from promptic_sdk.client import AsyncPrompticClient, PrompticAPIError, PrompticClient

RUN_ID = "2653356b-0ea9-4061-b1cf-926f42b686bc"
BASE_URL = "https://test.example/api/v1/"


def cancellation_transport(outcome, error_status=None):
    requests = []
    polls = iter(["canceling", "canceled"])

    def handler(request):
        requests.append(request)
        if request.method == "GET":
            assert request.url.path == f"/api/v1/experiments/{RUN_ID}"
            return httpx.Response(200, json={"id": RUN_ID, "experimentStatus": next(polls)})
        assert request.method == "POST"
        assert request.url.path == f"/api/v1/experiments/{RUN_ID}/cancel"
        assert request.content in (b"", b"null")
        if error_status:
            return httpx.Response(error_status, json={"error": "Cannot cancel this experiment"})
        return httpx.Response(200, json={"status": outcome})

    return httpx.MockTransport(handler), requests


@pytest.mark.parametrize("outcome", ["canceled", "canceling"])
def test_sync_cancellation_and_repeated_request(outcome):
    transport, requests = cancellation_transport(outcome)
    with PrompticClient(api_key="test-key", endpoint="https://test.example") as client:
        client._client.close()
        client._client = httpx.Client(transport=transport, base_url=BASE_URL)
        assert client.cancel_experiment(RUN_ID) == {"status": outcome}
        assert len(requests) == 1  # The method does not wait, retry, or start another run.
        assert client.cancel_experiment(RUN_ID) == {"status": outcome}
        if outcome == "canceling":
            assert client.get_experiment(RUN_ID)["experimentStatus"] == "canceling"
            assert client.get_experiment(RUN_ID)["experimentStatus"] == "canceled"


@pytest.mark.parametrize("outcome", ["canceled", "canceling"])
async def test_async_cancellation_and_repeated_request(outcome):
    transport, requests = cancellation_transport(outcome)
    async with AsyncPrompticClient(api_key="test-key", endpoint="https://test.example") as client:
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=transport, base_url=BASE_URL)
        assert await client.cancel_experiment(RUN_ID) == {"status": outcome}
        assert len(requests) == 1
        assert await client.cancel_experiment(RUN_ID) == {"status": outcome}
        if outcome == "canceling":
            assert (await client.get_experiment(RUN_ID))["experimentStatus"] == "canceling"
            assert (await client.get_experiment(RUN_ID))["experimentStatus"] == "canceled"


@pytest.mark.parametrize("status", [401, 404, 409])
def test_sync_cancellation_errors_are_not_retried(status):
    transport, requests = cancellation_transport(None, status)
    with PrompticClient(api_key="test-key", endpoint="https://test.example") as client:
        client._client.close()
        client._client = httpx.Client(transport=transport, base_url=BASE_URL)
        with pytest.raises(PrompticAPIError) as error:
            client.cancel_experiment(RUN_ID)
        assert error.value.status_code == status
        assert len(requests) == 1


@pytest.mark.parametrize("status", [401, 404, 409])
async def test_async_cancellation_errors_are_not_retried(status):
    transport, requests = cancellation_transport(None, status)
    async with AsyncPrompticClient(api_key="test-key", endpoint="https://test.example") as client:
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=transport, base_url=BASE_URL)
        with pytest.raises(PrompticAPIError) as error:
            await client.cancel_experiment(RUN_ID)
        assert error.value.status_code == status
        assert len(requests) == 1


@pytest.mark.parametrize("outcome", ["canceled", "canceling"])
@pytest.mark.parametrize("as_json", [False, True])
def test_cli_cancellation_output(outcome, as_json):
    client = MagicMock()
    client.__enter__.return_value = client
    client.cancel_experiment.return_value = {"status": outcome}
    with patch("promptic_sdk.cli.commands.experiments.get_client", return_value=client):
        args = ["experiments", "cancel", RUN_ID, "--yes"]
        if as_json:
            args.append("--json")
        result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    client.cancel_experiment.assert_called_once_with(RUN_ID)
    client.get_experiment.assert_not_called()
    if as_json:
        assert json.loads(result.stdout) == {"status": outcome}
    elif outcome == "canceled":
        assert "Experiment canceled." in result.stdout
    else:
        assert "Cancellation requested." in result.stdout
        assert "Experiment canceled." not in result.stdout
        assert "canceling" in result.stdout


@pytest.mark.parametrize("answer,accepted", [("n\n", False), ("y\n", True)])
def test_cli_requires_confirmation(answer, accepted):
    client = MagicMock()
    client.__enter__.return_value = client
    client.cancel_experiment.return_value = {"status": "canceling"}
    with patch("promptic_sdk.cli.commands.experiments.get_client", return_value=client) as factory:
        result = CliRunner().invoke(app, ["experiments", "cancel", RUN_ID], input=answer)
    if accepted:
        assert result.exit_code == 0
        client.cancel_experiment.assert_called_once_with(RUN_ID)
    else:
        assert result.exit_code != 0
        factory.assert_not_called()


@pytest.mark.parametrize("status", [401, 404, 409])
def test_cli_error_does_not_report_success(status):
    client = MagicMock()
    client.__enter__.return_value = client
    client.cancel_experiment.side_effect = PrompticAPIError(status, "Cannot cancel experiment")
    with patch("promptic_sdk.cli.commands.experiments.get_client", return_value=client):
        result = CliRunner().invoke(app, ["experiments", "cancel", RUN_ID, "--yes", "--json"])
    assert result.exit_code != 0
    assert "canceled" not in result.stdout
    assert isinstance(result.exception, PrompticAPIError)
    assert result.exception.status_code == status


def test_cancellation_types_are_exported():
    assert {"canceling", "canceled"} <= set(get_args(ExperimentStatus))
    assert set(get_args(get_type_hints(ExperimentCancellation)["status"])) == {
        "canceling",
        "canceled",
    }
    assert {"cancelRequestedAt", "cancelRequestedBy", "canceledAt"} <= Experiment.__optional_keys__
