"""Tests for the platform client."""

import httpx
import pytest

from promptic_sdk.client import AsyncPromenticClient, PromenticClient


class TestPromenticClient:
    def test_requires_api_key(self):
        with pytest.raises(ValueError, match="API key is required"):
            PromenticClient(api_key=None)

    def test_reads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test_key")
        client = PromenticClient()
        assert client.api_key == "pk_test_key"
        client.close()

    def test_custom_endpoint(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        client = PromenticClient(endpoint="https://custom.example.com")
        assert client.endpoint == "https://custom.example.com"
        # httpx normalizes base_url with trailing slash
        assert str(client._client.base_url).rstrip("/") == "https://custom.example.com/api/v1"
        client.close()

    def test_default_endpoint(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        client = PromenticClient()
        assert client.endpoint == "https://app.promptic.eu"
        client.close()

    def test_context_manager(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        with PromenticClient() as client:
            assert client.api_key == "pk_test"

    def test_list_traces(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        response_data = {"traces": [{"traceId": "abc123"}], "total": 1}

        with PromenticClient() as client:

            def handler(request: httpx.Request) -> httpx.Response:
                assert "/traces" in str(request.url)
                assert request.headers["authorization"] == "Bearer pk_test"
                return httpx.Response(200, json=response_data)

            client._client = httpx.Client(
                transport=httpx.MockTransport(handler),
                base_url="https://app.promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            result = client.list_traces(limit=10)
            assert result == response_data

    def test_get_trace(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        response_data = {"traceId": "abc123", "spans": []}

        with PromenticClient() as client:

            def handler(request: httpx.Request) -> httpx.Response:
                assert "/traces/abc123" in str(request.url)
                return httpx.Response(200, json=response_data)

            client._client = httpx.Client(
                transport=httpx.MockTransport(handler),
                base_url="https://app.promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            result = client.get_trace("abc123")
            assert result == response_data

    def test_get_stats(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        response_data = {
            "totalTraces": 100,
            "totalTokens": 50000,
            "totalCostUsd": 1.23,
            "errorRate": 0.05,
        }

        with PromenticClient() as client:

            def handler(request: httpx.Request) -> httpx.Response:
                assert "/traces/stats" in str(request.url)
                return httpx.Response(200, json=response_data)

            client._client = httpx.Client(
                transport=httpx.MockTransport(handler),
                base_url="https://app.promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            result = client.get_stats(days_back=7)
            assert result == response_data

    def test_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        client = PromenticClient(endpoint="https://example.com/")
        assert client.endpoint == "https://example.com"
        client.close()


class TestAsyncPromenticClient:
    def test_requires_api_key(self):
        with pytest.raises(ValueError, match="API key is required"):
            AsyncPromenticClient(api_key=None)

    def test_reads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test_key")
        client = AsyncPromenticClient()
        assert client.api_key == "pk_test_key"

    def test_custom_endpoint(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        client = AsyncPromenticClient(endpoint="https://custom.example.com")
        assert client.endpoint == "https://custom.example.com"
        assert str(client._client.base_url).rstrip("/") == "https://custom.example.com/api/v1"

    def test_default_endpoint(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        client = AsyncPromenticClient()
        assert client.endpoint == "https://app.promptic.eu"

    async def test_context_manager(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        async with AsyncPromenticClient() as client:
            assert client.api_key == "pk_test"

    async def test_list_traces(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        response_data = {"traces": [{"traceId": "abc123"}], "total": 1}

        async with AsyncPromenticClient() as client:

            async def handler(request: httpx.Request) -> httpx.Response:
                assert "/traces" in str(request.url)
                assert request.headers["authorization"] == "Bearer pk_test"
                return httpx.Response(200, json=response_data)

            client._client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler),
                base_url="https://app.promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            result = await client.list_traces(limit=10)
            assert result == response_data

    async def test_get_trace(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        response_data = {"traceId": "abc123", "spans": []}

        async with AsyncPromenticClient() as client:

            async def handler(request: httpx.Request) -> httpx.Response:
                assert "/traces/abc123" in str(request.url)
                return httpx.Response(200, json=response_data)

            client._client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler),
                base_url="https://app.promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            result = await client.get_trace("abc123")
            assert result == response_data

    async def test_get_stats(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        response_data = {
            "totalTraces": 100,
            "totalTokens": 50000,
            "totalCostUsd": 1.23,
            "errorRate": 0.05,
        }

        async with AsyncPromenticClient() as client:

            async def handler(request: httpx.Request) -> httpx.Response:
                assert "/traces/stats" in str(request.url)
                return httpx.Response(200, json=response_data)

            client._client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler),
                base_url="https://app.promptic.eu/api/v1",
                headers={"Authorization": "Bearer pk_test"},
            )

            result = await client.get_stats(days_back=7)
            assert result == response_data

    def test_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("PROMPTIC_API_KEY", "pk_test")
        client = AsyncPromenticClient(endpoint="https://example.com/")
        assert client.endpoint == "https://example.com"
