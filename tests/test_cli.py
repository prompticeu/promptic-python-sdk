"""Tests for the CLI."""

import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from promptic_sdk.cli.main import app

runner = CliRunner()


class TestConfigure:
    def test_configure_saves_config(self, tmp_path):
        with (
            patch("promptic_sdk.cli.commands.configure.save_config") as mock_save,
            patch(
                "promptic_sdk.cli.commands.configure.get_config_path",
                return_value=tmp_path / "config.toml",
            ),
        ):
            result = runner.invoke(
                app, ["configure", "--api-key", "pk_test", "--endpoint", "https://test.com"]
            )
            assert result.exit_code == 0
            mock_save.assert_called_once_with("pk_test", "https://test.com")


class TestTracesCommands:
    def _mock_config(self):
        return patch(
            "promptic_sdk.cli.commands.traces.load_config",
            return_value=MagicMock(api_key="pk_test", endpoint="https://test.com"),
        )

    def _mock_client(self, method_name, return_value):
        mock_client = MagicMock()
        getattr(mock_client, method_name).return_value = return_value
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        return patch(
            "promptic_sdk.cli.commands.traces.PromenticClient",
            return_value=mock_client,
        )

    def test_list_traces_json(self):
        data = {"traces": [{"traceId": "abc", "name": "test", "status": "ok"}], "total": 1}
        with self._mock_config(), self._mock_client("list_traces", data):
            result = runner.invoke(app, ["traces", "list", "--json"])
            assert result.exit_code == 0
            output = json.loads(result.stdout)
            assert output["total"] == 1
            assert output["traces"][0]["traceId"] == "abc"

    def test_list_traces_table(self):
        data = {
            "traces": [
                {
                    "traceId": "abc123def456",
                    "name": "test-trace",
                    "status": "ok",
                    "durationMs": 150,
                    "totalTokens": 500,
                    "totalCostUsd": 0.0012,
                    "startTime": "2025-01-01T00:00:00Z",
                }
            ],
            "total": 1,
        }
        with self._mock_config(), self._mock_client("list_traces", data):
            result = runner.invoke(app, ["traces", "list"])
            assert result.exit_code == 0
            assert "test-trace" in result.stdout

    def test_get_trace_json(self):
        data = {"traceId": "abc123", "name": "test", "status": "ok", "spans": []}
        with self._mock_config(), self._mock_client("get_trace", data):
            result = runner.invoke(app, ["traces", "get", "abc123", "--json"])
            assert result.exit_code == 0
            output = json.loads(result.stdout)
            assert output["traceId"] == "abc123"

    def test_get_trace_human(self):
        data = {
            "traceId": "abc123",
            "name": "test-trace",
            "status": "ok",
            "durationMs": 200,
            "totalTokens": 1000,
            "totalCostUsd": 0.005,
            "spans": [
                {
                    "name": "chat",
                    "kind": "llm",
                    "status": "ok",
                    "durationMs": 180,
                    "model": "gpt-4o",
                    "totalTokens": 1000,
                }
            ],
        }
        with self._mock_config(), self._mock_client("get_trace", data):
            result = runner.invoke(app, ["traces", "get", "abc123"])
            assert result.exit_code == 0
            assert "abc123" in result.stdout
            assert "test-trace" in result.stdout

    def test_stats_json(self):
        data = {
            "totalTraces": 100,
            "totalTokens": 50000,
            "totalCostUsd": 1.23,
            "errorRate": 0.05,
        }
        with self._mock_config(), self._mock_client("get_stats", data):
            result = runner.invoke(app, ["traces", "stats", "--json"])
            assert result.exit_code == 0
            output = json.loads(result.stdout)
            assert output["totalTraces"] == 100

    def test_stats_human(self):
        data = {
            "totalTraces": 100,
            "totalTokens": 50000,
            "totalCostUsd": 1.23,
            "errorRate": 0.05,
        }
        with self._mock_config(), self._mock_client("get_stats", data):
            result = runner.invoke(app, ["traces", "stats"])
            assert result.exit_code == 0
            assert "100" in result.stdout
            assert "50000" in result.stdout

    def test_no_config_exits_with_error(self):
        with patch("promptic_sdk.cli.commands.traces.load_config", return_value=None):
            result = runner.invoke(app, ["traces", "list"])
            assert result.exit_code == 1
