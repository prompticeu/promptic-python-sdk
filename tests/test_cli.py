"""Tests for the CLI."""

import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from promptic_sdk.cli.config import CliConfig
from promptic_sdk.cli.main import app

runner = CliRunner()


def _mock_config(**overrides):
    """Return a patch that makes get_client() use a mock config."""
    config = CliConfig(
        endpoint=overrides.get("endpoint", "https://test.com"),
        api_key=overrides.get("api_key", "pk_test"),
    )
    return patch("promptic_sdk.cli.load_config", return_value=config)


def _mock_config_none():
    """Return a patch that makes get_client() return None config."""
    return patch("promptic_sdk.cli.load_config", return_value=None)


def _mock_client(module_path, method_name, return_value):
    """Mock PrompticClient in the cli __init__ module."""
    mock_client = MagicMock()
    getattr(mock_client, method_name).return_value = return_value
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    return patch(
        "promptic_sdk.cli.PrompticClient",
        return_value=mock_client,
    )


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
    def test_list_traces_json(self):
        data = {"traces": [{"traceId": "abc", "name": "test", "status": "ok"}], "total": 1}
        with _mock_config(), _mock_client("traces", "list_traces", data):
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
        with _mock_config(), _mock_client("traces", "list_traces", data):
            result = runner.invoke(app, ["traces", "list"])
            assert result.exit_code == 0
            assert "test-trace" in result.stdout

    def test_get_trace_json(self):
        data = {"traceId": "abc123", "name": "test", "status": "ok", "spans": []}
        with _mock_config(), _mock_client("traces", "get_trace", data):
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
        with _mock_config(), _mock_client("traces", "get_trace", data):
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
        with _mock_config(), _mock_client("traces", "get_stats", data):
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
        with _mock_config(), _mock_client("traces", "get_stats", data):
            result = runner.invoke(app, ["traces", "stats"])
            assert result.exit_code == 0
            assert "100" in result.stdout
            assert "50000" in result.stdout

    def test_no_config_exits_with_error(self):
        with _mock_config_none():
            result = runner.invoke(app, ["traces", "list"])
            assert result.exit_code == 1


class TestExperimentsCommands:
    def _new_exp_payload(self) -> dict:
        return {
            "id": "new-exp-id",
            "name": "Run 2",
            "experimentStatus": "pending",
            "targetModel": "gpt-5.4-nano",
            "modelUnavailable": False,
        }

    def test_duplicate_calls_client_with_no_flags(self):
        with (
            _mock_config(),
            _mock_client("experiments", "duplicate_experiment", self._new_exp_payload()) as patched,
        ):
            result = runner.invoke(app, ["experiments", "duplicate", "src-exp-id", "--json"])
            assert result.exit_code == 0
            output = json.loads(result.stdout)
            assert output["id"] == "new-exp-id"

            mock_client = patched.return_value
            mock_client.duplicate_experiment.assert_called_once_with(
                "src-exp-id", initial_prompt_override=None
            )

    def test_duplicate_with_initial_prompt_override(self):
        with (
            _mock_config(),
            _mock_client("experiments", "duplicate_experiment", self._new_exp_payload()) as patched,
        ):
            result = runner.invoke(
                app,
                [
                    "experiments",
                    "duplicate",
                    "src-exp-id",
                    "-p",
                    "custom prompt",
                    "--json",
                ],
            )
            assert result.exit_code == 0
            mock_client = patched.return_value
            mock_client.duplicate_experiment.assert_called_once_with(
                "src-exp-id", initial_prompt_override="custom prompt"
            )

    def test_duplicate_with_start(self):
        with (
            _mock_config(),
            _mock_client("experiments", "duplicate_experiment", self._new_exp_payload()) as patched,
        ):
            mock_client = patched.return_value
            # Configure start_experiment too so the chained call lands on the same mock.
            mock_client.start_experiment.return_value = {"status": "scheduled"}

            result = runner.invoke(app, ["experiments", "duplicate", "src-exp-id", "--start"])
            assert result.exit_code == 0
            mock_client.duplicate_experiment.assert_called_once()
            mock_client.start_experiment.assert_called_once_with("new-exp-id")

    def test_continue_passes_continue_from_optimized(self):
        with (
            _mock_config(),
            _mock_client("experiments", "duplicate_experiment", self._new_exp_payload()) as patched,
        ):
            result = runner.invoke(app, ["experiments", "continue", "src-exp-id", "--json"])
            assert result.exit_code == 0
            mock_client = patched.return_value
            mock_client.duplicate_experiment.assert_called_once_with(
                "src-exp-id", continue_from_optimized=True
            )

    def test_continue_warns_on_unavailable_model(self):
        payload = self._new_exp_payload()
        payload["modelUnavailable"] = True
        with _mock_config(), _mock_client("experiments", "duplicate_experiment", payload):
            result = runner.invoke(app, ["experiments", "continue", "src-exp-id"])
            assert result.exit_code == 0
            assert "no longer available" in result.stdout
