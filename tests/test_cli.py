"""Tests for the CLI."""

import json
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from promptic_sdk.cli.commands.experiments import _load_json_array
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


class TestDatasetCommands:
    def test_create_prints_canonical_id_example(self):
        payload = {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "name": "Regression",
            "caseCount": 0,
        }
        with _mock_config(), _mock_client("datasets", "create_dataset", payload):
            result = runner.invoke(
                app,
                ["datasets", "create", "--component", "component-id", "--name", "Regression"],
            )

        assert result.exit_code == 0
        assert "Cases: 0" in result.stdout
        assert f"dataset_id='{payload['id']}'" in result.stdout

    def test_get_prints_canonical_cases(self):
        payload = {
            "id": "dataset-id",
            "name": "Regression",
            "description": None,
            "caseCount": 1,
            "cases": [
                {
                    "inputPayload": {
                        "input": "question",
                        "trace": "promptictrace://550e8400-e29b-41d4-a716-446655440001",
                    },
                    "expectedPayload": {"value": "answer"},
                }
            ],
        }
        with _mock_config(), _mock_client("datasets", "get_dataset", payload):
            result = runner.invoke(
                app,
                ["datasets", "get", "dataset-id", "--component", "component-id"],
            )

        assert result.exit_code == 0
        assert "Cases: 1" in result.stdout
        assert "question" in result.stdout


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

    def test_create_tool_selection_from_json_files(self, tmp_path):
        tools_file = tmp_path / "tools.json"
        tools_file.write_text(json.dumps([{"name": "search", "description": "Search documents"}]))
        cases_file = tmp_path / "cases.json"
        cases_file.write_text(
            json.dumps([{"query": "Find the invoice", "expected_tool": "search"}])
        )
        payload = {
            "id": "tool-exp-id",
            "name": "Tool routing",
            "experimentStatus": "pending",
            "targetModel": "gpt-4.1-nano",
        }

        with (
            _mock_config(),
            _mock_client("experiments", "create_tool_selection_experiment", payload) as patched,
        ):
            client = patched.return_value
            client.start_experiment.return_value = {"status": "scheduled"}
            result = runner.invoke(
                app,
                [
                    "experiments",
                    "create-tool-selection",
                    "--component-id",
                    "component-id",
                    "--tools",
                    str(tools_file),
                    "--test-cases",
                    str(cases_file),
                    "--target-model",
                    "gpt-4.1-nano",
                    "--system-prompt",
                    "Choose carefully.",
                    "--optimize-system-prompt",
                    "--epochs",
                    "4",
                    "--train-split-ratio",
                    "0.8",
                    "--start",
                    "--json",
                ],
            )

        assert result.exit_code == 0
        assert json.loads(result.stdout)["id"] == "tool-exp-id"
        client.create_tool_selection_experiment.assert_called_once_with(
            "component-id",
            tools=[{"name": "search", "description": "Search documents"}],
            test_cases=[{"query": "Find the invoice", "expected_tool": "search"}],
            target_model="gpt-4.1-nano",
            tool_source="manual",
            system_prompt="Choose carefully.",
            optimize_system_prompt=True,
            epochs=4,
            train_split_ratio=0.8,
            name=None,
            description=None,
        )
        client.start_experiment.assert_called_once_with("tool-exp-id")

    def test_create_tool_selection_rejects_non_array_tools(self, tmp_path):
        tools_file = tmp_path / "tools.json"
        tools_file.write_text(json.dumps({"name": "search"}))
        cases_file = tmp_path / "cases.json"
        cases_file.write_text(json.dumps([]))

        with _mock_config():
            result = runner.invoke(
                app,
                [
                    "experiments",
                    "create-tool-selection",
                    "--component-id",
                    "component-id",
                    "--tools",
                    str(tools_file),
                    "--test-cases",
                    str(cases_file),
                ],
                terminal_width=200,
            )

        assert result.exit_code == 2

        with pytest.raises(
            typer.BadParameter, match="--tools must contain a JSON array of objects"
        ):
            _load_json_array(tools_file, option_name="--tools")

    def test_create_tool_selection_rejects_missing_tool_description(self, tmp_path):
        tools_file = tmp_path / "tools.json"
        tools_file.write_text(json.dumps([{"name": "search"}]))
        cases_file = tmp_path / "cases.json"
        cases_file.write_text(
            json.dumps([{"query": "Find the invoice", "expected_tool": "search"}])
        )

        with _mock_config():
            result = runner.invoke(
                app,
                [
                    "experiments",
                    "create-tool-selection",
                    "--component-id",
                    "component-id",
                    "--tools",
                    str(tools_file),
                    "--test-cases",
                    str(cases_file),
                ],
            )

        assert result.exit_code == 2


class TestIterationsCommands:
    def test_get_displays_tool_selection_outputs(self):
        payload = {
            "id": 7,
            "experimentId": "tool-exp-id",
            "iterationNumber": 2,
            "prompt": "",
            "promptTokens": 12,
            "overallNormalizedScore": 0.9,
            "evalNormalizedScore": 0.8,
            "schemaSnapshot": None,
            "toolDescriptions": {
                "search[archive]": "Search [indexed] invoices and preserve broken[/] markup."
            },
            "selectionSystemPrompt": "Use [tools] only when needed; never parse broken[/] markup.",
            "createdAt": "2026-09-03T00:00:00Z",
            "updatedAt": "2026-09-03T00:00:00Z",
            "scores": [],
        }
        with _mock_config(), _mock_client("iterations", "get_iteration", payload):
            result = runner.invoke(app, ["iterations", "get", "tool-exp-id", "7"])

        assert result.exit_code == 0
        assert "Tool Descriptions" in result.stdout
        assert "search[archive]" in result.stdout
        assert "Search [indexed] invoices and preserve broken[/] markup." in result.stdout
        assert "Selection System Prompt" in result.stdout
        assert "Use [tools] only when needed; never parse broken[/] markup." in result.stdout
