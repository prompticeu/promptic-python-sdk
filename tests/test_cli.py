"""Tests for the CLI."""

import json
from unittest.mock import MagicMock, patch

import pytest
import typer
from click import unstyle
from typer.testing import CliRunner

from promptic_sdk.cli.commands.agent_gym import _evaluator
from promptic_sdk.cli.commands.experiments import _load_json_array
from promptic_sdk.cli.config import CliConfig
from promptic_sdk.cli.main import app

runner = CliRunner()


def test_agent_gym_cli_field_method():
    evaluator = _evaluator(
        {
            "kind": "field_level_judge",
            "fields": {
                "answer": {"method": "judge", "judgeInstructions": "Check evidence."},
                "citations": {"method": "array_judge"},
            },
        }
    )
    fields = evaluator.as_request()["fields"]
    assert fields["answer"]["method"] == "judge"
    assert fields["citations"]["method"] == "array_judge"
    assert "strategy" not in fields["citations"]


@pytest.mark.parametrize("legacy_key", ["strategy", "arrayStrategy", "array_strategy"])
def test_agent_gym_cli_rejects_legacy_field_keys(legacy_key):
    with pytest.raises(ValueError, match="unsupported field scoring key"):
        _evaluator(
            {
                "kind": "field_level_judge",
                "fields": {"answer": {legacy_key: "exact"}},
            }
        )


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
    method = mock_client
    for part in method_name.split("."):
        method = getattr(method, part)
    method.return_value = return_value
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


class TestModelsCommands:
    def test_list_models_json(self):
        payload = {
            "data": [
                {
                    "id": "judge-1",
                    "name": "Judge One",
                    "provider": "OpenAI",
                    "group": "openai",
                    "judgeEligible": True,
                }
            ],
        }
        with _mock_config(), _mock_client("models", "models.list", payload):
            result = runner.invoke(app, ["models", "list", "--json"])

        assert result.exit_code == 0
        assert json.loads(result.stdout) == payload

    def test_list_models_table(self):
        payload = {
            "data": [
                {
                    "id": "judge-1",
                    "name": "Judge One",
                    "provider": "OpenAI",
                    "group": "openai",
                    "judgeEligible": True,
                }
            ],
        }
        with _mock_config(), _mock_client("models", "models.list", payload):
            result = runner.invoke(app, ["models", "list"])

        assert result.exit_code == 0
        assert "judge-1" in result.stdout
        assert "yes" in result.stdout


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

    def test_cases_add_uploads_json_array(self, tmp_path):
        cases_file = tmp_path / "cases.json"
        cases = [
            {
                "inputPayload": {"message": "hello"},
                "expectedPayload": "greeting",
                "split": "train",
            }
        ]
        cases_file.write_text(json.dumps(cases))
        payload = {"data": [{"id": 7, **cases[0]}]}
        with (
            _mock_config(),
            _mock_client("datasets", "create_dataset_cases", payload) as patched,
        ):
            result = runner.invoke(
                app,
                [
                    "datasets",
                    "cases",
                    "add",
                    "dataset-id",
                    "--component",
                    "component-id",
                    "--file",
                    str(cases_file),
                    "--json",
                ],
            )

        assert result.exit_code == 0
        assert json.loads(result.stdout)["data"][0]["id"] == 7
        patched.return_value.create_dataset_cases.assert_called_once_with(
            "component-id", "dataset-id", cases
        )

    def test_cases_update_reads_json_object(self, tmp_path):
        case_file = tmp_path / "case.json"
        updates = {"expectedPayload": {"label": "updated"}, "split": "eval"}
        case_file.write_text(json.dumps(updates))
        payload = {"id": 7, "inputPayload": {"message": "hello"}, **updates}
        with (
            _mock_config(),
            _mock_client("datasets", "update_dataset_case", payload) as patched,
        ):
            result = runner.invoke(
                app,
                [
                    "datasets",
                    "cases",
                    "update",
                    "dataset-id",
                    "7",
                    "--component",
                    "component-id",
                    "--file",
                    str(case_file),
                    "--json",
                ],
            )

        assert result.exit_code == 0
        patched.return_value.update_dataset_case.assert_called_once_with(
            "component-id", "dataset-id", 7, **updates
        )

    def test_cases_list_get_and_delete_call_client(self):
        cases = {"data": [{"id": 7, "inputPayload": {}, "expectedPayload": None}]}
        with (
            _mock_config(),
            _mock_client("datasets", "list_dataset_cases", cases) as patched,
        ):
            result = runner.invoke(
                app,
                [
                    "datasets",
                    "cases",
                    "list",
                    "dataset-id",
                    "--component",
                    "component-id",
                    "--json",
                ],
            )
        assert result.exit_code == 0
        patched.return_value.list_dataset_cases.assert_called_once_with(
            "component-id", "dataset-id"
        )

        with (
            _mock_config(),
            _mock_client("datasets", "get_dataset_case", cases["data"][0]) as patched,
        ):
            result = runner.invoke(
                app,
                [
                    "datasets",
                    "cases",
                    "get",
                    "dataset-id",
                    "7",
                    "--component",
                    "component-id",
                    "--json",
                ],
            )
        assert result.exit_code == 0
        patched.return_value.get_dataset_case.assert_called_once_with(
            "component-id", "dataset-id", 7
        )

        with _mock_config(), _mock_client("datasets", "delete_dataset_case", None) as patched:
            result = runner.invoke(
                app,
                [
                    "datasets",
                    "cases",
                    "delete",
                    "dataset-id",
                    "7",
                    "--component",
                    "component-id",
                    "--force",
                ],
            )
        assert result.exit_code == 0
        patched.return_value.delete_dataset_case.assert_called_once_with(
            "component-id", "dataset-id", 7
        )


class TestAgentGymCommands:
    def test_apply_round_trips_finalized_verifier_contract(self, tmp_path):
        config = tmp_path / "agent.json"
        config.write_text(
            json.dumps(
                {
                    "aiApplicationId": "00000000-0000-4000-8000-000000000040",
                    "name": "Agent",
                    "goal": "Review a submitted result.",
                    "evaluators": [
                        {
                            "kind": "verifier_agent",
                            "instructions": "Inspect the selected evidence and score correctness.",
                            "model": "verifier-model",
                            "metrics": [
                                {
                                    "key": "correctness",
                                    "name": "Correctness",
                                    "instructions": "Check correctness.",
                                    "weight": 2,
                                    "threshold": 0.8,
                                }
                            ],
                            "evidence": {
                                "caseInputs": True,
                                "submittedOutput": True,
                                "expectedBehavior": False,
                                "expectedOutput": True,
                                "executionTrace": True,
                            },
                            "budget": {"maxSteps": 200},
                        }
                    ],
                }
            )
        )
        benchmark = MagicMock()
        benchmark.id = "00000000-0000-4000-8000-000000000041"
        benchmark.ready_for_submission = False
        benchmark.data = {}
        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.benchmarks.create.return_value = benchmark

        with patch("promptic_sdk.cli.commands.agent_gym.AgentGymClient", return_value=client):
            result = runner.invoke(app, ["agent-gym", "apply", str(config)])

        assert result.exit_code == 0
        evaluator = client.benchmarks.create.call_args.kwargs["evaluators"][0]
        assert evaluator.as_request() == {
            "kind": "verifier_agent",
            "required": False,
            "instructions": "Inspect the selected evidence and score correctness.",
            "model": "verifier-model",
            "evidence": {
                "caseInputs": True,
                "submittedOutput": True,
                "expectedBehavior": False,
                "expectedOutput": True,
                "executionTrace": True,
            },
            "metrics": [
                {
                    "key": "correctness",
                    "name": "Correctness",
                    "instructions": "Check correctness.",
                }
            ],
            "metricBindings": {"correctness": {"weight": 2.0, "threshold": 0.8}},
            "budget": {"maxSteps": 200},
        }

    def test_apply_preserves_explicit_field_scoring(self, tmp_path):
        config = tmp_path / "agent.json"
        config.write_text(
            json.dumps(
                {
                    "aiApplicationId": "00000000-0000-4000-8000-000000000040",
                    "name": "Agent",
                    "goal": "Extract an answer.",
                    "evaluators": [
                        {
                            "kind": "field_level_judge",
                            "fields": {
                                "answer": {
                                    "method": "judge",
                                    "judgeInstructions": "Check the evidence.",
                                    "weight": 2,
                                }
                            },
                        }
                    ],
                }
            )
        )
        benchmark = MagicMock()
        benchmark.id = "00000000-0000-4000-8000-000000000041"
        benchmark.ready_for_submission = False
        benchmark.data = {}
        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.benchmarks.create.return_value = benchmark

        with patch("promptic_sdk.cli.commands.agent_gym.AgentGymClient", return_value=client):
            result = runner.invoke(app, ["agent-gym", "apply", str(config)])

        assert result.exit_code == 0
        evaluator = client.benchmarks.create.call_args.kwargs["evaluators"][0]
        assert evaluator.as_request()["fields"] == {
            "answer": {
                "include": True,
                "method": "judge",
                "weight": 2.0,
                "judge_instructions": "Check the evidence.",
            }
        }

    def test_apply_round_trips_locked_expected_behavior_judge(self, tmp_path):
        config = tmp_path / "agent.json"
        config.write_text(
            json.dumps(
                {
                    "aiApplicationId": "00000000-0000-4000-8000-000000000040",
                    "name": "Agent",
                    "goal": "Verify a submission.",
                    "evaluators": [
                        {
                            "kind": "expected_behavior_judge",
                            "name": "Expected behavior",
                            "description": "Uses the platform preset.",
                            "model": "verifier-model",
                            "metricBindings": {
                                "behavior_compliance": {"weight": 2, "threshold": 0.8}
                            },
                            "required": True,
                            "threshold": 0.8,
                        }
                    ],
                }
            )
        )
        benchmark = MagicMock()
        benchmark.id = "00000000-0000-4000-8000-000000000041"
        benchmark.ready_for_submission = False
        benchmark.data = {}
        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.benchmarks.create.return_value = benchmark

        with patch("promptic_sdk.cli.commands.agent_gym.AgentGymClient", return_value=client):
            result = runner.invoke(app, ["agent-gym", "apply", str(config)])

        assert result.exit_code == 0
        evaluator = client.benchmarks.create.call_args.kwargs["evaluators"][0]
        assert evaluator.as_request() == {
            "kind": "expected_behavior_judge",
            "required": True,
            "name": "Expected behavior",
            "description": "Uses the platform preset.",
            "threshold": 0.8,
            "model": "verifier-model",
            "metricBindings": {"behavior_compliance": {"weight": 2.0, "threshold": 0.8}},
        }

    @pytest.mark.parametrize(
        ("evaluator", "message"),
        [
            (
                {
                    "kind": "verifier_agent",
                    "instructions": "Evaluate.",
                    "metrics": [{"key": "overall", "name": "Overall", "instructions": "Evaluate."}],
                    "tools": {"shell": False},
                },
                "unsupported verifier_agent field",
            ),
            (
                {
                    "kind": "verifier_agent",
                    "instructions": "Evaluate.",
                    "metrics": [{"key": "overall", "name": "Overall", "instructions": "Evaluate."}],
                    "evidence": {"required": ["case_input"]},
                },
                "unsupported verifier evidence field",
            ),
            (
                {"kind": "expected_behavior_judge", "instructions": "Override."},
                "expected_behavior_judge is locked",
            ),
            (
                {"kind": "expected_behavior_judge", "evidence": {"caseInputs": True}},
                "expected_behavior_judge is locked",
            ),
            (
                {
                    "kind": "expected_behavior_judge",
                    "metrics": [{"key": "overall", "name": "Overall", "instructions": "Override."}],
                },
                "expected_behavior_judge is locked",
            ),
            (
                {"kind": "expected_behavior_judge", "budget": {"maxSteps": 20}},
                "expected_behavior_judge is locked",
            ),
            ({"kind": "trajectory_judge"}, "unsupported evaluator kind"),
        ],
    )
    def test_apply_rejects_removed_or_locked_verifier_configuration(
        self, tmp_path, evaluator, message
    ):
        config = tmp_path / "agent.json"
        config.write_text(
            json.dumps(
                {
                    "name": "Agent",
                    "goal": "Verify a submission.",
                    "evaluators": [evaluator],
                }
            )
        )

        result = runner.invoke(app, ["agent-gym", "apply", str(config)])

        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)
        assert message in str(result.exception)

    def test_apply_rejects_shared_field_judge_rubric(self, tmp_path):
        config = tmp_path / "agent.json"
        config.write_text(
            json.dumps(
                {
                    "aiApplicationId": "00000000-0000-4000-8000-000000000040",
                    "name": "Agent",
                    "goal": "Extract an answer.",
                    "evaluators": [{"kind": "field_level_judge", "rubric": "Shared guidance"}],
                }
            )
        )

        result = runner.invoke(app, ["agent-gym", "apply", str(config)])

        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)
        assert "does not support shared instructions" in str(result.exception)

    def test_apply_explicitly_publishes_the_configured_revision(self, tmp_path):
        config = tmp_path / "agent.json"
        config.write_text(
            json.dumps(
                {
                    "aiApplicationId": "00000000-0000-4000-8000-000000000040",
                    "name": "Agent",
                    "goal": "Classify requests.",
                    "inputSchema": {"type": "object", "properties": {}},
                    "outputSchema": {"type": "object", "properties": {}},
                    "evaluators": [],
                    "cases": [{"input": {}, "output": {}}],
                }
            )
        )
        benchmark = MagicMock()
        benchmark.id = "00000000-0000-4000-8000-000000000041"
        benchmark.ready_for_submission = True
        benchmark.data = {"activeRevisionId": "00000000-0000-4000-8000-000000000042"}
        benchmark.cases.add_many.return_value = {"created": 1, "failed": 0, "results": []}
        benchmark.refresh.return_value = benchmark
        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.benchmarks.create.return_value = benchmark

        with patch("promptic_sdk.cli.commands.agent_gym.AgentGymClient", return_value=client):
            result = runner.invoke(app, ["agent-gym", "apply", str(config)])

        assert result.exit_code == 0
        assert "Active revision: 00000000-0000-4000-8000-000000000042" in result.stdout
        benchmark.publish.assert_called_once_with()

    def test_apply_no_longer_accepts_manual_publish_flag(self, tmp_path):
        config = tmp_path / "agent.json"
        config.write_text(json.dumps({"name": "Agent", "goal": "Classify requests."}))

        result = runner.invoke(app, ["agent-gym", "apply", str(config), "--publish"])

        assert result.exit_code != 0
        assert "No such option" in result.output

    def test_publish_draft_activates_ready_revision(self):
        benchmark = MagicMock()
        benchmark.publish.return_value.revision.id = "revision-2"
        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.benchmarks.get.return_value = benchmark

        with patch("promptic_sdk.cli.commands.agent_gym.AgentGymClient", return_value=client):
            result = runner.invoke(app, ["agent-gym", "publish-draft", "benchmark-1"])

        assert result.exit_code == 0
        assert "revision-2" in result.stdout
        benchmark.publish.assert_called_once_with()


class TestExperimentsCommands:
    def _new_exp_payload(self) -> dict:
        return {
            "id": "new-exp-id",
            "name": "Run 2",
            "experimentStatus": "pending",
            "targetModel": "gpt-5.4-nano",
            "modelUnavailable": False,
        }

    def test_create_structured_output_with_schema_hyperparameters_and_start(self, tmp_path):
        schema_file = tmp_path / "schema.json"
        schema = {
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
        }
        schema_file.write_text(json.dumps(schema))
        hyperparameters_file = tmp_path / "hyperparameters.json"
        hyperparameters = {"epochs": 2, "trainSplitRatio": 0.8, "enableCot": True}
        hyperparameters_file.write_text(json.dumps(hyperparameters))

        with (
            _mock_config(),
            _mock_client("experiments", "create_experiment", self._new_exp_payload()) as patched,
        ):
            result = runner.invoke(
                app,
                [
                    "experiments",
                    "create",
                    "--component-id",
                    "component-id",
                    "--target-model",
                    "gpt-4.1-nano",
                    "--task-type",
                    "structuredOutput",
                    "--initial-prompt",
                    "Classify {message}",
                    "--output-schema",
                    str(schema_file),
                    "--hyperparameters",
                    str(hyperparameters_file),
                    "--start",
                    "--json",
                ],
            )

        assert result.exit_code == 0
        client = patched.return_value
        client.create_experiment.assert_called_once_with(
            ai_component_id="component-id",
            target_model="gpt-4.1-nano",
            task_type="structuredOutput",
            initial_prompt="Classify {message}",
            name=None,
            description=None,
            provider="openai",
            optimizer="prompticV2",
            hyperparameters=hyperparameters,
            initial_prediction_model_schema=schema,
        )
        client.start_experiment.assert_called_once_with("new-exp-id")

    def test_create_structured_output_requires_schema(self):
        with _mock_config():
            result = runner.invoke(
                app,
                [
                    "experiments",
                    "create",
                    "--component-id",
                    "component-id",
                    "--target-model",
                    "gpt-4.1-nano",
                    "--task-type",
                    "structuredOutput",
                    "--initial-prompt",
                    "Classify {message}",
                ],
            )

        assert result.exit_code == 2
        normalized_output = "".join(unstyle(result.output).split())
        assert "--output-schemaisrequired" in normalized_output

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
