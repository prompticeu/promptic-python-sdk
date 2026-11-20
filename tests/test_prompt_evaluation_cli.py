"""CLI workflows for locally authored prompt candidates."""

import json
from unittest.mock import patch

from typer.testing import CliRunner

from promptic_sdk.cli.main import app

runner = CliRunner()


def test_create_evaluation_only_with_chat_messages(tmp_path):
    messages = [{"role": "system", "content": "Classify"}, {"role": "user", "content": "{input}"}]
    path = tmp_path / "messages.json"
    path.write_text(json.dumps(messages))
    with patch("promptic_sdk.cli.commands.experiments.get_client") as get_client:
        client = get_client.return_value.__enter__.return_value
        client.create_experiment.return_value = {"id": "candidate"}
        result = runner.invoke(
            app,
            [
                "experiments",
                "create",
                "-c",
                "component",
                "-m",
                "gpt-5.6-luna",
                "-t",
                "classification",
                "--messages",
                str(path),
                "--evaluate-only",
                "--json",
            ],
        )
    assert result.exit_code == 0, result.output
    args = client.create_experiment.call_args.kwargs
    assert args["evaluation_only"] is True
    assert args["prompt_format"] == "multi_message"
    assert args["initial_prompt_messages"] == messages
    client.start_experiment.assert_not_called()


def test_create_rejects_conflicting_evaluation_budget(tmp_path):
    path = tmp_path / "hyper.json"
    path.write_text(json.dumps({"epochs": 3, "numFewShots": 8}))
    with patch("promptic_sdk.cli.commands.experiments.get_client") as get_client:
        result = runner.invoke(
            app,
            [
                "experiments",
                "create",
                "-c",
                "component",
                "-m",
                "gpt-5.6-luna",
                "-t",
                "classification",
                "-p",
                "{input}",
                "--evaluate-only",
                "--hyperparameters",
                str(path),
            ],
        )
    assert result.exit_code == 2
    get_client.assert_not_called()


def test_duplicate_evaluates_only_before_explicit_start():
    original = {
        "id": "candidate",
        "hyperparameters": {"epochs": 3, "numFewShots": 2, "trainSplitRatio": 0.8},
    }
    with patch("promptic_sdk.cli.commands.experiments.get_client") as get_client:
        client = get_client.return_value.__enter__.return_value
        client.duplicate_experiment.return_value = original
        client.update_experiment.return_value = {"id": "candidate"}
        result = runner.invoke(
            app,
            [
                "experiments",
                "duplicate",
                "reference",
                "-p",
                "New {input}",
                "--evaluate-only",
                "--start",
                "--json",
            ],
        )
    assert result.exit_code == 0, result.output
    client.update_experiment.assert_called_once_with(
        "candidate", hyperparameters={"epochs": 0, "numFewShots": 2, "trainSplitRatio": 0.8}
    )
    assert [call[0] for call in client.method_calls] == [
        "duplicate_experiment",
        "update_experiment",
        "start_experiment",
    ]
    assert original["hyperparameters"]["epochs"] == 3


def test_update_zero_preserves_hyperparameters_and_uses_canonical_prompt():
    with patch("promptic_sdk.cli.commands.experiments.get_client") as get_client:
        client = get_client.return_value.__enter__.return_value
        client.get_experiment.return_value = {
            "hyperparameters": {"epochs": 3, "thinkingLevel": "low", "numFewShots": 4}
        }
        client.update_experiment.return_value = {"id": "candidate"}
        result = runner.invoke(
            app,
            ["experiments", "update", "candidate", "--epochs", "0", "-p", "New {input}", "--json"],
        )
    assert result.exit_code == 0, result.output
    client.update_experiment.assert_called_once_with(
        "candidate",
        promptFormat="single",
        initialPromptMessages=[{"role": "user", "content": "New {input}"}],
        hyperparameters={"epochs": 0, "thinkingLevel": "low", "numFewShots": 4},
    )


def test_tool_selection_accepts_zero_rounds(tmp_path):
    tools, cases = tmp_path / "tools.json", tmp_path / "cases.json"
    tools.write_text(json.dumps([{"name": "lookup", "description": "Look up"}]))
    cases.write_text(json.dumps([{"query": "Find it", "expected_tool": "lookup"}]))
    with patch("promptic_sdk.cli.commands.experiments.get_client") as get_client:
        client = get_client.return_value.__enter__.return_value
        client.create_tool_selection_experiment.return_value = {"id": "candidate"}
        result = runner.invoke(
            app,
            [
                "experiments",
                "create-tool-selection",
                "-c",
                "component",
                "--tools",
                str(tools),
                "--test-cases",
                str(cases),
                "--epochs",
                "0",
                "--json",
            ],
        )
    assert result.exit_code == 0, result.output
    assert client.create_tool_selection_experiment.call_args.kwargs["epochs"] == 0
    client.start_experiment.assert_not_called()


def test_cases_returns_a_bounded_json_page():
    page = {"data": [{"id": "1", "score": 0}], "nextCursor": "next", "total": 38}
    with patch("promptic_sdk.cli.commands.experiments.get_client") as get_client:
        client = get_client.return_value.__enter__.return_value
        client.list_experiment_cases.return_value = page
        result = runner.invoke(
            app,
            [
                "experiments",
                "cases",
                "candidate",
                "--iteration-id",
                "42",
                "--limit",
                "10",
                "--cursor",
                "previous",
                "--sort",
                "score",
                "--json",
            ],
        )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == page
    client.list_experiment_cases.assert_called_once_with(
        "candidate",
        iteration_id=42,
        split="all",
        limit=10,
        cursor="previous",
        sort="score",
        descending=False,
    )
