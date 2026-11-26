"""Agent Gym Code Function evaluators: typed model, validation, CLI and sandbox tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from typer.testing import CliRunner

from promptic_sdk import (
    AgentGymCodeFunctionTestRow,
    CodeFunction,
    CodeFunctionConfig,
    ExactMatch,
    PrompticClient,
)
from promptic_sdk.agent_gym.authoring_models import (
    MAX_CODE_FUNCTION_EVALUATORS,
    EvaluatorConfigurationError,
    validate_evaluators,
)
from promptic_sdk.cli.commands.agent_gym import _evaluator
from promptic_sdk.cli.main import app

SOURCE = 'def score(row):\n    return 1.0 if row["output"] == row["expected_output"] else 0.0\n'


def test_code_function_serializes_the_platform_contract() -> None:
    assert CodeFunction(SOURCE).as_request() == {
        "kind": "code_function",
        "weight": 1.0,
        "required": False,
        "language": "python",
        "source": SOURCE,
    }
    request = CodeFunction(
        "function score(row) { return 1 }",
        "javascript",
        name="Totals",
        weight=2,
        required=True,
        scale_min=0,
        scale_max=10,
        id="totals",
        enabled=False,
    ).as_request()
    assert request == {
        "kind": "code_function",
        "id": "totals",
        "enabled": False,
        "weight": 2,
        "required": True,
        "name": "Totals",
        "language": "javascript",
        "source": "function score(row) { return 1 }",
        "scaleMin": 0,
        "scaleMax": 10,
    }
    assert "threshold" not in request


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"source": "  "}, "cannot be empty"),
        ({"source": "x" * (64 * 1024 + 1)}, "at most 64 KiB"),
        ({"source": SOURCE, "language": "ruby"}, "python' or 'javascript"),
        ({"source": SOURCE, "scale_min": 1, "scale_max": 1}, "must differ"),
        ({"source": SOURCE, "weight": 0}, "weight must be positive"),
    ],
)
def test_code_function_rejects_invalid_configuration(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        CodeFunction(**kwargs)


def test_code_functions_repeat_with_unique_names_and_a_limit() -> None:
    validate_evaluators([CodeFunction(SOURCE, name="A"), CodeFunction(SOURCE, name="B")])
    with pytest.raises(EvaluatorConfigurationError) as duplicate:
        validate_evaluators([CodeFunction(SOURCE, name="A"), CodeFunction(SOURCE, name="A")])
    assert duplicate.value.code == "duplicate_evaluator_name"
    with pytest.raises(EvaluatorConfigurationError) as too_many:
        validate_evaluators(
            [
                CodeFunction(SOURCE, name=f"Check {index}")
                for index in range(MAX_CODE_FUNCTION_EVALUATORS + 1)
            ]
        )
    assert too_many.value.code == "too_many_code_function_evaluators"
    # Code functions and targeted checks share the unique-name space.
    with pytest.raises(EvaluatorConfigurationError):
        validate_evaluators([CodeFunction(SOURCE, name="A"), ExactMatch(name="A")])


def test_cli_agent_config_reads_inline_and_file_sources(tmp_path) -> None:
    (tmp_path / "checks").mkdir()
    (tmp_path / "checks" / "totals.js").write_text("function score(row) { return 1 }")
    inline = _evaluator({"kind": "code_function", "name": "Inline", "source": SOURCE}, tmp_path)
    from_file = _evaluator(
        {
            "kind": "code_function",
            "name": "From file",
            "language": "javascript",
            "sourceFile": "checks/totals.js",
            "scaleMax": 10,
            "weight": 2,
        },
        tmp_path,
    )
    assert isinstance(inline, CodeFunction)
    assert inline.source == SOURCE
    assert isinstance(from_file, CodeFunction)
    assert from_file.source == "function score(row) { return 1 }"
    assert from_file.as_request()["scaleMax"] == 10
    assert from_file.weight == 2


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"kind": "code_function"}, "exactly one of source or sourceFile"),
        (
            {"kind": "code_function", "source": SOURCE, "sourceFile": "a.py"},
            "exactly one of source or sourceFile",
        ),
        ({"kind": "code_function", "sourceFile": "missing.py"}, "readable UTF-8 file"),
        ({"kind": "code_function", "source": SOURCE, "threshold": 0.5}, "threshold"),
        ({"kind": "code_function", "source": SOURCE, "instructions": "x"}, "instructions"),
    ],
)
def test_cli_agent_config_rejects_invalid_code_functions(tmp_path, value, message) -> None:
    with pytest.raises(ValueError, match=message):
        _evaluator(value, tmp_path)


def test_client_sends_agent_gym_rows_with_their_row_contract() -> None:
    config: CodeFunctionConfig = {"language": "python", "source": SOURCE}
    row: AgentGymCodeFunctionTestRow = {"inputs": {"q": 1}, "output": {"answer": 2}}
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(202, json={"id": "test-id", "status": "pending"})

    with PrompticClient(api_key="ptc_test") as client:
        client._client.close()
        client._client = httpx.Client(
            transport=httpx.MockTransport(handler), base_url="https://promptic.test/api/v1"
        )
        client.test_code_evaluator(config, row, row_contract="agent_gym")
    assert bodies == [{"config": config, "row_contract": "agent_gym", "row": row}]


def test_cli_tests_agent_gym_rows(tmp_path) -> None:
    source = tmp_path / "score.py"
    source.write_text(SOURCE, encoding="utf-8")
    sample = tmp_path / "row.json"
    row = {"inputs": {"q": 1}, "output": {"answer": 2}, "expected_output": {"answer": 2}}
    sample.write_text(json.dumps(row), encoding="utf-8")
    client = MagicMock()
    client.__enter__.return_value = client
    client.test_code_evaluator.return_value = {"id": "test-id", "status": "pending"}
    arguments = [
        "evaluators",
        "test",
        "--language",
        "python",
        "--source-file",
        str(source),
        "--row-file",
        str(sample),
        "--row-contract",
        "agent_gym",
        "--json",
    ]
    with patch("promptic_sdk.cli.commands.evaluators.get_client", return_value=client):
        result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    client.test_code_evaluator.assert_called_once_with(
        {"language": "python", "source": SOURCE, "function_name": "score"},
        row,
        row_contract="agent_gym",
    )

    sample.write_text(json.dumps({"input_variables": {}, "expected": "", "predicted": ""}))
    with patch("promptic_sdk.cli.commands.evaluators.get_client", return_value=client):
        rejected = CliRunner().invoke(app, arguments)
    assert rejected.exit_code != 0


def test_code_functions_example_is_valid() -> None:
    with open("examples/agent_gym_code_functions.json") as handle:
        payload = json.load(handle)
    evaluators = [_evaluator(item, Path("examples")) for item in payload["evaluators"]]
    validate_evaluators(evaluators)
    code_functions = [item for item in evaluators if isinstance(item, CodeFunction)]
    assert [item.language for item in code_functions] == ["python", "javascript"]
    assert "def score(row)" in code_functions[0].source
