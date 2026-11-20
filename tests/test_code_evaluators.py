"""Code evaluator SDK contracts and safe CLI source-file handling."""

import json
from typing import get_args
from unittest.mock import MagicMock, patch

import httpx
import pytest
from rich.text import Text
from typer.testing import CliRunner

import promptic_sdk
from promptic_sdk import (
    AsyncPrompticClient,
    CodeFunctionConfig,
    CodeFunctionTestRow,
    PrompticClient,
)
from promptic_sdk.cli.main import app
from promptic_sdk.models import EvaluatorType

SOURCES = {
    "python": 'def score(row):\n    return {"score": 1.0, "reason": "Python passed"}',
    "javascript": 'function score(row) { return { score: 1, reason: "JS passed" }; }',
}


def evaluator(language):
    config: CodeFunctionConfig = {"language": language, "source": SOURCES[language]}
    return {
        "name": f"{language} check",
        "type": "codeFunction",
        "scaleMin": 0,
        "scaleMax": 1,
        "weight": 1,
        "config": config,
    }


def test_code_evaluator_types_are_exported():
    assert "codeFunction" in get_args(EvaluatorType)
    assert "CodeFunctionConfig" in promptic_sdk.__all__
    assert CodeFunctionConfig.__required_keys__ == {"language", "source"}
    assert CodeFunctionConfig.__optional_keys__ == {"function_name"}
    assert CodeFunctionTestRow.__required_keys__ == {"input_variables", "expected", "predicted"}
    assert CodeFunctionTestRow.__optional_keys__ == {"id"}


@pytest.mark.parametrize("language", SOURCES)
def test_sync_client_creates_updates_and_reads_code_evaluators(language):
    payload = evaluator(language)
    calls = []

    def handler(request):
        calls.append(request)
        if request.method == "POST":
            assert json.loads(request.content) == [payload]
            return httpx.Response(201, json={"data": [{"id": "check", **payload}]})
        if request.method == "PATCH":
            assert json.loads(request.content) == {"config": payload["config"]}
            return httpx.Response(200, json={"id": "check", **payload})
        return httpx.Response(200, json={"data": [{"id": "check", **payload}]})

    with PrompticClient(api_key="ptc_test") as client:
        client._client.close()
        client._client = httpx.Client(
            transport=httpx.MockTransport(handler), base_url="https://promptic.test/api/v1"
        )
        assert client.create_evaluators("exp", [payload])["data"][0]["config"] == payload["config"]
        assert (
            client.update_evaluator("exp", "check", config=payload["config"])["type"]
            == "codeFunction"
        )
        assert client.list_evaluators("exp")["data"][0]["type"] == "codeFunction"
    assert [request.url.path for request in calls] == [
        "/api/v1/experiments/exp/evaluators",
        "/api/v1/experiments/exp/evaluators/check",
        "/api/v1/experiments/exp/evaluators",
    ]


@pytest.mark.parametrize("language", SOURCES)
async def test_async_client_creates_and_updates_code_evaluators(language):
    payload = evaluator(language)

    def handler(request):
        if request.method == "POST":
            assert json.loads(request.content) == [payload]
            return httpx.Response(201, json={"data": [payload]})
        assert json.loads(request.content) == {"config": payload["config"]}
        return httpx.Response(200, json=payload)

    async with AsyncPrompticClient(api_key="ptc_test") as client:
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://promptic.test/api/v1"
        )
        assert (await client.create_evaluators("exp", [payload]))["data"] == [payload]
        assert await client.update_evaluator("exp", "check", config=payload["config"]) == payload


def add(*args):
    client = MagicMock()
    client.__enter__.return_value = client
    client.create_evaluators.return_value = {"data": [{"id": "check"}]}
    with patch("promptic_sdk.cli.commands.evaluators.get_client", return_value=client):
        result = CliRunner().invoke(app, ["evaluators", "add", "exp", "--name", "Check", *args])
    return result, client


@pytest.mark.parametrize("language", SOURCES)
def test_cli_uploads_source_without_executing_it(tmp_path, language):
    source = tmp_path / "score.txt"
    # Top-level throwing code must be uploaded, not evaluated locally.
    value = SOURCES[language] + (
        "\nraise Exception('not local')" if language == "python" else "\nthrow Error('not local');"
    )
    source.write_text(value, encoding="utf-8")
    result, client = add(
        "--type",
        "codeFunction",
        "--language",
        language,
        "--source-file",
        str(source),
        "--disabled",
        "--weight",
        "2",
        "--json",
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"data": [{"id": "check"}]}
    [item] = client.create_evaluators.call_args.args[1]
    assert item["config"] == {"language": language, "source": value, "function_name": "score"}
    assert item["enabled"] is False and item["weight"] == 2


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--type", "codeFunction"], "choose python or javascript"),
        (["--type", "codeFunction", "--language", "typescript"], "choose python or javascript"),
        (["--type", "codeFunction", "--language", "python"], "required for codeFunction"),
        (["--type", "exactMatch", "--language", "python"], "require --type codeFunction"),
        (["--type", "codeFunction", "--field", "answer"], "only supported for"),
        (["--type", "codeFunction", "--instructions", "Check"], "only supported for"),
    ],
)
def test_cli_rejects_incompatible_options(args, message):
    result, client = add(*args)
    assert result.exit_code == 2
    output = Text.from_ansi(result.output).plain
    assert message in " ".join(output.replace("│", " ").split())
    client.create_evaluators.assert_not_called()


@pytest.mark.parametrize("language", SOURCES)
def test_cli_preserves_source_whitespace(tmp_path, language):
    source = tmp_path / "score.txt"
    value = "\n\n" + SOURCES[language] + "\n\n"
    source.write_bytes(value.encode("utf-8"))
    result, client = add(
        "--type", "codeFunction", "--language", language, "--source-file", str(source)
    )
    assert result.exit_code == 0, result.output
    assert client.create_evaluators.call_args.args[1][0]["config"]["source"] == value


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b" \n", "cannot be empty"),
        (b"\xff", "UTF-8"),
        (b"x" * (65536 + 1), "64 KiB"),
        ("\u00e9".encode() * 32769, "64 KiB"),
    ],
)
def test_cli_rejects_invalid_files_before_request(tmp_path, content, message):
    source = tmp_path / "score.py"
    source.write_bytes(content)
    result, client = add(
        "--type", "codeFunction", "--language", "python", "--source-file", str(source)
    )
    assert result.exit_code == 2
    assert message in " ".join(result.output.split())
    client.create_evaluators.assert_not_called()


def test_cli_rejects_missing_source_file(tmp_path):
    result, client = add(
        "--type", "codeFunction", "--language", "python", "--source-file", str(tmp_path / "absent")
    )
    assert result.exit_code == 2
    client.create_evaluators.assert_not_called()


def test_cli_accepts_exact_utf8_size_limit(tmp_path):
    source = tmp_path / "score.py"
    source.write_bytes(b"x" * 65536)
    result, client = add(
        "--type", "codeFunction", "--language", "python", "--source-file", str(source)
    )
    assert result.exit_code == 0, result.output
    assert len(client.create_evaluators.call_args.args[1][0]["config"]["source"]) == 65536


@pytest.mark.parametrize("language", SOURCES)
def test_sync_client_can_queue_and_poll_sandbox_probe(language):
    config = evaluator(language)["config"]
    row: CodeFunctionTestRow = {
        "input_variables": {"question": "Hi"},
        "expected": "OK",
        "predicted": "OK",
    }

    def handler(request):
        if request.method == "POST":
            assert request.url.path == "/api/v1/code-evaluator-tests"
            assert json.loads(request.content) == {"config": config, "row": row}
            return httpx.Response(202, json={"id": "test-id", "status": "pending"})
        assert request.url.path == "/api/v1/code-evaluator-tests/test-id"
        return httpx.Response(
            200,
            json={
                "id": "test-id",
                "status": "failed",
                "result": {
                    "error": "row_execution_failed",
                    "error_type": "AttributeError",
                    "line": 2,
                },
            },
        )

    with PrompticClient(api_key="ptc_test") as client:
        client._client.close()
        client._client = httpx.Client(
            transport=httpx.MockTransport(handler), base_url="https://promptic.test/api/v1"
        )
        test = client.test_code_evaluator(config, row)
        assert client.get_code_evaluator_test(test["id"])["result"]["line"] == 2


async def test_async_client_can_queue_and_poll_sandbox_probe():
    config = evaluator("python")["config"]
    row: CodeFunctionTestRow = {"input_variables": {}, "expected": "OK", "predicted": "OK"}

    def handler(request):
        if request.method == "POST":
            assert json.loads(request.content) == {"config": config, "row": row}
            return httpx.Response(202, json={"id": "test-id", "status": "pending"})
        return httpx.Response(200, json={"status": "completed", "result": {"score": 1}})

    async with AsyncPrompticClient(api_key="ptc_test") as client:
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://promptic.test/api/v1"
        )
        test = await client.test_code_evaluator(config, row)
        assert (await client.get_code_evaluator_test(test["id"]))["result"]["score"] == 1


@pytest.mark.parametrize("language", SOURCES)
def test_cli_queues_and_polls_one_row_probe(tmp_path, language):
    source = tmp_path / "score.txt"
    source.write_text(SOURCES[language], encoding="utf-8")
    sample = tmp_path / "row.json"
    row = {"input_variables": {}, "expected": "OK", "predicted": "OK"}
    sample.write_text(json.dumps(row), encoding="utf-8")
    client = MagicMock()
    client.__enter__.return_value = client
    client.test_code_evaluator.return_value = {"id": "test-id", "status": "pending"}
    client.get_code_evaluator_test.return_value = {
        "id": "test-id",
        "status": "completed",
        "result": {"score": 1},
    }
    with patch("promptic_sdk.cli.commands.evaluators.get_client", return_value=client):
        result = CliRunner().invoke(
            app,
            [
                "evaluators",
                "test",
                "--language",
                language,
                "--source-file",
                str(source),
                "--row-file",
                str(sample),
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["id"] == "test-id"
        client.test_code_evaluator.assert_called_once_with(
            {"language": language, "source": SOURCES[language], "function_name": "score"}, row
        )
        result = CliRunner().invoke(app, ["evaluators", "test-result", "test-id", "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["result"]["score"] == 1


@pytest.mark.parametrize("content", [b"bad json", b"[]", b"{}", b"\xff", b"x" * 65537])
def test_cli_rejects_invalid_probe_rows_before_submission(tmp_path, content):
    source = tmp_path / "score.py"
    source.write_text(SOURCES["python"], encoding="utf-8")
    sample = tmp_path / "row.json"
    sample.write_bytes(content)
    with patch("promptic_sdk.cli.commands.evaluators.get_client") as client:
        result = CliRunner().invoke(
            app,
            [
                "evaluators",
                "test",
                "--language",
                "python",
                "--source-file",
                str(source),
                "--row-file",
                str(sample),
            ],
        )
    assert result.exit_code == 2, result.output
    client.assert_not_called()


@pytest.mark.parametrize("source_text", ["#" * 65536, '"' * 65536])
def test_cli_rejects_oversized_combined_probe_before_submission(tmp_path, source_text):
    source = tmp_path / "score.py"
    source.write_text(source_text, encoding="utf-8")
    sample = tmp_path / "row.json"
    sample.write_text(
        json.dumps({"input_variables": {}, "expected": "x" * 65470, "predicted": ""}),
        encoding="utf-8",
    )
    assert source.stat().st_size <= 65536
    assert sample.stat().st_size <= 65536
    with patch("promptic_sdk.cli.commands.evaluators.get_client") as client:
        result = CliRunner().invoke(
            app,
            [
                "evaluators",
                "test",
                "--language",
                "python",
                "--source-file",
                str(source),
                "--row-file",
                str(sample),
            ],
        )
    assert result.exit_code == 2, result.output
    assert "combined JSON request" in Text.from_ansi(result.output).plain
    client.assert_not_called()
