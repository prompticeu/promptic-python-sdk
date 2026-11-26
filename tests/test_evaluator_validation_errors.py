"""Regression tests for validation diagnostics seen by SDK users and coding agents."""

from __future__ import annotations

import importlib
import json
import sys
from unittest.mock import MagicMock

import httpx
import pytest

from promptic_sdk import (
    AgentGymClient,
    AIJudge,
    AsyncAgentGymClient,
    ClassificationF1,
    ExactMatch,
    ExpectedBehaviorJudge,
    FieldLevelJudge,
    VerifierAgent,
    VerifierMetric,
)
from promptic_sdk.agent_gym._http import _api_error
from promptic_sdk.agent_gym._validation import EvaluatorConfigurationError
from promptic_sdk.agent_gym.authoring import AgentGymBenchmark, AsyncAgentGymBenchmark
from promptic_sdk.agent_gym.authoring_models import validate_evaluators

BENCHMARK_ID = "00000000-0000-4000-8000-000000000041"
# Older deployments may still return duplicate-kind errors for field-level judges.
CORRECTION = (
    "Only one field-level evaluator is supported. Combine your field rules into one evaluator."
)
ISSUE = {
    "code": "duplicate_evaluator_kind",
    "path": ["evaluators", 1, "kind"],
    "kind": "field_level_judge",
    "firstIndex": 0,
    "message": CORRECTION,
}
BODIES = [
    {
        "error": "Invalid body",
        "details": {"fieldErrors": {"evaluators": [CORRECTION]}, "formErrors": []},
    },
    {"error": "duplicate_evaluator_kind", "message": CORRECTION, "details": {"issues": [ISSUE]}},
    {
        "error": "Invalid evaluator configuration.",
        "issues": [ISSUE],
        "validFieldPaths": ["answer", "pages"],
    },
]


@pytest.mark.parametrize("operation", ["create", "configure"])
def test_sync_rejects_duplicate_explicit_ids_before_any_request(operation, monkeypatch):
    with AgentGymClient(api_key="ptc_test") as client:
        request = MagicMock(side_effect=AssertionError("must not send a request"))
        monkeypatch.setattr(client._transport, "request", request)
        judges = [
            AIJudge(name="Answer", id="shared"),
            AIJudge(name="Pages", id="shared"),
        ]
        with pytest.raises(EvaluatorConfigurationError) as raised:
            if operation == "create":
                client.benchmarks.create(name="QA", goal="Answer questions", evaluators=judges)
            else:
                benchmark = AgentGymBenchmark(client.benchmarks, {"id": BENCHMARK_ID})
                benchmark.configure(name="Must not save", evaluators=judges)
        assert raised.value.code == "duplicate_evaluator_id"
        assert "evaluators[1].id" in str(raised.value)
        assert "evaluators[0].id" in str(raised.value)
        assert "Choose a distinct id or omit it" in str(raised.value)
        request.assert_not_called()


@pytest.mark.parametrize("operation", ["create", "configure"])
async def test_async_rejects_duplicate_explicit_ids_before_any_request(operation, monkeypatch):
    async with AsyncAgentGymClient(api_key="ptc_test") as client:
        request = MagicMock(side_effect=AssertionError("must not send a request"))
        monkeypatch.setattr(client._transport, "request", request)
        judges = [
            AIJudge(name="Answer", id="shared"),
            AIJudge(name="Pages", id="shared"),
        ]
        with pytest.raises(EvaluatorConfigurationError) as raised:
            if operation == "create":
                await client.benchmarks.create(
                    name="QA", goal="Answer questions", evaluators=judges
                )
            else:
                benchmark = AsyncAgentGymBenchmark(client.benchmarks, {"id": BENCHMARK_ID})
                await benchmark.configure(name="Must not save", evaluators=judges)
        assert raised.value.code == "duplicate_evaluator_id"
        assert "evaluators[1].id" in str(raised.value)
        assert "evaluators[0].id" in str(raised.value)
        assert "Choose a distinct id or omit it" in str(raised.value)
        request.assert_not_called()


def _singleton(kind, **kwargs):
    if kind == "f1":
        return ClassificationF1(("label",), **kwargs)
    if kind == "verifier_agent":
        return VerifierAgent(
            instructions="Inspect the answer.",
            metrics=(VerifierMetric("quality", "Quality", "Check accuracy."),),
            **kwargs,
        )
    return ExpectedBehaviorJudge(**kwargs)


@pytest.mark.parametrize("kind", ["f1", "verifier_agent", "expected_behavior_judge"])
@pytest.mark.parametrize("operation", ["create", "configure"])
def test_sync_retains_singleton_restrictions_with_distinct_ids(kind, operation, monkeypatch):
    with AgentGymClient(api_key="ptc_test") as client:
        request = MagicMock(side_effect=AssertionError("must not send a request"))
        monkeypatch.setattr(client._transport, "request", request)
        evaluators = [_singleton(kind, id="first"), _singleton(kind, id="second")]
        with pytest.raises(EvaluatorConfigurationError) as raised:
            if operation == "create":
                client.benchmarks.create(name="QA", goal="Answer questions", evaluators=evaluators)
            else:
                benchmark = AgentGymBenchmark(client.benchmarks, {"id": BENCHMARK_ID})
                benchmark.configure(evaluators=evaluators)
        assert raised.value.code == "duplicate_evaluator_kind"
        assert raised.value.details["issues"][0]["kind"] == kind
        assert "evaluators[1].kind" in str(raised.value)
        request.assert_not_called()


@pytest.mark.parametrize("kind", ["f1", "verifier_agent", "expected_behavior_judge"])
@pytest.mark.parametrize("operation", ["create", "configure"])
async def test_async_retains_singleton_restrictions_with_distinct_ids(kind, operation, monkeypatch):
    async with AsyncAgentGymClient(api_key="ptc_test") as client:
        request = MagicMock(side_effect=AssertionError("must not send a request"))
        monkeypatch.setattr(client._transport, "request", request)
        evaluators = [_singleton(kind, id="first"), _singleton(kind, id="second")]
        with pytest.raises(EvaluatorConfigurationError) as raised:
            if operation == "create":
                await client.benchmarks.create(
                    name="QA", goal="Answer questions", evaluators=evaluators
                )
            else:
                benchmark = AsyncAgentGymBenchmark(client.benchmarks, {"id": BENCHMARK_ID})
                await benchmark.configure(evaluators=evaluators)
        assert raised.value.code == "duplicate_evaluator_kind"
        assert raised.value.details["issues"][0]["kind"] == kind
        assert "evaluators[1].kind" in str(raised.value)
        request.assert_not_called()


def test_duplicate_ids_are_unique_across_kinds_and_report_every_conflict():
    with pytest.raises(EvaluatorConfigurationError) as raised:
        validate_evaluators(
            [
                AIJudge(id="shared"),
                ClassificationF1(("label",), id="shared"),
                ClassificationF1(("other",), id="shared"),
            ]
        )
    assert raised.value.code == "duplicate_evaluator_id"
    issues = raised.value.details["issues"]
    assert [issue["path"] for issue in issues] == [
        ["evaluators", 1, "id"],
        ["evaluators", 2, "id"],
        ["evaluators", 2, "kind"],
    ]
    assert [issue["firstIndex"] for issue in issues] == [0, 0, 1]


@pytest.mark.parametrize(
    "kind", ["f1", "verifier_agent", "expected_behavior_judge", "field_level_judge"]
)
@pytest.mark.parametrize("invalid_id", ["", 1, True, []])
def test_typed_evaluators_reject_invalid_explicit_ids(kind, invalid_id):
    with pytest.raises(ValueError, match="id must be a non-empty string"):
        if kind == "field_level_judge":
            FieldLevelJudge(id=invalid_id)
        else:
            _singleton(kind, id=invalid_id)


def test_field_level_judge_warns_that_it_is_deprecated():
    with pytest.warns(DeprecationWarning, match="AIJudge"):
        FieldLevelJudge(fields={"answer": "judge"})


@pytest.mark.filterwarnings("ignore:FieldLevelJudge:DeprecationWarning")
@pytest.mark.parametrize("operation", ["create", "configure"])
@pytest.mark.parametrize("enabled", [True, False])
def test_deprecated_field_level_judge_is_rejected_before_any_request(
    operation, enabled, monkeypatch
):
    with AgentGymClient(api_key="ptc_test") as client:
        request = MagicMock(side_effect=AssertionError("must not send a request"))
        monkeypatch.setattr(client._transport, "request", request)
        evaluators = [ExactMatch("pages"), FieldLevelJudge(enabled=enabled)]
        with pytest.raises(EvaluatorConfigurationError) as raised:
            if operation == "create":
                client.benchmarks.create(name="QA", goal="Answer questions", evaluators=evaluators)
            else:
                benchmark = AgentGymBenchmark(client.benchmarks, {"id": BENCHMARK_ID})
                benchmark.configure(evaluators=evaluators)
        assert raised.value.code == "deprecated_evaluator_kind"
        assert raised.value.details["issues"][0]["path"] == ["evaluators", 1, "kind"]
        assert "can no longer be configured" in str(raised.value)
        request.assert_not_called()


@pytest.mark.parametrize("body", BODIES)
def test_api_exception_renders_validation_details_for_old_and_new_servers(body):
    error = _api_error(httpx.Response(400, json=body))
    assert CORRECTION in str(error)
    assert CORRECTION in error.message
    assert "evaluators" in error.message
    assert error.status_code == 400
    if body.get("details"):
        assert error.details == body["details"]
    else:
        assert error.details is not None
        assert error.details["issues"] == [ISSUE]


def test_api_retains_top_level_issues_alongside_details_and_prefers_code():
    error = _api_error(
        httpx.Response(
            400,
            json={
                "error": "Invalid body",
                "code": "duplicate_evaluator_kind",
                "details": {"validFieldPaths": ["answer"]},
                "issues": [ISSUE],
            },
        )
    )
    assert error.code == "duplicate_evaluator_kind"
    assert error.details == {"validFieldPaths": ["answer"], "issues": [ISSUE]}
    assert "evaluators[1].kind" in error.message


@pytest.mark.parametrize(
    "body",
    [{"error": "Not found"}, {"error": "bad", "details": {"issues": [None, 2], "fieldErrors": []}}],
)
def test_non_validation_and_malformed_details_remain_readable(body):
    error = _api_error(httpx.Response(400, json=body))
    assert error.message == body["error"]


def _run_apply(monkeypatch, tmp_path, payload, *, error_format="text", api_error=None):
    cli = importlib.import_module("promptic_sdk.cli.main")
    command = importlib.import_module("promptic_sdk.cli.commands.agent_gym")
    client = MagicMock()
    client.return_value.__enter__.return_value.benchmarks.create.side_effect = api_error
    monkeypatch.setattr(command, "AgentGymClient", client)
    config = tmp_path / "agent.json"
    config.write_text(json.dumps({"name": "QA", "goal": "Answer questions", **payload}))
    monkeypatch.setattr(
        sys, "argv", ["promptic", "--error-format", error_format, "agent-gym", "apply", str(config)]
    )
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 1
    return client


@pytest.mark.parametrize("error_format", ["text", "json"])
@pytest.mark.parametrize("duplicate", ["id", "kind", "name"])
def test_cli_local_duplicate_error_is_actionable_and_makes_no_client(
    monkeypatch, tmp_path, capsys, error_format, duplicate
):
    evaluators = {
        "id": [
            {"kind": "ai_judge", "id": "shared", "name": "Answer"},
            {"kind": "exact_match", "id": "shared", "name": "Pages"},
        ],
        "kind": [
            {"kind": "f1", "id": "first", "fieldPaths": ["label"]},
            {"kind": "f1", "id": "second", "fieldPaths": ["label"]},
        ],
        "name": [
            {"kind": "exact_match", "field": "answer", "name": "Answer"},
            {"kind": "ai_judge", "field": "answer", "name": "Answer"},
        ],
    }[duplicate]
    client = _run_apply(
        monkeypatch,
        tmp_path,
        {"evaluators": evaluators},
        error_format=error_format,
    )
    client.assert_not_called()
    output = capsys.readouterr()
    assert output.out == ""
    assert "Traceback" not in output.err
    if error_format == "json":
        body = json.loads(output.err)
        assert body["error"] == f"duplicate_evaluator_{duplicate}"
        assert body["status_code"] is None
        assert body["details"]["issues"][0]["path"] == ["evaluators", 1, duplicate]
        assert body["details"]["issues"][0]["firstIndex"] == 0
    else:
        assert f"evaluators[1].{duplicate}" in output.err
        assert {
            "id": "Choose a distinct id or omit it",
            "kind": "Only one evaluator of kind 'f1'",
            "name": "Give each targeted evaluator and code function a unique name",
        }[duplicate] in " ".join(output.err.split())


@pytest.mark.parametrize("body", BODIES)
@pytest.mark.parametrize("error_format", ["text", "json"])
def test_cli_remote_errors_include_correction(monkeypatch, tmp_path, capsys, body, error_format):
    error = _api_error(httpx.Response(400, json=body))
    _run_apply(monkeypatch, tmp_path, {}, error_format=error_format, api_error=error)
    output = capsys.readouterr()
    assert output.out == ""
    if error_format == "json":
        diagnostic = json.loads(output.err)
        assert diagnostic["status_code"] == 400
        assert diagnostic["details"] == error.details
        assert CORRECTION in diagnostic["message"]
    else:
        assert "Error (400):" in output.err
        assert CORRECTION in output.err


def test_cli_preserves_brackets_in_server_validation_text(monkeypatch, tmp_path, capsys):
    error = _api_error(
        httpx.Response(
            400,
            json={
                "error": "invalid_configuration",
                "message": "Keep [bold]answer[/bold] literal.",
            },
        )
    )
    _run_apply(monkeypatch, tmp_path, {}, api_error=error)
    assert "[bold]answer[/bold]" in capsys.readouterr().err


def test_explicit_error_code_preserves_human_error_without_message():
    error = _api_error(
        httpx.Response(
            400, json={"code": "invalid_judge_model", "error": "Choose an eligible judge model."}
        )
    )
    assert error.code == "invalid_judge_model"
    assert error.message == "Choose an eligible judge model."


@pytest.mark.parametrize("error_format", ["text", "json"])
def test_mixed_validation_formats_keep_all_distinct_corrections(
    monkeypatch, tmp_path, capsys, error_format
):
    details = {
        "issues": [ISSUE, {"message": "Choose a valid configuration."}],
        "fieldErrors": {
            "evaluators[1].kind": [CORRECTION],
            "name": ["Required", "Required"],
            "goal": ["Required"],
        },
        "formErrors": ["Choose a valid configuration.", "Provide at least one case."],
    }
    error = _api_error(httpx.Response(400, json={"error": "Invalid body", "details": details}))
    expected = [
        f"evaluators[1].kind: {CORRECTION}",
        "Choose a valid configuration.",
        "name: Required",
        "goal: Required",
        "Provide at least one case.",
    ]
    assert error.message.splitlines() == expected
    assert error.details == details
    _run_apply(monkeypatch, tmp_path, {}, error_format=error_format, api_error=error)
    output = capsys.readouterr()
    assert output.out == ""
    if error_format == "json":
        diagnostic = json.loads(output.err)
        assert diagnostic["message"].splitlines() == expected
        assert diagnostic["details"] == details
    else:
        for correction in expected:
            assert output.err.count(correction) == 1
