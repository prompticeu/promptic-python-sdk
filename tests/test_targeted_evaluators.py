"""Targeted evaluators: one output field, or the whole response, scored with one method."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from typer.testing import CliRunner

import promptic_sdk
from promptic_sdk import (
    AgentGymClient,
    AIJudge,
    ClassificationF1,
    ContainsText,
    ExactMatch,
    ExpectedBehaviorJudge,
    FieldLevelJudge,
    SemanticSimilarity,
    VerifierAgent,
    VerifierMetric,
)
from promptic_sdk.agent_gym._validation import EvaluatorConfigurationError
from promptic_sdk.agent_gym.authoring_models import MAX_TARGETED_EVALUATORS, validate_evaluators
from promptic_sdk.cli.commands.agent_gym import _evaluator
from promptic_sdk.cli.config import CliConfig
from promptic_sdk.cli.main import app

runner = CliRunner()

TARGETED = [
    (ExactMatch, "exact_match"),
    (ContainsText, "contains_text"),
    (SemanticSimilarity, "semantic_similarity"),
    (AIJudge, "ai_judge"),
]


# ── Typed Agent Gym models ───────────────────────────────────────────


@pytest.mark.parametrize(("model", "kind"), TARGETED)
def test_targeted_evaluator_defaults_score_the_whole_response(model, kind):
    assert model().as_request() == {"kind": kind, "weight": 1.0, "required": False}


@pytest.mark.parametrize(("model", "kind"), TARGETED)
def test_targeted_evaluator_serializes_every_common_setting(model, kind):
    evaluator = model(
        " user.name ",
        "Name check",
        "Checks the name.",
        2,
        True,
        0.5,
        id="name-check",
        enabled=False,
    )
    assert evaluator.field == "user.name"
    assert evaluator.as_request() == {
        "kind": kind,
        "id": "name-check",
        "enabled": False,
        "weight": 2,
        "required": True,
        "name": "Name check",
        "description": "Checks the name.",
        "threshold": 0.5,
        "field": "user.name",
    }


def test_ai_judge_sends_guidance_and_model():
    judge = AIJudge("answer", instructions="Amounts must match.", model="judge-model")
    assert judge.as_request() == {
        "kind": "ai_judge",
        "weight": 1.0,
        "required": False,
        "field": "answer",
        "instructions": "Amounts must match.",
        "model": "judge-model",
    }


@pytest.mark.parametrize("model", [ExactMatch, ContainsText, SemanticSimilarity])
def test_non_judge_targeted_evaluators_take_no_guidance(model):
    with pytest.raises(TypeError):
        model("answer", instructions="Be strict.")


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("", "non-empty dotted output path"),
        ("   ", "non-empty dotted output path"),
        ("people[].name", "points into list items"),
        ("items[0]", "points into list items"),
        ("user..name", "dotted path"),
        (".answer", "dotted path"),
    ],
)
@pytest.mark.parametrize(("model", "_kind"), TARGETED)
def test_targeted_evaluators_reject_malformed_fields(model, _kind, field, message):
    with pytest.raises(ValueError, match=message):
        model(field)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"weight": 0}, "weight must be positive"),
        ({"threshold": 1.5}, "threshold must be between 0 and 1"),
        ({"id": ""}, "id must be a non-empty string"),
        ({"enabled": "no"}, "enabled must be a boolean"),
    ],
)
@pytest.mark.parametrize(("model", "_kind"), TARGETED)
def test_targeted_evaluators_validate_common_settings(model, _kind, kwargs, message):
    with pytest.raises(ValueError, match=message):
        model(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"instructions": " "}, "instructions cannot be empty"),
        ({"model": ""}, "model must be a non-empty string"),
    ],
)
def test_ai_judge_rejects_blank_guidance_and_model(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AIJudge(**kwargs)


@pytest.mark.filterwarnings("ignore:FieldLevelJudge:DeprecationWarning")
def test_existing_evaluators_serialize_enabled_only_when_disabled():
    verifier_metrics = (VerifierMetric("quality", "Quality", "Check quality."),)
    evaluators = [
        ClassificationF1(("label",)),
        FieldLevelJudge(),
        VerifierAgent(instructions="Inspect.", metrics=verifier_metrics),
        ExpectedBehaviorJudge(),
    ]
    for evaluator in evaluators:
        assert "enabled" not in evaluator.as_request()
    disabled = [
        ClassificationF1(("label",), enabled=False),
        FieldLevelJudge(enabled=False),
        VerifierAgent(instructions="Inspect.", metrics=verifier_metrics, enabled=False),
        ExpectedBehaviorJudge(enabled=False),
    ]
    for evaluator in disabled:
        assert evaluator.as_request()["enabled"] is False


def test_targeted_models_are_exported_from_both_packages():
    for name in ("ExactMatch", "ContainsText", "SemanticSimilarity", "AIJudge"):
        assert name in promptic_sdk.__all__
        assert name in promptic_sdk.agent_gym.__all__
    for name in ("TargetedEvaluatorConfig", "TargetedEvaluatorType"):
        assert name in promptic_sdk.__all__


# ── Local uniqueness rules ───────────────────────────────────────────


def test_targeted_kinds_repeat_on_the_same_and_different_fields():
    validate_evaluators(
        [
            ExactMatch("count"),
            ExactMatch("answer"),
            ExactMatch("answer", "Strict answer"),
            AIJudge("answer"),
            AIJudge("answer", "Answer judged"),
            ContainsText(),
            ContainsText("source"),
            SemanticSimilarity(),
            SemanticSimilarity("answer"),
            ClassificationF1(("label",)),
        ]
    )


def test_targeted_names_must_be_unique_across_targeted_kinds():
    with pytest.raises(EvaluatorConfigurationError) as raised:
        validate_evaluators(
            [
                ExactMatch("answer", "Answer"),
                AIJudge("answer", " Answer "),
            ]
        )
    assert raised.value.code == "duplicate_evaluator_name"
    [issue] = raised.value.details["issues"]
    assert issue["path"] == ["evaluators", 1, "name"]
    assert issue["firstIndex"] == 0
    assert issue["kind"] == "ai_judge"
    assert "evaluators[1].name" in str(raised.value)


def test_names_are_only_unique_among_targeted_evaluators():
    validate_evaluators(
        [
            ClassificationF1(("label",), "Answer"),
            ExpectedBehaviorJudge("Answer"),
            ExactMatch("answer", "Answer"),
        ]
    )


def test_ids_stay_unique_across_targeted_and_fixed_kinds():
    with pytest.raises(EvaluatorConfigurationError) as raised:
        validate_evaluators([ClassificationF1(("label",), id="check"), ExactMatch(id="check")])
    assert raised.value.code == "duplicate_evaluator_id"
    assert raised.value.details["issues"][0]["path"] == ["evaluators", 1, "id"]


def test_singleton_kinds_keep_one_slot_including_disabled():
    with pytest.raises(EvaluatorConfigurationError) as raised:
        validate_evaluators(
            [ClassificationF1(("label",), enabled=False), ClassificationF1(("other",))]
        )
    assert raised.value.code == "duplicate_evaluator_kind"


def test_targeted_evaluator_limit():
    validate_evaluators([ExactMatch(f"field_{index}") for index in range(MAX_TARGETED_EVALUATORS)])
    with pytest.raises(EvaluatorConfigurationError) as raised:
        validate_evaluators(
            [
                ClassificationF1(("label",)),
                *(ExactMatch(f"field_{index}") for index in range(MAX_TARGETED_EVALUATORS + 1)),
            ]
        )
    assert raised.value.code == "too_many_targeted_evaluators"
    assert f"At most {MAX_TARGETED_EVALUATORS}" in str(raised.value)


# ── Agent Gym CLI configuration files ────────────────────────────────


@pytest.mark.parametrize(("model", "kind"), TARGETED)
def test_cli_builds_targeted_evaluators(model, kind):
    value = {
        "kind": kind,
        "id": "check",
        "name": "Check",
        "field": "answer",
        "weight": 2,
        "enabled": False,
    }
    evaluator = _evaluator(value)
    assert isinstance(evaluator, model)
    assert evaluator.as_request() == {**value, "required": False}


def test_cli_builds_ai_judge_with_guidance_and_model():
    value = {"kind": "ai_judge", "instructions": "Amounts must match.", "model": "judge-model"}
    assert _evaluator(value).as_request() == {
        **value,
        "weight": 1.0,
        "required": False,
    }


@pytest.mark.parametrize("kind", ["exact_match", "contains_text", "semantic_similarity"])
@pytest.mark.parametrize("key", ["instructions", "model", "fields", "fieldPaths", "rubric"])
def test_cli_rejects_settings_targeted_kinds_do_not_accept(kind, key):
    with pytest.raises(ValueError, match=f"unsupported {kind} field"):
        _evaluator({"kind": kind, key: {}})


@pytest.mark.parametrize("key", ["fields", "evidence", "metrics", "budget"])
def test_cli_rejects_verifier_settings_on_ai_judge(key):
    with pytest.raises(ValueError, match="unsupported ai_judge field"):
        _evaluator({"kind": "ai_judge", key: {}})


def test_cli_rejects_non_boolean_enabled():
    with pytest.raises(ValueError, match="enabled must be true or false"):
        _evaluator({"kind": "exact_match", "enabled": "yes"})


def test_cli_passes_enabled_to_existing_kinds():
    evaluator = _evaluator({"kind": "f1", "fieldPaths": ["label"], "enabled": False})
    assert evaluator.as_request()["enabled"] is False


def test_apply_sends_targeted_evaluators(tmp_path):
    evaluators = [
        {"kind": "exact_match", "field": "invoice_id"},
        {"kind": "exact_match", "field": "total", "name": "Total"},
        {"kind": "ai_judge", "field": "answer", "instructions": "Accept paraphrases."},
        {"kind": "contains_text", "field": "source", "enabled": False},
        {"kind": "semantic_similarity"},
    ]
    config = tmp_path / "agent.json"
    config.write_text(json.dumps({"name": "QA", "goal": "Answer", "evaluators": evaluators}))
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(201, json={"id": "benchmark-id"})

    client = AgentGymClient(api_key="ptc_test")
    client._transport.api.close()
    client._transport.api = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://promptic.test/api/v1"
    )
    with patch("promptic_sdk.cli.commands.agent_gym.AgentGymClient", return_value=client):
        result = runner.invoke(app, ["agent-gym", "apply", str(config)])

    assert result.exit_code == 0, result.output
    assert json.loads(requests[0].content)["evaluators"] == [
        {**evaluator, "weight": 1.0, "required": False} for evaluator in evaluators
    ]


def test_targeted_checks_example_is_valid():
    with open("examples/agent_gym_targeted_checks.json") as handle:
        payload = json.load(handle)
    evaluators = [_evaluator(item) for item in payload["evaluators"]]
    validate_evaluators(evaluators)
    assert {evaluator.kind for evaluator in evaluators} >= {
        "exact_match",
        "contains_text",
        "semantic_similarity",
        "ai_judge",
    }


# ── Prompt Optimization CLI ──────────────────────────────────────────


def _mock_client():
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.create_evaluators.return_value = {"data": [{"id": "evaluator-id"}]}
    client.update_evaluator.return_value = {"id": "evaluator-id", "enabled": False}
    config = CliConfig(endpoint="https://test.com", api_key="pk_test")
    return (
        patch("promptic_sdk.cli.load_config", return_value=config),
        patch("promptic_sdk.cli.PrompticClient", return_value=client),
        client,
    )


def _add(*args):
    config_patch, client_patch, client = _mock_client()
    with config_patch, client_patch:
        result = runner.invoke(
            app, ["evaluators", "add", "experiment-id", "--name", "Check", *args]
        )
    return result, client


def test_cli_adds_targeted_evaluator_for_a_field():
    result, client = _add("--type", "exactMatch", "--field", " invoice.id ")
    assert result.exit_code == 0, result.output
    client.create_evaluators.assert_called_once_with(
        "experiment-id",
        [
            {
                "name": "Check",
                "type": "exactMatch",
                "scaleMin": 0,
                "scaleMax": 1,
                "weight": 1,
                "config": {"field": "invoice.id"},
            }
        ],
    )


def test_cli_adds_ai_judge_with_guidance_on_the_whole_response():
    result, client = _add("--type", "aiJudge", "--instructions", "Amounts must match.")
    assert result.exit_code == 0, result.output
    [evaluator] = client.create_evaluators.call_args.args[1]
    assert evaluator["config"] == {"instructions": "Amounts must match."}


def test_cli_adds_disabled_evaluator_without_config():
    result, client = _add("--type", "containsText", "--disabled")
    assert result.exit_code == 0, result.output
    [evaluator] = client.create_evaluators.call_args.args[1]
    assert evaluator["enabled"] is False
    assert "config" not in evaluator


def test_cli_keeps_legacy_types_unchanged():
    result, client = _add("--type", "structuredOutput")
    assert result.exit_code == 0, result.output
    [evaluator] = client.create_evaluators.call_args.args[1]
    assert evaluator == {
        "name": "Check",
        "type": "structuredOutput",
        "scaleMin": 0,
        "scaleMax": 1,
        "weight": 1,
    }


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--type", "generalJudge", "--field", "answer"], "only supported for"),
        (["--type", "similarity", "--instructions", "x"], "only supported for"),
        (["--type", "exactMatch", "--instructions", "x"], "only aiJudge accepts"),
        (["--type", "aiJudge", "--instructions", " "], "cannot be empty"),
        (["--type", "containsText", "--field", "people[].name"], "points into list items"),
        (["--type", "exactMatch", "--field", "a..b"], "dotted path"),
    ],
)
def test_cli_rejects_invalid_targeted_options_before_any_request(args, message):
    result, client = _add(*args)
    assert result.exit_code == 2
    assert message in " ".join(result.output.split())
    client.create_evaluators.assert_not_called()


@pytest.mark.parametrize(("command", "enabled"), [("enable", True), ("disable", False)])
def test_cli_switches_evaluators_on_and_off(command, enabled):
    config_patch, client_patch, client = _mock_client()
    with config_patch, client_patch:
        result = runner.invoke(app, ["evaluators", command, "experiment-id", "evaluator-id"])
    assert result.exit_code == 0, result.output
    client.update_evaluator.assert_called_once_with(
        "experiment-id", "evaluator-id", enabled=enabled
    )
    assert "evaluator-id" in result.output


def test_cli_lists_target_and_enabled_state():
    config_patch, client_patch, client = _mock_client()
    base = {"scaleMin": 0, "scaleMax": 1, "weight": 1}
    client.list_evaluators.return_value = {
        "data": [
            {**base, "id": "e1", "name": "Id", "type": "exactMatch", "config": {"field": "id"}},
            {**base, "id": "e2", "name": "Whole", "type": "aiJudge", "config": None},
            {**base, "id": "e3", "name": "Old", "type": "generalJudge", "enabled": False},
        ]
    }
    with config_patch, client_patch:
        result = runner.invoke(app, ["evaluators", "list", "experiment-id"])
    assert result.exit_code == 0, result.output
    lines = {line.split()[1]: line for line in result.output.splitlines() if "│ e" in line}
    assert "id" in lines["e1"].split("│")[4] and "yes" in lines["e1"]
    assert "whole response" in lines["e2"]
    assert "no" in lines["e3"].split("│")[-2]
