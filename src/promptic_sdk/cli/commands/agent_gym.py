"""Agent component setup commands for coding agents and CI."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, cast

import typer
from rich.console import Console

from promptic_sdk.agent_gym import (
    AgentEvaluator,
    AgentGymClient,
    BenchmarkCase,
    BenchmarkFile,
    ClassificationF1,
    EvidenceKind,
    EvidencePolicy,
    ExpectedBehaviorJudge,
    FieldLevelJudge,
    FieldScoring,
    InvestigationBudget,
    MetricBinding,
    VerifierAgent,
    VerifierMetric,
)

agent_gym_app = typer.Typer(help="Configure Agent Optimization components and revisions.")
console = Console()

_VERIFIER_EVIDENCE_FIELDS = (
    (EvidenceKind.CASE_INPUT, "caseInputs"),
    (EvidenceKind.SUBMITTED_OUTPUT, "submittedOutput"),
    (EvidenceKind.EXPECTED_BEHAVIOR, "expectedBehavior"),
    (EvidenceKind.EXPECTED_OUTPUT, "expectedOutput"),
    (EvidenceKind.EXECUTION_TRACE, "executionTrace"),
)
_COMMON_EVALUATOR_KEYS = {"kind", "name", "description", "weight", "required", "threshold"}


def _files(value: Any, base: Path) -> Any:
    if isinstance(value, dict):
        if set(value) == {"$file"} and isinstance(value["$file"], str):
            source = (base / value["$file"]).resolve()
            return BenchmarkFile(source, path=value["$file"])
        return {key: _files(child, base) for key, child in value.items()}
    if isinstance(value, list):
        return [_files(child, base) for child in value]
    return value


def _verifier_evidence(value: Any) -> EvidencePolicy:
    if value is None:
        return EvidencePolicy()
    if not isinstance(value, dict):
        raise ValueError("verifier evidence must be a JSON object")
    allowed = {key for _, key in _VERIFIER_EVIDENCE_FIELDS}
    if unknown := set(value) - allowed:
        raise ValueError(f"unsupported verifier evidence field(s): {', '.join(sorted(unknown))}")
    defaults = EvidencePolicy().selected
    selected: list[EvidenceKind] = []
    for kind, key in _VERIFIER_EVIDENCE_FIELDS:
        enabled = value.get(key, kind in defaults)
        if not isinstance(enabled, bool):
            raise ValueError(f"verifier evidence {key} must be a boolean")
        if enabled:
            selected.append(kind)
    return EvidencePolicy(selected=selected)


def _investigation_budget(value: Any) -> InvestigationBudget:
    if value is None:
        return InvestigationBudget()
    if not isinstance(value, dict):
        raise ValueError("verifier budget must be a JSON object")
    if unknown := set(value) - {"maxSteps"}:
        raise ValueError(f"unsupported verifier budget field(s): {', '.join(sorted(unknown))}")
    return InvestigationBudget(max_steps=value.get("maxSteps", 20))


def _verifier_metrics(value: Any) -> tuple[VerifierMetric, ...]:
    if not isinstance(value, list):
        raise ValueError("verifier metrics must be a JSON array")
    metrics: list[VerifierMetric] = []
    for item in value:
        required = {"key", "name", "instructions"}
        allowed = required | {"weight", "threshold"}
        if not isinstance(item, dict) or not required <= set(item) or set(item) - allowed:
            raise ValueError(
                "each verifier metric requires key, name, and instructions and may include "
                "weight and threshold"
            )
        metrics.append(
            VerifierMetric(
                item["key"],
                item["name"],
                item["instructions"],
                weight=item.get("weight", 1),
                threshold=item.get("threshold"),
            )
        )
    return tuple(metrics)


def _metric_bindings(value: Any) -> dict[str, MetricBinding]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("metricBindings must be a JSON object")
    bindings: dict[str, MetricBinding] = {}
    for key, configuration in value.items():
        if not isinstance(configuration, dict):
            raise ValueError(f"metric binding {key} must be a JSON object")
        if unknown := set(configuration) - {"weight", "threshold"}:
            raise ValueError(
                f"unsupported metric binding field(s) for {key}: {', '.join(sorted(unknown))}"
            )
        bindings[key] = MetricBinding(
            weight=configuration.get("weight", 1),
            threshold=configuration.get("threshold"),
        )
    return bindings


def _evaluator(value: dict[str, Any]) -> AgentEvaluator:
    fields = {}
    for path, configuration in value.get("fields", {}).items():
        if not isinstance(configuration, dict):
            fields[path] = configuration
            continue
        allowed = {"method", "include", "judgeInstructions", "judge_instructions", "weight"}
        if unknown := set(configuration) - allowed:
            raise ValueError(f"unsupported field scoring key(s): {', '.join(sorted(unknown))}")
        fields[path] = FieldScoring(
            method=configuration.get("method", "exact"),
            include=bool(configuration.get("include", True)),
            judge_instructions=configuration.get(
                "judgeInstructions", configuration.get("judge_instructions")
            ),
            weight=float(configuration.get("weight", 1)),
        )
    kind = value["kind"]
    name = cast(str | None, value.get("name"))
    description = cast(str | None, value.get("description"))
    weight = float(value.get("weight", 1))
    required = bool(value.get("required", False))
    threshold = cast(float | None, value.get("threshold"))
    if kind == "f1":
        return ClassificationF1(
            tuple(value.get("fieldPaths", ())), name, description, weight, required, threshold
        )
    if kind == "field_level_judge":
        if "rubric" in value or "instructions" in value:
            raise ValueError(
                "field_level_judge does not support shared instructions; configure "
                "judgeInstructions on each judged field"
            )
        return FieldLevelJudge(
            fields, value.get("model"), name, description, weight, required, threshold
        )
    if kind == "verifier_agent":
        allowed = _COMMON_EVALUATOR_KEYS | {
            "instructions",
            "model",
            "evidence",
            "metrics",
            "metricBindings",
            "budget",
        }
        if unknown := set(value) - allowed:
            raise ValueError(f"unsupported verifier_agent field(s): {', '.join(sorted(unknown))}")
        return VerifierAgent(
            instructions=value.get("instructions"),
            model=value.get("model"),
            name=name,
            description=description,
            weight=weight,
            required=required,
            threshold=threshold,
            evidence=_verifier_evidence(value.get("evidence")),
            metrics=_verifier_metrics(value.get("metrics")),
            metric_bindings=_metric_bindings(value.get("metricBindings")),
            budget=_investigation_budget(value.get("budget")),
        )
    if kind == "expected_behavior_judge":
        allowed = _COMMON_EVALUATOR_KEYS | {"model", "metricBindings"}
        if unknown := set(value) - allowed:
            raise ValueError(
                "expected_behavior_judge is locked; unsupported field(s): "
                f"{', '.join(sorted(unknown))}"
            )
        return ExpectedBehaviorJudge(
            model=value.get("model"),
            metric_bindings=_metric_bindings(value.get("metricBindings")),
            name=name,
            description=description,
            weight=weight,
            required=required,
            threshold=threshold,
        )
    raise ValueError(f"unsupported evaluator kind: {kind}")


def _callback(reference: str) -> Any:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("callback must use module:function syntax")
    callback = getattr(importlib.import_module(module_name), attribute)
    if not callable(callback):
        raise TypeError(f"{reference} is not callable")
    return callback


@agent_gym_app.command("apply")
def apply_agent_config(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
) -> None:
    """Create and fully configure an Agent component from one JSON file."""
    payload = json.loads(config.read_text())
    evaluators = [_evaluator(item) for item in payload.get("evaluators", [])]
    with AgentGymClient(ai_application_id=payload.get("aiApplicationId")) as client:
        agent = client.benchmarks.create(
            name=payload["name"],
            goal=payload["goal"],
            description=payload.get("description"),
            input_schema=payload.get("inputSchema"),
            output_schema=payload.get("outputSchema"),
            evaluators=evaluators,
        )
        cases = [
            BenchmarkCase(
                title=item.get("title"),
                input=_files(item.get("input", {}), config.parent),
                output=_files(item["output"], config.parent) if "output" in item else None,
                expected_behavior=item.get("expectedBehavior"),
            )
            for item in payload.get("cases", [])
        ]
        if cases:
            result = agent.cases.add_many(cases)
            if result["failed"]:
                raise RuntimeError(f"{result['failed']} test case(s) failed: {result['results']}")
            agent.publish()
            agent.refresh()
    console.print(f"[green]Agent configured:[/green] {agent.id}")
    console.print(f"Ready for submission: {'yes' if agent.ready_for_submission else 'no'}")
    if agent.data.get("activeRevisionId"):
        console.print(f"Active revision: {agent.data['activeRevisionId']}")


@agent_gym_app.command("status")
def agent_status(agent_id: str = typer.Argument(..., help="Agent component benchmark ID.")) -> None:
    """Show external-submission readiness and recommended evaluator configuration."""
    with AgentGymClient() as client:
        agent = client.benchmarks.get(agent_id)
    console.print_json(data=agent.data)


@agent_gym_app.command("revisions")
def agent_revisions(
    agent_id: str = typer.Argument(..., help="Agent component benchmark ID."),
) -> None:
    """List published benchmark revisions and active draft-fingerprint status."""
    with AgentGymClient() as client:
        data = client.benchmarks.get(agent_id).revisions()
    console.print_json(data=data)


@agent_gym_app.command("dataset-pull")
def pull_agent_dataset(
    agent_id: str = typer.Argument(..., help="Agent component benchmark ID."),
    output: Path = typer.Option(..., "--output", "-o", help="Persistent private destination."),
    revision_id: str | None = typer.Option(None, "--revision", help="Immutable revision ID."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace differing local files."),
) -> None:
    """Download public frozen inputs without running an agent."""
    with AgentGymClient() as client:
        dataset = client.download_dataset(
            agent_id, output, revision_id=revision_id, overwrite=overwrite
        )
    console.print(f"[green]Dataset downloaded:[/green] {dataset.root}")
    console.print(f"Revision: {dataset.revision['id']}")
    console.print(f"Cases: {len(dataset.cases)}")


@agent_gym_app.command("draft")
def agent_revision_draft(
    agent_id: str = typer.Argument(..., help="Agent component benchmark ID."),
) -> None:
    """Show the persisted schema-migration draft and grouped conflicts."""
    with AgentGymClient() as client:
        draft = client.benchmarks.get(agent_id).revision_draft()
    console.print_json(data={"draft": draft})


@agent_gym_app.command("resolve-draft")
def resolve_agent_revision_draft(
    agent_id: str = typer.Argument(..., help="Agent component benchmark ID."),
    field: str = typer.Argument(..., help="Conflict key, for example output.name."),
    kind: str = typer.Option(
        ...,
        "--kind",
        help="Resolution: make_optional, default_value, or edit_cases.",
    ),
    value: str | None = typer.Option(
        None,
        "--value",
        help="JSON value required by default_value (for example '\"unknown\"').",
    ),
) -> None:
    """Resolve one persisted schema-migration conflict."""
    kwargs: dict[str, Any] = {"kind": kind}
    if value is not None:
        kwargs["value"] = json.loads(value)
    with AgentGymClient() as client:
        draft = client.benchmarks.get(agent_id).resolve_revision_draft(field, **kwargs)
    console.print_json(data={"draft": draft})


@agent_gym_app.command("abandon-draft")
def abandon_agent_revision_draft(
    agent_id: str = typer.Argument(..., help="Agent component benchmark ID."),
    yes: bool = typer.Option(False, "--yes", help="Confirm discarding the persisted draft."),
) -> None:
    """Discard a persisted schema-migration draft without changing the active revision."""
    if not yes:
        raise typer.BadParameter("pass --yes to confirm abandoning the revision draft")
    with AgentGymClient() as client:
        client.benchmarks.get(agent_id).abandon_revision_draft()
    console.print("[green]Revision draft abandoned.[/green]")


@agent_gym_app.command("publish-draft")
def publish_agent_revision_draft(
    agent_id: str = typer.Argument(..., help="Agent component benchmark ID."),
) -> None:
    """Activate a ready benchmark draft as one immutable revision."""
    with AgentGymClient() as client:
        result = client.benchmarks.get(agent_id).publish()
    console.print(f"[green]Active revision:[/green] {result.revision.id}")


@agent_gym_app.command("reevaluate")
def reevaluate_agent_run(
    agent_id: str = typer.Argument(..., help="Agent component benchmark ID."),
    run_id: str = typer.Argument(..., help="Execution-current benchmark run ID."),
) -> None:
    """Re-score a variant run after evaluator-only benchmark changes."""
    with AgentGymClient() as client:
        result = client.reevaluate_run(agent_id, run_id)
    console.print(
        f"[green]Re-evaluation queued:[/green] {result['run_id']} "
        f"(operation {result['evaluation_run_id']})"
    )


@agent_gym_app.command("run")
def run_agent(
    agent_id: str = typer.Argument(..., help="Agent component benchmark ID."),
    callback: str = typer.Argument(..., help="Trusted Python callback as module:function."),
    name: str = typer.Option(..., "--name", help="Candidate architecture name."),
    version: str = typer.Option(..., "--version", help="Candidate architecture version."),
    architecture: str = typer.Option(
        ..., "--architecture", help="Architecture description or path to a Markdown file."
    ),
) -> None:
    """Execute a trusted local Agent and submit its predictions and traces."""
    description_path = Path(architecture)
    description = description_path.read_text() if description_path.is_file() else architecture
    with AgentGymClient() as client:
        result = client.run_and_submit(
            agent_id,
            _callback(callback),
            name=name,
            version=version,
            architecture_description=description,
        )
    console.print(f"[green]Run submitted:[/green] {result.run_id}")
