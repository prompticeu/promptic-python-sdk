"""Evaluator commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import httpx
import typer
from rich.console import Console
from rich.table import Table

from promptic_sdk._evaluator_fields import TARGETED_EVALUATOR_TYPES, validate_target_field
from promptic_sdk.cli import get_client
from promptic_sdk.client import code_function_test_body
from promptic_sdk.models import (
    AgentGymCodeFunctionTestRow,
    CodeFunctionConfig,
    CodeFunctionRowContract,
    CodeFunctionTestRow,
)

evaluators_app = typer.Typer(help="Manage evaluators.")
console = Console()


def _target_label(evaluator: Any) -> str:
    """Name the field a targeted evaluator scores; other types have no single target."""
    if evaluator["type"] not in TARGETED_EVALUATOR_TYPES:
        return ""
    config = evaluator.get("config") or {}
    return config.get("field") or "whole response"


@evaluators_app.command("list")
def list_evaluators(
    experiment_id: str = typer.Argument(help="Experiment ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List evaluators for an experiment."""
    with get_client() as client:
        result = client.list_evaluators(experiment_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    items = result["data"]
    if not items:
        console.print("No evaluators found.", style="dim")
        return

    table = Table(title=f"Evaluators ({len(items)})")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Target")
    table.add_column("Scale")
    table.add_column("Weight", justify="right")
    table.add_column("Enabled")

    for e in items:
        scale = f"{e['scaleMin']}-{e['scaleMax']}"
        table.add_row(
            e["id"],
            e["name"],
            e["type"],
            _target_label(e),
            scale,
            str(e["weight"]),
            "yes" if e.get("enabled", True) else "no",
        )

    console.print(table)


@evaluators_app.command("add")
def add_evaluator(
    experiment_id: str = typer.Argument(help="Experiment ID."),
    name: str = typer.Option(..., "--name", "-n", help="Evaluator name."),
    eval_type: str = typer.Option(
        ...,
        "--type",
        "-t",
        help=(
            "Type. Targeted checks on one field or the whole response: exactMatch, "
            "containsText, semanticSimilarity (Embedding Similarity), aiJudge. "
            "Judges: referenceJudge (Deviation Judge), comparisonJudge, generalJudge. "
            "Also codeFunction (Python/JavaScript), f1 (classification), similarity "
            "and structuredOutput (existing setups)."
        ),
    ),
    target_field: str | None = typer.Option(
        None,
        "--field",
        help=(
            "Targeted types only: dotted output path to score, such as 'answer' or "
            "'user.name'. Omit to score the whole response."
        ),
    ),
    instructions: str | None = typer.Option(
        None,
        "--instructions",
        help="aiJudge only: guidance added to the built-in equivalence rubric.",
    ),
    language: str | None = typer.Option(
        None, "--language", help="codeFunction only: python or javascript."
    ),
    source_file: Path | None = typer.Option(
        None,
        "--source-file",
        help="codeFunction only: UTF-8 file defining score(row), at most 64 KiB.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    scale_min: float = typer.Option(0, "--scale-min", help="Scale minimum."),
    scale_max: float = typer.Option(1, "--scale-max", help="Scale maximum."),
    weight: float = typer.Option(1, "--weight", "-w", help="Weight."),
    enabled: bool = typer.Option(
        True, "--enabled/--disabled", help="Disabled evaluators are kept but not run."
    ),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Add an evaluator to an experiment."""
    evaluator: dict[str, Any] = {
        "name": name,
        "type": eval_type,
        "scaleMin": scale_min,
        "scaleMax": scale_max,
        "weight": weight,
    }
    if not enabled:
        evaluator["enabled"] = False
    config = _targeted_config(eval_type, target_field, instructions)
    code_config = _code_function_config(eval_type, language, source_file)
    if code_config is not None:
        config = code_config
    if config is not None:
        evaluator["config"] = config

    with get_client() as client:
        result = client.create_evaluators(experiment_id, [evaluator])

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    created = result["data"]
    if created:
        console.print(f"[green]Evaluator created:[/green] {created[0]['id']}")
    else:
        console.print("[green]Evaluator created.[/green]")


def _code_function_config(
    eval_type: str, language: str | None, source_file: Path | None
) -> dict[str, str] | None:
    """Read source as data, never execute or import it in the CLI process."""
    if eval_type != "codeFunction":
        if language is not None or source_file is not None:
            raise typer.BadParameter(
                "--language and --source-file require --type codeFunction", param_hint="--type"
            )
        return None
    if language not in {"python", "javascript"}:
        raise typer.BadParameter("choose python or javascript", param_hint="--language")
    if source_file is None:
        raise typer.BadParameter("required for codeFunction", param_hint="--source-file")
    try:
        with source_file.open("rb") as handle:
            raw = handle.read(64 * 1024 + 1)
        if len(raw) > 64 * 1024:
            raise typer.BadParameter("must be at most 64 KiB", param_hint="--source-file")
        source = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise typer.BadParameter(
            "must be a readable UTF-8 file", param_hint="--source-file"
        ) from exc
    if not source.strip():
        raise typer.BadParameter("cannot be empty", param_hint="--source-file")
    return {"language": language, "source": source, "function_name": "score"}


@evaluators_app.command("test")
def test_code_evaluator(
    language: str = typer.Option(..., "--language", help="python or javascript."),
    source_file: Path = typer.Option(
        ..., "--source-file", exists=True, dir_okay=False, readable=True
    ),
    row_file: Path = typer.Option(..., "--row-file", exists=True, dir_okay=False, readable=True),
    row_contract: str = typer.Option(
        "experiment",
        "--row-contract",
        help=(
            "experiment (input_variables, expected, predicted) or agent_gym "
            "(inputs, output, expected_output, expected_behavior)."
        ),
    ),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Queue a sandbox probe with an explicit sample output; usage may be billed."""
    if row_contract not in {"experiment", "agent_gym"}:
        raise typer.BadParameter("choose experiment or agent_gym", param_hint="--row-contract")
    config = _code_function_config("codeFunction", language, source_file)
    try:
        with row_file.open("rb") as handle:
            raw = handle.read(64 * 1024 + 1)
        if len(raw) > 64 * 1024:
            raise typer.BadParameter("must be at most 64 KiB", param_hint="--row-file")
        row = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise typer.BadParameter(
            "must be a readable UTF-8 JSON object", param_hint="--row-file"
        ) from exc
    if row_contract == "agent_gym":
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("inputs"), dict)
            or "output" not in row
            or not isinstance(row.get("expected_behavior"), str | None)
        ):
            raise typer.BadParameter(
                "requires inputs (object) and output; expected_behavior must be a string",
                param_hint="--row-file",
            )
    elif (
        not isinstance(row, dict)
        or not isinstance(row.get("input_variables"), dict)
        or not isinstance(row.get("expected"), str)
        or not isinstance(row.get("predicted"), str)
    ):
        raise typer.BadParameter(
            "requires input_variables (object), expected and predicted (strings)",
            param_hint="--row-file",
        )
    contract = cast(CodeFunctionRowContract, row_contract)
    typed_config = cast(CodeFunctionConfig, config)
    typed_row = cast(CodeFunctionTestRow | AgentGymCodeFunctionTestRow, row)
    payload = httpx.Request(
        "POST", "https://localhost", json=code_function_test_body(typed_config, typed_row, contract)
    )
    if len(payload.content) > 128 * 1024:
        raise typer.BadParameter(
            "combined JSON request must be at most 128 KiB, including escaping and envelope",
            param_hint="--source-file / --row-file",
        )
    with get_client() as client:
        result = client.test_code_evaluator(typed_config, typed_row, row_contract=contract)
    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        console.print(f"Function test queued: {result['id']}. Use evaluators test-result to poll.")


@evaluators_app.command("test-result")
def get_code_evaluator_test(
    test_id: str = typer.Argument(help="Function test ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Read a function probe's status, score/reason, and error diagnostics."""
    with get_client() as client:
        result = client.get_code_evaluator_test(test_id)
    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        console.print_json(data=result)


def _targeted_config(
    eval_type: str, target_field: str | None, instructions: str | None
) -> dict[str, str] | None:
    """Build a targeted evaluator config, rejecting options the type does not accept."""
    if eval_type not in TARGETED_EVALUATOR_TYPES:
        if target_field is not None or instructions is not None:
            raise typer.BadParameter(
                "--field and --instructions are only supported for "
                + ", ".join(TARGETED_EVALUATOR_TYPES),
                param_hint="--type",
            )
        return None
    if instructions is not None and eval_type != "aiJudge":
        raise typer.BadParameter("only aiJudge accepts --instructions", param_hint="--instructions")
    if instructions is not None and not instructions.strip():
        raise typer.BadParameter("cannot be empty", param_hint="--instructions")
    try:
        path = validate_target_field(target_field)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--field") from exc
    config: dict[str, str] = {}
    if path is not None:
        config["field"] = path
    if instructions is not None:
        config["instructions"] = instructions
    return config or None


def _set_enabled(experiment_id: str, evaluator_id: str, enabled: bool, output_json: bool) -> None:
    with get_client() as client:
        result = client.update_evaluator(experiment_id, evaluator_id, enabled=enabled)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    state = "enabled" if enabled else "disabled"
    console.print(f"[green]Evaluator {state}:[/green] {evaluator_id}")


@evaluators_app.command("enable")
def enable_evaluator(
    experiment_id: str = typer.Argument(help="Experiment ID."),
    evaluator_id: str = typer.Argument(help="Evaluator ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Switch an evaluator on so it runs and counts toward the score."""
    _set_enabled(experiment_id, evaluator_id, True, output_json)


@evaluators_app.command("disable")
def disable_evaluator(
    experiment_id: str = typer.Argument(help="Experiment ID."),
    evaluator_id: str = typer.Argument(help="Evaluator ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Switch an evaluator off: it keeps its configuration but is not run or scored."""
    _set_enabled(experiment_id, evaluator_id, False, output_json)


@evaluators_app.command("delete")
def delete_evaluator(
    experiment_id: str = typer.Argument(help="Experiment ID."),
    evaluator_id: str = typer.Argument(help="Evaluator ID."),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation."),
) -> None:
    """Delete an evaluator."""
    if not force:
        typer.confirm(f"Delete evaluator {evaluator_id}?", abort=True)

    with get_client() as client:
        client.delete_evaluator(experiment_id, evaluator_id)

    console.print("[green]Evaluator deleted.[/green]")
