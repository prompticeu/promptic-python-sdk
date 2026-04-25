"""Evaluator commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from promptic_sdk.cli import get_client

evaluators_app = typer.Typer(help="Manage evaluators.")
console = Console()
err_console = Console(stderr=True)


def _resolve_config(instructions: str | None, config_file: Path | None) -> dict[str, Any] | None:
    """Translate the --instructions / --config-file flags into a config dict.

    Mutually exclusive: ``--instructions`` is shorthand for
    ``{"instructions": <text>}`` (the shape used by comparison- and reference-
    judge evaluators). ``--config-file`` reads arbitrary JSON for evaluators
    that need a richer shape (e.g. ``generalJudge`` with ``messages``).
    Returns ``None`` when neither flag is set so the caller can omit the key.
    """
    if instructions is not None and config_file is not None:
        err_console.print(
            "[red]Error:[/red] --instructions and --config-file are mutually exclusive.",
        )
        raise typer.Exit(2)
    if instructions is not None:
        return {"instructions": instructions}
    if config_file is not None:
        try:
            with config_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            err_console.print(f"[red]Failed to load --config-file:[/red] {exc}")
            raise typer.Exit(2) from exc
        if not isinstance(data, dict):
            err_console.print("[red]--config-file must contain a JSON object.[/red]")
            raise typer.Exit(2)
        return data
    return None


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
    table.add_column("Scale")
    table.add_column("Weight", justify="right")

    for e in items:
        scale = f"{e['scaleMin']}-{e['scaleMax']}"
        table.add_row(
            e["id"],
            e["name"],
            e["type"],
            scale,
            str(e["weight"]),
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
            "Type (e.g. f1, similarity, structuredOutput, comparisonJudge, "
            "referenceJudge, generalJudge)."
        ),
    ),
    scale_min: float = typer.Option(0, "--scale-min", help="Scale minimum."),
    scale_max: float = typer.Option(1, "--scale-max", help="Scale maximum."),
    weight: float = typer.Option(1, "--weight", "-w", help="Weight."),
    description: Annotated[
        str | None, typer.Option("--description", "-d", help="Optional description.")
    ] = None,
    instructions: Annotated[
        str | None,
        typer.Option(
            "--instructions",
            "-i",
            help=(
                "Judge instructions text (shorthand for "
                "config={'instructions': ...}). Use for comparisonJudge / "
                "referenceJudge evaluators. Mutually exclusive with --config-file."
            ),
        ),
    ] = None,
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config-file",
            "-c",
            help=(
                "JSON file with the full ``config`` payload — use for richer "
                "shapes (e.g. generalJudge expects {'messages': [...]})."
            ),
        ),
    ] = None,
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
    if description is not None:
        evaluator["description"] = description
    config = _resolve_config(instructions, config_file)
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


@evaluators_app.command("update")
def update_evaluator(
    experiment_id: str = typer.Argument(help="Experiment ID."),
    evaluator_id: str = typer.Argument(help="Evaluator ID."),
    name: Annotated[str | None, typer.Option("--name", "-n", help="New name.")] = None,
    eval_type: Annotated[
        str | None, typer.Option("--type", "-t", help="New evaluator type.")
    ] = None,
    scale_min: Annotated[
        float | None, typer.Option("--scale-min", help="New scale minimum.")
    ] = None,
    scale_max: Annotated[
        float | None, typer.Option("--scale-max", help="New scale maximum.")
    ] = None,
    weight: Annotated[float | None, typer.Option("--weight", "-w", help="New weight.")] = None,
    description: Annotated[
        str | None, typer.Option("--description", "-d", help="New description.")
    ] = None,
    instructions: Annotated[
        str | None,
        typer.Option(
            "--instructions",
            "-i",
            help=(
                "Replace judge instructions (shorthand for "
                "config={'instructions': ...}). Mutually exclusive with --config-file."
            ),
        ),
    ] = None,
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config-file",
            "-c",
            help="JSON file with the full ``config`` payload to replace the existing config.",
        ),
    ] = None,
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Update an evaluator in place.

    Only flags that are set are sent — others stay untouched on the server.
    Use ``--instructions`` to swap a judge's prompt without recreating it,
    or ``--config-file`` to replace the whole config block.
    """
    updates: dict[str, Any] = {}
    if name is not None:
        updates["name"] = name
    if eval_type is not None:
        updates["type"] = eval_type
    if scale_min is not None:
        updates["scaleMin"] = scale_min
    if scale_max is not None:
        updates["scaleMax"] = scale_max
    if weight is not None:
        updates["weight"] = weight
    if description is not None:
        updates["description"] = description
    config = _resolve_config(instructions, config_file)
    if config is not None:
        updates["config"] = config

    if not updates:
        err_console.print("[yellow]No updates specified.[/yellow]")
        raise typer.Exit(1)

    with get_client() as client:
        result = client.update_evaluator(experiment_id, evaluator_id, **updates)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    console.print("[green]Evaluator updated.[/green]")


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
