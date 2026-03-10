"""Evaluator commands."""

from __future__ import annotations

import json
import sys

import typer
from rich.console import Console
from rich.table import Table

from promptic_sdk.cli import get_client

evaluators_app = typer.Typer(help="Manage evaluators.")
console = Console()


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
        ..., "--type", "-t", help="Type (f1, judge, similarity, structuredOutput)."
    ),
    scale_min: float = typer.Option(0, "--scale-min", help="Scale minimum."),
    scale_max: float = typer.Option(1, "--scale-max", help="Scale maximum."),
    weight: float = typer.Option(1, "--weight", "-w", help="Weight."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Add an evaluator to an experiment."""
    evaluator = {
        "name": name,
        "type": eval_type,
        "scaleMin": scale_min,
        "scaleMax": scale_max,
        "weight": weight,
    }

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
