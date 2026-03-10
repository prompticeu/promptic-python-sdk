"""Iteration commands."""

from __future__ import annotations

import json
import sys

import typer
from rich.console import Console
from rich.table import Table

from promptic_sdk.cli import get_client
from promptic_sdk.models import IterationWithScores

iterations_app = typer.Typer(help="View experiment iterations (results).")
console = Console()


@iterations_app.command("list")
def list_iterations(
    experiment_id: str = typer.Argument(help="Experiment ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List iterations for an experiment."""
    with get_client() as client:
        result = client.list_iterations(experiment_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    items = result["data"]
    if not items:
        console.print("No iterations found.", style="dim")
        return

    table = Table(title=f"Iterations ({len(items)})")
    table.add_column("#", justify="right")
    table.add_column("ID", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Prompt Tokens", justify="right")
    table.add_column("Created")

    for it in items:
        score_str = f"{it['overallNormalizedScore']:.4f}"
        table.add_row(
            str(it["iterationNumber"]),
            str(it["id"]),
            score_str,
            str(it["promptTokens"] or "-"),
            it["createdAt"],
        )

    console.print(table)


@iterations_app.command("get")
def get_iteration(
    experiment_id: str = typer.Argument(help="Experiment ID."),
    iteration_id: int = typer.Argument(help="Iteration ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Get an iteration with evaluator scores."""
    with get_client() as client:
        result = client.get_iteration(experiment_id, iteration_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    _print_iteration(result)


@iterations_app.command("best")
def best_iteration(
    experiment_id: str = typer.Argument(help="Experiment ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Get the best-scoring iteration for an experiment."""
    with get_client() as client:
        result = client.get_best_iteration(experiment_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    _print_iteration(result)


def _print_iteration(it: IterationWithScores) -> None:
    score_str = f"{it['overallNormalizedScore']:.4f}"

    console.print(f"\n[bold]Iteration #{it['iterationNumber']}[/bold]")
    console.print(f"  ID:     {it['id']}")
    console.print(f"  Score:  [green]{score_str}[/green]")
    console.print(f"  Tokens: {it['promptTokens'] or '-'}")

    prompt = it["prompt"]
    if prompt:
        preview = prompt[:200]
        console.print(f"\n[bold]Prompt:[/bold]\n{preview}{'...' if len(prompt) > 200 else ''}")

    scores = it.get("scores", [])
    if scores:
        console.print("\n[bold]Evaluator Scores:[/bold]")
        table = Table()
        table.add_column("Evaluator")
        table.add_column("Type")
        table.add_column("Raw", justify="right")
        table.add_column("Normalized", justify="right")
        for s in scores:
            table.add_row(
                s["evaluatorName"],
                s["evaluatorType"],
                f"{s['rawScore']:.4f}",
                f"{s['score']:.4f}",
            )
        console.print(table)
