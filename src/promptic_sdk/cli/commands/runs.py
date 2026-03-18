"""Runs commands — create, list, get, and delete."""

from __future__ import annotations

import json
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from promptic_sdk.cli import get_client

runs_app = typer.Typer(help="Manage agent runs.")
console = Console()
err_console = Console(stderr=True)


@runs_app.command("create")
def create_run(
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
    dataset_id: str = typer.Option(..., "--dataset", help="Dataset ID."),
    name: Annotated[str | None, typer.Option(help="Run name.")] = None,
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Create a new run for a dataset."""
    with get_client() as client:
        result = client.create_run(component_id, dataset_id, name=name)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    console.print(f"[green]Run created:[/green] {result['id']}")
    if result.get("name"):
        console.print(f"  Name: {result['name']}")
    console.print(f"  Status: {result['status']}")
    console.print(f"  Traces: {result['traceCount']}")


@runs_app.command("list")
def list_runs(
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List runs for an AI component."""
    with get_client() as client:
        result = client.list_runs(component_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    runs = result["data"]
    if not runs:
        console.print("No runs found.", style="dim")
        return

    table = Table(title=f"Runs ({len(runs)})")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Traces", justify="right")
    table.add_column("Created")

    for run in runs:
        table.add_row(
            run["id"],
            run.get("name") or "-",
            run["status"],
            str(run["traceCount"]),
            run["createdAt"],
        )

    console.print(table)


@runs_app.command("get")
def get_run(
    run_id: str = typer.Argument(help="Run ID."),
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Get a run with its traces."""
    with get_client() as client:
        result = client.get_run(component_id, run_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    console.print(f"\n[bold]Run:[/bold] {result.get('name') or result['id']}")
    console.print(f"[bold]ID:[/bold] {result['id']}")
    console.print(f"[bold]Status:[/bold] {result['status']}")
    console.print(f"[bold]Traces:[/bold] {result['traceCount']}")

    traces = result.get("traces", [])
    if traces:
        console.print(f"\n[bold]Traces ({len(traces)}):[/bold]")
        trace_table = Table()
        trace_table.add_column("ID", style="cyan")
        trace_table.add_column("Name")
        trace_table.add_column("Status")
        trace_table.add_column("Duration (ms)", justify="right")

        for trace in traces:
            trace_table.add_row(
                trace["id"],
                trace.get("name") or "-",
                trace["status"],
                str(trace.get("durationMs") or "-"),
            )
        console.print(trace_table)


@runs_app.command("delete")
def delete_run(
    run_id: str = typer.Argument(help="Run ID."),
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
) -> None:
    """Delete a run."""
    with get_client() as client:
        client.delete_run(component_id, run_id)
    console.print("[green]Run deleted.[/green]")
