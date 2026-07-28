"""Datasets commands — create, list, and get."""

from __future__ import annotations

import json
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from promptic_sdk.cli import get_client

datasets_app = typer.Typer(help="Manage reusable datasets.")
console = Console()
err_console = Console(stderr=True)


@datasets_app.command("create")
def create_dataset(
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
    name: str = typer.Option(..., "--name", help="Dataset name."),
    description: Annotated[str | None, typer.Option(help="Dataset description.")] = None,
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Create a new canonical dataset."""
    with get_client() as client:
        result = client.create_dataset(
            component_id,
            name,
            description=description,
        )

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    console.print(f"[green]Dataset created:[/green] {result['name']}")
    console.print(f"  ID: {result['id']}")
    console.print(f"  Cases: {result['caseCount']}")


@datasets_app.command("list")
def list_datasets(
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List datasets for an AI component."""
    with get_client() as client:
        result = client.list_datasets(component_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    datasets = result["data"]
    if not datasets:
        console.print("No datasets found.", style="dim")
        return

    table = Table(title=f"Datasets ({len(datasets)})")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Cases", justify="right")
    table.add_column("Created")

    for ds in datasets:
        table.add_row(
            ds["id"],
            ds["name"],
            str(ds["caseCount"]),
            ds["createdAt"],
        )

    console.print(table)


@datasets_app.command("get")
def get_dataset(
    dataset_id: str = typer.Argument(help="Dataset ID."),
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Get a dataset with its cases."""
    with get_client() as client:
        result = client.get_dataset(component_id, dataset_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    console.print(f"\n[bold]Dataset:[/bold] {result['name']}")
    console.print(f"[bold]ID:[/bold] {result['id']}")
    console.print(f"[bold]Cases:[/bold] {result['caseCount']}")
    if result.get("description"):
        console.print(f"[bold]Description:[/bold] {result['description']}")

    cases = result.get("cases", [])
    if cases:
        console.print(f"\n[bold]Cases ({len(cases)}):[/bold]")
        case_table = Table()
        case_table.add_column("ID", style="cyan")
        case_table.add_column("Input", max_width=50)
        case_table.add_column("Expected", max_width=50)

        for case in cases:
            case_table.add_row(
                str(case["id"]),
                json.dumps(case["inputPayload"], default=str)[:100],
                json.dumps(case["expectedPayload"], default=str)[:100],
            )
        console.print(case_table)


@datasets_app.command("delete")
def delete_dataset(
    dataset_id: str = typer.Argument(help="Dataset ID."),
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
) -> None:
    """Delete a dataset."""
    with get_client() as client:
        client.delete_dataset(component_id, dataset_id)
    console.print("[green]Dataset deleted.[/green]")
