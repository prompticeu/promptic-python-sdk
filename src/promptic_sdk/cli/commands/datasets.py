"""Datasets commands — create, list, and get."""

from __future__ import annotations

import json
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from promptic_sdk.cli import get_client

datasets_app = typer.Typer(help="Manage agent datasets.")
console = Console()
err_console = Console(stderr=True)


@datasets_app.command("create")
def create_dataset(
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
    name: str = typer.Option(..., "--name", help="Dataset name."),
    description: Annotated[str | None, typer.Option(help="Dataset description.")] = None,
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Create a new dataset from traces."""
    with get_client() as client:
        # First create the dataset
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
    console.print(f"  Items: {result['itemCount']}")
    console.print()
    console.print(
        "[dim]Tip: Add traces via SDK with "
        "promptic_sdk.ai_component('...', dataset='...')"
        " or use the API.[/dim]"
    )


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
    table.add_column("Items", justify="right")
    table.add_column("Created")

    for ds in datasets:
        table.add_row(
            ds["id"],
            ds["name"],
            str(ds["itemCount"]),
            ds["createdAt"],
        )

    console.print(table)


@datasets_app.command("get")
def get_dataset(
    dataset_id: str = typer.Argument(help="Dataset ID."),
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Get a dataset with its items."""
    with get_client() as client:
        result = client.get_dataset(component_id, dataset_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    console.print(f"\n[bold]Dataset:[/bold] {result['name']}")
    console.print(f"[bold]ID:[/bold] {result['id']}")
    console.print(f"[bold]Items:[/bold] {result['itemCount']}")
    if result.get("description"):
        console.print(f"[bold]Description:[/bold] {result['description']}")

    items = result.get("items", [])
    if items:
        console.print(f"\n[bold]Items ({len(items)}):[/bold]")
        item_table = Table()
        item_table.add_column("Trace ID", style="cyan")
        item_table.add_column("Input", max_width=40)
        item_table.add_column("Output", max_width=40)

        for item in items:
            item_table.add_row(
                item["traceDbId"],
                (item["input"] or "-")[:80],
                (item["output"] or "-")[:80],
            )
        console.print(item_table)


@datasets_app.command("delete")
def delete_dataset(
    dataset_id: str = typer.Argument(help="Dataset ID."),
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
) -> None:
    """Delete a dataset."""
    with get_client() as client:
        client.delete_dataset(component_id, dataset_id)
    console.print("[green]Dataset deleted.[/green]")
