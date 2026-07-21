"""Datasets commands — create, list, and get."""

from __future__ import annotations

import json
import sys
from typing import Annotated, cast

import typer
from rich.console import Console
from rich.table import Table

from promptic_sdk.cli import get_client

datasets_app = typer.Typer(help="Manage agent datasets.")
console = Console()
err_console = Console(stderr=True)


def _display_payload(value: object, preferred_key: str | None = None) -> str:
    """Render canonical JSON payloads compactly for the terminal."""
    if preferred_key and isinstance(value, dict):
        preferred = cast(dict[str, object], value).get(preferred_key)
        if isinstance(preferred, str):
            return preferred
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


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
    console.print(f"  Cases: {result['caseCount']}")
    console.print()
    console.print(
        "[dim]Tip: Add traces via SDK with "
        f"promptic_sdk.ai_component('...', dataset_id='{result['id']}')"
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
    """Get a dataset with its canonical cases."""
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
        case_table.add_column("Trace ID", style="cyan")
        case_table.add_column("Input", max_width=40)
        case_table.add_column("Expected", max_width=40)

        for dataset_case in cases:
            trace_references = dataset_case["traceReferences"]
            source_trace = next(
                (reference for reference in trace_references if reference["role"] == "source"),
                trace_references[0] if trace_references else None,
            )
            case_table.add_row(
                source_trace["traceDbId"] if source_trace else "-",
                _display_payload(dataset_case["inputPayload"], "input")[:80],
                _display_payload(dataset_case["expectedPayload"], "value")[:80],
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
