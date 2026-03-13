"""Annotations commands — create, list, and delete."""

from __future__ import annotations

import json
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from promptic_sdk.cli import get_client

annotations_app = typer.Typer(help="Manage trace annotations.")
console = Console()
err_console = Console(stderr=True)


@annotations_app.command("create")
def create_annotation(
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
    run_id: str = typer.Option(..., "--run", help="Run ID."),
    trace_db_id: str = typer.Option(..., "--trace", help="Trace DB ID."),
    rating: Annotated[
        str | None,
        typer.Option(help="Rating (positive/negative)."),
    ] = None,
    comment: Annotated[str | None, typer.Option(help="Comment text.")] = None,
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Create or update an annotation for a trace in a run."""
    with get_client() as client:
        result = client.upsert_annotation(
            component_id,
            run_id,
            trace_db_id,
            rating=rating,
            comment=comment,
        )

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    console.print(f"[green]Annotation saved:[/green] {result['id']}")
    if result.get("rating"):
        console.print(f"  Rating: {result['rating']}")
    if result.get("comment"):
        console.print(f"  Comment: {result['comment']}")


@annotations_app.command("list")
def list_annotations(
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
    run_id: str = typer.Option(..., "--run", help="Run ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List annotations for a run."""
    with get_client() as client:
        result = client.list_annotations(component_id, run_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    annotations = result["data"]
    if not annotations:
        console.print("No annotations found.", style="dim")
        return

    table = Table(title=f"Annotations ({len(annotations)})")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Trace ID")
    table.add_column("Rating")
    table.add_column("Comment", max_width=40)
    table.add_column("Created")

    for ann in annotations:
        table.add_row(
            ann["id"],
            ann["traceDbId"],
            ann.get("rating") or "-",
            (ann.get("comment") or "-")[:80],
            ann["createdAt"],
        )

    console.print(table)


@annotations_app.command("delete")
def delete_annotation(
    annotation_id: str = typer.Argument(help="Annotation ID."),
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
    run_id: str = typer.Option(..., "--run", help="Run ID."),
) -> None:
    """Delete an annotation."""
    with get_client() as client:
        client.delete_annotation(component_id, run_id, annotation_id)
    console.print("[green]Annotation deleted.[/green]")
