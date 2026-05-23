"""Artifact commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from promptic_sdk.cli import get_client

artifacts_app = typer.Typer(help="Fetch trace artifacts.")
console = Console()
err_console = Console(stderr=True)


@artifacts_app.command("get")
def get_artifact(
    artifact_id: str = typer.Argument(help="Artifact ID."),
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write bytes to file.")
    ] = None,
    output_json: bool = typer.Option(False, "--json", help="Output metadata as JSON."),
) -> None:
    """Get artifact metadata or download its bytes."""
    with get_client() as client:
        try:
            if output:
                client.download_artifact(artifact_id, output)
                console.print(f"Wrote {output}")
                return
            result = client.get_artifact(artifact_id)
        except Exception as e:
            err_console.print(f"Error: {e}", style="red")
            raise typer.Exit(1) from e

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    console.print(f"[bold]Artifact:[/bold] {result['id']}")
    console.print(f"[bold]MIME:[/bold] {result['mimeType']}")
    console.print(f"[bold]Size:[/bold] {result['sizeBytes']} bytes")
    console.print(f"[bold]Source:[/bold] {result['sourcePath']}")
    if result.get("traceId"):
        console.print(f"[bold]Trace:[/bold] {result['traceId']}")
    if result.get("spanId"):
        console.print(f"[bold]Span:[/bold] {result['spanId']}")
