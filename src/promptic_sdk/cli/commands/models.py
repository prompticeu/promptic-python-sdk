"""Available-model commands."""

from __future__ import annotations

import json
import sys

import typer
from rich.console import Console
from rich.table import Table

from promptic_sdk.cli import get_client

models_app = typer.Typer(help="Discover models available to the AI Application.")
console = Console()


@models_app.command("list")
def list_models(
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List models; Judge=yes marks IDs valid for benchmark judge evaluators."""
    with get_client() as client:
        result = client.models.list()

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    items = result["data"]
    if not items:
        console.print("No models available.", style="dim")
        return

    table = Table(title=f"Available models ({len(items)})")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Provider")
    table.add_column("Group")
    table.add_column("Judge")

    for model in items:
        judge = "yes" if model["judgeEligible"] else "no"
        table.add_row(model["id"], model["name"], model["provider"], model["group"], judge)

    console.print(table)
