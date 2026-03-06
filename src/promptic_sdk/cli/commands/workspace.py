"""Workspace commands."""

from __future__ import annotations

import json
import sys

import typer
from rich.console import Console

from promptic_sdk.cli.config import load_config
from promptic_sdk.client import PromenticClient

workspace_app = typer.Typer(help="Workspace information.")
console = Console()
err_console = Console(stderr=True)


def _get_client() -> PromenticClient:
    config = load_config()
    if not config:
        err_console.print(
            "No configuration found. Run 'promptic configure' or set PROMPTIC_API_KEY.",
            style="red",
        )
        raise typer.Exit(1)
    return PromenticClient(api_key=config.api_key, endpoint=config.endpoint)


@workspace_app.command("info")
def info(
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Show workspace details for the current API key."""
    with _get_client() as client:
        result = client.get_workspace()

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    console.print(f"\n[bold]Workspace:[/bold] {result['name']}")
    console.print(f"  ID:          {result['id']}")
    if result["description"]:
        console.print(f"  Description: {result['description']}")
    console.print(f"  Created:     {result['createdAt']}")
