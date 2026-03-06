"""AI component commands."""

from __future__ import annotations

import json
import sys

import typer
from rich.console import Console
from rich.table import Table

from promptic_sdk.cli.config import load_config
from promptic_sdk.client import PromenticClient

components_app = typer.Typer(help="Manage AI components.")
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


@components_app.command("list")
def list_components(
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List all AI components in the workspace."""
    with _get_client() as client:
        result = client.list_components()

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    items = result["data"]
    if not items:
        console.print("No components found.", style="dim")
        return

    table = Table(title=f"AI Components ({len(items)})")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Description")
    table.add_column("Created")

    for c in items:
        table.add_row(
            c["id"],
            c["name"],
            c["description"] or "-",
            c["createdAt"],
        )

    console.print(table)


@components_app.command("create")
def create_component(
    name: str = typer.Argument(help="Component name."),
    description: str | None = typer.Option(
        None, "--description", "-d", help="Optional description."
    ),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Create a new AI component."""
    with _get_client() as client:
        result = client.create_component(name, description=description)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    console.print(f"[green]Component created:[/green] {result['id']}")


@components_app.command("get")
def get_component(
    component_id: str = typer.Argument(help="Component ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Get details of an AI component."""
    with _get_client() as client:
        result = client.get_component(component_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    console.print(f"\n[bold]Component:[/bold] {result['name']}")
    console.print(f"  ID:          {result['id']}")
    if result["description"]:
        console.print(f"  Description: {result['description']}")
    console.print(f"  Created:     {result['createdAt']}")


@components_app.command("delete")
def delete_component(
    component_id: str = typer.Argument(help="Component ID."),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation."),
) -> None:
    """Delete an AI component."""
    if not force:
        typer.confirm(f"Delete component {component_id}?", abort=True)

    with _get_client() as client:
        client.delete_component(component_id)

    console.print("[green]Component deleted.[/green]")
