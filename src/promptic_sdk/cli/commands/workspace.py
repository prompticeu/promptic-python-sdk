"""Workspace commands."""

from __future__ import annotations

import json
import sys

import httpx
import typer
from rich.console import Console
from rich.table import Table

from promptic_sdk.cli import get_client
from promptic_sdk.cli.config import load_config, save_workspace

workspace_app = typer.Typer(help="Workspace information.")
console = Console()
err_console = Console(stderr=True)


@workspace_app.command("info")
def info(
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Show workspace details for the current API key."""
    with get_client() as client:
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


@workspace_app.command("list")
def list_workspaces(
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List all workspaces accessible to the current user."""
    config = load_config()
    if not config or not config.access_token:
        err_console.print(
            "Session login required. Run 'promptic login' first.",
            style="red",
        )
        raise typer.Exit(1)

    try:
        resp = httpx.get(
            f"{config.endpoint.rstrip('/')}/api/v1/workspaces",
            headers={"Authorization": f"Bearer {config.access_token}"},
            timeout=15,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        err_console.print(f"[red]Failed to list workspaces:[/red] {exc}")
        raise typer.Exit(1) from exc

    data = resp.json()
    workspaces = data.get("data", [])

    if output_json:
        json.dump(workspaces, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    if not workspaces:
        console.print("No workspaces found.")
        return

    selected = config.workspace_id

    table = Table(title="Workspaces")
    table.add_column("", width=2)
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Created")

    for ws in workspaces:
        marker = "*" if ws["id"] == selected else ""
        table.add_row(marker, ws["id"], ws["name"], str(ws.get("createdAt", "")))

    console.print(table)
    if selected:
        console.print("\n[dim]* = currently selected workspace[/dim]")
    else:
        console.print("\n[dim]Run 'promptic workspace select <id>' to select a workspace.[/dim]")


@workspace_app.command("select")
def select_workspace(
    workspace_id: str = typer.Argument(help="Workspace ID to select."),
) -> None:
    """Select a workspace for CLI commands."""
    save_workspace(workspace_id)
    console.print(f"Workspace [bold]{workspace_id}[/bold] selected.", style="green")
