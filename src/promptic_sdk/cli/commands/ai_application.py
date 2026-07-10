"""AI Application commands."""

from __future__ import annotations

import json
import sys

import httpx
import typer
from rich.console import Console
from rich.table import Table

from promptic_sdk.cli import get_client
from promptic_sdk.cli.config import load_config, save_ai_application

ai_application_app = typer.Typer(help="AI Application information.")
console = Console()
err_console = Console(stderr=True)


@ai_application_app.command("info")
def info(
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Show AI Application details for the current API key."""
    with get_client() as client:
        result = client.get_ai_application()

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    console.print(f"\n[bold]AI Application:[/bold] {result['name']}")
    console.print(f"  ID:          {result['id']}")
    if result["description"]:
        console.print(f"  Description: {result['description']}")
    console.print(f"  Created:     {result['createdAt']}")


@ai_application_app.command("list")
def list_ai_applications(
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List all AI Applications accessible to the current user."""
    config = load_config()
    if not config or not config.access_token:
        err_console.print(
            "Session login required. Run 'promptic login' first.",
            style="red",
        )
        raise typer.Exit(1)

    try:
        resp = httpx.get(
            f"{config.endpoint.rstrip('/')}/api/v1/ai-applications",
            headers={"Authorization": f"Bearer {config.access_token}"},
            timeout=15,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        err_console.print(f"[red]Failed to list AI applications:[/red] {exc}")
        raise typer.Exit(1) from exc

    data = resp.json()
    ai_applications = data.get("data", [])

    if output_json:
        json.dump(ai_applications, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    if not ai_applications:
        console.print("No AI applications found.")
        return

    selected = config.ai_application_id

    table = Table(title="AI Applications")
    table.add_column("", width=2)
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Created")

    for app in ai_applications:
        marker = "*" if app["id"] == selected else ""
        table.add_row(marker, app["id"], app["name"], str(app.get("createdAt", "")))

    console.print(table)
    if selected:
        console.print("\n[dim]* = currently selected AI application[/dim]")
    else:
        console.print(
            "\n[dim]Run 'promptic ai-application select <id>' to select an AI application.[/dim]"
        )


@ai_application_app.command("select")
def select_ai_application(
    ai_application_id: str = typer.Argument(help="AI Application ID to select."),
) -> None:
    """Select an AI Application for CLI commands."""
    save_ai_application(ai_application_id)
    console.print(f"AI Application [bold]{ai_application_id}[/bold] selected.", style="green")
