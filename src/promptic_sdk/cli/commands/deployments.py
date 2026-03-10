"""Deployment commands."""

from __future__ import annotations

import json
import sys

import typer
from rich.console import Console

from promptic_sdk.cli import get_client

deployments_app = typer.Typer(help="Manage deployments.")
console = Console()


@deployments_app.command("status")
def deployment_status(
    component_id: str = typer.Argument(help="Component ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Show current deployment for a component."""
    with get_client() as client:
        result = client.get_deployment(component_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    if not result:
        console.print("No deployment active.", style="dim")
        return

    exp = result["experiment"]
    console.print("\n[bold]Active Deployment[/bold]")
    console.print(f"  Component:   {result['aiComponentId']}")
    console.print(f"  Experiment:  {result['experimentId']}")
    console.print(f"  Exp. Name:   {exp['name'] or '-'}")
    console.print(f"  Model:       {exp['targetModel']}")
    console.print(f"  Status:      {exp['experimentStatus']}")


@deployments_app.command("deploy")
def deploy(
    component_id: str = typer.Argument(help="Component ID."),
    experiment_id: str = typer.Argument(help="Experiment ID to deploy."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Deploy an experiment to a component."""
    with get_client() as client:
        result = client.deploy(component_id, experiment_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    console.print(
        f"[green]Deployed[/green] experiment {result['experimentId']} "
        f"to component {result['aiComponentId']}"
    )


@deployments_app.command("prompt")
def deployment_prompt(
    component_id: str = typer.Argument(help="Component ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Show the deployed prompt for a component."""
    with get_client() as client:
        result = client.get_deployed_prompt(component_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    if not result:
        console.print("No deployment active.", style="dim")
        return

    console.print(f"\n[bold]{result['componentName'] or '-'}[/bold]")
    console.print(f"  Model:       {result['model']}")
    console.print(f"  Score:       {result['score']}")
    console.print(f"  Experiment:  {result['experimentId']}")
    console.print("\n[bold]Prompt:[/bold]\n")
    console.print(result["prompt"])


@deployments_app.command("undeploy")
def undeploy(
    component_id: str = typer.Argument(help="Component ID."),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation."),
) -> None:
    """Remove deployment from a component."""
    if not force:
        typer.confirm(f"Undeploy from component {component_id}?", abort=True)

    with get_client() as client:
        client.undeploy(component_id)

    console.print("[green]Deployment removed.[/green]")
