"""Experiment commands."""

from __future__ import annotations

import json
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from promptic_sdk.cli.config import load_config
from promptic_sdk.client import PromenticClient

experiments_app = typer.Typer(help="Manage experiments.")
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


@experiments_app.command("list")
def list_experiments(
    component_id: Annotated[
        str | None, typer.Option("--component-id", "-c", help="Filter by component.")
    ] = None,
    status: Annotated[str | None, typer.Option(help="Filter by status.")] = None,
    limit: int = typer.Option(20, help="Number of results."),
    offset: int = typer.Option(0, help="Offset for pagination."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List experiments."""
    with _get_client() as client:
        result = client.list_experiments(
            component_id=component_id, status=status, limit=limit, offset=offset
        )

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    items = result["data"]
    if not items:
        console.print("No experiments found.", style="dim")
        return

    table = Table(title=f"Experiments ({len(items)})")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Model")
    table.add_column("Task Type")
    table.add_column("Created")

    for e in items:
        st = e["experimentStatus"]
        st_style = {"completed": "green", "failed": "red", "running": "yellow"}.get(st, "dim")
        table.add_row(
            e["id"],
            e["name"] or "-",
            f"[{st_style}]{st}[/{st_style}]",
            e["targetModel"],
            e["taskType"],
            e["createdAt"],
        )

    console.print(table)


@experiments_app.command("create")
def create_experiment(
    component_id: Annotated[
        str | None, typer.Option("--component-id", "-c", help="AI component ID.")
    ] = None,
    target_model: Annotated[
        str | None, typer.Option("--target-model", "-m", help="Target model (e.g. gpt-4.1-nano).")
    ] = None,
    task_type: Annotated[
        str | None,
        typer.Option("--task-type", "-t", help="Task type (classification, generation, etc.)."),
    ] = None,
    initial_prompt: Annotated[
        str | None, typer.Option("--initial-prompt", "-p", help="Initial prompt text.")
    ] = None,
    name: Annotated[str | None, typer.Option("--name", help="Experiment name.")] = None,
    description: Annotated[str | None, typer.Option("--description", help="Description.")] = None,
    provider: str = typer.Option("openai", help="Model provider."),
    optimizer: str = typer.Option("prompticV2", help="Optimizer type."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Create a new experiment.

    If required flags are missing, an interactive wizard prompts for them.
    """
    # Interactive wizard for missing required fields
    if not component_id:
        component_id = typer.prompt("AI component ID")
    if not target_model:
        target_model = typer.prompt("Target model (e.g. gpt-4.1-nano)")
    if not task_type:
        task_type = typer.prompt(
            "Task type",
            default="classification",
        )
    if not initial_prompt:
        initial_prompt = typer.prompt("Initial prompt", default="")
        if not initial_prompt:
            initial_prompt = None

    with _get_client() as client:
        result = client.create_experiment(
            ai_component_id=component_id,
            target_model=target_model,
            task_type=task_type,
            initial_prompt=initial_prompt,
            name=name,
            description=description,
            provider=provider,
            optimizer=optimizer,
        )

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    console.print(f"[green]Experiment created:[/green] {result['id']}")
    console.print(f"  Name:   {result['name'] or '-'}")
    console.print(f"  Model:  {result['targetModel']}")
    console.print(f"  Status: {result['experimentStatus']}")


@experiments_app.command("get")
def get_experiment(
    experiment_id: str = typer.Argument(help="Experiment ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Get experiment details."""
    with _get_client() as client:
        result = client.get_experiment(experiment_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    st = result["experimentStatus"]
    st_style = {"completed": "green", "failed": "red", "running": "yellow"}.get(st, "dim")

    console.print(f"\n[bold]Experiment:[/bold] {result['name'] or '-'}")
    console.print(f"  ID:       {result['id']}")
    console.print(f"  Status:   [{st_style}]{st}[/{st_style}]")
    console.print(f"  Model:    {result['targetModel']}")
    console.print(f"  Provider: {result['provider']}")
    console.print(f"  Task:     {result['taskType']}")
    if result["initialPrompt"]:
        prompt_preview = result["initialPrompt"][:100]
        console.print(
            f"  Prompt:   {prompt_preview}{'...' if len(result['initialPrompt']) > 100 else ''}"
        )
    console.print(f"  Created:  {result['createdAt']}")


@experiments_app.command("update")
def update_experiment(
    experiment_id: str = typer.Argument(help="Experiment ID."),
    name: Annotated[str | None, typer.Option(help="New name.")] = None,
    initial_prompt: Annotated[
        str | None, typer.Option("--initial-prompt", "-p", help="New initial prompt.")
    ] = None,
    target_model: Annotated[
        str | None, typer.Option("--target-model", "-m", help="New target model.")
    ] = None,
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Update a pending experiment."""
    updates: dict[str, str] = {}
    if name is not None:
        updates["name"] = name
    if initial_prompt is not None:
        updates["initialPrompt"] = initial_prompt
    if target_model is not None:
        updates["targetModel"] = target_model

    if not updates:
        err_console.print("No updates specified.", style="yellow")
        raise typer.Exit(1)

    with _get_client() as client:
        result = client.update_experiment(experiment_id, **updates)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    console.print("[green]Experiment updated.[/green]")


@experiments_app.command("delete")
def delete_experiment(
    experiment_id: str = typer.Argument(help="Experiment ID."),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation."),
) -> None:
    """Delete an experiment."""
    if not force:
        typer.confirm(f"Delete experiment {experiment_id}?", abort=True)

    with _get_client() as client:
        client.delete_experiment(experiment_id)

    console.print("[green]Experiment deleted.[/green]")


@experiments_app.command("start")
def start_experiment(
    experiment_id: str = typer.Argument(help="Experiment ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Start a pending experiment (enqueue for training)."""
    with _get_client() as client:
        result = client.start_experiment(experiment_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    console.print(f"[green]Experiment started.[/green] Status: {result['status']}")
