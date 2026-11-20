"""Experiment commands."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict, cast

import typer
from rich.console import Console
from rich.table import Table

from promptic_sdk.cli import get_client
from promptic_sdk.models import (
    OptimizationBrief,
    PromptExperimentTaskType,
    PromptMessage,
    ToolSelectionTestCase,
    ToolSelectionTool,
    ToolSource,
)

experiments_app = typer.Typer(help="Manage experiments.")
console = Console()
err_console = Console(stderr=True)


def _load_json_array(path: Path, *, option_name: str) -> list[dict[str, Any]]:
    """Load a JSON array of objects for a CLI option."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"Could not read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise typer.BadParameter(f"{option_name} must contain a JSON array of objects")
    return value


def _load_json_object(path: Path, *, option_name: str) -> dict[str, Any]:
    """Load a JSON object for a CLI option."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"Could not read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise typer.BadParameter(f"{option_name} must contain a JSON object")
    return value


class _GuidanceOptions(TypedDict, total=False):
    """Optional SDK keyword arguments for CLI guidance."""

    optimization_guidance: OptimizationBrief


def _guidance_options(path: Path | None) -> _GuidanceOptions:
    if path is None:
        return {}
    return {
        "optimization_guidance": cast(
            OptimizationBrief, _load_json_object(path, option_name="--guidance")
        )
    }


def _validate_tool_selection_tools(
    tools: list[dict[str, Any]], *, option_name: str = "--tools"
) -> None:
    """Validate required tool fields before passing them to the SDK client."""
    for index, tool in enumerate(tools):
        for field in ("name", "description"):
            value = tool.get(field)
            if not isinstance(value, str) or not value.strip():
                raise typer.BadParameter(
                    f"{option_name}[{index}].{field} must be a non-empty string"
                )


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
    with get_client() as client:
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
        st_style = {
            "completed": "green",
            "failed": "red",
            "running": "yellow",
            "canceling": "yellow",
            "canceled": "cyan",
        }.get(st, "dim")
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
        typer.Option(
            "--task-type",
            "-t",
            help=(
                "Task type: classification, textGeneration, or structuredOutput. "
                "Tool selection requires create_tool_selection_experiment()."
            ),
        ),
    ] = None,
    initial_prompt: Annotated[
        str | None, typer.Option("--initial-prompt", "-p", help="Initial prompt text.")
    ] = None,
    name: Annotated[str | None, typer.Option("--name", help="Experiment name.")] = None,
    description: Annotated[str | None, typer.Option("--description", help="Description.")] = None,
    provider: str = typer.Option("openai", help="Model provider."),
    optimizer: str = typer.Option("prompticV2", help="Optimizer type."),
    hyperparameters_file: Annotated[
        Path | None,
        typer.Option(
            "--hyperparameters",
            exists=True,
            dir_okay=False,
            readable=True,
            help="JSON file containing experiment hyperparameters.",
        ),
    ] = None,
    evaluation_only: bool = typer.Option(
        False, "--evaluate-only", help="Evaluate the supplied prompt once without optimizing it."
    ),
    messages_file: Annotated[
        Path | None,
        typer.Option(
            "--messages",
            exists=True,
            dir_okay=False,
            readable=True,
            help="JSON array of ordered role/content messages for a multi-message prompt.",
        ),
    ] = None,
    output_schema_file: Annotated[
        Path | None,
        typer.Option(
            "--output-schema",
            exists=True,
            dir_okay=False,
            readable=True,
            help="JSON Schema file for structured-output predictions.",
        ),
    ] = None,
    start: bool = typer.Option(False, "--start", help="Start the experiment after creating it."),
    guidance_file: Annotated[
        Path | None,
        typer.Option(
            "--guidance",
            exists=True,
            dir_okay=False,
            readable=True,
            help="JSON file with optimization rules.",
        ),
    ] = None,
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
    supported_task_types = {"classification", "textGeneration", "structuredOutput"}
    if task_type not in supported_task_types:
        raise typer.BadParameter(
            "--task-type must be classification, textGeneration, or structuredOutput; "
            "tool selection requires create_tool_selection_experiment()"
        )
    if initial_prompt is not None and messages_file is not None:
        raise typer.BadParameter("Use --initial-prompt or --messages, not both")
    if not initial_prompt and messages_file is None:
        initial_prompt = typer.prompt("Initial prompt", default="")
        if not initial_prompt:
            initial_prompt = None

    hyperparameters = (
        _load_json_object(hyperparameters_file, option_name="--hyperparameters")
        if hyperparameters_file
        else None
    )
    output_schema = (
        _load_json_object(output_schema_file, option_name="--output-schema")
        if output_schema_file
        else None
    )
    if task_type == "structuredOutput" and output_schema is None:
        raise typer.BadParameter("--output-schema is required for structuredOutput experiments")

    extras: dict[str, Any] = {}
    if evaluation_only:
        if hyperparameters is not None and hyperparameters.get("epochs", 0) != 0:
            raise typer.BadParameter(
                "--evaluate-only conflicts with nonzero epochs in --hyperparameters"
            )
        extras["evaluation_only"] = True
    if messages_file is not None:
        extras["prompt_format"] = "multi_message"
        extras["initial_prompt_messages"] = cast(
            "list[PromptMessage]", _load_json_array(messages_file, option_name="--messages")
        )

    with get_client() as client:
        result = client.create_experiment(
            ai_component_id=component_id,
            target_model=target_model,
            task_type=cast("PromptExperimentTaskType", task_type),
            initial_prompt=initial_prompt,
            name=name,
            description=description,
            provider=provider,
            optimizer=optimizer,
            hyperparameters=hyperparameters,
            initial_prediction_model_schema=output_schema,
            **_guidance_options(guidance_file),
            **extras,
        )
        if start:
            client.start_experiment(result["id"])

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    console.print(f"[green]Experiment created:[/green] {result['id']}")
    console.print(f"  Name:   {result['name'] or '-'}")
    console.print(f"  Model:  {result['targetModel']}")
    console.print(f"  Status: {result['experimentStatus']}")
    if start:
        console.print("[green]Experiment scheduled.[/green]")


@experiments_app.command("create-tool-selection")
def create_tool_selection_experiment(
    component_id: Annotated[str, typer.Option("--component-id", "-c", help="AI component ID.")],
    tools_file: Annotated[
        Path,
        typer.Option(
            "--tools",
            exists=True,
            dir_okay=False,
            readable=True,
            help="JSON file containing tool definitions.",
        ),
    ],
    test_cases_file: Annotated[
        Path,
        typer.Option(
            "--test-cases",
            exists=True,
            dir_okay=False,
            readable=True,
            help="JSON file containing query and expected-tool cases.",
        ),
    ],
    target_model: Annotated[
        str | None, typer.Option("--target-model", "-m", help="Target model.")
    ] = None,
    tool_source: Annotated[str, typer.Option(help="Tool provenance: manual or mcp.")] = "manual",
    system_prompt: Annotated[
        str | None, typer.Option(help="Optional system prompt used during evaluation.")
    ] = None,
    optimize_system_prompt: bool = typer.Option(
        False,
        "--optimize-system-prompt",
        help="Optimize the system prompt together with tool descriptions.",
    ),
    epochs: Annotated[
        int | None, typer.Option(min=0, max=5, help="Optimization rounds; 0 evaluates only.")
    ] = None,
    evaluation_only: bool = typer.Option(
        False, "--evaluate-only", help="Evaluate supplied tools without rewriting them."
    ),
    train_split_ratio: Annotated[
        float | None,
        typer.Option(min=0.5, max=0.9, help="Held-out evaluation split."),
    ] = None,
    name: Annotated[str | None, typer.Option(help="Experiment name.")] = None,
    description: Annotated[str | None, typer.Option(help="Experiment description.")] = None,
    start: bool = typer.Option(False, "--start", help="Start the experiment after creating it."),
    guidance_file: Annotated[
        Path | None,
        typer.Option(
            "--guidance",
            exists=True,
            dir_okay=False,
            readable=True,
            help="JSON file with optimization rules.",
        ),
    ] = None,
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Create a complete tool-selection optimization experiment."""
    if evaluation_only:
        if epochs not in (None, 0):
            raise typer.BadParameter("--evaluate-only conflicts with nonzero --epochs")
        epochs = 0
    if tool_source not in {"manual", "mcp"}:
        raise typer.BadParameter("--tool-source must be manual or mcp")

    raw_tools = _load_json_array(tools_file, option_name="--tools")
    _validate_tool_selection_tools(raw_tools)
    tools = cast("list[ToolSelectionTool]", raw_tools)
    test_cases = cast(
        "list[ToolSelectionTestCase]",
        _load_json_array(test_cases_file, option_name="--test-cases"),
    )

    with get_client() as client:
        result = client.create_tool_selection_experiment(
            component_id,
            tools=tools,
            test_cases=test_cases,
            target_model=target_model,
            tool_source=cast("ToolSource", tool_source),
            system_prompt=system_prompt,
            optimize_system_prompt=optimize_system_prompt,
            **_guidance_options(guidance_file),
            epochs=epochs,
            train_split_ratio=train_split_ratio,
            name=name,
            description=description,
        )
        if start:
            client.start_experiment(result["id"])

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    console.print(f"[green]Tool-selection experiment created:[/green] {result['id']}")
    if start:
        console.print("[green]Experiment scheduled.[/green]")


@experiments_app.command("get")
def get_experiment(
    experiment_id: str = typer.Argument(help="Experiment ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Get experiment details."""
    with get_client() as client:
        result = client.get_experiment(experiment_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    st = result["experimentStatus"]
    st_style = {
        "completed": "green",
        "failed": "red",
        "running": "yellow",
        "canceling": "yellow",
        "canceled": "cyan",
    }.get(st, "dim")

    console.print(f"\n[bold]Experiment:[/bold] {result['name'] or '-'}")
    console.print(f"  ID:       {result['id']}")
    console.print(f"  Status:   [{st_style}]{st}[/{st_style}]")
    console.print(f"  Model:    {result['targetModel']}")
    console.print(f"  Provider: {result['provider']}")
    console.print(f"  Task:     {result['taskType']}")
    prompt_format = result.get("promptFormat", "single")
    prompt_messages = result.get("initialPromptMessages") or []
    if prompt_messages:
        if prompt_format == "multi_message":
            console.print("  Prompt:")
            for msg in prompt_messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                preview = content[:100]
                suffix = "..." if len(content) > 100 else ""
                console.print(f"    [cyan]{role}:[/cyan] {preview}{suffix}")
        else:
            content = prompt_messages[0].get("content", "")
            preview = content[:100]
            suffix = "..." if len(content) > 100 else ""
            console.print(f"  Prompt:   {preview}{suffix}")
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
    guidance_file: Annotated[
        Path | None,
        typer.Option(
            "--guidance",
            exists=True,
            dir_okay=False,
            readable=True,
            help="JSON file with optimization rules.",
        ),
    ] = None,
    epochs: Annotated[
        int | None, typer.Option(min=0, max=10, help="Optimization rounds; 0 evaluates only.")
    ] = None,
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Update a pending experiment."""
    updates: dict[str, Any] = {}
    if guidance_file is not None:
        updates["optimizationGuidance"] = _load_json_object(guidance_file, option_name="--guidance")
    if name is not None:
        updates["name"] = name
    if initial_prompt is not None:
        updates["promptFormat"] = "single"
        updates["initialPromptMessages"] = [{"role": "user", "content": initial_prompt}]
    if target_model is not None:
        updates["targetModel"] = target_model

    if not updates and epochs is None:
        err_console.print("No updates specified.", style="yellow")
        raise typer.Exit(1)

    with get_client() as client:
        if epochs is not None:
            current = client.get_experiment(experiment_id)
            updates["hyperparameters"] = {
                "numFewShots": 8,
                **current.get("hyperparameters", {}),
                "epochs": epochs,
            }
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

    with get_client() as client:
        client.delete_experiment(experiment_id)

    console.print("[green]Experiment deleted.[/green]")


@experiments_app.command("start")
def start_experiment(
    experiment_id: str = typer.Argument(help="Experiment ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Start a pending experiment in its configured evaluation or optimization mode."""
    with get_client() as client:
        result = client.start_experiment(experiment_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    console.print(f"[green]Experiment started.[/green] Status: {result['status']}")


@experiments_app.command("cancel")
def cancel_experiment(
    experiment_id: str = typer.Argument(help="Experiment ID."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
    output_json: bool = typer.Option(
        False, "--json", help="Output the cancellation response as JSON."
    ),
) -> None:
    """Request cancellation; running work stops at a safe boundary, not immediately."""
    if not yes:
        typer.confirm(f"Request cancellation of experiment {experiment_id}?", abort=True)

    with get_client() as client:
        result = client.cancel_experiment(experiment_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    if result["status"] == "canceled":
        console.print("[cyan]Experiment canceled.[/cyan]")
    else:
        console.print(
            "[yellow]Cancellation requested.[/yellow] Status: canceling. "
            "In-flight work may still finish and incur usage charges."
        )
        console.print(f"Check status: promptic experiments get {experiment_id}")


def _print_duplicated_experiment(result: Mapping[str, Any], *, source_id: str, kind: str) -> None:
    """Render the new experiment after a duplicate/continue call."""
    console.print(f"[green]Experiment {kind} from[/green] {source_id}")
    console.print(f"  New ID:  {result['id']}")
    console.print(f"  Name:    {result['name'] or '-'}")
    console.print(f"  Status:  {result['experimentStatus']}")
    console.print(f"  Model:   {result['targetModel']}")
    if result.get("modelUnavailable"):
        console.print(
            "[yellow]Warning:[/yellow] the source's target model is no longer "
            "available in this AI Application; update it before starting."
        )


@experiments_app.command("duplicate")
def duplicate_experiment(
    experiment_id: str = typer.Argument(help="Source experiment ID."),
    initial_prompt: Annotated[
        str | None,
        typer.Option(
            "--initial-prompt",
            "-p",
            help="Override the initial prompt of the new experiment.",
        ),
    ] = None,
    evaluation_only: bool = typer.Option(
        False, "--evaluate-only", help="Configure the new candidate to evaluate without optimizing."
    ),
    start: bool = typer.Option(
        False, "--start", help="Immediately start the new experiment after creating it."
    ),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Duplicate an experiment (clones dataset cases + evaluators).

    The new experiment lives on the same AI component and starts from the
    source's initial prompt (or ``--initial-prompt`` if provided). Use
    ``promptic experiments continue`` to seed from the source's best
    optimized prompt instead.
    """
    with get_client() as client:
        result = client.duplicate_experiment(experiment_id, initial_prompt_override=initial_prompt)
        if evaluation_only:
            hyperparameters = {"numFewShots": 8, **result.get("hyperparameters", {}), "epochs": 0}
            result = {
                **result,
                **client.update_experiment(result["id"], hyperparameters=hyperparameters),
            }
        if start:
            client.start_experiment(result["id"])

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    _print_duplicated_experiment(result, source_id=experiment_id, kind="duplicated")
    if start:
        console.print("[green]New experiment started.[/green]")


@experiments_app.command("continue")
def continue_experiment(
    experiment_id: str = typer.Argument(help="Source experiment ID."),
    start: bool = typer.Option(
        False, "--start", help="Immediately start the new experiment after creating it."
    ),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Continue from an experiment's best iteration.

    Creates a new experiment under the same AI component, copying
    dataset cases + evaluators from the source, and seeds the initial
    prompt from the source's best optimized iteration. Useful for chaining
    optimization runs after promising results.
    """
    with get_client() as client:
        result = client.duplicate_experiment(experiment_id, continue_from_optimized=True)
        if start:
            client.start_experiment(result["id"])

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    _print_duplicated_experiment(result, source_id=experiment_id, kind="continued")
    if start:
        console.print("[green]New experiment started.[/green]")


@experiments_app.command("cases")
def list_experiment_cases(
    experiment_id: str = typer.Argument(help="Experiment ID."),
    iteration_id: int = typer.Option(
        ..., "--iteration-id", min=1, help="Persisted iteration ID, not the iteration number."
    ),
    split: str = typer.Option("all", help="Case split: all, train or eval."),
    limit: Annotated[int, typer.Option(min=1, max=100)] = 25,
    cursor: str | None = typer.Option(None, help="Cursor from the previous page."),
    sort: str | None = typer.Option(None, help="Server-side sort, e.g. score or latencyMs."),
    descending: bool = typer.Option(False, "--descending", help="Sort highest values first."),
    output_json: bool = typer.Option(False, "--json", help="Output the complete page as JSON."),
) -> None:
    """Read a page of predictions and per-evaluator case results."""
    if split not in {"all", "train", "eval"}:
        raise typer.BadParameter("--split must be all, train or eval")
    with get_client() as client:
        result = client.list_experiment_cases(
            experiment_id,
            iteration_id=iteration_id,
            split=cast(Literal["all", "train", "eval"], split),
            limit=limit,
            cursor=cursor,
            sort=sort,
            descending=descending,
        )
    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return
    table = Table(title=f"Case results ({len(result['data'])} / {result['total']})")
    for name in ["ID", "Expected", "Prediction", "Score"]:
        table.add_column(name)
    for row in result["data"]:
        score = row.get("score")
        table.add_row(
            str(row["id"]),
            str(row.get("expected", "")),
            str(row.get("prediction", "")),
            f"{score:.3f}" if score is not None else "-",
        )
    console.print(table)
    if result.get("nextCursor"):
        console.print(f"Next cursor: {result['nextCursor']}")
