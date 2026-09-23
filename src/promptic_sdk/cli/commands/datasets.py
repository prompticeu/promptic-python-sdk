"""Datasets commands — create, list, and get."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from rich.console import Console
from rich.table import Table

from promptic_sdk.cli import get_client
from promptic_sdk.models import DatasetCaseCreate

datasets_app = typer.Typer(help="Manage agent datasets.")
cases_app = typer.Typer(help="Manage canonical cases in a dataset.")
datasets_app.add_typer(cases_app, name="cases")
console = Console()
err_console = Console(stderr=True)


def _load_json(path: Path) -> Any:
    """Load JSON from a CLI file option with a useful validation error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"Could not read valid JSON from {path}: {exc}") from exc


def _load_case_list(path: Path) -> list[dict[str, Any]]:
    """Load one case object or an array of case objects."""
    value = _load_json(path)
    if isinstance(value, dict):
        return [cast(dict[str, Any], value)]
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        return cast(list[dict[str, Any]], value)
    raise typer.BadParameter("--file must contain a case object or a non-empty array of cases")


def _load_case_update(path: Path) -> dict[str, Any]:
    """Load one case update object."""
    value = _load_json(path)
    if not isinstance(value, dict):
        raise typer.BadParameter("--file must contain one case object")
    return cast(dict[str, Any], value)


def _display_payload(value: object, preferred_key: str | None = None) -> str:
    """Render canonical JSON payloads compactly for the terminal."""
    if preferred_key and isinstance(value, dict):
        preferred = cast(dict[str, object], value).get(preferred_key)
        if isinstance(preferred, str):
            return preferred
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@datasets_app.command("create")
def create_dataset(
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
    name: str = typer.Option(..., "--name", help="Dataset name."),
    description: Annotated[str | None, typer.Option(help="Dataset description.")] = None,
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Create a new dataset from traces."""
    with get_client() as client:
        # First create the dataset
        result = client.create_dataset(
            component_id,
            name,
            description=description,
        )

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    console.print(f"[green]Dataset created:[/green] {result['name']}")
    console.print(f"  ID: {result['id']}")
    console.print(f"  Cases: {result['caseCount']}")
    console.print()
    console.print(
        "[dim]Tip: Add traces via SDK with "
        f"promptic_sdk.ai_component('...', dataset_id='{result['id']}')"
        " or use the API.[/dim]"
    )


@datasets_app.command("list")
def list_datasets(
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List datasets for an AI component."""
    with get_client() as client:
        result = client.list_datasets(component_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    datasets = result["data"]
    if not datasets:
        console.print("No datasets found.", style="dim")
        return

    table = Table(title=f"Datasets ({len(datasets)})")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Cases", justify="right")
    table.add_column("Created")

    for ds in datasets:
        table.add_row(
            ds["id"],
            ds["name"],
            str(ds["caseCount"]),
            ds["createdAt"],
        )

    console.print(table)


@datasets_app.command("get")
def get_dataset(
    dataset_id: str = typer.Argument(help="Dataset ID."),
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Get a dataset with its canonical cases."""
    with get_client() as client:
        result = client.get_dataset(component_id, dataset_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    console.print(f"\n[bold]Dataset:[/bold] {result['name']}")
    console.print(f"[bold]ID:[/bold] {result['id']}")
    console.print(f"[bold]Cases:[/bold] {result['caseCount']}")
    if result.get("description"):
        console.print(f"[bold]Description:[/bold] {result['description']}")

    cases = result.get("cases", [])
    if cases:
        console.print(f"\n[bold]Cases ({len(cases)}):[/bold]")
        case_table = Table()
        case_table.add_column("Input", max_width=40)
        case_table.add_column("Expected", max_width=40)

        for dataset_case in cases:
            case_table.add_row(
                _display_payload(dataset_case["inputPayload"], "input")[:80],
                _display_payload(dataset_case["expectedPayload"], "value")[:80],
            )
        console.print(case_table)


@datasets_app.command("delete")
def delete_dataset(
    dataset_id: str = typer.Argument(help="Dataset ID."),
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
) -> None:
    """Delete a dataset."""
    with get_client() as client:
        client.delete_dataset(component_id, dataset_id)
    console.print("[green]Dataset deleted.[/green]")


@cases_app.command("list")
def list_dataset_cases(
    dataset_id: str = typer.Argument(help="Dataset ID."),
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List canonical cases in a dataset."""
    with get_client() as client:
        result = client.list_dataset_cases(component_id, dataset_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    cases = result["data"]
    if not cases:
        console.print("No dataset cases found.", style="dim")
        return
    table = Table(title=f"Dataset cases ({len(cases)})")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Input", max_width=40)
    table.add_column("Expected", max_width=40)
    table.add_column("Split")
    for dataset_case in cases:
        table.add_row(
            str(dataset_case["id"]),
            _display_payload(dataset_case["inputPayload"], "input")[:80],
            _display_payload(dataset_case["expectedPayload"], "value")[:80],
            dataset_case.get("split") or "-",
        )
    console.print(table)


@cases_app.command("get")
def get_dataset_case(
    dataset_id: str = typer.Argument(help="Dataset ID."),
    case_id: int = typer.Argument(help="Dataset case ID."),
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Get one canonical dataset case."""
    with get_client() as client:
        result = client.get_dataset_case(component_id, dataset_id, case_id)
    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return
    console.print_json(data=result)


@cases_app.command("add")
def add_dataset_cases(
    dataset_id: str = typer.Argument(help="Dataset ID."),
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
    cases_file: Path = typer.Option(
        ...,
        "--file",
        exists=True,
        dir_okay=False,
        readable=True,
        help="JSON case object or array of case objects.",
    ),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Add one or more canonical cases from a JSON file."""
    cases = cast("list[DatasetCaseCreate]", _load_case_list(cases_file))
    with get_client() as client:
        result = client.create_dataset_cases(component_id, dataset_id, cases)
    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return
    console.print(f"[green]Added {len(result['data'])} dataset case(s).[/green]")


@cases_app.command("update")
def update_dataset_case(
    dataset_id: str = typer.Argument(help="Dataset ID."),
    case_id: int = typer.Argument(help="Dataset case ID."),
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
    case_file: Path = typer.Option(
        ...,
        "--file",
        exists=True,
        dir_okay=False,
        readable=True,
        help="JSON object containing the fields to update.",
    ),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Update one canonical dataset case from a JSON file."""
    updates = _load_case_update(case_file)
    with get_client() as client:
        result = client.update_dataset_case(component_id, dataset_id, case_id, **updates)
    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return
    console.print(f"[green]Dataset case {result['id']} updated.[/green]")


@cases_app.command("delete")
def delete_dataset_case(
    dataset_id: str = typer.Argument(help="Dataset ID."),
    case_id: int = typer.Argument(help="Dataset case ID."),
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation."),
) -> None:
    """Delete one canonical dataset case."""
    if not force:
        typer.confirm(f"Delete dataset case {case_id}?", abort=True)
    with get_client() as client:
        client.delete_dataset_case(component_id, dataset_id, case_id)
    console.print("[green]Dataset case deleted.[/green]")
