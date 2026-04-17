"""Observation commands."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from promptic_sdk.cli import get_client

observations_app = typer.Typer(help="Manage observations (training data).")
console = Console()
err_console = Console(stderr=True)


# Columns that are observation metadata, not input variables
_META_COLUMNS = {"expected", "split", "idx"}


def _load_from_file(path: Path) -> list[dict[str, Any]]:
    """Load observations from CSV or JSONL file.

    Non-meta columns (everything except expected, split, idx) are wrapped
    into the ``variables`` dict.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            raw_rows = [dict(row) for row in reader]
    elif suffix in (".jsonl", ".ndjson"):
        raw_rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_rows.append(json.loads(line))
    elif suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            raw_rows = data
        else:
            err_console.print("JSON file must contain an array of objects.", style="red")
            raise typer.Exit(1)
    else:
        err_console.print(
            f"Unsupported file format: {suffix}. Use .csv, .jsonl, or .json", style="red"
        )
        raise typer.Exit(1)

    # Wrap non-meta columns into variables
    observations: list[dict[str, Any]] = []
    for row in raw_rows:
        obs: dict[str, Any] = {}
        variables: dict[str, str] = {}
        for k, v in row.items():
            if k in _META_COLUMNS:
                obs[k] = v
            elif k == "variables" and isinstance(v, dict):
                variables.update(v)
            else:
                variables[k] = v
        obs["variables"] = variables
        observations.append(obs)
    return observations


def _format_variables(variables: Any) -> str:
    if not isinstance(variables, dict) or not variables:
        return "-"
    if set(variables) == {"input"}:
        return str(variables["input"])
    return json.dumps(variables, ensure_ascii=False)


@observations_app.command("list")
def list_observations(
    experiment_id: str = typer.Argument(help="Experiment ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List observations for an experiment."""
    with get_client() as client:
        result = client.list_observations(experiment_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    items = result["data"]
    if not items:
        console.print("No observations found.", style="dim")
        return

    table = Table(title=f"Observations ({len(items)})")
    table.add_column("ID", style="cyan")
    table.add_column("Idx", justify="right")
    table.add_column("Variables", max_width=40)
    table.add_column("Expected", max_width=40)
    table.add_column("Split")

    for o in items:
        variables = _format_variables(o.get("variables"))
        exp = o["expected"]
        table.add_row(
            str(o["id"]),
            str(o["idx"]),
            variables[:40] + ("..." if len(variables) > 40 else ""),
            exp[:40] + ("..." if len(exp) > 40 else ""),
            o["split"],
        )

    console.print(table)


@observations_app.command("add")
def add_observations(
    experiment_id: str = typer.Argument(help="Experiment ID."),
    from_file: Annotated[
        Path | None, typer.Option("--from-file", "-f", help="Import from CSV/JSONL/JSON file.")
    ] = None,
    input_text: Annotated[
        str | None, typer.Option("--input", "-i", help="Input text (single observation).")
    ] = None,
    expected_text: Annotated[
        str | None, typer.Option("--expected", "-e", help="Expected output (single observation).")
    ] = None,
    split: str = typer.Option("eval", help="Split type (train/eval)."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Add observations to an experiment.

    Use --from-file for bulk import (CSV with input,expected columns, or JSONL).
    Use --input and --expected for a single observation.
    """
    observations: list[dict[str, Any]] = []

    if from_file:
        if not from_file.exists():
            err_console.print(f"File not found: {from_file}", style="red")
            raise typer.Exit(1)
        observations = _load_from_file(from_file)
        console.print(f"Loaded {len(observations)} observations from {from_file}")
    elif input_text and expected_text:
        observations = [
            {"variables": {"input": input_text}, "expected": expected_text, "split": split}
        ]
    else:
        err_console.print(
            "Provide --from-file or both --input and --expected.",
            style="red",
        )
        raise typer.Exit(1)

    # Ensure split is set
    for obs in observations:
        if "split" not in obs:
            obs["split"] = split

    with get_client() as client:
        result = client.create_observations(experiment_id, observations)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    created = result["data"]
    console.print(f"[green]Created {len(created)} observation(s).[/green]")


@observations_app.command("delete")
def delete_observation(
    experiment_id: str = typer.Argument(help="Experiment ID."),
    observation_id: int = typer.Argument(help="Observation ID."),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation."),
) -> None:
    """Delete an observation."""
    if not force:
        typer.confirm(f"Delete observation {observation_id}?", abort=True)

    with get_client() as client:
        client.delete_observation(experiment_id, observation_id)

    console.print("[green]Observation deleted.[/green]")
