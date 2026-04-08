"""Evaluations commands — run, list, and get."""

from __future__ import annotations

import json
import sys
import time
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from promptic_sdk.models import AgentEvaluation

import typer
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table

from promptic_sdk.cli import get_client

evaluations_app = typer.Typer(help="Manage agent evaluations.")
console = Console()
err_console = Console(stderr=True)


SEVERITY_STYLES = {
    "high": "red bold",
    "medium": "yellow",
    "low": "dim",
}


@evaluations_app.command("run")
def run_evaluation(
    component_id: str = typer.Argument(help="AI Component ID."),
    dataset_id: str = typer.Option(..., "--dataset", help="Dataset ID to evaluate."),
    name: Annotated[str | None, typer.Option(help="Evaluation name.")] = None,
    run_id: str = typer.Option(..., "--run", help="Run ID to associate."),
    no_wait: bool = typer.Option(False, "--no-wait", help="Don't wait for completion."),
    timeout: float = typer.Option(300, "--timeout", help="Max seconds to wait for completion."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Start an evaluation and display results."""
    with get_client() as client:
        result = client.create_evaluation(component_id, dataset_id, name=name, run_id=run_id)
        evaluation_id = result["id"]

        if result["status"] not in ("completed", "failed") and not no_wait:
            result = _poll_with_spinner(client, component_id, evaluation_id, timeout=timeout)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    status = result["status"]
    if status == "completed" and result.get("results"):
        _print_insights(result)
    elif status == "failed":
        err_console.print("[red]Evaluation failed.[/red]")
        raise typer.Exit(1)
    else:
        console.print(f"Evaluation {evaluation_id} — status: {status}")
        console.print("[dim]Run 'promptic evaluations get' to check results.[/dim]")


def _poll_with_spinner(
    client: object,
    component_id: str,
    evaluation_id: str,
    *,
    timeout: float = 300,
    poll_interval: float = 2,
) -> AgentEvaluation:
    """Poll evaluation status with a Rich spinner until terminal state."""
    from promptic_sdk.client import PrompticClient

    assert isinstance(client, PrompticClient)  # noqa: S101
    deadline = time.monotonic() + timeout
    with Live(Spinner("dots", text="Evaluating..."), console=err_console, transient=True):
        while True:
            result = client.get_evaluation(component_id, evaluation_id)
            if result["status"] in ("completed", "failed"):
                return result
            if time.monotonic() >= deadline:
                err_console.print(
                    f"[yellow]Timed out after {timeout}s — evaluation still running.[/yellow]"
                )
                return result
            time.sleep(poll_interval)
    return result  # unreachable but keeps type checkers happy


@evaluations_app.command("list")
def list_evaluations(
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List evaluations for an AI component."""
    with get_client() as client:
        result = client.list_evaluations(component_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    evaluations = result["data"]
    if not evaluations:
        console.print("No evaluations found.", style="dim")
        return

    table = Table(title=f"Evaluations ({len(evaluations)})")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Insights", justify="right")
    table.add_column("Created")

    for ev in evaluations:
        status_style = (
            "green"
            if ev["status"] == "completed"
            else "red"
            if ev["status"] == "failed"
            else "yellow"
        )
        insight_count = "-"
        results = ev.get("results")
        if results and results.get("insights"):
            insight_count = str(len(results["insights"]))

        table.add_row(
            ev["id"],
            ev.get("name") or "-",
            f"[{status_style}]{ev['status']}[/{status_style}]",
            insight_count,
            ev["createdAt"],
        )

    console.print(table)


@evaluations_app.command("get")
def get_evaluation(
    evaluation_id: str = typer.Argument(help="Evaluation ID."),
    component_id: str = typer.Option(..., "--component", help="AI Component ID."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Get evaluation results."""
    with get_client() as client:
        result = client.get_evaluation(component_id, evaluation_id)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    if result["status"] == "completed" and result.get("results"):
        _print_insights(result)
    else:
        console.print(f"Evaluation {result['id']} — status: {result['status']}")


def _print_insights(evaluation: AgentEvaluation) -> None:
    """Pretty-print evaluation insights."""
    results = evaluation["results"]
    assert results is not None  # noqa: S101
    meta = results["meta"]
    insights = results["insights"]

    console.print(f"\n[bold]Evaluation:[/bold] {evaluation.get('name') or evaluation['id']}")
    console.print(
        f"  Analyzed {meta['totalRuns']} runs | "
        f"{meta['totalTokens']} tokens | "
        f"${meta['totalCostUsd']:.4f} cost | "
        f"{meta['errorRate']:.1%} error rate"
    )

    if not insights:
        console.print("\n[green]No issues found.[/green]")
        return

    console.print(f"\n[bold]Insights ({len(insights)}):[/bold]\n")
    for i, insight in enumerate(insights, 1):
        severity = insight["severity"]
        style = SEVERITY_STYLES.get(severity, "")
        tag = f"[{style}]{severity.upper()}[/{style}]"
        freq = f"{insight['frequency']:.0%}" if insight["frequency"] > 0 else ""
        freq_str = f" ({freq} of runs)" if freq else ""

        console.print(f"  {i}. {tag} {insight['title']}{freq_str}")
        console.print(f"     {insight['description']}")
        if insight.get("suggestedFix"):
            console.print(f"     [dim]Fix: {insight['suggestedFix']}[/dim]")
        console.print()
