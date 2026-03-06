"""Traces commands — list, get, and stats."""

from __future__ import annotations

import json
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from promptic_sdk.cli.config import load_config
from promptic_sdk.client import PromenticClient

traces_app = typer.Typer(help="Manage and inspect traces.")
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


@traces_app.command("list")
def list_traces(
    limit: int = typer.Option(20, help="Number of traces to show."),
    offset: int = typer.Option(0, help="Offset for pagination."),
    status: Annotated[str | None, typer.Option(help="Filter by status (ok/error).")] = None,
    start_after: Annotated[
        str | None, typer.Option(help="Only traces after this ISO timestamp.")
    ] = None,
    start_before: Annotated[
        str | None, typer.Option(help="Only traces before this ISO timestamp.")
    ] = None,
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List recent traces."""
    with _get_client() as client:
        result = client.list_traces(
            limit=limit,
            offset=offset,
            status=status,
            start_after=start_after,
            start_before=start_before,
        )

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    traces = result["traces"]
    total = result["total"]

    if not traces:
        console.print("No traces found.", style="dim")
        return

    table = Table(title=f"Traces ({len(traces)} of {total})")
    table.add_column("Trace ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Duration", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Time")

    for t in traces:
        status_style = "red" if t["status"] == "error" else "green"
        duration = f"{t['durationMs']}ms" if t["durationMs"] is not None else "-"
        tokens = str(t["totalTokens"] or "-")
        cost = f"${t['totalCostUsd']:.4f}" if t["totalCostUsd"] else "-"

        table.add_row(
            t["traceId"],
            t["name"] or "-",
            f"[{status_style}]{t['status']}[/{status_style}]",
            duration,
            tokens,
            cost,
            t["startTime"],
        )

    console.print(table)


@traces_app.command("get")
def get_trace(
    trace_id: str = typer.Argument(help="OTel trace ID to look up."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Get a single trace with all spans and events."""
    with _get_client() as client:
        try:
            result = client.get_trace(trace_id)
        except Exception as e:
            err_console.print(f"Error: {e}", style="red")
            raise typer.Exit(1) from e

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    # Human-readable trace summary
    console.print(f"\n[bold]Trace:[/bold] {result['traceId']}")
    console.print(f"[bold]Name:[/bold] {result['name'] or '-'}")
    status = result["status"]
    status_style = "red" if status == "error" else "green"
    console.print(f"[bold]Status:[/bold] [{status_style}]{status}[/{status_style}]")
    if result["durationMs"] is not None:
        console.print(f"[bold]Duration:[/bold] {result['durationMs']}ms")
    if result["totalTokens"]:
        console.print(f"[bold]Tokens:[/bold] {result['totalTokens']}")
    if result["totalCostUsd"]:
        console.print(f"[bold]Cost:[/bold] ${result['totalCostUsd']:.4f}")

    # Spans table
    spans = result["spans"]
    if spans:
        console.print(f"\n[bold]Spans ({len(spans)}):[/bold]")
        span_table = Table()
        span_table.add_column("Name")
        span_table.add_column("Kind")
        span_table.add_column("Status")
        span_table.add_column("Duration", justify="right")
        span_table.add_column("Model")
        span_table.add_column("Tokens", justify="right")

        for s in spans:
            s_status = s["status"]
            s_style = "red" if s_status == "error" else "green"
            duration = f"{s['durationMs']}ms" if s["durationMs"] is not None else "-"
            tokens = str(s["totalTokens"] or "-")

            span_table.add_row(
                s["name"],
                s["kind"],
                f"[{s_style}]{s_status}[/{s_style}]",
                duration,
                s["model"] or "-",
                tokens,
            )

        console.print(span_table)


@traces_app.command("stats")
def stats(
    days_back: int = typer.Option(30, help="Number of days to look back."),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Show aggregated tracing statistics."""
    with _get_client() as client:
        result = client.get_stats(days_back=days_back)

    if output_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    console.print(f"\n[bold]Tracing Stats (last {days_back} days):[/bold]")
    console.print(f"  Total traces:  {result['totalTraces']}")
    console.print(f"  Total tokens:  {result['totalTokens']}")
    console.print(f"  Total cost:    ${result['totalCostUsd']:.4f}")
    error_rate = result["errorRate"]
    rate_style = "red" if error_rate > 0.1 else "green"
    console.print(f"  Error rate:    [{rate_style}]{error_rate:.1%}[/{rate_style}]")
